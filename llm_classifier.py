"""
Instruction-tuned LLMs as zero-shot sentiment classifiers, behind the same
interface as a Hugging Face sequence classifier, so the audit code
(ValenceModel, eec_reproduction.audit_one, targeted_probe.audit_one) runs on
them unchanged.

Model names take the form  "<hf-repo>@<scheme>", e.g.
    Qwen/Qwen2.5-1.5B-Instruct@2cls   answer options: positive / negative
    Qwen/Qwen2.5-1.5B-Instruct@3cls   answer options: positive / neutral / negative
Running the same weights with both prompts isolates the effect of the label
scheme, which the classifier fleet confounds with model and training data.

Scoring: the sentence is wrapped in the model's chat template with an
instruction to answer with one word; the class logits are the next-token logits
of each label word at the start of the answer (log-sum-exp over the lower-case
and capitalised forms). A softmax over these gives the label distribution.

Speed: the instruction prefix, identical for every sentence, is run once and its
key/value cache reused for each batch (verified to match full-prompt logits to
~3e-5). Every sentence's class logits are stored in an SQLite cache per model
variant (results_llm_cache/<variant>.sqlite), so a sentence is scored once
across all scripts and runs, and an interrupted run resumes.
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import copy
import os
import sqlite3
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = "results_llm_cache"
LABELS = {"2cls": ["negative", "positive"], "3cls": ["negative", "neutral", "positive"]}
# Prompt wordings. "2cls"/"3cls" are the original prompts; the "-p2"/"-p3"
# variants were written to differ in structure, wording and answer-option order,
# and were frozen (prompts_frozen.json, with SHA-256) before any variant was run.
# All variants are reported, whatever their results.
PROMPT = {
    "2cls": ("What is the sentiment of the following sentence? Answer with exactly one "
             "word: positive or negative.\n\nSentence:\n{s}"),
    "3cls": ("What is the sentiment of the following sentence? Answer with exactly one "
             "word: positive, neutral, or negative.\n\nSentence:\n{s}"),
    "2cls-p2": ("Classify the text below as negative or positive.\n\nText: {s}\n\n"
                "Reply with a single word."),
    "3cls-p2": ("Classify the text below as negative, neutral, or positive.\n\nText: {s}\n\n"
                "Reply with a single word."),
    "2cls-p3": ("Read this sentence and decide whether its overall tone is positive or "
                "negative:\n\"{s}\"\nGive only the one-word label."),
    "3cls-p3": ("Read this sentence and decide whether its overall tone is neutral, positive, "
                "or negative:\n\"{s}\"\nGive only the one-word label."),
}


def is_llm(name: str) -> bool:
    return "@" in name


class LLMTokenizer:
    """Stand-in for a tokenizer call: passes the raw sentences through, because
    the classifier needs them as cache keys and builds its own prompts."""

    def __call__(self, sentences, **_):
        return {"sentences": list(sentences)}


class LLMClassifier:
    def __init__(self, name: str, batch_size: int = 32):
        repo, variant = name.split("@")
        self.repo, self.variant = repo, variant
        self.scheme = variant.split("-")[0]  # "3cls-p2" -> label scheme "3cls"
        labels = LABELS[self.scheme]
        self.config = SimpleNamespace(id2label=dict(enumerate(labels)), num_labels=len(labels))
        self.batch_size = batch_size
        self._lm = None
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._db = sqlite3.connect(os.path.join(CACHE_DIR, name.split("/")[-1] + ".sqlite"))
        self._db.execute("CREATE TABLE IF NOT EXISTS logits (sentence TEXT PRIMARY KEY, v BLOB)")

    def eval(self):
        return self

    def _load(self) -> None:
        if self._lm is not None:
            return
        torch.set_num_threads(os.cpu_count() or 1)
        self.tok = AutoTokenizer.from_pretrained(self.repo)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self._lm = AutoModelForCausalLM.from_pretrained(self.repo, dtype=torch.float32).eval()
        full = self.tok.apply_chat_template(
            [{"role": "user", "content": PROMPT[self.variant].format(s="<<S>>")}],
            tokenize=False, add_generation_prompt=True)
        self._prefix, self._suffix = full.split("<<S>>")
        self._prefix_ids = self.tok(self._prefix, return_tensors="pt",
                                    add_special_tokens=False)["input_ids"]
        with torch.inference_mode():
            self._prefix_cache = self._lm(self._prefix_ids, use_cache=True).past_key_values
        self._label_ids = []
        for word in LABELS[self.scheme]:
            ids = {self.tok.encode(v, add_special_tokens=False)[0] for v in (word, word.capitalize())}
            self._label_ids.append(sorted(ids))
        firsts = [i for ids in self._label_ids for i in ids]
        assert len(firsts) == len(set(firsts)), f"label first-tokens collide: {self._label_ids}"

    @torch.inference_mode()
    def _score(self, sentences: list[str]) -> np.ndarray:
        self._load()
        out = np.empty((len(sentences), self.config.num_labels), dtype=np.float64)
        for start in range(0, len(sentences), self.batch_size):
            chunk = sentences[start : start + self.batch_size]
            enc = self.tok([s + self._suffix for s in chunk], return_tensors="pt", padding=True,
                           add_special_tokens=False)
            cache = copy.deepcopy(self._prefix_cache)
            cache.batch_repeat_interleave(len(chunk))
            mask = torch.cat([torch.ones(len(chunk), self._prefix_ids.shape[1], dtype=enc["attention_mask"].dtype),
                              enc["attention_mask"]], dim=1)
            logits = self._lm(input_ids=enc["input_ids"], attention_mask=mask,
                              past_key_values=cache, use_cache=True).logits
            last = logits[torch.arange(len(chunk)), enc["attention_mask"].sum(1) - 1].double()
            for j, ids in enumerate(self._label_ids):
                out[start : start + len(chunk), j] = torch.logsumexp(last[:, ids], dim=-1).numpy()
        return out

    def __call__(self, sentences: list[str], **_):
        found = {}
        for i in range(0, len(sentences), 500):  # stay under SQLite's variable limit
            part = sentences[i : i + 500]
            cur = self._db.execute(
                f"SELECT sentence, v FROM logits WHERE sentence IN ({','.join('?' * len(part))})", part)
            found.update((s, np.frombuffer(v, dtype=np.float64)) for s, v in cur)
        missing = sorted({s for s in sentences if s not in found}, key=len)
        for i in range(0, len(missing), 256):  # commit as we go so an interrupted run resumes
            part = missing[i : i + 256]
            vals = self._score(part)
            self._db.executemany("INSERT OR REPLACE INTO logits VALUES (?, ?)",
                                 [(s, v.tobytes()) for s, v in zip(part, vals)])
            self._db.commit()
            found.update(zip(part, vals))
        return SimpleNamespace(logits=torch.tensor(np.stack([found[s] for s in sentences])))


def load_llm(name: str):
    return LLMTokenizer(), LLMClassifier(name)
