"""
Score every probe sentence once per model and cache the full output
distributions, so later analyses (label-flip rates, uncertainty-targeted
probes) never need to re-run a model.

Covers the minimal-pair probe (minimal_pair_experiment.build_records: sentences
and their counterfactuals) and the EEC + EEC-Neutral corpus
(eec_reproduction). Each model's cache is one compressed .npz:

    sentences  (n,)    the scored strings
    probs      (n, k)  softmax output per sentence
    weights    (k,)    valence weight of each class (minimal_pair_experiment.valence_weights)

A rerun skips models that already have a cache file.

Outputs: ./results_probs/<model>.npz

Run:
    python prob_cache.py --models distilbert-base-uncased-finetuned-sst-2-english
    python prob_cache.py            # every model in the fleet
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import argparse
import gc
import os
import traceback

import numpy as np
import pandas as pd
import torch

import eec_reproduction as E
import minimal_pair_experiment as M

CACHE_DIR = "results_probs"


def probe_sentences() -> list[str]:
    records = M.build_records()
    eec = E.load_eec()
    corpus = pd.concat([eec[["Sentence"]], E.build_eec_neutral(eec)[["Sentence"]]])
    return sorted(set(records["sentence"]) | set(records["swapped"]) | set(corpus["Sentence"]))


@torch.inference_mode()
def score(name: str, sentences: list[str], batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    tok, model = M.load_classifier(name)
    model.eval()
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    probs = np.empty((len(sentences), model.config.num_labels), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        enc = tok([sentences[i] for i in idx], padding=True, truncation=True, max_length=64,
                  return_tensors="pt")
        probs[idx] = torch.softmax(model(**enc).logits.double(), dim=-1).numpy()
    return probs, M.valence_weights(model.config.id2label)


def load(name: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """sentence -> probability vector, and the class valence weights."""
    z = np.load(os.path.join(CACHE_DIR, f"{name.split('/')[-1]}.npz"), allow_pickle=False)
    return dict(zip(z["sentences"].tolist(), z["probs"])), z["weights"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=M.DEFAULT_MODELS)
    args = ap.parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True)
    sentences = probe_sentences()
    print(f"{len(sentences)} unique sentences", flush=True)
    for name in args.models:
        out = os.path.join(CACHE_DIR, f"{name.split('/')[-1]}.npz")
        if os.path.exists(out):
            print(f"--- {name}: cached", flush=True)
            continue
        print(f"--- {name}", flush=True)
        try:
            probs, weights = score(name, sentences)
            np.savez_compressed(out, sentences=np.array(sentences), probs=probs, weights=weights)
            print(f"    saved {probs.shape}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    SKIPPED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        gc.collect()


if __name__ == "__main__":
    main()
