"""
Salience-competition experiment asks whether identity bias becomes stronger when there are no clear sentiment words in a sentence.

This work builds on the findings of Kiritchenko & Mohammad (2018).
K&M found a surprising result, but they did not explain why it happens: In the four EEC sentence templates that contain no emotion word,
Changing the identity term causes a larger change in the model's sentiment score than in sentences that contain emotion words.
However, far fewer models are identified as significantly biased. 
They explain this by saying that the neutral templates have lower statistical power.
This script tests one possible explanation for why identity has a larger effect in neutral sentences:

    A sentiment classifier spends its decision mass on the most predictive
    tokens. When a strong emotion word is present, it dominates and the identity
    term is CROWDED OUT. On neutral text there is no competitor, so the identity
    term drives the prediction. Bias is therefore not constant -- an identity
    term's influence is INVERSELY related to the salience of the legitimate
    sentiment evidence in the same sentence.

Two testable predictions:
    H1 (competition):  the identity term's attribution SHARE falls as
                       the emotion word becomes stronger.
    H2 (neutral peak): identity attribution share is highest on neutral
                       sentences (no emotion word at all).

Method contribution: This work also improves the attribution method by fixing a problem in occlusion-based auditing.

Occlusion measures a token's importance by DELETING it ("Lakisha feels angry." ->
"feels angry."), which is ungrammatical and off-distribution.  The EEC hands us a
grammatical alternative for free: attribute the person term by SUBSTITUTING its
matched counterfactual pair ("this woman" <-> "this man") instead of deleting
it. We compute both and show they rank terms differently, which means published
occlusion-based bias explanations may be partly a grammaticality artifact.

Outputs (./results_salience/)
    attributions.csv        per-sentence attributions and shares
    summary.json            H1 regression, H2 contrast, occlusion-vs-substitution divergence
    eec_neutral_preview.txt  expanded neutral-template corpus (restores audit power)

Run:
    python salience_experiment.py

Dependencies: numpy, pandas, torch, transformers  (same as the original audit).
No scipy required -- Spearman/OLS/bootstrap are implemented in numpy.
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import itertools
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
RESULTS_DIR = "results_salience"
SEED = 20260713



### Model wrapper (This class works like the original audit code, but it remembers previous results so the 
### experiment can efficiently process tens of thousands of perturbed sentences on a CPU.)
def resolve_polarity(id2label: dict[int, str]) -> tuple[int, int]:
    """This function finds which model output represents the positive class and which represents the negative class. 
    It works with public sentiment models: binary POSITIVE/NEGATIVE,
    3-class negative/neutral/positive, 5-class star ratings, and LABEL_0/1."""
    items = {i: str(name).lower() for i, name in id2label.items()}
    pos = neg = None
    for i, name in items.items():
        if name.startswith("pos") or "positive" in name:
            pos = i
        if name.startswith("neg") or "negative" in name:
            neg = i
    if pos is None or neg is None:
        # star ratings: '1 star' .. '5 stars' -> extremes are neg/pos
        star = {i: int(name[0]) for i, name in items.items() if name[:1].isdigit()}
        if star:
            pos = max(star, key=star.get)
            neg = min(star, key=star.get)
        else:  # LABEL_0/LABEL_1 fallback: assume 0=neg, 1=pos
            neg, pos = 0, max(items)
    return pos, neg


class SentimentModel:
    """Thin wrapper exposing P(positive) and a logit margin, with a cache.
    Handles binary, 3-class, and 5-star sentiment models via resolve_polarity."""

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.batch_size = batch_size
        self.positive_index, self.negative_index = resolve_polarity(self.model.config.id2label)
        self._cache_prob: dict[str, float] = {}
        self._cache_margin: dict[str, float] = {}
        self._cache_maxprob: dict[str, float] = {}

    @torch.no_grad()
    def _forward(self, sentences: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each sentence, this function returns three numbers:
        1. P(positive): The probability that the sentence is positive.
        2. logit margin = (logit_pos - logit_neg): The difference between the positive and negative logits.
        3. max class probability (Max prob is a class-count-agnostic saturation):
        The model's highest probability, which shows how confident the model is, regardless of which class it chooses.
        """
        prob = np.empty(len(sentences), dtype=np.float64)
        margin = np.empty(len(sentences), dtype=np.float64)
        maxprob = np.empty(len(sentences), dtype=np.float64)
        for start in range(0, len(sentences), self.batch_size):
            chunk = sentences[start : start + self.batch_size]
            enc = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=64, return_tensors="pt"
            )
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            sl = slice(start, start + len(chunk))
            prob[sl] = probs[:, self.positive_index].cpu().numpy()
            neg = self.negative_index if self.negative_index is not None else 0
            margin[sl] = (logits[:, self.positive_index] - logits[:, neg]).cpu().numpy()
            maxprob[sl] = probs.max(dim=-1).values.cpu().numpy()
        return prob, margin, maxprob

    def _ensure(self, sentences: list[str]) -> None:
        missing = [s for s in sentences if s not in self._cache_prob]
        if missing:
            uniq = list(dict.fromkeys(missing))
            prob, margin, maxprob = self._forward(uniq)
            self._cache_prob.update(zip(uniq, prob))
            self._cache_margin.update(zip(uniq, margin))
            self._cache_maxprob.update(zip(uniq, maxprob))

    def max_prob(self, sentences: list[str]) -> np.ndarray:
        """Max class probability per sentence (saturation diagnostic)."""
        self._ensure(sentences)
        return np.array([self._cache_maxprob[s] for s in sentences], dtype=np.float64)

    def positive_score(self, sentences: list[str]) -> np.ndarray:
        """P(positive) for each sentence, memoised. Used for reporting."""
        self._ensure(sentences)
        return np.array([self._cache_prob[s] for s in sentences], dtype=np.float64)

    def margin_score(self, sentences: list[str]) -> np.ndarray:
        """Logit margin (logit_pos - logit_neg), memoised. Used for attribution:
        unbounded, so it does not saturate to the 0/1 rails the way softmax does."""
        self._ensure(sentences)
        return np.array([self._cache_margin[s] for s in sentences], dtype=np.float64)



### EEC vocabulary (subset chosen so every person term is an inflection-free,
### 1:1-swappable (one-to-one swappable) noun phrase or first name. This avoids problems like
### "she" ↔ "he" or "a" ↔ "an".)

# Gendered noun-phrase pairs: that can be swapped directly to create
# counterfactual sentence pairs while keeping the sentence grammatically correct.
GENDER_PAIRS = [
    ("this woman", "this man"),
    ("my sister", "my brother"),
    ("my mother", "my father"),
    ("my daughter", "my son"),
    ("my aunt", "my uncle"),
]

# First names taken from K&M Table 2 (based on Caliskan et al., 2017).
# These names are used to measure the attribution-share regression. 
# Since occlusion removes one word at a time, it does not need a matching name.
# The matching names of different races below are used when comparing predictions after swapping names
# (substitution-divergence analysis).
NAMES = {
    ("female", "european_american"): ["Amanda", "Betsy", "Ellen", "Katie", "Nancy"],
    ("female", "african_american"): ["Ebony", "Jasmine", "Lakisha", "Latoya", "Tanisha"],
    ("male", "european_american"): ["Adam", "Alan", "Frank", "Jack", "Roger"],
    ("male", "african_american"): ["Alonzo", "Darnell", "Jerome", "Lamar", "Leroy"],
}

# Emotion state words from K&M Table 4 with aAPPROXIMATE affect-intensity scores between 0 and 1.
# These scores are placeholders (temporary values) and only keep the correct order of emotion strength
# (for example: annoyed < angry < furious). 
# For the final published version, replace them with the official NRC Affect Intensity Lexicon scores
# (Mohammad, 2018). 
# The code will automatically use whatever scores are stored here.
EMOTION_INTENSITY = {
    # anger
    "irritated": 0.52, "annoyed": 0.55, "angry": 0.68, "furious": 0.87, "enraged": 0.92,
    # fear
    "discouraged": 0.50, "anxious": 0.63, "scared": 0.73, "fearful": 0.77, "terrified": 0.93,
    # joy
    "glad": 0.58, "relieved": 0.62, "happy": 0.73, "excited": 0.80, "ecstatic": 0.92,
    # sadness
    "disappointed": 0.55, "sad": 0.68, "miserable": 0.78, "depressed": 0.80, "devastated": 0.90,
}
NRC_PATH = os.path.join("data", "NRC-AffectIntensity-Lexicon.txt")
# Words the NRC lexicon does not score (filled in by load_nrc_intensities).
# Analyses that depend on intensity must exclude these rather than fall back to
# the placeholder values above.
NRC_MISSING: set[str] = set()


def load_nrc_intensities(path: str = NRC_PATH, allow_placeholders: bool = False) -> dict[str, float]:
    """
Return the official NRC Affect Intensity Lexicon (v0.5) scores for the 20 EEC
emotion words. The file is tab-separated: word<TAB>score<TAB>emotion, after a
free-text header.

A word can appear under several emotions (e.g. 'depressed' is scored for anger,
fear AND sadness), so the lookup is keyed on (word, EEC emotion category) --
an earlier version keyed on the word alone and silently took whichever row came
last. Words the lexicon does not contain are recorded in NRC_MISSING.

If the file is absent this raises, unless allow_placeholders=True: an
intensity analysis must never run on the placeholder values without saying so.
"""
    if not os.path.exists(path):
        if allow_placeholders:
            print(f"WARNING: {path} not found -- using PLACEHOLDER intensities.")
            NRC_MISSING.update(EMOTION_INTENSITY)
            return dict(EMOTION_INTENSITY)
        raise FileNotFoundError(
            f"NRC Affect Intensity Lexicon not found at {path}. Download it from "
            "http://saifmohammad.com/WebDocs/NRC-AffectIntensity-Lexicon.txt")
    scored: dict[tuple[str, str], float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            try:
                scored[(parts[0], parts[2])] = float(parts[1])
            except ValueError:
                continue  # header lines
    vals = dict(EMOTION_INTENSITY)
    NRC_MISSING.clear()
    for word in vals:
        key = (word, EMOTION_OF[word])
        if key in scored:
            vals[word] = scored[key]
        else:
            NRC_MISSING.add(word)
    if NRC_MISSING:
        print(f"NRC lexicon has no score for {sorted(NRC_MISSING)} "
              "-- exclude them from intensity analyses.")
    return vals


EMOTION_OF = {
    **{w: "anger" for w in ["irritated", "annoyed", "angry", "furious", "enraged"]},
    **{w: "fear" for w in ["discouraged", "anxious", "scared", "fearful", "terrified"]},
    **{w: "joy" for w in ["glad", "relieved", "happy", "excited", "ecstatic"]},
    **{w: "sadness" for w in ["disappointed", "sad", "miserable", "depressed", "devastated"]},
}

# Templates where the person term is inflection-free in every position, , so it can be swapped without changing the grammar.
EMOTION_TEMPLATES = [
    "{person} feels {emotion}.",
    "the situation makes {person} feel {emotion}.",
    "{person} made me feel {emotion}.",
]
NEUTRAL_TEMPLATES = [
    "i saw {person} in the market.",
    "i talked to {person} yesterday.",
    "{person} goes to the school in our neighborhood.",
    "{person} has two children.",
]



### Attribution primitives (span-aware: a multi-word person term is ONE token)
def _norm(tok: str) -> str:
    """Normalise a token for keyed lookup: drop surrounding punctuation so
    'terrified.' and 'terrified' match. (This mismatch silently zeroed the
    emotion-word attribution in an earlier version.)"""
    return tok.strip(".,!?;:'\"").lower()


def atomic_tokens(sentence: str, person_phrase: str) -> list[str]:
    """Split into whitespace tokens but keep the (possibly multi-word) person
    phrase as a single atomic token, so occlusion removes it as a unit."""
    marker = "\x00"
    tmp = sentence.replace(person_phrase, marker)
    return [person_phrase if tok == marker else tok for tok in tmp.split()]


def occlusion_attributions(
    model: SentimentModel, sentence: str, person_phrase: str
) -> tuple[list[tuple[str, float]], float, float]:
    """attribution(token) = margin(full) - margin(full without token), signed.
    Attribution is computed on the logit margin (unsaturated). Returns a list of
    (normalised_token, attribution), plus the base probability and base margin."""
    toks = atomic_tokens(sentence, person_phrase)
    base_prob = float(model.positive_score([sentence])[0])
    base_margin = float(model.margin_score([sentence])[0])
    variants = [" ".join(toks[:i] + toks[i + 1 :]) for i in range(len(toks))]
    margins = model.margin_score(variants)
    attrs = [(_norm(toks[i]), base_margin - float(margins[i])) for i in range(len(toks))]
    return attrs, base_prob, base_margin


def lookup(attrs: list[tuple[str, float]], target: str) -> float:
    """Signed attribution of the first token matching `target` (normalised)."""
    t = _norm(target)
    for tok, val in attrs:
        if tok == t:
            return val
    return 0.0


def substitution_attribution(
    model: SentimentModel, sentence: str, person_phrase: str, counterpart: str
) -> float:
    """Grammatical counterfactual attribution of the person term on the margin:
    margin(sentence) - margin(sentence with person swapped for its pair)."""
    swapped = sentence.replace(person_phrase, counterpart)
    base = float(model.margin_score([sentence])[0])
    alt = float(model.margin_score([swapped])[0])
    return base - alt



### numpy-only statistics (no scipy dependency)
###Spearman's rank correlation coefficient (ρ) without using SciPy
def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a))
    # average ties
    #If two numbers are equal, they should receive the average rank
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rank(np.asarray(x, float)), _rank(np.asarray(y, float))
    return float(np.corrcoef(rx, ry)[0, 1])


def ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, Pearson r)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return float(slope), float(intercept), r


def permutation_pvalue(x: np.ndarray, y: np.ndarray, n: int, rng) -> float:
    """Two-sided p-value for the OLS slope under label permutation."""
    obs, _, _ = ols_slope(x, y)
    y = np.asarray(y, float)
    count = 0
    for _ in range(n):
        perm, _, _ = ols_slope(x, rng.permutation(y))
        if abs(perm) >= abs(obs):
            count += 1
    return (count + 1) / (n + 1)


def bootstrap_slope_ci(x, y, n, rng, alpha=0.05) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    slopes = np.empty(n)
    idx = np.arange(len(x))
    for i in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        slopes[i], _, _ = ols_slope(x[s], y[s])
    lo, hi = np.percentile(slopes, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)



### Build the probe (test) corpus
@dataclass
class PersonTerm:
    surface: str
    kind: str              # "noun_phrase" | "name"
    gender: str
    race: str | None = None
    gender_pair: str | None = None   # counterpart across gender (noun phrases) # the opposite-gender version
    race_pair: str | None = None     # counterpart across race (names) # This field stores the counterpart used for counterfactual substitution.



def build_person_terms() -> list[PersonTerm]:
    terms: list[PersonTerm] = []
    for fem, masc in GENDER_PAIRS:
        terms.append(PersonTerm(fem, "noun_phrase", "female", gender_pair=masc))
        terms.append(PersonTerm(masc, "noun_phrase", "male", gender_pair=fem))
    # Names: pair each name with the same-position name of the other race (same
    # gender) to enable a race-axis substitution attribution.
    for gender in ("female", "male"):
        ea = NAMES[(gender, "european_american")]
        aa = NAMES[(gender, "african_american")]
        for e, a in zip(ea, aa):
            terms.append(PersonTerm(e, "name", gender, "european_american", race_pair=a))
            terms.append(PersonTerm(a, "name", gender, "african_american", race_pair=e))
    return terms


def build_records(persons: list[PersonTerm]) -> pd.DataFrame:
    rows = []
    for p in persons:
        for tmpl in EMOTION_TEMPLATES:
            for word in EMOTION_INTENSITY:
                rows.append(
                    dict(
                        sentence=tmpl.format(person=p.surface, emotion=word),
                        template=tmpl, person=p.surface, kind=p.kind,
                        gender=p.gender, race=p.race,
                        gender_pair=p.gender_pair, race_pair=p.race_pair,
                        emotion_word=word, emotion=EMOTION_OF[word],
                        intensity=EMOTION_INTENSITY[word], has_emotion=True,
                    )
                )
        for tmpl in NEUTRAL_TEMPLATES:
            rows.append(
                dict(
                    sentence=tmpl.format(person=p.surface),
                    template=tmpl, person=p.surface, kind=p.kind,
                    gender=p.gender, race=p.race,
                    gender_pair=p.gender_pair, race_pair=p.race_pair,
                    emotion_word=None, emotion="neutral",
                    intensity=0.0, has_emotion=False,
                )
            )
    return pd.DataFrame(rows)



### Core measurement
def measure(model: SentimentModel, df: pd.DataFrame) -> pd.DataFrame:
    attr_person, attr_emotion, ias_all, comp_ratio, base_scores, attr_person_sub = (
        [], [], [], [], [], []
    )
    for row in df.itertuples(index=False):
        attrs, base, _ = occlusion_attributions(model, row.sentence, row.person)
        a_person = lookup(attrs, row.person)
        total = sum(abs(v) for _, v in attrs) or 1e-12
        attr_person.append(a_person)
        base_scores.append(base)
        ias_all.append(abs(a_person) / total)

        if row.has_emotion:
            a_emo = lookup(attrs, row.emotion_word)
            attr_emotion.append(a_emo)
            comp_ratio.append(abs(a_person) / (abs(a_person) + abs(a_emo) + 1e-12))
        else:
            attr_emotion.append(np.nan)
            comp_ratio.append(np.nan)

        # Grammatical substitution attribution for the person term (gender axis
        # for noun phrases, race axis for names).
        counterpart = row.gender_pair if row.kind == "noun_phrase" else row.race_pair
        attr_person_sub.append(
            substitution_attribution(model, row.sentence, row.person, counterpart)
            if counterpart else np.nan
        )

    out = df.copy()
    out["base_score"] = base_scores
    out["attr_person_occ"] = attr_person
    out["attr_emotion_occ"] = attr_emotion
    out["ias_all"] = ias_all              # identity attribution share (all tokens)
    out["comp_ratio"] = comp_ratio        # person vs emotion word only
    out["attr_person_sub"] = attr_person_sub
    return out



### EEC-Neutral: restore the statistical power K&M lose on neutral text
EEC_NEUTRAL_TEMPLATES = NEUTRAL_TEMPLATES + [
    "i met {person} at the station.",
    "{person} lives on the next street.",
    "{person} works at the bakery downtown.",
    "we visited {person} last weekend.",
    "{person} parked the car outside.",
    "i handed the keys to {person}.",
    "{person} ordered a coffee.",
    "the letter was addressed to {person}.",
    "{person} walked to the office.",
    "i sat next to {person} on the train.",
    "{person} bought a newspaper.",
    "{person} answered the phone.",
]


def write_eec_neutral_preview(persons: list[PersonTerm], path: str) -> int:
    lines, n = [], 0
    for tmpl in EEC_NEUTRAL_TEMPLATES:
        for p in persons:
            lines.append(tmpl.format(person=p.surface))
            n += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return n



### Orchestration
def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    EMOTION_INTENSITY.update(load_nrc_intensities())

    print(f"Loading classifier under audit: {MODEL_NAME}")
    model = SentimentModel()

    persons = build_person_terms()
    df = build_records(persons)
    print(f"Scoring {len(df)} probe sentences (+ occlusion/substitution variants)...")
    res = measure(model, df)
    res.to_csv(os.path.join(RESULTS_DIR, "attributions.csv"), index=False)

    # ---- H1: identity attribution share vs emotion intensity (emotion rows) ----
    emo = res[res["has_emotion"]].dropna(subset=["comp_ratio"])
    slope, intercept, r = ols_slope(emo["intensity"], emo["comp_ratio"])
    p_perm = permutation_pvalue(emo["intensity"].values, emo["comp_ratio"].values, 2000, rng)
    ci_lo, ci_hi = bootstrap_slope_ci(emo["intensity"].values, emo["comp_ratio"].values, 2000, rng)

    # ---- H2: neutral vs emotional identity attribution share (all tokens) ----
    ias_neutral = res.loc[~res["has_emotion"], "ias_all"]
    ias_emotion = res.loc[res["has_emotion"], "ias_all"]

    def bootstrap_mean_diff(a, b, n, rng):
        a, b = np.asarray(a), np.asarray(b)
        diffs = np.empty(n)
        for i in range(n):
            diffs[i] = (rng.choice(a, len(a), replace=True).mean()
                        - rng.choice(b, len(b), replace=True).mean())
        return float(diffs.mean()), tuple(np.percentile(diffs, [2.5, 97.5]))

    h2_diff, (h2_lo, h2_hi) = bootstrap_mean_diff(ias_neutral, ias_emotion, 2000, rng)

    # ---- Occlusion vs substitution divergence (idea 5) ----
    both = res.dropna(subset=["attr_person_sub"])
    rho = spearman(both["attr_person_occ"].abs().values, both["attr_person_sub"].abs().values)

    # ---- EEC-Neutral corpus ----
    n_neutral = write_eec_neutral_preview(
        persons, os.path.join(RESULTS_DIR, "eec_neutral_preview.txt")
    )
    n_pairs_full = 3 * len(EMOTION_INTENSITY) * len(persons)  # rough paired-comparison budget
    n_pairs_neutral = len(EEC_NEUTRAL_TEMPLATES) * len(persons)

    summary = {
        "model_audited": MODEL_NAME,
        "n_probe_sentences": int(len(df)),
        "H1_competition": {
            "hypothesis": "comp_ratio (identity share vs emotion word) DECREASES with emotion intensity",
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "pearson_r": round(r, 4),
            "permutation_p_2000": round(p_perm, 5),
            "bootstrap_slope_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
            "supported": bool(slope < 0 and ci_hi < 0),
        },
        "H2_neutral_peak": {
            "hypothesis": "identity attribution share (all tokens) is HIGHER on neutral sentences",
            "mean_ias_neutral": round(float(ias_neutral.mean()), 4),
            "mean_ias_emotional": round(float(ias_emotion.mean()), 4),
            "bootstrap_mean_diff_95ci": [round(h2_lo, 4), round(h2_hi, 4)],
            "supported": bool(h2_lo > 0),
        },
        "occlusion_vs_substitution": {
            "note": "rank agreement between deletion-based and grammatical-substitution "
                    "attribution of the identity term; gap in mean magnitude shows deletion "
                    "and substitution disagree on HOW MUCH the term matters even when they "
                    "agree on ranking (idea 5)",
            "spearman_rho_abs_attr": round(rho, 4),
            "mean_abs_attr_occlusion": round(float(both["attr_person_occ"].abs().mean()), 4),
            "mean_abs_attr_substitution": round(float(both["attr_person_sub"].abs().mean()), 4),
        },
        "eec_neutral": {
            "n_templates": len(EEC_NEUTRAL_TEMPLATES),
            "n_generated_sentences": n_neutral,
            "paired_comparisons_emotional_probe": n_pairs_full,
            "paired_comparisons_neutral_probe": n_pairs_neutral,
            "note": "K&M flag only 44 neutral paired comparisons vs 1,584 emotional; "
                    "this expansion restores audit power on the regime where bias is largest",
        },
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---- Console report ----
    print("\n=== H1  identity share vs emotion intensity (crowding-out) ===")
    print(f"  slope={slope:+.4f}  r={r:+.3f}  perm-p={p_perm:.4f}  "
          f"95%CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  supported={summary['H1_competition']['supported']}")
    print("=== H2  neutral vs emotional identity attribution share ===")
    print(f"  neutral={ias_neutral.mean():.3f}  emotional={ias_emotion.mean():.3f}  "
          f"diff95CI=[{h2_lo:+.4f},{h2_hi:+.4f}]  supported={summary['H2_neutral_peak']['supported']}")
    print("=== idea 5  occlusion vs substitution ===")
    print(f"  Spearman rho(|occ|,|sub|)={rho:+.3f}  "
          f"|occ|mean={both['attr_person_occ'].abs().mean():.4f}  "
          f"|sub|mean={both['attr_person_sub'].abs().mean():.4f}")
    print(f"\nEEC-Neutral: {n_neutral} sentences from {len(EEC_NEUTRAL_TEMPLATES)} templates "
          f"({n_pairs_neutral} paired comparisons vs K&M's 44)")
    print(f"\nDone. Results in ./{RESULTS_DIR}/")


if __name__ == "__main__":
    main()
