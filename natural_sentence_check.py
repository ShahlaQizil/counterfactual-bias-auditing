"""
Natural-sentence check: does the template finding survive outside templates?

Real review sentences that mention a person through gendered words are given a
full gender flip (he<->she, his<->her, son<->daughter, ...), so every sentence
yields a male and a female version that differ only in gender. The analysis
plan (PLAN below) is written to results_natural/analysis_plan.json with its
SHA-256 by the `build` stage, before any model is scored.

Source: unlabelled split of the IMDB review corpus (Maas et al., 2011), 4,000
reviews at 40 evenly spaced offsets, fetched through the Hugging Face rows API.
Sentiment checkpoints are fine-tuned on labelled data, so this split is not in
their training sets. The sentences are written to data/ (not redistributed);
this script rebuilds them deterministically.

Stages:
    python natural_sentence_check.py build     # sentences + frozen plan
    python natural_sentence_check.py score     # per-model probabilities (cached)
    python natural_sentence_check.py analyse   # tests in PLAN

Outputs: ./data/natural_sentences.csv, ./results_natural/
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import argparse
import gc
import hashlib
import json
import os
import re
import time
import traceback
import urllib.request
import zlib

import numpy as np
import pandas as pd
from scipy import stats

import minimal_pair_experiment as M

RESULTS_DIR = "results_natural"
DATA_PATH = os.path.join("data", "natural_sentences.csv")
NRC_PATH = os.path.join("data", "NRC-AffectIntensity-Lexicon.txt")
SEED = M.SEED
N_BOOT = 2000
PER_STRATUM = 500
LLMS = ["Qwen/Qwen2.5-0.5B-Instruct@2cls", "Qwen/Qwen2.5-0.5B-Instruct@3cls",
        "Qwen/Qwen2.5-1.5B-Instruct@2cls", "Qwen/Qwen2.5-1.5B-Instruct@3cls"]
SYSTEMS = M.DEFAULT_MODELS + LLMS

PLAN = {
    "purpose": "Test whether the template findings hold on natural sentences.",
    "data": ("IMDB unsupervised split, 100 reviews at each of 40 evenly spaced offsets; "
             "<br /> removed; regex sentence split; keep 6-35 words, exactly one gender "
             "(only male or only female terms from GENDER_PAIRS, none from UNSWAPPABLE), no capitalised "
             "word except 'I' after the first, first word attested in lower case (removes names "
             "and titles), deduplicated."),
    "swap": "Full gender flip via GENDER_PAIRS; her->his before a content word, else him; "
            "his->hers when not followed by a word. Each sentence gives a male and a female version.",
    "strata": ("Affect load = max NRC Affect Intensity over the sentence's words (gender terms "
               "excluded). low = no lexicon word; high = max intensity >= 0.5. "
               f"{PER_STRATUM} sentences sampled per stratum (seed {SEED})."),
    "measures": "Sensitivity = TV distance between the two versions' output distributions; "
                "uncertainty u = 1 - max P, averaged over the two versions; "
                "direction = logit(e_female) - logit(e_male).",
    "systems": "18 classifiers + 4 LLM configurations (original prompts) = 22.",
    "H1": "Within each system, Spearman(sensitivity, u) over sentences is positive. "
          "Report count positive and count significant after Holm over 22.",
    "H2": "Across systems, log(sens_low/sens_high) correlates positively with "
          "log(u_low/u_high) (Spearman, 22 systems). Direction agreement is counted among "
          "systems whose sensitivity ratio is significant (sentence bootstrap, 2000, Holm over 22).",
    "H3": "Directional bias per system and stratum: mean direction, sentence-bootstrap "
          "percentile CI, Bonferroni alpha = 0.05/44. Count systems whose verdict "
          "(detected/not, sign) differs between strata.",
    "revisions": ("v2 (before any model was scored): after reading sample swaps, added actor/"
                  "actress pairs, dropped sentences with unswappable gendered words or an initial "
                  "word never seen in lower case (names), extended the function-word list for her."),
    "falsification": "H2 is not supported if rho <= 0.3 or direction agreement < 2/3; "
                     "whatever the outcome, it is reported.",
}

GENDER_PAIRS = [
    ("he", "she"), ("him", "her"), ("his", "her"), ("himself", "herself"),
    ("man", "woman"), ("men", "women"), ("boy", "girl"), ("boys", "girls"),
    ("son", "daughter"), ("sons", "daughters"), ("father", "mother"), ("fathers", "mothers"),
    ("dad", "mom"), ("brother", "sister"), ("brothers", "sisters"),
    ("husband", "wife"), ("husbands", "wives"), ("boyfriend", "girlfriend"),
    ("boyfriends", "girlfriends"), ("uncle", "aunt"), ("nephew", "niece"),
    ("king", "queen"), ("prince", "princess"), ("gentleman", "lady"), ("gentlemen", "ladies"),
    ("grandfather", "grandmother"), ("grandson", "granddaughter"),
    ("male", "female"), ("males", "females"), ("actor", "actress"), ("actors", "actresses"),
    ("hers", "his"),
]
# gendered words without a clean counterpart: sentences containing them are dropped
UNSWAPPABLE = set("guy guys dude dudes waitress waitresses mr mrs ms miss madam sir lad lads gal gals".split())
MALE = {m for m, f in GENDER_PAIRS if m != "hers"}
FEMALE = {f for m, f in GENDER_PAIRS if f != "his"} | {"hers"}
TO_FEMALE = {m: f for m, f in GENDER_PAIRS if m != "hers"}
TO_MALE = {f: m for m, f in GENDER_PAIRS if f not in ("her", "his")} | {"hers": "his"}
# after "her", these words mean it is an object pronoun (-> him), not a possessive (-> his)
FUNCTION_WORDS = set("""a an the to and or but in on at for with as from by of into off up down out
over again back so too that this when while because if than then is was be been away around about
through after before like very just all more also even not no anymore now here there well what how
why where who which until much enough either though alone yet most less ever any some once twice
instead anyway onto upon against without within during since along across behind toward towards
under among are were has had have will would can could should may might must does did do""".split())
TOKEN = re.compile(r"[A-Za-z]+|[^A-Za-z]+")


### Building the sentences
def fetch_reviews() -> list[str]:
    cache = os.path.join("data", "imdb_unsupervised_sample.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    url = "https://datasets-server.huggingface.co/rows?dataset=stanfordnlp/imdb&config=plain_text&split=unsupervised&offset={}&length=100"
    texts = []
    for off in np.linspace(0, 50000 - 100, 40).astype(int):
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url.format(off), timeout=60) as r:
                    texts += [row["row"]["text"] for row in json.load(r)["rows"]]
                break
            except Exception:  # noqa: BLE001
                time.sleep(20 * (attempt + 1))
        else:
            raise RuntimeError(f"could not fetch offset {off}")
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(texts, f)
    return texts


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)]


def words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", s)


def gender_of(s: str) -> str | None:
    ws = {w.lower() for w in words(s)}
    m, f = bool(ws & MALE), bool(ws & FEMALE)
    return "male" if m and not f else "female" if f and not m else None


def keep(s: str, lower_vocab: set[str]) -> bool:
    ws = words(s)
    if not 6 <= len(ws) <= 35 or not s[-1:] in ".!?" or '"' in s:
        return False
    if any(w[0].isupper() and w != "I" for w in ws[1:]):
        return False
    # a sentence-initial word never seen in lower case is a name or title
    if ws[0].lower() not in lower_vocab or {w.lower() for w in ws} & UNSWAPPABLE:
        return False
    return gender_of(s) is not None


def flip(s: str, to: str) -> str:
    toks = TOKEN.findall(s)
    out = []
    for i, t in enumerate(toks):
        low = t.lower()
        nxt = next((x.lower() for x in toks[i + 1:] if x.strip()), "")
        nxt_is_word = nxt.isalpha()
        if to == "female" and low in TO_FEMALE:
            new = TO_FEMALE[low]
            if low == "his" and not nxt_is_word:
                new = "hers"
        elif to == "male" and (low in TO_MALE or low == "her"):
            if low == "her":
                new = "his" if nxt_is_word and nxt not in FUNCTION_WORDS else "him"
            else:
                new = TO_MALE[low]
        else:
            out.append(t)
            continue
        out.append(new.capitalize() if t[0].isupper() else new)
    return "".join(out)


def load_nrc() -> dict[str, float]:
    best: dict[str, float] = {}
    with open(NRC_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                try:
                    v = float(parts[1])
                except ValueError:
                    continue
                best[parts[0]] = max(best.get(parts[0], 0.0), v)
    return best


def build() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plan = json.dumps({"plan": PLAN, "gender_pairs": GENDER_PAIRS}, indent=2, sort_keys=True)
    with open(os.path.join(RESULTS_DIR, "analysis_plan.json"), "w", encoding="utf-8") as f:
        f.write(plan)
    sha = hashlib.sha256(plan.encode()).hexdigest()
    with open(os.path.join(RESULTS_DIR, "analysis_plan.sha256"), "w", encoding="utf-8") as f:
        f.write(f"{sha}  analysis_plan.json  frozen {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"plan frozen, sha256 {sha}")

    reviews = fetch_reviews()
    nrc = load_nrc()
    lower_vocab = {w for t in reviews for w in re.findall(r"(?<![A-Za-z])[a-z]+(?![A-Za-z])", t)}
    rows, seen = [], set()
    for rid, text in enumerate(reviews):
        for s in split_sentences(text):
            if s in seen or not keep(s, lower_vocab):
                continue
            seen.add(s)
            g = gender_of(s)
            lex = [nrc[w.lower()] for w in words(s)
                   if w.lower() in nrc and w.lower() not in MALE | FEMALE]
            aff = max(lex, default=0.0)
            male, female = (s, flip(s, "female")) if g == "male" else (flip(s, "male"), s)
            rows.append({"review": rid, "original_gender": g, "affect": aff,
                         "n_affect_words": len(lex), "male": male, "female": female})
    df = pd.DataFrame(rows)
    df["stratum"] = np.where(df["n_affect_words"] == 0, "low",
                             np.where(df["affect"] >= 0.5, "high", "mid"))
    print(f"{len(reviews)} reviews -> {len(df)} candidate sentences; strata "
          f"{df['stratum'].value_counts().to_dict()}")
    rng = np.random.default_rng(SEED)
    parts = []
    for st in ("low", "high"):
        pool = df[df["stratum"] == st]
        parts.append(pool.iloc[rng.permutation(len(pool))[:PER_STRATUM]])
    out = pd.concat(parts).reset_index(drop=True)
    out.insert(0, "sid", range(len(out)))
    out.to_csv(DATA_PATH, index=False)
    print(f"saved {len(out)} sentences to {DATA_PATH}: "
          f"{out.groupby(['stratum', 'original_gender']).size().to_dict()}")


### Scoring
def score() -> None:
    import distribution_shift_check as D
    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    sentences = df["male"].tolist() + df["female"].tolist()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name in SYSTEMS:
        out = os.path.join(RESULTS_DIR, f"probs_{name.split('/')[-1]}.npz")
        if os.path.exists(out):
            print(f"--- {name}: cached", flush=True)
            continue
        print(f"--- {name}", flush=True)
        t0 = time.time()
        try:
            probs, weights = D.probabilities(name, sentences, batch_size=32 if "@" in name else 128)
            np.savez_compressed(out, male=np.stack([probs[s] for s in df["male"]]),
                                female=np.stack([probs[s] for s in df["female"]]), weights=weights)
            print(f"    saved ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    SKIPPED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        gc.collect()


### Analysis
def logit_e(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    e = np.clip(p @ w, 1e-12, 1 - 1e-12)
    return np.log(e) - np.log1p(-e)


def holm(p: list[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def boot_p(ratio_samples: np.ndarray) -> float:
    """Two-sided bootstrap p for log ratio = 0."""
    lr = np.log(ratio_samples)
    return float(min(1.0, 2 * min((lr <= 0).mean(), (lr >= 0).mean()) + 1 / len(lr)))


def analyse() -> None:
    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    low = (df["stratum"] == "low").to_numpy()
    rows, per_sentence = [], []
    for name in SYSTEMS:
        path = os.path.join(RESULTS_DIR, f"probs_{name.split('/')[-1]}.npz")
        if not os.path.exists(path):
            print(f"missing {name}")
            continue
        z = np.load(path)
        a, b, w = z["male"], z["female"], z["weights"]
        tv = 0.5 * np.abs(a - b).sum(axis=1)
        u = 1 - 0.5 * (a.max(axis=1) + b.max(axis=1))
        d = logit_e(b, w) - logit_e(a, w)
        rho, p_rho = stats.spearmanr(tv, u)
        rng = np.random.default_rng([SEED, zlib.crc32(name.encode())])
        il, ih = np.flatnonzero(low), np.flatnonzero(~low)
        bl = rng.choice(il, (N_BOOT, len(il)))
        bh = rng.choice(ih, (N_BOOT, len(ih)))
        sens_ratio_b = tv[bl].mean(1) / tv[bh].mean(1)
        row = {"model": name,
               "rho_within": float(rho), "p_within": float(p_rho),
               "tv_low": float(tv[low].mean()), "tv_high": float(tv[~low].mean()),
               "u_low": float(u[low].mean()), "u_high": float(u[~low].mean()),
               "sens_ratio": float(tv[low].mean() / tv[~low].mean()),
               "sens_ratio_ci": [float(x) for x in np.percentile(sens_ratio_b, [2.5, 97.5])],
               "p_sens_ratio": boot_p(sens_ratio_b),
               "unc_ratio": float(u[low].mean() / u[~low].mean())}
        for st, idx, bi in (("low", il, bl), ("high", ih, bh)):
            m = d[bi].mean(1)
            lo, hi = np.percentile(m, [100 * 0.025 / 44, 100 * (1 - 0.025 / 44)])
            row[f"dir_{st}"] = float(d[idx].mean())
            row[f"dir_{st}_ci"] = [float(lo), float(hi)]
            row[f"verdict_{st}"] = "F>M" if lo > 0 else "M>F" if hi < 0 else "none"
        rows.append(row)
        per_sentence.append(pd.DataFrame({"model": name, "sid": df["sid"], "stratum": df["stratum"],
                                          "tv": tv, "u": u, "direction": d}))
    res = pd.DataFrame(rows)
    res["p_within_holm"] = holm(res["p_within"])
    res["p_sens_ratio_holm"] = holm(res["p_sens_ratio"])
    res["verdicts_differ"] = res["verdict_low"] != res["verdict_high"]
    res.to_csv(os.path.join(RESULTS_DIR, "natural_summary.csv"), index=False)
    pd.concat(per_sentence).to_csv(os.path.join(RESULTS_DIR, "natural_per_sentence.csv"), index=False)

    ls, lu = np.log(res["sens_ratio"]), np.log(res["unc_ratio"])
    rho_x, p_x = stats.spearmanr(ls, lu)
    sig = res["p_sens_ratio_holm"] < 0.05
    agree = int((np.sign(ls[sig]) == np.sign(lu[sig])).sum())
    clf = ~res["model"].str.contains("@")
    rho_c, p_c = stats.spearmanr(ls[clf], lu[clf])
    summary = {
        "n_sentences": int(len(df)), "n_low": int(low.sum()), "n_high": int((~low).sum()),
        "n_systems": int(len(res)),
        "H1_positive": int((res["rho_within"] > 0).sum()),
        "H1_significant_holm": int(((res["rho_within"] > 0) & (res["p_within_holm"] < 0.05)).sum()),
        "H1_median_rho": float(res["rho_within"].median()),
        "H2_rho": float(rho_x), "H2_p": float(p_x),
        "H2_rho_classifiers_only": float(rho_c), "H2_p_classifiers_only": float(p_c),
        "H2_n_sig_ratio": int(sig.sum()), "H2_direction_agree": agree,
        "sens_ratio_gt1": int((res["sens_ratio"] > 1).sum()),
        "sens_ratio_gt1_sig": int((sig & (res["sens_ratio"] > 1)).sum()),
        "sens_ratio_lt1_sig": int((sig & (res["sens_ratio"] < 1)).sum()),
        "H3_detected_low": int((res["verdict_low"] != "none").sum()),
        "H3_detected_high": int((res["verdict_high"] != "none").sum()),
        "H3_verdicts_differ": int(res["verdicts_differ"].sum()),
        "H3_sign_conflicts": int(((res["verdict_low"] == "F>M") & (res["verdict_high"] == "M>F")
                                  | (res["verdict_low"] == "M>F") & (res["verdict_high"] == "F>M")).sum()),
    }
    with open(os.path.join(RESULTS_DIR, "natural_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.set_option("display.width", 220)
    print(res[["model", "rho_within", "p_within_holm", "tv_low", "tv_high", "sens_ratio",
               "p_sens_ratio_holm", "unc_ratio", "verdict_low", "verdict_high"]]
          .assign(model=res["model"].str.split("/").str[-1]).round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "score", "analyse"])
    args = ap.parse_args()
    {"build": build, "score": score, "analyse": analyse}[args.stage]()


if __name__ == "__main__":
    main()
