"""
Scheme-agnostic check for the minimal-pair result.

The valence score e = sum_c P(c) w_c can hide identity effects in 3-class and
5-class models: if swapping the person term moves probability out of 'neutral'
equally into 'positive' and 'negative', e does not change although the model's
output did. This script repeats analysis A with the total-variation distance
between the full output distributions,

    TV(x, x') = 0.5 * sum_c |P(c | x) - P(c | x')|,

which responds to any change in the prediction, on the same minimal-pair probe
and the same two-way (template x person-pair) bootstrap.

If 3-class models still show filler/emotion < 1 under TV, their reversal is
real; if TV flips it back above 1, the reversal was an artefact of e.

Outputs (./results_minimal_pair/):
    tv_<model>.csv     per-pair TV distances and neutral-class probability
    tv_<model>.json    per-model result (a rerun skips models that have one)
    tv_summary.json    per-model filler/emotion, bare/emotion ratios under TV

Run:
    python distribution_shift_check.py
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import argparse
import gc
import json
import os
import traceback
import zlib

import numpy as np
import pandas as pd
import torch

import minimal_pair_experiment as M

RESULTS_DIR = M.RESULTS_DIR


@torch.inference_mode()
def probabilities(name: str, sentences: list[str], batch_size: int = 128) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Full output distribution per sentence, plus the valence weight of each class."""
    tok, model = M.load_classifier(name)
    model.eval()
    uniq = sorted(set(sentences), key=len)
    out = {}
    for start in range(0, len(uniq), batch_size):
        chunk = uniq[start : start + batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=64, return_tensors="pt")
        probs = torch.softmax(model(**enc).logits.double(), dim=-1).numpy()
        out.update(zip(chunk, probs))
    return out, M.valence_weights(model.config.id2label)


def audit_one(name: str, records: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    pairs = records[records["anchor"]].copy()
    probs, weights = probabilities(name, pairs["sentence"].tolist() + pairs["swapped"].tolist())
    a = np.stack([probs[s] for s in pairs["sentence"]])
    b = np.stack([probs[s] for s in pairs["swapped"]])
    pairs["tv"] = 0.5 * np.abs(a - b).sum(axis=1)
    # probability on the neutral / middle class (valence weight 0.5); 0 for binary models
    pairs["neutral_mass"] = a[:, np.isclose(weights, 0.5)].sum(axis=1)
    mats = M.cell_means(pairs, "tv")
    rng = np.random.default_rng([M.SEED, zlib.crc32(name.encode())])
    row = {"model": name, "status": "ok",
           "mean_tv": {c: round(float(m.mean()), 4) for c, m in mats.items()},
           "filler_vs_emotion": M.two_way_bootstrap(mats, "filler", "emotion", rng),
           "bare_vs_emotion": M.two_way_bootstrap(mats, "bare", "emotion", rng),
           "neutral_mass": {c: round(float(g.mean()), 4)
                            for c, g in pairs.groupby("condition")["neutral_mass"]}}
    return row, pairs[["base_id", "pair_id", "condition", "form", "emotion_word", "sentence",
                       "swapped", "tv", "neutral_mass"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=M.DEFAULT_MODELS)
    args = ap.parse_args()
    records = M.build_records()
    rows = []
    for name in args.models:
        tag = name.split("/")[-1]
        saved = os.path.join(RESULTS_DIR, f"tv_{tag}.json")
        if os.path.exists(saved):
            with open(saved, encoding="utf-8") as f:
                rows.append(json.load(f))
            print(f"--- {name}: loaded saved result", flush=True)
            continue
        print(f"--- {name}", flush=True)
        try:
            row, pairs = audit_one(name, records)
            pairs.to_csv(os.path.join(RESULTS_DIR, f"tv_{tag}.csv"), index=False)
            with open(saved, "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)
            fe = row["filler_vs_emotion"]
            print(f"    [tv] bare={row['mean_tv']['bare']:.4f} filler={row['mean_tv']['filler']:.4f} "
                  f"emotion={row['mean_tv']['emotion']:.4f}  filler/emotion={fe['ratio']:.2f} "
                  f"CI{fe['ratio_ci']} ({fe['n_templates_higher']}/{fe['n_templates']} templates)  "
                  f"neutral mass filler={row['neutral_mass']['filler']:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    SKIPPED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            row = {"model": name, "status": f"failed: {type(e).__name__}: {e}"}
        rows.append(row)
        gc.collect()
    with open(os.path.join(RESULTS_DIR, "tv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Results in ./{RESULTS_DIR}/tv_summary.json")


if __name__ == "__main__":
    main()
