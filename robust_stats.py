"""
All three sensitivity measures for every model, with multiplicity correction.

For each model and each measure -- |Delta_e| (valence, [0,1] scale),
|Delta_logit| (logit of valence) and TV distance -- this recomputes the
filler/emotion ratio with the two-way (template x person-pair) bootstrap and a
two-sided bootstrap p-value for ratio = 1, then applies Holm's correction across
the 18 models separately per measure. A model is classed 'effect' (ratio > 1),
'reversed' (ratio < 1) or 'null' only by the Holm-adjusted p-value, so no
measure is chosen after seeing the results.

Inputs:  results_minimal_pair/<model>.csv, results_minimal_pair/tv_<model>.csv
Outputs: results_minimal_pair/robust_stats.csv

Run:
    python robust_stats.py
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import glob
import os
import json
import zlib

import numpy as np
import pandas as pd

import minimal_pair_experiment as M

N_BOOT = 4000  # two-sided p resolution 1/2000, below the smallest Holm threshold 0.05/18
OUT = os.path.join(M.RESULTS_DIR, "robust_stats.csv")


def bootstrap_ratio(mats: dict[str, np.ndarray], rng) -> tuple[float, float, float, float]:
    """filler/emotion ratio, 95% CI and two-sided bootstrap p-value for ratio = 1."""
    a, b = mats["filler"], mats["emotion"]
    nt, npair = a.shape
    r = np.empty(N_BOOT)
    for i in range(N_BOOT):
        t = rng.integers(0, nt, nt)
        p = rng.integers(0, npair, npair)
        r[i] = a[np.ix_(t, p)].mean() / b[np.ix_(t, p)].mean()
    p_two = min(1.0, 2 * min((r <= 1).mean(), (r >= 1).mean()))
    p_two = max(p_two, 1 / N_BOOT)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(a.mean() / b.mean()), float(lo), float(hi), float(p_two)


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    rows = []
    for tv_path in sorted(glob.glob(os.path.join(M.RESULTS_DIR, "tv_*.csv"))):
        tag = os.path.basename(tv_path)[3:-4]
        main_csv = os.path.join(M.RESULTS_DIR, f"{tag}.csv")
        if not os.path.exists(main_csv):
            continue
        res = pd.read_csv(main_csv)
        pairs = res[res["anchor"]]
        tv = pd.read_csv(tv_path)
        k = len(json.load(open(os.path.join(M.RESULTS_DIR, f"{tag}.json")))["valence_weights"])
        rng = np.random.default_rng([M.SEED, zlib.crc32(tag.encode())])
        for measure, df, col in (("valence_e", pairs, "sub_e"), ("logit", pairs, "sub_logit"),
                                 ("tv", tv, "tv")):
            ratio, lo, hi, p = bootstrap_ratio(M.cell_means(df, col), rng)
            rows.append({"model": tag, "k": k, "measure": measure, "ratio": ratio,
                         "lo": lo, "hi": hi, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby("measure")["p"].transform(lambda s: holm(s.values))
    out["category"] = np.where(out["p_holm"] >= 0.05, "null",
                               np.where(out["ratio"] > 1, "effect", "reversed"))
    out.to_csv(OUT, index=False)
    print(out.groupby(["measure", "category"]).size().unstack(fill_value=0))
    wide = out.pivot(index="model", columns="measure", values="category")
    print("\nmodels whose category differs between measures:")
    print(wide[wide.nunique(axis=1) > 1].to_string())
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
