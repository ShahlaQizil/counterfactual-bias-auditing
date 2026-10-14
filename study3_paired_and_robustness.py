"""
Two analyses reported in the paper that are computed from saved results:

1. Study 3, model-level paired comparisons. For each identity axis (gender,
   race) and each comparison set (random, EEC-Neutral, emotional), an exact
   McNemar test on the models where exactly one of the two template sets flags a
   directional bias (two-sided binomial test on the discordant counts). Holm
   correction over these six comparisons. Also the Wilcoxon signed-rank tests on
   per-model |d_z| (exploratory, uncorrected).

2. Robustness of the uncertainty--sensitivity correlation (Study 2): Spearman
   rho between the log sensitivity ratio and the log uncertainty ratio across the
   18 classifiers, its leave-one-model-out range, and a bootstrap 95% interval
   over models (10,000 resamples), for TV and for the primary measure.

Inputs:  results_targeted/<model>.csv, results_minimal_pair/paper_table.csv,
         results_minimal_pair/robust_stats.csv
Outputs: results_stats/paired_tests.csv, results_stats/correlation_robustness.json

Run:
    python study3_paired_and_robustness.py
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon

SEED = 20260713
N_BOOT = 10_000


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_tests() -> pd.DataFrame:
    t = pd.concat([pd.read_csv(f) for f in glob.glob("results_targeted/*.csv")
                   if os.path.basename(f) != "summary.csv"])
    t["tag"] = t["model"].str.split("/").str[-1]
    rows = []
    for axis in ("gender", "race"):
        flagged = t[t.axis == axis].pivot(index="tag", columns="set", values="flagged").astype(bool)
        dz = t[t.axis == axis].pivot(index="tag", columns="set", values="d_z").abs()
        for other in ("random", "neutral", "emotional"):
            only_t = int((flagged.targeted & ~flagged[other]).sum())
            only_o = int((~flagged.targeted & flagged[other]).sum())
            p = binomtest(only_t, only_t + only_o, 0.5).pvalue if only_t + only_o else 1.0
            rows.append({"axis": axis, "comparison": f"targeted vs {other}",
                         "flagged_targeted": int(flagged.targeted.sum()), "flagged_other": int(flagged[other].sum()),
                         "only_targeted": only_t, "only_other": only_o, "p_mcnemar_exact": p,
                         "wilcoxon_p_abs_dz": float(wilcoxon(dz.targeted, dz[other]).pvalue),
                         "n_models_dz_larger": int((dz.targeted > dz[other]).sum())})
    out = pd.DataFrame(rows)
    out["p_mcnemar_holm"] = holm(out["p_mcnemar_exact"].values)
    return out


def correlation_robustness() -> dict:
    t = pd.read_csv("results_minimal_pair/paper_table.csv", keep_default_na=False)
    r = pd.read_csv("results_minimal_pair/robust_stats.csv", keep_default_na=False)
    prim = r[((r.k == 2) & (r.measure == "logit")) | ((r.k > 2) & (r.measure == "tv"))]
    prim = prim.merge(t[["tag", "unc_f", "unc_e"]], left_on="model", right_on="tag")
    rng = np.random.default_rng(SEED)
    out = {}
    for name, x, y in (("tv", np.log(t.tv.values), np.log((t.unc_f / t.unc_e).values)),
                       ("primary", np.log(prim.ratio.values), np.log((prim.unc_f / prim.unc_e).values))):
        loo = [spearmanr(np.delete(x, i), np.delete(y, i)).statistic for i in range(len(x))]
        boot = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(x), len(x))
            boot.append(spearmanr(x[i], y[i]).statistic)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        out[name] = {"rho": float(spearmanr(x, y).statistic), "loo_min": float(min(loo)),
                     "loo_max": float(max(loo)), "boot_lo": float(lo), "boot_hi": float(hi),
                     "n_models": int(len(x))}
    return out


def main() -> None:
    pt = paired_tests()
    os.makedirs("results_stats", exist_ok=True)
    pt.to_csv("results_stats/paired_tests.csv", index=False)
    print(pt.round(4).to_string(index=False))
    cr = correlation_robustness()
    with open("results_stats/correlation_robustness.json", "w", encoding="utf-8") as f:
        json.dump(cr, f, indent=2)
    print(json.dumps(cr, indent=1))


if __name__ == "__main__":
    main()
