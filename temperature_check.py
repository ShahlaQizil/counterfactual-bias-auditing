"""
Exploratory: how much of the probability-scale direction (Study 2) depends on saturation?

Temperature scaling divides a model's logits by T: the predicted class is unchanged, but
outputs become less saturated as T grows. For each classifier and T in {1, 2, 4, 8, 16},
we recompute, from the cached output distributions (logits = log P up to a constant),
  - the TV ratio (filler over emotion) and its direction,
  - the uncertainty ratio (1 - max P) and its direction,
and compare them with the direction in logit space (mean absolute change in centred
log-probabilities, which does not depend on T up to a constant factor).

Outputs: results_stats/temperature_check.json
Run:
    python temperature_check.py
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import json

import numpy as np
import pandas as pd
from scipy import stats

import minimal_pair_experiment as M
import prob_cache

TEMPS = [1, 2, 4, 8, 16]


def main() -> None:
    rec = M.build_records()
    rec = rec[rec["anchor"] & rec["condition"].isin(["filler", "emotion"])].reset_index(drop=True)
    is_f = (rec["condition"] == "filler").to_numpy()
    per_model = []
    for name in M.DEFAULT_MODELS:
        tag = name.split("/")[-1]
        probs, _ = prob_cache.load(tag)
        lp_a = np.log(np.clip(np.stack([probs[s] for s in rec["sentence"]]).astype(np.float64), 1e-30, 1))
        lp_b = np.log(np.clip(np.stack([probs[s] for s in rec["swapped"]]).astype(np.float64), 1e-30, 1))
        ca, cb = lp_a - lp_a.mean(1, keepdims=True), lp_b - lp_b.mean(1, keepdims=True)
        dlog = np.abs(ca - cb).mean(1)
        row = {"model": tag, "k": int(lp_a.shape[1]),
               "log_dir": float(np.sign(np.log(dlog[is_f].mean() / dlog[~is_f].mean())))}
        for T in TEMPS:
            pa = np.exp(lp_a / T); pa /= pa.sum(1, keepdims=True)
            pb = np.exp(lp_b / T); pb /= pb.sum(1, keepdims=True)
            tv = 0.5 * np.abs(pa - pb).sum(1)
            u = 1 - pa.max(1)
            row[f"tv_ratio_T{T}"] = float(tv[is_f].mean() / tv[~is_f].mean())
            row[f"u_ratio_T{T}"] = float(u[is_f].mean() / u[~is_f].mean())
        per_model.append(row)
    d = pd.DataFrame(per_model)
    out = {"note": "exploratory; not in a frozen plan", "temps": TEMPS, "per_T": {}}
    for T in TEMPS:
        tvd, ud = np.sign(np.log(d[f"tv_ratio_T{T}"])), np.sign(np.log(d[f"u_ratio_T{T}"]))
        out["per_T"][str(T)] = {
            "tv_direction_changed_vs_T1": int((tvd != np.sign(np.log(d.tv_ratio_T1))).sum()),
            "tv_direction_agrees_with_log_space": int((tvd == d.log_dir).sum()),
            "tv_direction_agrees_with_uncertainty": int((tvd == ud).sum()),
            "rho_tv_vs_u_ratio": float(stats.spearmanr(np.log(d[f"tv_ratio_T{T}"]), np.log(d[f"u_ratio_T{T}"]))[0]),
        }
    out["models"] = int(len(d))
    out["log_space_direction_up"] = int((d.log_dir > 0).sum())
    d.to_csv(_os.path.join("results_stats", "temperature_check.csv"), index=False)
    with open(_os.path.join("results_stats", "temperature_check.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
