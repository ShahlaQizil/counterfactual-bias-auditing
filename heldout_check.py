"""
Held-out check for the uncertainty account (Study 2, 18 classifiers).

In the main analysis, uncertainty and identity sensitivity are measured on the same
sentences. Here they are measured on disjoint person terms: for each of 200 random
splits of the 15 minimal-pair person pairs into halves A and B, the uncertainty ratio
(filler over emotion, 1 - max P) comes from the sentences of pairs in A, and the
sensitivity ratio (filler over emotion; TV and |Delta_l|) from pairs in B only. We
report the Spearman correlation across the 18 classifiers and how many of the models
with a Holm-significant primary ratio get the direction right, per split.

This is an exploratory check added after the main analyses; it uses saved results only.

Inputs:  results_minimal_pair/<model>.csv, tv_<model>.csv, paper_table.csv, robust_stats.csv
Outputs: results_stats/heldout_prediction.json

Run:
    python heldout_check.py
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

N_SPLITS = 200
SEED = 7


def main() -> None:
    t = pd.read_csv("results_minimal_pair/paper_table.csv", keep_default_na=False)
    r = pd.read_csv("results_minimal_pair/robust_stats.csv", keep_default_na=False)
    prim = r[((r.k == 2) & (r.measure == "logit")) | ((r.k > 2) & (r.measure == "tv"))].set_index("model")
    sig = np.array([prim.at[tag, "category"] != "null" for tag in t.tag])

    data = {}
    for tag in t.tag:
        d = pd.read_csv(f"results_minimal_pair/{tag}.csv", keep_default_na=False)
        d = d[d.condition.isin(["filler", "emotion"]) & (d.anchor.astype(str) == "True")]
        tv = pd.read_csv(f"results_minimal_pair/tv_{tag}.csv", keep_default_na=False)
        data[tag] = (d, tv[tv.condition.isin(["filler", "emotion"])])
    pairs = sorted(data[t.tag[0]][0].pair_id.unique())

    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(N_SPLITS):
        a = set(rng.choice(pairs, len(pairs) // 2, replace=False))
        ur, s_tv, s_lg = [], [], []
        for tag in t.tag:
            d, tv = data[tag]
            u = 1 - d[d.pair_id.isin(a)].groupby("condition").maxprob.mean()
            ur.append(u["filler"] / u["emotion"])
            m = tv[~tv.pair_id.isin(a)].groupby("condition").tv.mean()
            s_tv.append(m["filler"] / m["emotion"])
            db = d[~d.pair_id.isin(a)]
            g = db.assign(x=db.sub_logit.astype(float).abs()).groupby("condition").x.mean()
            s_lg.append(g["filler"] / g["emotion"])
        ur, s_tv, s_lg = map(np.log, (ur, s_tv, s_lg))
        rows.append({"rho_tv": stats.spearmanr(ur, s_tv)[0], "rho_logit": stats.spearmanr(ur, s_lg)[0],
                     "agree_sig_tv": int((np.sign(ur[sig]) == np.sign(s_tv[sig])).sum()),
                     "agree_sig_logit": int((np.sign(ur[sig]) == np.sign(s_lg[sig])).sum())})
    res = pd.DataFrame(rows)
    out = {
        "n_pairs": len(pairs), "n_splits": N_SPLITS, "n_significant_models": int(sig.sum()),
        "rho_tv_median": round(float(res.rho_tv.median()), 4),
        "rho_tv_p05": round(float(res.rho_tv.quantile(0.05)), 4),
        "rho_logit_median": round(float(res.rho_logit.median()), 4),
        "rho_logit_p05": round(float(res.rho_logit.quantile(0.05)), 4),
        "agree_sig_tv_min": int(res.agree_sig_tv.min()),
        "agree_sig_tv_all_splits": bool((res.agree_sig_tv == sig.sum()).all()),
        "agree_sig_logit_median": int(res.agree_sig_logit.median()),
        "agree_sig_logit_min": int(res.agree_sig_logit.min()),
    }
    os.makedirs("results_stats", exist_ok=True)
    with open(os.path.join("results_stats", "heldout_prediction.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
