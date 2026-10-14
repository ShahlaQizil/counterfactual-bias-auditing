"""
Exploratory reanalysis of the template-family test (NOT part of its frozen plan).

Question: for multi-class models, does the uncertainty predictor still give the right
direction when sensitivity is measured on a scale that does not saturate?

Measure, fixed before computing: for a counterfactual pair with output distributions P and
P', the mean absolute change in centred log-probabilities,
    D(P, P') = (1/k) * sum_c | (log P_c - mean_j log P_j) - (log P'_c - mean_j log P'_j) |,
i.e. the L1 distance between the two outputs in logit space (invariant to the softmax's
additive constant, and not compressed near probability 0 or 1). Cells, clarity rule
(two-way base x pair bootstrap, 95% CI excludes 0), predictor (held-out-name uncertainty
ratio), discordance (versus the Study 2 direction) and baselines are exactly as in
template_family_test.py. Binary models are reported separately, since for them D is
proportional to |Delta logit|.

Outputs: results_template_family/logspace.json

Run:
    python template_family_logspace.py
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

import minimal_pair_experiment as M
import template_family_test as X

R = X.OUT


def cells() -> pd.DataFrame:
    f, pairs = X.frames(), X.test_pairs()
    rng = np.random.default_rng(X.SEED)
    rows = []
    for name in M.DEFAULT_MODELS:
        tag = name.split("/")[-1]
        z = np.load(f"{R}/out_{tag}.npz")
        idx = {p: i for i, p in enumerate(z["persons"].tolist())}
        ia, ib = [idx[a] for a in pairs.a], [idx[b] for b in pairs.b]
        lp = np.log(np.clip(z["probs"], 1e-300, 1.0))
        clr = lp - lp.mean(-1, keepdims=True)
        sens = np.abs(clr[:, ia] - clr[:, ib]).mean(-1)                      # (templates, pairs)
        for fam in X.FAMILIES:
            m = (f.family == fam).to_numpy()
            fs = sens[np.flatnonzero(m & (f.condition == "filler").to_numpy())]
            for level in ("high", "low"):
                emo = np.stack([np.flatnonzero(m & (f.intensity == level).to_numpy() & (f.base == b).to_numpy())
                                for b in range(8)])
                es = sens[emo].mean(1)
                point = float(np.log(fs.mean() / es.mean()))
                bb = rng.integers(0, 8, (X.N_CELL_BOOT, 8))
                pp = rng.integers(0, len(pairs), (X.N_CELL_BOOT, len(pairs)))
                lr = np.log(np.array([fs[np.ix_(bb[i], pp[i])].mean() / es[np.ix_(bb[i], pp[i])].mean()
                                      for i in range(X.N_CELL_BOOT)]))
                lo, hi = np.percentile(lr, [2.5, 97.5])
                rows.append({"model": tag, "k": z["probs"].shape[-1], "family": fam, "intensity": level,
                             "log_s_ratio": point, "clear": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def main() -> None:
    d = cells().merge(X.uncertainty_cells(), on=["model", "family", "intensity"]).merge(X.study2_baselines(), on=["model", "k"])
    d["pred"] = np.sign(d.log_u_ratio)
    rng = np.random.default_rng(X.SEED + 2)
    out = {"note": "exploratory; not part of the frozen plan; measure fixed before computing"}
    for kind, g in (("all", d), ("multiclass", d[d.k > 2]), ("binary", d[d.k == 2])):
        disc = g[g.pred != g.usual]
        ci = X.cluster_boot(g, X.accuracy, rng)
        ci2 = X.cluster_boot(disc, X.accuracy, rng) if len(disc) else np.array([np.nan])
        cl = g[g.clear]
        out[kind] = {
            "cells": int(len(g)), "clear": int(g.clear.sum()),
            "acc": X.accuracy(g), "acc_ci": [float(np.nanpercentile(ci, 2.5)), float(np.nanpercentile(ci, 97.5))],
            "baselines": {b: X.accuracy(g, b) for b in ("usual", "label", "neutral_mass", "majority")},
            "discordant": int(len(disc)), "discordant_clear": int(disc.clear.sum()),
            "acc_discordant": X.accuracy(disc) if len(disc) else None,
            "acc_discordant_ci": [float(np.nanpercentile(ci2, 2.5)), float(np.nanpercentile(ci2, 97.5))],
            "clear_cells_down": float((cl.log_s_ratio < 0).mean()) if len(cl) else None,
        }
    with open(f"{R}/logspace.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
