"""
Exploratory follow-up to the template-family test (NOT part of its frozen plan).

Added after the frozen analysis to examine two questions the audit raised:
  1. Does restricting accuracy to cells with a clear direction flatter the predictor?
     -> accuracy of the uncertainty prediction on all 288 cells, using the point estimate.
  2. Why do the primary and logit outcomes differ?
     -> agreement of their directions by label scheme; share of clear logit cells pointing
        down in multi-class models; whether neutral-class probability (Study 2) relates to
        how often the two outcomes disagree.

Inputs:  results_template_family/cells_primary.csv, cells_logit.csv, unc_*.npz,
         results_minimal_pair/paper_table.csv (via template_family_test)
Outputs: results_template_family/exploratory.json

Run:
    python template_family_exploratory.py
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

import template_family_test as X

R = X.OUT


def main() -> None:
    u = X.uncertainty_cells()
    base = X.study2_baselines()
    keys = ["model", "family", "intensity"]
    p = pd.read_csv(f"{R}/cells_primary.csv").rename(columns={"log_s_ratio": "ls_p", "clear": "clear_p"})
    lg = pd.read_csv(f"{R}/cells_logit.csv").rename(columns={"log_s_ratio": "ls_l", "clear": "clear_l"})
    d = u.merge(p[keys + ["ls_p", "clear_p"]], on=keys).merge(lg[keys + ["ls_l", "clear_l"]], on=keys).merge(base, on="model")
    d["pred"] = np.sign(d.log_u_ratio)
    t2 = pd.read_csv("results_minimal_pair/paper_table.csv", keep_default_na=False).set_index("tag")

    out = {"note": "exploratory; not part of the frozen plan",
           "acc_all_cells_primary": float((d.pred == np.sign(d.ls_p)).mean()),
           "acc_all_cells_logit": float((d.pred == np.sign(d.ls_l)).mean())}
    for kind, g in (("binary", d[d.k == 2]), ("multiclass", d[d.k > 2])):
        out[f"{kind}_cells"] = int(len(g))
        out[f"{kind}_primary_logit_direction_agree"] = float((np.sign(g.ls_p) == np.sign(g.ls_l)).mean())
        cl = g[g.clear_l]
        out[f"{kind}_clear_logit_cells_down"] = float((cl.ls_l < 0).mean())
    mc = d[d.k > 2].assign(disagree=lambda x: np.sign(x.ls_p) != np.sign(x.ls_l))
    per = mc.groupby("model").disagree.mean()
    out["multiclass_spearman_pneu_vs_disagreement"] = float(stats.spearmanr(t2.loc[per.index, "nm_f"], per)[0])
    with open(f"{R}/exploratory.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
