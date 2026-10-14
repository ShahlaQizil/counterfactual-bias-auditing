"""
Exploratory: can each template set detect a bias of KNOWN size?

A directional bias of fixed size delta is injected into every female sentence of the EEC
(gender axis) by shifting its logits towards positive: z_c <- z_c + delta * (w_c - mean w),
where w are the valence weights (for binary models this adds delta to the logit margin).
Using the cached outputs (logits = log P up to a constant), we recompute the paired
differences on the probability scale (Delta_e, the EEC's original score scale) and on the
logit scale (Delta_l), and count the models in which the injected direction (female > male)
is detected at alpha = 0.05/36 by the paired t-test and by the cluster-robust test with
templates and person pairs as clusters, on emotional templates and on EEC-Neutral.
delta = 0 gives the baseline (real effects only). We also relate, across models, the size
of the injected effect on Delta_e to each template set's mean uncertainty.

Outputs: results_stats/bias_injection.csv, bias_injection.json
Run:
    python bias_injection.py
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

import cluster_robust as C
import eec_reproduction as E
import minimal_pair_experiment as M
import prob_cache

DELTAS = [0.0, 0.25, 0.5, 1.0]


def main() -> None:
    eec = E.load_eec()
    pidx = E.person_index(eec)
    corpus = pd.concat([eec[["Sentence", "Template", "Person", "Gender", "Race", "emotion_word", "subset"]],
                        E.build_eec_neutral(eec)], ignore_index=True)
    female = corpus["Person"].map(lambda p: pidx[p][0] == "female").to_numpy()
    rows = []
    for name in M.DEFAULT_MODELS:
        tag = name.split("/")[-1]
        probs, w = prob_cache.load(tag)
        z = np.log(np.clip(np.stack([probs[s] for s in corpus["Sentence"]]).astype(np.float64), 1e-30, 1))
        shift = w - w.mean()
        for delta in DELTAS:
            zz = z + delta * female[:, None] * shift[None, :]
            p = np.exp(zz - zz.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            e = np.clip(p @ w, 1e-12, 1 - 1e-12)
            u = 1 - p.max(1)
            for scale, val in (("prob", e), ("logit", np.log(e) - np.log1p(-e))):
                d = corpus.assign(lg=val)
                for sub in ("emotional", "eec_neutral"):
                    sd = d[d["subset"] == sub]
                    pr = C.eec_pairs(sd, pidx, "gender")
                    res = C.tests(pr.delta.to_numpy(), pr.template.to_numpy(), pr.pair.to_numpy())
                    rows.append({"model": tag, "delta": delta, "scale": scale, "subset": sub,
                                 "mean": res["mean"],
                                 "u_mean": float(u[(corpus["subset"] == sub).to_numpy()].mean()),
                                 "detect_t": bool(res["flag_original"] and res["mean"] > 0),
                                 "detect_robust": bool(res["flag_cluster_twoway"] and res["mean"] > 0)})
    df = pd.DataFrame(rows)
    df.to_csv(_os.path.join("results_stats", "bias_injection.csv"), index=False)
    out = {"note": "exploratory; bias injected into cached outputs", "deltas": DELTAS, "counts": {}}
    for (scale, sub, delta), g in df.groupby(["scale", "subset", "delta"]):
        out["counts"][f"{scale}|{sub}|{delta}"] = {"t": int(g.detect_t.sum()), "robust": int(g.detect_robust.sum())}
    # injected effect on the probability scale versus uncertainty, across models
    base = df[(df.delta == 0) & (df.scale == "prob")].set_index(["model", "subset"])["mean"]
    for delta in DELTAS[1:]:
        g = df[(df.delta == delta) & (df.scale == "prob")].set_index(["model", "subset"])
        eff = g["mean"] - base
        out[f"rho_injected_effect_vs_uncertainty_delta{delta}"] = float(stats.spearmanr(eff, g["u_mean"])[0])
    with open(_os.path.join("results_stats", "bias_injection.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
