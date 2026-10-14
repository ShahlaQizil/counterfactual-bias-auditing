"""
Cluster-robust re-analysis of the directional tests in Study 1 (EEC replication) and
Study 3 (template sets), added after a reviewer pointed out that paired t-tests over
template x person pairs treat dependent observations as independent.

Fixed before computing:
  primary    one-way cluster-robust test of mean Delta_l = 0 with TEMPLATES as clusters
             (CR1 variance, t with G-1 degrees of freedom), Bonferroni alpha = 0.05/36 as in
             the paper; the identity terms are the benchmark's fixed set.
  secondary  two-way cluster-robust test (templates x person pairs; Cameron-Gelbach-Miller
             variance V_T + V_P - V_TP, df = min(G_T, G_P) - 1). For race, each EEC
             instantiation has a single averaged pair, so only the one-way test applies.
  original   the paper's paired t-test, kept for comparability with Kiritchenko & Mohammad.

Study 1 uses the cached EEC outputs (results_probs); Study 3 uses the cached per-frame
Delta_l of the 320-frame pool (results_targeting_control) for the targeted and random sets,
and scores the EEC-Neutral and emotional sets (16 templates x 30 persons) afresh.

Outputs: results_stats/cluster_robust_eec.csv, cluster_robust_targeted.csv,
         cluster_robust.json
Run:
    python cluster_robust.py
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import json
import os
import zlib

import numpy as np
import pandas as pd
from scipy import stats

import eec_reproduction as E
import minimal_pair_experiment as M
import prob_cache
import salience_experiment as S
import targeted_probe as T

ALPHA = 0.05 / 36
OUT = "results_stats"


def cluster_var(r: np.ndarray, g: np.ndarray) -> tuple[float, int]:
    s = pd.Series(r).groupby(g).sum()
    G = len(s)
    return float((s ** 2).sum() / len(r) ** 2 * G / (G - 1)), G


def tests(d: np.ndarray, tmpl: np.ndarray, pair: np.ndarray | None) -> dict:
    n = len(d)
    out = {"n": n, "mean": float(d.mean())}
    out["p_original"] = float(stats.ttest_1samp(d, 0.0).pvalue)
    r = d - d.mean()
    vt, gt = cluster_var(r, tmpl)
    t1 = d.mean() / np.sqrt(vt) if vt > 0 else 0.0
    out["p_cluster_template"] = float(2 * stats.t.sf(abs(t1), gt - 1))
    out["G_template"] = gt
    if pair is not None:
        vp, gp = cluster_var(r, pair)
        vtp, _ = cluster_var(r, np.array([f"{a}|{b}" for a, b in zip(tmpl, pair)]))
        v2 = vt + vp - vtp
        if v2 <= 0:
            v2 = max(vt, vp)
        t2 = d.mean() / np.sqrt(v2)
        out["p_cluster_twoway"] = float(2 * stats.t.sf(abs(t2), min(gt, gp) - 1))
        out["G_pair"] = gp
    else:
        out["p_cluster_twoway"] = out["p_cluster_template"]
        out["G_pair"] = 1
    for k in ("original", "cluster_template", "cluster_twoway"):
        out[f"flag_{k}"] = out[f"p_{k}"] < ALPHA
    return out


def eec_pairs(d: pd.DataFrame, pidx: dict, axis: str) -> pd.DataFrame:
    """Kiritchenko & Mohammad pairing (as in eec_reproduction), keeping a pair identifier."""
    d = d.assign(g=d["Person"].map(lambda p: pidx[p][0]), r=d["Person"].map(lambda p: pidx[p][1]),
                 np_i=d["Person"].map(lambda p: pidx[p][2]))
    rows = []
    for (tmpl, word), g in d.groupby(["Template", "emotion_word"]):
        if axis == "gender":
            nps = g[g["r"] == "np"]
            f, m = nps[nps["g"] == "female"].set_index("np_i"), nps[nps["g"] == "male"].set_index("np_i")
            for i in f.index.intersection(m.index):
                rows.append((tmpl, f"np{i}", f.at[i, "lg"] - m.at[i, "lg"]))
            names = g[g["r"] != "np"]
            rows.append((tmpl, "names", names[names["g"] == "female"]["lg"].mean() - names[names["g"] == "male"]["lg"].mean()))
        else:
            aa, ea = g[g["r"] == "African-American"], g[g["r"] == "European"]
            rows.append((tmpl, "race", aa["lg"].mean() - ea["lg"].mean()))
    return pd.DataFrame(rows, columns=["template", "pair", "delta"])


def study1() -> pd.DataFrame:
    eec = E.load_eec()
    pidx = E.person_index(eec)
    corpus = pd.concat([eec[["Sentence", "Template", "Person", "Gender", "Race", "emotion_word", "subset"]],
                        E.build_eec_neutral(eec)], ignore_index=True)
    rows = []
    for name in M.DEFAULT_MODELS:
        probs, w = prob_cache.load(name.split("/")[-1])
        p = np.stack([probs[s] for s in corpus["Sentence"]]).astype(np.float64)
        e = np.clip(p @ w, 1e-12, 1 - 1e-12)
        d = corpus.assign(lg=np.log(e) - np.log1p(-e))
        subsets = {"full": d[d["subset"] != "eec_neutral"], "emotional": d[d["subset"] == "emotional"],
                   "neutral": d[d["subset"] == "neutral"], "eec_neutral": d[d["subset"] == "eec_neutral"]}
        for sub, sd in subsets.items():
            for axis in ("gender", "race"):
                pr = eec_pairs(sd, pidx, axis)
                res = tests(pr.delta.to_numpy(), pr.template.to_numpy(),
                            pr.pair.to_numpy() if axis == "gender" else None)
                rows.append({"model": name, "subset": sub, "axis": axis, **res})
    return pd.DataFrame(rows)


def study3() -> pd.DataFrame:
    pool, pairs = T.frames(), T.test_pairs()
    words = list(S.EMOTION_INTENSITY)
    rows = []
    for name in M.DEFAULT_MODELS:
        tag = name.split("/")[-1]
        z = np.load(os.path.join("results_targeting_control", f"{tag}.npz"))
        sel, delta = z["sel_unc"], z["delta"]                      # (320,), (320, 31)
        rng = np.random.default_rng([M.SEED, zlib.crc32(name.encode())])
        targeted = np.argsort(-sel, kind="stable")[:16]
        random_idx = pool.sample(16, random_state=int(rng.integers(1 << 31))).index.to_numpy()
        emotional = [t.rstrip(".") + f", and then i felt {words[rng.integers(len(words))]}."
                     for t in S.EEC_NEUTRAL_TEMPLATES]
        model = M.ValenceModel(name)
        extra = {}
        for set_name, temps in (("neutral", list(S.EEC_NEUTRAL_TEMPLATES)), ("emotional", emotional)):
            mat = np.empty((len(temps), len(pairs)))
            for j, (a, b) in enumerate(zip(pairs.a, pairs.b)):
                sa = model.scores([T.fill(t, a) for t in temps])[:, 1]
                sb = model.scores([T.fill(t, b) for t in temps])[:, 1]
                mat[:, j] = sa - sb
            extra[set_name] = mat
        sets = {"targeted": delta[targeted], "random": delta[random_idx], **extra}
        for set_name, mat in sets.items():
            for axis in ("gender", "race", "placebo"):
                cols = np.flatnonzero(pairs.axis.to_numpy() == axis)
                sub = mat[:, cols]
                tm = np.repeat(np.arange(sub.shape[0]), sub.shape[1])
                pr = np.tile(cols, sub.shape[0])
                rows.append({"model": name, "set": set_name, "axis": axis, **tests(sub.ravel(), tm, pr)})
        del model
    return pd.DataFrame(rows)


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return {"only_first": n10, "only_second": n01, "p": float(p)}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    s1 = study1()
    s1.to_csv(os.path.join(OUT, "cluster_robust_eec.csv"), index=False)
    s3 = study3()
    s3.to_csv(os.path.join(OUT, "cluster_robust_targeted.csv"), index=False)

    # sanity: original-protocol flags must reproduce the published ones
    pub1 = pd.read_csv("results_eec/eec_audit.csv")
    pub1 = pub1[pub1.scale == "logit"].set_index(["model", "subset", "axis"]).flagged
    rep1 = s1.set_index(["model", "subset", "axis"]).flag_original
    pub3 = pd.concat([pd.read_csv(f"results_targeted/{n.split('/')[-1]}.csv") for n in M.DEFAULT_MODELS])
    pub3 = pub3.set_index(["model", "set", "axis"]).flagged
    rep3 = s3.set_index(["model", "set", "axis"]).flag_original
    summary = {"reproduces_published_flags": {"study1_mismatches": int((pub1.reindex(rep1.index) != rep1).sum()),
                                              "study3_mismatches": int((pub3.reindex(rep3.index) != rep3).sum())}}
    for kind in ("original", "cluster_template", "cluster_twoway"):
        col = f"flag_{kind}"
        summary[kind] = {
            "study1": {f"{sub}_{ax}": int(g[col].sum()) for (sub, ax), g in s1.groupby(["subset", "axis"])},
            "study3": {f"{st}_{ax}": int(g[col].sum()) for (st, ax), g in s3.groupby(["set", "axis"])},
        }
        w1 = {ax: s1[s1.axis == ax].pivot(index="model", columns="subset", values=col) for ax in ("gender", "race")}
        summary[kind]["study1_derived"] = {
            ax: {"full_not_eecn": int((w1[ax].full & ~w1[ax].eec_neutral).sum()),
                 "eecn_not_neutral4": int((w1[ax].eec_neutral & ~w1[ax].neutral).sum()),
                 "eecn_not_full": int((w1[ax].eec_neutral & ~w1[ax].full).sum())}
            for ax in ("gender", "race")}
        w3 = {ax: s3[s3.axis == ax].pivot(index="model", columns="set", values=col) for ax in ("gender", "race")}
        summary[kind]["study3_mcnemar_targeted_vs"] = {
            f"{ax}_{ref}": mcnemar(w3[ax].targeted.to_numpy(), w3[ax][ref].to_numpy())
            for ax in ("gender", "race") for ref in ("random", "neutral", "emotional")}
    # sign reversal in twitter-roberta race (both subsets significant with opposite signs)
    tr = s1[(s1.model == "cardiffnlp/twitter-roberta-base-sentiment") & (s1.axis == "race")].set_index("subset")
    summary["twitter_roberta_race"] = {sub: {"mean": float(tr.at[sub, "mean"]),
                                             "flag_template": bool(tr.at[sub, "flag_cluster_template"])}
                                       for sub in ("emotional", "eec_neutral")}
    with open(os.path.join(OUT, "cluster_robust.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
