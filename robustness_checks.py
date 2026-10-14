"""
Two robustness checks computed from cached model outputs (no model is re-run).

A. Uncertainty definition. The paper measures uncertainty as u = 1 - max_c P(c|x).
   We repeat the uncertainty analyses with two alternatives:
     entropy  H(P) / log k   (normalised to [0, 1])
     margin   1 - (P_(1) - P_(2))   (one minus the gap between the two most probable classes)
   A1  across the 18 classifiers: Spearman between the log sensitivity ratio (filler over
       emotion; TV and |Delta_l|) and the log uncertainty ratio, per definition;
   A2  within each classifier: Spearman between pair-level TV and pair uncertainty
       (mean over the two sentences of the pair), per definition;
   A3  natural sentences (22 systems): the within-system correlation, per definition.

B. Equal-size template sets in the EEC replication. EEC-Neutral has 176 gender and 16
   race pairs, the emotional templates 1,540 and 140, so "detected on emotional, not on
   EEC-Neutral" mixes template type with power. For each model and axis we draw 2,000
   random subsets of emotional pairs of the EEC-Neutral size and record how often the
   paired t-test on Delta_l is significant at the EEC threshold (0.05/36): the detection
   rate an emotional audit of the same size would have. Compared with the EEC-Neutral
   verdict, this separates the template effect from the size effect.

Inputs:  results_probs/*.npz, results_minimal_pair/tv_*.csv, paper_table.csv,
         results_eec/eec_audit.csv, results_natural/probs_*.npz, data/ (EEC)
Outputs: results_stats/uncertainty_definitions.csv, results_stats/eec_equal_size.csv,
         results_stats/robustness_checks.json

Run:
    python robustness_checks.py
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
from scipy import stats

import eec_reproduction as E
import minimal_pair_experiment as M
import prob_cache

OUT = "results_stats"
ALPHA = 0.05 / 36
N_DRAWS = 2000
SEED = 20261012


### A. Uncertainty definitions
def uncertainties(p: np.ndarray) -> dict[str, np.ndarray]:
    k = p.shape[1]
    srt = np.sort(p, axis=1)
    ent = -(p * np.log(np.clip(p, 1e-300, 1))).sum(axis=1) / np.log(k)
    return {"maxprob": 1 - srt[:, -1], "entropy": ent, "margin": 1 - (srt[:, -1] - srt[:, -2])}


def part_a() -> tuple[pd.DataFrame, dict]:
    # one member of each counterfactual pair, as for the paper's uncertainty columns
    records = M.build_records()
    records = records[records["anchor"]].reset_index(drop=True)
    table = pd.read_csv("results_minimal_pair/paper_table.csv", keep_default_na=False).set_index("tag")
    rows = []
    for tag in table.index:
        probs, _ = prob_cache.load(tag)
        p_all = np.stack([probs[s] for s in records["sentence"]])
        u = uncertainties(p_all)
        row = {"model": tag}
        for name, val in u.items():
            f = val[(records["condition"] == "filler").to_numpy()].mean()
            e = val[(records["condition"] == "emotion").to_numpy()].mean()
            row[f"ratio_{name}"] = f / e
        # pair level: TV against mean uncertainty of the two sentences
        tv = pd.read_csv(f"results_minimal_pair/tv_{tag}.csv", keep_default_na=False)
        pa = np.stack([probs[s] for s in tv["sentence"]])
        pb = np.stack([probs[s] for s in tv["swapped"]])
        ua, ub = uncertainties(pa), uncertainties(pb)
        for name in ua:
            row[f"within_{name}"] = stats.spearmanr(tv["tv"], 0.5 * (ua[name] + ub[name]))[0]
        rows.append(row)
    res = pd.DataFrame(rows).set_index("model")
    # sanity: the max-prob ratio must reproduce the paper's uncertainty ratio
    paper_ratio = (table["unc_f"] / table["unc_e"]).reindex(res.index)
    repro = float(np.max(np.abs(res["ratio_maxprob"] / paper_ratio - 1)))

    summary = {"maxprob_ratio_max_rel_deviation_from_paper": repro}
    for name in ("maxprob", "entropy", "margin"):
        lu = np.log(res[f"ratio_{name}"])
        summary[f"rho_tv_{name}"] = float(stats.spearmanr(np.log(table.loc[res.index, "tv"]), lu)[0])
        summary[f"rho_logit_{name}"] = float(stats.spearmanr(np.log(table.loc[res.index, "lg"]), lu)[0])
        summary[f"within_median_{name}"] = float(res[f"within_{name}"].median())
        summary[f"within_positive_{name}"] = int((res[f"within_{name}"] > 0).sum())
    # agreement in direction between definitions
    summary["direction_agree_entropy"] = int((np.sign(np.log(res.ratio_entropy)) == np.sign(np.log(res.ratio_maxprob))).sum())
    summary["direction_agree_margin"] = int((np.sign(np.log(res.ratio_margin)) == np.sign(np.log(res.ratio_maxprob))).sum())

    # A3: natural sentences
    nat = []
    for path in sorted(glob.glob("results_natural/probs_*.npz")):
        z = np.load(path)
        a, b = z["male"], z["female"]
        tvn = 0.5 * np.abs(a - b).sum(axis=1)
        ua, ub = uncertainties(a), uncertainties(b)
        nat.append({name: stats.spearmanr(tvn, 0.5 * (ua[name] + ub[name]))[0] for name in ua})
    nat = pd.DataFrame(nat)
    for name in nat:
        summary[f"natural_within_median_{name}"] = float(nat[name].median())
        summary[f"natural_within_positive_{name}"] = int((nat[name] > 0).sum())
    summary["natural_systems"] = int(len(nat))
    return res.reset_index(), summary


### B. Equal-size template sets
def part_b() -> tuple[pd.DataFrame, dict]:
    eec = E.load_eec()
    pidx = E.person_index(eec)
    corpus = pd.concat([eec[["Sentence", "Template", "Person", "Gender", "Race", "emotion_word", "subset"]],
                        E.build_eec_neutral(eec)], ignore_index=True)
    audit = pd.read_csv("results_eec/eec_audit.csv")
    audit = audit[audit.scale == "logit"].set_index(["model", "subset", "axis"])
    rng = np.random.default_rng(SEED)
    rows, max_dev = [], 0.0
    for name in sorted(audit.index.get_level_values(0).unique()):
        probs, w = prob_cache.load(name.split("/")[-1])
        p = np.stack([probs[s] for s in corpus["Sentence"]]).astype(np.float64)
        e = np.clip(p @ w, 1e-12, 1 - 1e-12)
        d = corpus.assign(score_e=e, score_logit=np.log(e) - np.log1p(-e))
        for axis, fn in (("gender", E.gender_pairs), ("race", E.race_pairs)):
            emo = fn(d[d["subset"] == "emotional"], pidx)
            neu = fn(d[d["subset"] == "eec_neutral"], pidx)
            de = (emo["a_logit"] - emo["b_logit"]).to_numpy()
            dn = (neu["a_logit"] - neu["b_logit"]).to_numpy()
            # reproduce the published d_z from the cache
            for sub, delta in (("emotional", de), ("eec_neutral", dn)):
                pub = audit.loc[(name, sub, axis), "d_z"]
                max_dev = max(max_dev, abs(delta.mean() / delta.std(ddof=1) - pub))
            n = len(dn)
            idx = np.array([rng.choice(len(de), n, replace=False) for _ in range(N_DRAWS)])
            sub = de[idx]
            t = sub.mean(1) / (sub.std(1, ddof=1) / np.sqrt(n))
            pv = 2 * stats.t.sf(np.abs(t), n - 1)
            rows.append({"model": name, "axis": axis, "n_neutral": n, "n_emotional": len(de),
                         "flag_full": bool(audit.loc[(name, "full", axis), "flagged"]),
                         "flag_emotional": bool(audit.loc[(name, "emotional", axis), "flagged"]),
                         "flag_eec_neutral": bool(audit.loc[(name, "eec_neutral", axis), "flagged"]),
                         "dz_emotional": float(de.mean() / de.std(ddof=1)),
                         "dz_eec_neutral": float(dn.mean() / dn.std(ddof=1)),
                         "equal_size_emotional_rate": float((pv < ALPHA).mean())})
    res = pd.DataFrame(rows)
    summary = {"dz_max_abs_deviation_from_published": float(max_dev)}
    for axis in ("gender", "race"):
        r = res[res.axis == axis]
        missed = r[r.flag_full & ~r.flag_eec_neutral]
        summary[axis] = {
            "eec_neutral_flags": int(r.flag_eec_neutral.sum()),
            "equal_size_emotional_expected_flags": round(float(r.equal_size_emotional_rate.sum()), 1),
            "equal_size_emotional_majority_flags": int((r.equal_size_emotional_rate > 0.5).sum()),
            "missed_by_eec_neutral": int(len(missed)),
            "missed_rates": [round(x, 2) for x in missed.equal_size_emotional_rate],
            "missed_detected_by_equal_size_emotional_majority": int((missed.equal_size_emotional_rate > 0.5).sum()),
            "eec_neutral_only_vs_equal_size": int((r.flag_eec_neutral & (r.equal_size_emotional_rate < 0.5)).sum()),
        }
    return res, summary


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    a, sa = part_a()
    a.to_csv(os.path.join(OUT, "uncertainty_definitions.csv"), index=False)
    b, sb = part_b()
    b.to_csv(os.path.join(OUT, "eec_equal_size.csv"), index=False)
    summary = {"uncertainty_definitions": sa, "eec_equal_size": sb}
    with open(os.path.join(OUT, "robustness_checks.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(b.assign(model=b.model.str.split("/").str[-1].str[:26]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
