"""
Cross-model targeting control for Study 3 (uncertainty-aware template selection).

Question: do targeted templates flag more directional disparities because they are
selected FOR THE AUDITED MODEL, or because some clauses in the pool make every model
uncertain? The random-frame control in targeted_probe.py holds the clause pool fixed
but not this.

For each classifier we score all 320 frames of the Study 3 pool with the held-out
selection names (frame uncertainty) and with every test person (signed Delta_l per
frame and person pair), so any 16-frame set can be evaluated offline. Sets compared,
all of 16 frames from the same pool and tested with the same pairs and threshold as
Study 3 (paired t-test on Delta_l, alpha = 0.05/36):

  targeted    the 16 frames most uncertain for the audited model (as in Study 3)
  consensus   the 16 frames with the best mean uncertainty RANK across the other 17
              models (uncertain in general, but not selected for this model)
  donor       each other model's targeted set applied to this model; we report the
              expected number of flags (mean over the 17 donors)

The plan (PLAN) is written to results_targeting_control/analysis_plan.json with its
SHA-256 before any model is scored.

Run:
    python targeting_control.py score      # per-model cache (re-runs each classifier)
    python targeting_control.py analyse
Outputs: results_targeting_control/
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import argparse
import gc
import hashlib
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
from scipy import stats

import minimal_pair_experiment as M
import targeted_probe as T

OUT = "results_targeting_control"
ALPHA = 0.05 / 36
N_SELECT = 16

PLAN = {
    "question": "Is the Study 3 detection gain due to per-model targeting or to clauses "
                "that make every model uncertain?",
    "sets": "targeted (top-16 frames by the audited model's uncertainty on held-out names); "
            "consensus (top-16 frames by mean uncertainty rank across the other 17 models); "
            "donor (each other model's targeted set; expected flags = mean over 17 donors).",
    "test": "Same test pairs and threshold as Study 3: paired t-test on Delta_l over "
            "frames x pairs, alpha = 0.05/36, per axis (gender, race, placebo).",
    "comparison": "Models flagged per set and axis; exact McNemar targeted vs consensus "
                  "per axis (two tests, Holm); clause composition and overlap of sets.",
    "decision_rule": "If targeted flags clearly more models than consensus for gender "
                     "(discordant models favour targeted), the gain is attributed to "
                     "per-model targeting. If targeted and consensus flag similar numbers, "
                     "the claim is narrowed to 'templates that are uncertain in general'. "
                     "The result is reported either way.",
    "sanity": "The targeted set rebuilt from the cache must reproduce the Study 3 flags.",
}


def freeze_plan() -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "analysis_plan.json")
    if os.path.exists(path):
        return
    text = json.dumps(PLAN, indent=2, sort_keys=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    with open(os.path.join(OUT, "analysis_plan.sha256"), "w", encoding="utf-8") as f:
        f.write(f"{sha}  analysis_plan.json  frozen {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"plan frozen, sha256 {sha}")


def score() -> None:
    freeze_plan()
    pool, pairs = T.frames(), T.test_pairs()
    persons = sorted(set(pairs["a"]) | set(pairs["b"]))
    for name in M.DEFAULT_MODELS:
        out = os.path.join(OUT, f"{name.split('/')[-1]}.npz")
        if os.path.exists(out):
            print(f"--- {name}: cached", flush=True)
            continue
        print(f"--- {name}", flush=True)
        t0 = time.time()
        try:
            model = M.ValenceModel(name)
            sel = [T.fill(t, n) for t in pool["template"] for n in T.SELECTION_NAMES]
            sel_unc = (1 - model.scores(sel)[:, 2]).reshape(len(pool), len(T.SELECTION_NAMES)).mean(1)
            sents = [T.fill(t, p) for t in pool["template"] for p in persons]
            lg = model.scores(sents)[:, 1].reshape(len(pool), len(persons))
            idx = {p: i for i, p in enumerate(persons)}
            delta = lg[:, [idx[a] for a in pairs["a"]]] - lg[:, [idx[b] for b in pairs["b"]]]
            np.savez_compressed(out, sel_unc=sel_unc, delta=delta)
            print(f"    saved ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    SKIPPED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        gc.collect()


def test_set(delta: np.ndarray, frames_idx: np.ndarray, axes: np.ndarray) -> dict[str, bool]:
    out = {}
    for axis in ("gender", "race", "placebo"):
        d = delta[np.ix_(frames_idx, np.flatnonzero(axes == axis))].ravel()
        p = stats.ttest_1samp(d, 0.0).pvalue
        out[axis] = bool(p < ALPHA)
    return out


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return n10, n01, float(p)


def analyse() -> None:
    pool, pairs = T.frames(), T.test_pairs()
    axes = pairs["axis"].to_numpy()
    tags = [n.split("/")[-1] for n in M.DEFAULT_MODELS]
    data = {t: np.load(os.path.join(OUT, f"{t}.npz")) for t in tags}
    sel = {t: data[t]["sel_unc"] for t in tags}
    ranks = {t: stats.rankdata(-sel[t]) for t in tags}          # 1 = most uncertain
    targeted = {t: np.argsort(-sel[t], kind="stable")[:N_SELECT] for t in tags}
    clause_of = pool["frame"].str.extract(r"c(\d+)")[0].astype(int).to_numpy()
    category = np.where(clause_of < 8, "mild positive", np.where(clause_of < 15, "mild negative", "factual"))

    study3 = pd.concat([pd.read_csv(f"results_targeted/{t}.csv") for t in tags])
    study3 = study3[study3.set == "targeted"].assign(tag=lambda d: d.model.str.split("/").str[-1])
    s3 = study3.set_index(["tag", "axis"])["flagged"]

    rows, repro_ok = [], True
    for t in tags:
        others = [o for o in tags if o != t]
        consensus = np.argsort(np.mean([ranks[o] for o in others], axis=0), kind="stable")[:N_SELECT]
        delta = data[t]["delta"]
        ft = test_set(delta, targeted[t], axes)
        fc = test_set(delta, consensus, axes)
        donor = [test_set(delta, targeted[o], axes) for o in others]
        for axis in ("gender", "race", "placebo"):
            repro_ok &= bool(ft[axis] == bool(s3[(t, axis)]))
        rows.append({
            "model": t,
            **{f"targeted_{a}": ft[a] for a in ft},
            **{f"consensus_{a}": fc[a] for a in fc},
            **{f"donor_rate_{a}": float(np.mean([d[a] for d in donor])) for a in ft},
            "unc_targeted": float(sel[t][targeted[t]].mean()),
            "unc_consensus": float(sel[t][consensus].mean()),
            "overlap_targeted_consensus": int(len(set(targeted[t]) & set(consensus))),
            "targeted_mild_negative": int((category[targeted[t]] == "mild negative").sum()),
            "targeted_mild_positive": int((category[targeted[t]] == "mild positive").sum()),
            "targeted_factual": int((category[targeted[t]] == "factual").sum()),
        })
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "targeting_control.csv"), index=False)

    jacc = [len(set(targeted[a]) & set(targeted[b])) / len(set(targeted[a]) | set(targeted[b]))
            for i, a in enumerate(tags) for b in tags[i + 1:]]
    mc = {a: mcnemar(res[f"targeted_{a}"].to_numpy(), res[f"consensus_{a}"].to_numpy())
          for a in ("gender", "race")}
    holm = sorted(mc.items(), key=lambda kv: kv[1][2])
    p_holm = {}
    for i, (a, (_, _, p)) in enumerate(holm):
        p_holm[a] = min(1.0, max([min(1.0, (2 - j) * holm[j][1][2]) for j in range(i + 1)]))
    summary = {
        "study3_targeted_flags_reproduced": bool(repro_ok),
        "flags": {s: {a: int(res[f"{s}_{a}"].sum()) for a in ("gender", "race", "placebo")}
                  for s in ("targeted", "consensus")},
        "donor_expected_flags": {a: round(float(res[f"donor_rate_{a}"].sum()), 1)
                                 for a in ("gender", "race", "placebo")},
        "mcnemar_targeted_vs_consensus": {a: {"targeted_only": mc[a][0], "consensus_only": mc[a][1],
                                              "p": round(mc[a][2], 4), "p_holm": round(p_holm[a], 4)}
                                          for a in mc},
        "median_unc_targeted": round(float(res.unc_targeted.median()), 3),
        "median_unc_consensus": round(float(res.unc_consensus.median()), 3),
        "median_overlap_targeted_consensus": float(res.overlap_targeted_consensus.median()),
        "mean_pairwise_jaccard_targeted_sets": round(float(np.mean(jacc)), 3),
        "targeted_clause_mix_total": {c: int(res[f"targeted_{c.replace(' ', '_')}"].sum())
                                      for c in ("mild negative", "mild positive", "factual")},
    }
    with open(os.path.join(OUT, "targeting_control.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(res.drop(columns=[c for c in res if c.startswith("targeted_m") or c == "targeted_factual"])
          .round(2).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["score", "analyse"])
    {"score": score, "analyse": analyse}[ap.parse_args().stage]()


if __name__ == "__main__":
    main()
