"""
Reproduce the Kiritchenko & Mohammad (2018) EEC audit on the sentiment fleet,
and run the same audit on EEC-Neutral.

What K&M actually reported (Sec. 5.1, Tables 5-6)
-------------------------------------------------
Gender only: on the four emotion-word-free templates, average female-male score
differences "tend to be higher" than on the FULL set, while far fewer
submissions are significant (anger: 34/46 -> 3/46). They attribute the drop to
power: 44 comparisons on the subset vs 1,584 on the full set (the 1,584 already
includes the 44; the emotional templates alone give 1,540).

Protocol (template and pairing design replicated; output measures adapted to
sentiment classifiers -- the original systems were emotion-intensity regressors)
---------------------------
Gender: for every template x emotion-word instantiation, ten noun-phrase pairs
(my daughter / my son, ...) plus one pair of averages over female vs male first
names -> 11 pairs per instantiation. Race: one pair per instantiation, the
average over African-American names vs the average over European-American
names. Paired t-test per model, Bonferroni over (models x 2 axes).

Reported per model, subset (full / emotional / neutral / EEC-Neutral) and scale
(expected valence e, and logit(e); see minimal_pair_experiment.py):
    n pairs, mean |delta|, mean signed delta, Cohen's d_z, p-value, flagged.
d_z does not depend on n, so it separates effect size from power -- which is
exactly what the flag counts in K&M conflate.

EEC-Neutral uses the 16 templates from salience_experiment.py with the full EEC
person set (subject/object forms taken from the EEC itself), capitalised like
the EEC sentences.

Data: data/Equity-Evaluation-Corpus/Equity-Evaluation-Corpus.csv (NOT
redistributable -- see its README; do not include it in releases).

Run:
    python eec_reproduction.py
    python eec_reproduction.py --models distilbert-base-uncased-finetuned-sst-2-english
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import argparse
import gc
import glob
import json
import os
import traceback

import numpy as np
import pandas as pd
from scipy import stats

import salience_experiment as S
from minimal_pair_experiment import ValenceModel
from run_fleet import FLEET

EEC_PATH = os.path.join("data", "Equity-Evaluation-Corpus", "Equity-Evaluation-Corpus.csv")
RESULTS_DIR = "results_eec"



### Corpus
def load_eec() -> pd.DataFrame:
    d = pd.read_csv(EEC_PATH)
    d["emotion_word"] = d["Emotion word"].fillna("")
    d["subset"] = np.where(d["Emotion"].isna(), "neutral", "emotional")
    return d


def person_index(eec: pd.DataFrame) -> dict[str, tuple[str, str, int | None]]:
    """person surface form -> (gender, race or 'np', noun-phrase pair index).
    The EEC lists male and female noun phrases in matching order (he/she,
    this man/this woman, ...), in subject and object form."""
    out = {}
    for tmpl_prefix in ("<person subject> feels", "I saw"):
        rows = eec[eec["Template"].str.startswith(tmpl_prefix)].drop_duplicates("Person")
        for gender in ("male", "female"):
            nps = rows[(rows["Gender"] == gender) & rows["Race"].isna()]["Person"].tolist()
            for i, p in enumerate(nps):
                out[p] = (gender, "np", i)
            for p in rows[(rows["Gender"] == gender) & rows["Race"].notna()]["Person"]:
                race = rows.loc[rows["Person"] == p, "Race"].iloc[0]
                out[p] = (gender, race, None)
    return out


def build_eec_neutral(eec: pd.DataFrame) -> pd.DataFrame:
    """EEC-Neutral: 16 emotion-word-free templates x the 60 EEC persons."""
    forms = {}
    for tmpl_prefix, slot in (("<person subject> feels", "subject"), ("I saw", "object")):
        rows = eec[eec["Template"].str.startswith(tmpl_prefix)].drop_duplicates("Person")
        forms[slot] = rows[["Person", "Gender", "Race"]].reset_index(drop=True)
    rows = []
    for tmpl in S.EEC_NEUTRAL_TEMPLATES:
        slot = "subject" if tmpl.startswith("{person}") else "object"
        for r in forms[slot].itertuples(index=False):
            s = tmpl.format(person=r.Person)
            rows.append({"Sentence": s[0].upper() + s[1:], "Template": tmpl, "Person": r.Person,
                         "Gender": r.Gender, "Race": r.Race, "emotion_word": "",
                         "subset": "eec_neutral"})
    return pd.DataFrame(rows)



### K&M pairing
def gender_pairs(d: pd.DataFrame, pidx: dict) -> pd.DataFrame:
    """11 (female, male) score pairs per instantiation: 10 noun phrases + 1 name average."""
    d = d.assign(g=d["Person"].map(lambda p: pidx[p][0]),
                 r=d["Person"].map(lambda p: pidx[p][1]),
                 np_i=d["Person"].map(lambda p: pidx[p][2]))
    out = []
    for (tmpl, word), g in d.groupby(["Template", "emotion_word"]):
        nps = g[g["r"] == "np"]
        f = nps[nps["g"] == "female"].set_index("np_i")
        m = nps[nps["g"] == "male"].set_index("np_i")
        for i in f.index.intersection(m.index):
            out.append((tmpl, word, f.at[i, "score_e"], m.at[i, "score_e"],
                        f.at[i, "score_logit"], m.at[i, "score_logit"]))
        names = g[g["r"] != "np"]
        fn, mn = names[names["g"] == "female"], names[names["g"] == "male"]
        out.append((tmpl, word, fn["score_e"].mean(), mn["score_e"].mean(),
                    fn["score_logit"].mean(), mn["score_logit"].mean()))
    return pd.DataFrame(out, columns=["template", "word", "a_e", "b_e", "a_logit", "b_logit"])


def race_pairs(d: pd.DataFrame, pidx: dict) -> pd.DataFrame:
    """1 (African-American avg, European-American avg) pair per instantiation."""
    d = d.assign(r=d["Person"].map(lambda p: pidx[p][1]))
    out = []
    for (tmpl, word), g in d.groupby(["Template", "emotion_word"]):
        aa, ea = g[g["r"] == "African-American"], g[g["r"] == "European"]
        out.append((tmpl, word, aa["score_e"].mean(), ea["score_e"].mean(),
                    aa["score_logit"].mean(), ea["score_logit"].mean()))
    return pd.DataFrame(out, columns=["template", "word", "a_e", "b_e", "a_logit", "b_logit"])


def paired_stats(p: pd.DataFrame, scale: str) -> dict:
    delta = (p[f"a_{scale}"] - p[f"b_{scale}"]).values
    n = len(delta)
    sd = delta.std(ddof=1) if n > 1 else np.nan
    pval = float(stats.ttest_rel(p[f"a_{scale}"], p[f"b_{scale}"]).pvalue) if n > 1 else np.nan
    return {"n": n, "mean_abs_delta": float(np.abs(delta).mean()),
            "mean_delta": float(delta.mean()),
            "d_z": float(delta.mean() / sd) if sd and sd > 0 else 0.0,
            "p": pval}



### Orchestration
def audit_one(name: str, corpus: pd.DataFrame, pidx: dict) -> list[dict]:
    model = ValenceModel(name)
    sc = model.scores(corpus["Sentence"].tolist())
    d = corpus.assign(score_e=sc[:, 0], score_logit=sc[:, 1])
    subsets = {"full": d[d["subset"] != "eec_neutral"],
               "emotional": d[d["subset"] == "emotional"],
               "neutral": d[d["subset"] == "neutral"],
               "eec_neutral": d[d["subset"] == "eec_neutral"]}
    rows = []
    for sub_name, sd in subsets.items():
        for axis, fn in (("gender", gender_pairs), ("race", race_pairs)):
            pairs = fn(sd, pidx)
            for scale in ("e", "logit"):
                rows.append({"model": name, "subset": sub_name, "axis": axis, "scale": scale,
                             **paired_stats(pairs, scale)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=FLEET)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    eec = load_eec()
    pidx = person_index(eec)
    corpus = pd.concat([eec[["Sentence", "Template", "Person", "Gender", "Race", "emotion_word", "subset"]],
                        build_eec_neutral(eec)], ignore_index=True)
    print(f"EEC: {len(eec)} sentences; EEC-Neutral: {(corpus['subset'] == 'eec_neutral').sum()}; "
          f"{len(args.models)} models\n")

    failed = []
    for name in args.models:
        out = os.path.join(RESULTS_DIR, f"model_{name.split('/')[-1]}.csv")
        if os.path.exists(out):
            print(f"--- {name}: loaded saved result", flush=True)
            continue
        print(f"--- {name}", flush=True)
        try:
            r = audit_one(name, corpus, pidx)
            pd.DataFrame(r).to_csv(out, index=False)
            for x in r:
                if x["scale"] == "e" and x["axis"] == "gender":
                    print(f"    gender {x['subset']:11s} n={x['n']:4d} |d|={x['mean_abs_delta']:.4f} "
                          f"d_z={x['d_z']:+.2f} p={x['p']:.2g}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    SKIPPED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            failed.append({"model": name, "error": f"{type(e).__name__}: {e}"})
        gc.collect()

    # Aggregate every model audited so far, so one-model-at-a-time runs build up
    # the fleet table. Bonferroni uses the full planned fleet (models x {gender,
    # race}), as K&M did, so a model's flag does not change as others are added.
    res = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RESULTS_DIR, "model_*.csv"))],
                    ignore_index=True)
    n_models = res["model"].nunique()
    alpha = 0.05 / (2 * len(FLEET))
    res["flagged"] = res["p"] < alpha
    res.to_csv(os.path.join(RESULTS_DIR, "eec_audit.csv"), index=False)

    # Fleet view: does the fleet reproduce K&M's pattern?
    agg = (res.groupby(["axis", "scale", "subset"])
              .agg(n_pairs=("n", "first"), models_flagged=("flagged", "sum"),
                   median_abs_delta=("mean_abs_delta", "median"), median_abs_dz=("d_z", lambda x: np.median(np.abs(x))))
              .reset_index())
    wide = res.pivot_table(index=["model", "axis", "scale"], columns="subset", values="mean_abs_delta").reset_index()
    wide["neutral_gt_full"] = wide["neutral"] > wide["full"]
    agg_nf = wide.groupby(["axis", "scale"])["neutral_gt_full"].sum().rename("models_neutral_absdelta_gt_full")
    summary = {"n_models": int(n_models), "bonferroni_alpha": alpha, "failed": failed,
               "fleet": agg.round(4).to_dict(orient="records"),
               "neutral_vs_full": agg_nf.reset_index().to_dict(orient="records")}
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n" + agg.round(4).to_string(index=False))
    print("\n" + agg_nf.to_string())
    print(f"\nResults in ./{RESULTS_DIR}/")


if __name__ == "__main__":
    main()
