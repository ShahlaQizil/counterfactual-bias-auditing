"""
Uncertainty-targeted probe: do templates chosen to make a model uncertain
detect more DIRECTIONAL bias than standard template sets of the same size?

This tests the paper's account without circularity. Identity sensitivity is
mechanically bounded by uncertainty, but a consistent signed shift against one
group (the EEC paired t-test) is not, so "targeted templates flag more biases"
is a falsifiable prediction.

Design
------
Frame pool   16 base templates x 20 mildly valenced clauses (", and it was okay.",
             ", and it was a bit dull.", ...) = 320 frames.
Selection    For each model, the 16 frames with the highest mean predictive
             uncertainty, measured with 20 HELD-OUT EEC first names that never
             appear in testing. Selection never looks at identity contrasts.
Comparison   Template sets of 16 frames each:
               targeted   top-16 uncertainty frames (per model)
               random     16 frames drawn at random from the same pool (control)
               neutral    the 16 EEC-Neutral templates, no clause
               emotional  the 16 base templates + an emotion clause (random word)
Test persons 5 gender noun-phrase pairs and 10 same-gender race name pairs (the
             minimal-pair probe's persons); placebo pairs swap names WITHIN the
             same race and gender group, where no directional effect is expected.
Outcome      Paired t-test on signed Delta_logit per axis (gender, race,
             placebo) and template set; Bonferroni alpha = 0.05/36 as in the
             EEC reproduction. Also mean |Delta_logit|, d_z, and uncertainty.

Outputs: results_targeted/<model>.csv (per set x axis), results_targeted/summary.csv

Run:
    python targeted_probe.py --models distilbert-base-uncased-finetuned-sst-2-english
    python targeted_probe.py            # all models
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
import os
import traceback
import zlib

import numpy as np
import pandas as pd
from scipy import stats

import minimal_pair_experiment as M
import salience_experiment as S

OUT_DIR = "results_targeted"
N_SELECT = 16
ALPHA = 0.05 / 36

CLAUSES = [
    # mild positive
    "it was nice", "it was pleasant", "it was fine", "it was okay", "it was alright",
    "it was not bad", "it was decent", "it was quite calm",
    # mild negative
    "it was a bit dull", "it was somewhat tiring", "it was a little annoying",
    "it was not great", "it was rather slow", "it was a bit cold", "it was crowded",
    # weakly valenced / factual
    "it was raining", "it was late", "it was early", "it was on time", "it was noisy",
]

# EEC first names NOT used in the test persons, for uncertainty-based selection only
SELECTION_NAMES = ["Torrance", "Malik", "Jamel", "Alphonse", "Terrence", "Harry", "Josh",
                   "Justin", "Ryan", "Andrew", "Nichelle", "Shereen", "Latisha", "Shaniqua",
                   "Tia", "Courtney", "Heather", "Melanie", "Kristin", "Stephanie"]


def frames() -> pd.DataFrame:
    rows = []
    for b, tmpl in enumerate(S.EEC_NEUTRAL_TEMPLATES):
        stem = tmpl.rstrip(".")
        for c, clause in enumerate(CLAUSES):
            rows.append({"frame": f"b{b}c{c}", "base_id": b, "template": f"{stem}, and {clause}."})
    return pd.DataFrame(rows)


def test_pairs() -> pd.DataFrame:
    """(axis, a, b) person pairs: gender NP pairs, race name pairs, placebo name pairs."""
    rows = [("gender", f, m) for f, m in S.GENDER_PAIRS]
    for gender in ("female", "male"):
        ea, aa = S.NAMES[(gender, "european_american")], S.NAMES[(gender, "african_american")]
        rows += [("race", a, e) for e, a in zip(ea, aa)]  # African-American minus European-American
        for group in (ea, aa):  # placebo: neighbouring names within one race-gender group
            rows += [("placebo", group[i], group[i + 1]) for i in range(len(group) - 1)]
    return pd.DataFrame(rows, columns=["axis", "a", "b"])


def fill(template: str, person: str) -> str:
    s = template.format(person=person)
    return s[0].upper() + s[1:]


def template_sets(model: M.ValenceModel, pool: pd.DataFrame, rng) -> dict[str, list[str]]:
    # selection: mean uncertainty over held-out names only
    sents = [fill(t, n) for t in pool["template"] for n in SELECTION_NAMES]
    unc = 1 - model.scores(sents)[:, 2]
    pool = pool.assign(sel_unc=unc.reshape(len(pool), len(SELECTION_NAMES)).mean(axis=1))
    targeted = pool.nlargest(N_SELECT, "sel_unc")["template"].tolist()
    random_ = pool.sample(N_SELECT, random_state=int(rng.integers(1 << 31)))["template"].tolist()
    words = list(S.EMOTION_INTENSITY)
    emotional = [t.rstrip(".") + f", and then i felt {words[rng.integers(len(words))]}."
                 for t in S.EEC_NEUTRAL_TEMPLATES]
    return {"targeted": targeted, "random": random_, "neutral": list(S.EEC_NEUTRAL_TEMPLATES),
            "emotional": emotional}


def audit_one(name: str, pool: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng([M.SEED, zlib.crc32(name.encode())])
    model = M.ValenceModel(name)
    rows = []
    for set_name, temps in template_sets(model, pool, rng).items():
        for axis, g in pairs.groupby("axis"):
            a = [fill(t, p) for t in temps for p in g["a"]]
            b = [fill(t, p) for t in temps for p in g["b"]]
            sa, sb = model.scores(a), model.scores(b)
            delta = sa[:, 1] - sb[:, 1]
            sd = delta.std(ddof=1)
            p = float(stats.ttest_rel(sa[:, 1], sb[:, 1]).pvalue)
            rows.append({"model": name, "set": set_name, "axis": axis, "n": len(delta),
                         "mean_abs_delta": float(np.abs(delta).mean()),
                         "d_z": float(delta.mean() / sd) if sd > 0 else 0.0, "p": p,
                         "flagged": p < ALPHA,
                         "mean_unc": float(1 - np.concatenate([sa[:, 2], sb[:, 2]]).mean())})
    return pd.DataFrame(rows)


def summarise() -> None:
    files = [f for f in glob.glob(os.path.join(OUT_DIR, "*.csv")) if not f.endswith("summary.csv")]
    if not files:
        return
    res = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    summ = (res.groupby(["set", "axis"])
               .agg(models=("model", "nunique"), flagged=("flagged", "sum"),
                    median_abs_dz=("d_z", lambda x: float(np.median(np.abs(x)))),
                    median_unc=("mean_unc", "median"))
               .reset_index())
    summ.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print(summ.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=M.DEFAULT_MODELS)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    pool, pairs = frames(), test_pairs()
    for name in args.models:
        out = os.path.join(OUT_DIR, f"{name.split('/')[-1]}.csv")
        if os.path.exists(out):
            print(f"--- {name}: saved", flush=True)
            continue
        print(f"--- {name}", flush=True)
        try:
            r = audit_one(name, pool, pairs)
            r.to_csv(out, index=False)
            print(r[["set", "axis", "n", "mean_abs_delta", "d_z", "p", "flagged", "mean_unc"]]
                  .round(4).to_string(index=False), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    SKIPPED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        gc.collect()
    summarise()


if __name__ == "__main__":
    main()
