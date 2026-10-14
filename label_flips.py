"""
Label-flip rates: how often does swapping the identity term change the model's
predicted label (argmax class)? This ties identity sensitivity to decisions.

Reads the probability caches from prob_cache.py and reports, per model:
  minimal-pair probe  flip rate in the bare, filler and emotion conditions
                      (anchor pairs only, each counterfactual pair counted once)
  EEC                 flip rate over the gender noun-phrase pairs (sentence-level
                      counterfactuals; name pairs are averaged in the EEC
                      protocol and have no single predicted label) on emotional
                      templates, the original 4 neutral templates and EEC-Neutral

Outputs: results_probs/label_flips.csv

Run:
    python label_flips.py
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import glob
import os

import numpy as np
import pandas as pd

import eec_reproduction as E
import minimal_pair_experiment as M
import prob_cache as C


def flip_rate(probs: dict[str, np.ndarray], a: pd.Series, b: pd.Series) -> float:
    pa = np.stack([probs[s] for s in a]).argmax(axis=1)
    pb = np.stack([probs[s] for s in b]).argmax(axis=1)
    return float((pa != pb).mean())


def eec_np_pairs() -> pd.DataFrame:
    """Gender noun-phrase counterfactual pairs (female, male sentence) per EEC subset."""
    eec = E.load_eec()
    corpus = pd.concat([eec[["Sentence", "Template", "Person", "Gender", "Race", "emotion_word", "subset"]],
                        E.build_eec_neutral(eec)], ignore_index=True)
    pidx = E.person_index(eec)
    nps = corpus[corpus["Person"].map(lambda p: pidx[p][1] == "np")].copy()
    nps["g"] = nps["Person"].map(lambda p: pidx[p][0])
    nps["i"] = nps["Person"].map(lambda p: pidx[p][2])
    f = nps[nps["g"] == "female"].set_index(["subset", "Template", "emotion_word", "i"])["Sentence"]
    m = nps[nps["g"] == "male"].set_index(["subset", "Template", "emotion_word", "i"])["Sentence"]
    pairs = pd.concat([f.rename("a"), m.rename("b")], axis=1).dropna().reset_index()
    # the original 4 neutral templates also appear in EEC-Neutral; keep both labels
    return pairs


def main() -> None:
    records = M.build_records()
    anchors = records[records["anchor"]]
    eec_pairs = eec_np_pairs()
    rows = []
    for path in sorted(glob.glob(os.path.join(C.CACHE_DIR, "*.npz"))):
        tag = os.path.basename(path)[:-4]
        probs, weights = C.load(tag)
        row = {"model": tag, "k": len(weights)}
        for cond in ("bare", "filler", "emotion"):
            g = anchors[anchors["condition"] == cond]
            row[f"mp_{cond}"] = flip_rate(probs, g["sentence"], g["swapped"])
        for sub in ("emotional", "neutral", "eec_neutral"):
            g = eec_pairs[eec_pairs["subset"] == sub]
            row[f"eec_{sub}"] = flip_rate(probs, g["a"], g["b"])
            row[f"n_eec_{sub}"] = len(g)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.CACHE_DIR, "label_flips.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
