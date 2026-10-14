"""
Audit instruction-tuned LLMs, used as zero-shot sentiment classifiers, with
the same three studies as the classifier fleet.

Each LLM is run with a binary and a three-class prompt (llm_classifier.py), so
the label scheme varies while the weights stay fixed.

Stages (each saved separately, so an interrupted run resumes):
  mp        minimal-pair probe with 4 emotion words (one per emotion, as in the
            occlusion subset) instead of 20, to keep CPU time manageable:
            filler/emotion and bare/emotion ratios on |Delta_logit|, TV and
            |Delta_e| with the two-way bootstrap, mean uncertainty, neutral-class
            mass and label-flip rates per condition. No control-token occlusion.
  eec       exact EEC protocol + EEC-Neutral (eec_reproduction.audit_one)
  targeted  uncertainty-targeted probe with placebo (targeted_probe.audit_one)

Outputs (./results_llm/):
  <variant>_mp.json, <variant>_mp.csv     minimal-pair summary and per-pair values
  <variant>_eec.csv                        EEC rows (as results_eec/model_*.csv)
  <variant>_targeted.csv                   targeted-probe rows
  summary.csv                              one row per variant, Holm across variants

Run (CPU; Qwen2.5-0.5B ~1.5 h per variant, Qwen2.5-1.5B ~4-5 h):
    python llm_audit.py
    python llm_audit.py --models Qwen/Qwen2.5-0.5B-Instruct@3cls --stages mp
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import argparse
import gc
import json
import os
import zlib

import numpy as np
import pandas as pd

import eec_reproduction as E
import minimal_pair_experiment as M
import robust_stats as R
import targeted_probe as T

OUT_DIR = "results_llm"
LLM_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct@2cls",
    "Qwen/Qwen2.5-0.5B-Instruct@3cls",
    "Qwen/Qwen2.5-1.5B-Instruct@2cls",
    "Qwen/Qwen2.5-1.5B-Instruct@3cls",
]
# Prompt-robustness variants (wordings frozen in prompts_frozen.json before
# running). Run on the minimal-pair stage:
#   python llm_audit.py --models <PROMPT_VARIANTS...> --stages mp
PROMPT_VARIANTS = [f"Qwen/Qwen2.5-{size}-Instruct@{scheme}-{p}"
                   for size in ("0.5B", "1.5B") for p in ("p2", "p3") for scheme in ("2cls", "3cls")]
EMOTION_WORDS = M.OCC_EMOTION_WORDS


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def stage_mp(name: str, tag: str, quick: bool) -> None:
    rec = M.build_records()
    rec = rec[rec["emotion_word"].isna() | rec["emotion_word"].isin(EMOTION_WORDS)]
    if quick:
        rec = rec[rec["base_id"] < 2]
    model = M.ValenceModel(name)
    res = M.substitution(model, rec)
    clf = model.model
    pa = softmax(clf(res["sentence"].tolist()).logits.numpy())
    pb = softmax(clf(res["swapped"].tolist()).logits.numpy())
    res["tv"] = 0.5 * np.abs(pa - pb).sum(axis=1)
    res["flip"] = pa.argmax(axis=1) != pb.argmax(axis=1)
    res["unc"] = 1 - pa.max(axis=1)
    res["neutral_mass"] = pa[:, 1] if pa.shape[1] == 3 else 0.0
    pairs = res[res["anchor"]]
    rng = np.random.default_rng([M.SEED, zlib.crc32(name.encode())])
    out = {"model": name, "k": int(pa.shape[1]), "n_pairs": int(len(pairs))}
    for measure, col in (("logit", "sub_logit"), ("tv", "tv"), ("valence_e", "sub_e")):
        mats = M.cell_means(pairs, col)
        ratio, lo, hi, p = R.bootstrap_ratio(mats, rng)
        out[measure] = {"filler_over_emotion": ratio, "lo": lo, "hi": hi, "p": p,
                        "bare_over_emotion": float(mats["bare"].mean() / mats["emotion"].mean())}
    by = pairs.groupby("condition")
    out["uncertainty"] = by["unc"].mean().round(4).to_dict()
    out["neutral_mass"] = by["neutral_mass"].mean().round(4).to_dict()
    out["flip_rate"] = by["flip"].mean().round(4).to_dict()
    out["mean_e"] = by["e"].mean().round(4).to_dict()
    pairs.drop(columns=["swapped"]).to_csv(os.path.join(OUT_DIR, f"{tag}_mp.csv"), index=False)
    with open(os.path.join(OUT_DIR, f"{tag}_mp.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1), flush=True)


def stage_eec(name: str, tag: str) -> None:
    eec = E.load_eec()
    pidx = E.person_index(eec)
    corpus = pd.concat([eec[["Sentence", "Template", "Person", "Gender", "Race", "emotion_word", "subset"]],
                        E.build_eec_neutral(eec)], ignore_index=True)
    rows = pd.DataFrame(E.audit_one(name, corpus, pidx))
    rows["flagged"] = rows["p"] < 0.05 / 36
    rows.to_csv(os.path.join(OUT_DIR, f"{tag}_eec.csv"), index=False)
    print(rows[rows["scale"] == "logit"][["axis", "subset", "n", "mean_abs_delta", "d_z", "p", "flagged"]]
          .round(4).to_string(index=False), flush=True)


def stage_targeted(name: str, tag: str) -> None:
    rows = T.audit_one(name, T.frames(), T.test_pairs())
    rows.to_csv(os.path.join(OUT_DIR, f"{tag}_targeted.csv"), index=False)
    print(rows[["set", "axis", "n", "mean_abs_delta", "d_z", "p", "flagged", "mean_unc"]]
          .round(4).to_string(index=False), flush=True)


def summarise() -> None:
    rows = []
    for path in sorted(os.listdir(OUT_DIR)):
        if not path.endswith("_mp.json"):
            continue
        r = json.load(open(os.path.join(OUT_DIR, path)))
        primary = "logit" if r["k"] == 2 else "tv"
        rows.append({"model": r["model"], "k": r["k"], "primary": primary,
                     "ratio": r[primary]["filler_over_emotion"], "lo": r[primary]["lo"],
                     "hi": r[primary]["hi"], "p": r[primary]["p"],
                     "logit_ratio": r["logit"]["filler_over_emotion"],
                     "tv_ratio": r["tv"]["filler_over_emotion"],
                     "unc_filler": r["uncertainty"].get("filler"), "unc_emotion": r["uncertainty"].get("emotion"),
                     "flip_filler": r["flip_rate"].get("filler"), "flip_emotion": r["flip_rate"].get("emotion"),
                     "neutral_mass_filler": r["neutral_mass"].get("filler")})
    if not rows:
        return
    s = pd.DataFrame(rows)
    s["p_holm"] = R.holm(s["p"].values)
    s["category"] = np.where(s["p_holm"] >= 0.05, "null", np.where(s["ratio"] > 1, "effect", "reversed"))
    s.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print(s.round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=LLM_MODELS)
    ap.add_argument("--stages", nargs="*", default=["mp", "eec", "targeted"])
    ap.add_argument("--quick", action="store_true", help="mp stage on 2 base templates only (smoke test)")
    ap.add_argument("--robustness", action="store_true",
                    help="run the frozen prompt variants (PROMPT_VARIANTS) on the mp stage")
    args = ap.parse_args()
    if args.robustness:
        args.models, args.stages = PROMPT_VARIANTS, ["mp"]
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in args.models:
        tag = name.split("/")[-1] + ("_quick" if args.quick else "")
        for stage in args.stages:
            done = os.path.join(OUT_DIR, f"{tag}_{stage}." + ("json" if stage == "mp" else "csv"))
            if os.path.exists(done):
                print(f"--- {name} [{stage}]: done", flush=True)
                continue
            print(f"--- {name} [{stage}]", flush=True)
            if stage == "mp":
                stage_mp(name, tag, args.quick)
            elif stage == "eec":
                stage_eec(name, tag)
            elif stage == "targeted":
                stage_targeted(name, tag)
            gc.collect()
    if not args.quick:
        summarise()


if __name__ == "__main__":
    main()
