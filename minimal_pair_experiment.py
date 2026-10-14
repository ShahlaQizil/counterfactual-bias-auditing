"""
Minimal-pair experiment: does removing the emotion word -- and ONLY the emotion
word -- raise counterfactual identity sensitivity?

Why this replaces the neutral-vs-emotional comparison in salience_experiment.py
------------------------------------------------------------------------------
The original comparison set four neutral templates against three different
emotional templates. Those differ in wording, length and where the person term
sits, so any neutral/emotional gap is confounded with template identity (and a
single neutral template carried most of the effect). Here every sentence is
built from the SAME base template in three conditions:

    bare     i saw {person} in the market.
    filler   i saw {person} in the market, and then i went home.
    emotion  i saw {person} in the market, and then i felt {emotion}.

The filler clause has the same structure as the emotion clause, so
filler-vs-emotion isolates the emotion word itself; bare-vs-filler measures
what adding any clause does.

Three analyses
--------------
A  Counterfactual sensitivity |sub| = |s(x) - s(x with person swapped for its
   matched counterpart)|, compared across conditions within template. Primary
   contrast: filler vs emotion. Inference: two-way cluster bootstrap over base
   templates and person pairs.
B  Output scale. Every quantity is computed on two scales: the expected
   valence e in [0, 1] (the analogue of the score an EEC audit compares) and
   logit(e), which does not compress near the rails. If the gap exists on e but
   vanishes on logit(e), range compression -- not the emotion word -- explains
   the EEC neutral-template pattern.
C  Control tokens. Occlusion attribution of the identity term vs the other
   base-template tokens, which appear unchanged in every condition. If the
   control tokens lose attribution in the emotion condition as much as the
   identity term does, the effect is generic, not identity-specific.

Plus an exploratory intensity analysis (formerly H1) on official NRC scores,
split by emotion. The 20 EEC words span a narrow intensity range, so a flat
slope here is not evidence of a threshold effect.

Valence across label schemes
----------------------------
e = sum_c P(c) * w_c, where w_c places each class on [0, 1]: binary neg/pos ->
0/1; 3-class adds neutral 0.5; 5-class and star ratings are evenly spaced.
logit(e) is computed stably from the logits. For binary models logit(e) equals
the pos-neg logit margin exactly. (The old pos-neg margin ignored the middle
classes and, for tabularisai, compared 'Very Positive' against 'Negative'.)

Outputs (./results_minimal_pair/)
    <model>.csv          per-sentence scores and substitution sensitivities
    <model>_occ.csv      per-token occlusion attributions (subset)
    fleet_table.csv      one row per model
    summary.json         per-model results + fleet aggregate

Run:
    python minimal_pair_experiment.py                  # full fleet
    python minimal_pair_experiment.py --models distilbert-base-uncased-finetuned-sst-2-english
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
import re
import traceback
import zlib

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import salience_experiment as S
from run_fleet import FLEET

RESULTS_DIR = "results_minimal_pair"
SEED = S.SEED
N_BOOT = 2000

BASE_TEMPLATES = S.EEC_NEUTRAL_TEMPLATES  # 16 emotion-word-free templates
CLAUSES = {
    # form: (filler clause, emotion clause) -- same structure, differ only in the tail
    "a": (", and then i went home.", ", and then i felt {emotion}."),
    "b": (", which was on a tuesday.", ", which made me feel {emotion}."),
}
OCC_EMOTION_WORDS = ["angry", "scared", "happy", "sad"]  # occlusion subset, one per emotion
# the two roberta-large models take several hours each on CPU, so they run last
LARGE = {"siebert/sentiment-roberta-large-english", "j-hartmann/sentiment-roberta-large-english-3-classes"}
DEFAULT_MODELS = ([m for m in FLEET if m not in LARGE]
                  + [m for m in FLEET if m in LARGE])



### Valence model
def valence_weights(id2label: dict[int, str]) -> np.ndarray:
    """Place each output class on the negative(0)..positive(1) axis."""
    k = len(id2label)
    names = [str(id2label[i]).strip().lower() for i in range(k)]
    if all(re.fullmatch(r"label_\d+", n) for n in names):
        # Assumption: LABEL_0 is the most negative class (true for
        # cardiffnlp/twitter-roberta-base-sentiment; check model cards otherwise).
        return np.array([int(n.split("_")[1]) / (k - 1) for n in names])
    if all(n[:1].isdigit() for n in names):
        # star ratings ('1 star'..'5 stars') or Sentiment140-style '0'/'4'
        v = np.array([int(re.match(r"\d+", n).group()) for n in names], dtype=float)
        return (v - v.min()) / (v.max() - v.min())
    graded = any("very" in n for n in names)
    w = []
    for n in names:
        if "neg" in n:
            w.append(0.0 if ("very" in n or not graded) else 0.25)
        elif "pos" in n:
            w.append(1.0 if ("very" in n or not graded) else 0.75)
        elif "neu" in n:
            w.append(0.5)
        else:
            raise ValueError(f"cannot place label {n!r} on the valence axis")
    return np.array(w)


def load_classifier(model_name: str):
    """Tokenizer and model. Old BERT checkpoints (e.g. yiyanghkust/finbert-tone)
    have no `model_type` in config.json, which the Auto classes require.
    Names of the form "<repo>@2cls" / "<repo>@3cls" are instruction-tuned LLMs
    used as zero-shot classifiers (llm_classifier.py)."""
    import llm_classifier
    if llm_classifier.is_llm(model_name):
        return llm_classifier.load_llm(model_name)
    try:
        return (AutoTokenizer.from_pretrained(model_name),
                AutoModelForSequenceClassification.from_pretrained(model_name))
    except ValueError:
        from transformers import BertForSequenceClassification, BertTokenizerFast
        return (BertTokenizerFast.from_pretrained(model_name),
                BertForSequenceClassification.from_pretrained(model_name))


class ValenceModel:
    """Scores sentences on the expected-valence scale e and on logit(e), memoised."""

    def __init__(self, model_name: str, batch_size: int = 128):
        self.name = model_name
        self.tokenizer, self.model = load_classifier(model_name)
        self.model.eval()
        self.batch_size = batch_size
        self.weights = valence_weights(self.model.config.id2label)
        w = torch.tensor(self.weights, dtype=torch.float64)
        self._log_w = torch.log(w)          # -inf where w == 0
        self._log_1mw = torch.log1p(-w)     # -inf where w == 1
        self._cache: dict[str, tuple[float, float, float]] = {}

    @torch.inference_mode()
    def _forward(self, sentences: list[str]) -> None:
        # length-sorted batches waste far less compute on padding
        sentences = sorted(sentences, key=len)
        for start in range(0, len(sentences), self.batch_size):
            chunk = sentences[start : start + self.batch_size]
            enc = self.tokenizer(chunk, padding=True, truncation=True, max_length=64,
                                 return_tensors="pt")
            logits = self.model(**enc).logits.double()
            probs = torch.softmax(logits, dim=-1)
            e = probs @ torch.tensor(self.weights, dtype=torch.float64)
            # logit(e) = log sum_c p_c w_c - log sum_c p_c (1 - w_c), from logits directly
            lg = (torch.logsumexp(logits + self._log_w, dim=-1)
                  - torch.logsumexp(logits + self._log_1mw, dim=-1))
            mx = probs.max(dim=-1).values
            for s, a, b, c in zip(chunk, e.tolist(), lg.tolist(), mx.tolist()):
                self._cache[s] = (a, b, c)

    def scores(self, sentences: list[str]) -> np.ndarray:
        """(n, 3) array: expected valence e, logit(e), max class probability."""
        missing = list(dict.fromkeys(s for s in sentences if s not in self._cache))
        if missing:
            self._forward(missing)
        return np.array([self._cache[s] for s in sentences], dtype=np.float64)



### Probe corpus
def build_records() -> pd.DataFrame:
    """One row per (base template, person, condition, form, emotion word).
    Every person appears; `anchor` marks one member of each counterfactual pair
    so pair-level statistics count each pair once."""
    rows = []
    for p in S.build_person_terms():
        counterpart = p.gender_pair if p.kind == "noun_phrase" else p.race_pair
        anchor = (p.gender == "female") if p.kind == "noun_phrase" else (p.race == "european_american")
        pair_id = p.surface if anchor else counterpart
        for b, tmpl in enumerate(BASE_TEMPLATES):
            stem = tmpl.rstrip(".")
            variants = [("bare", None, None, tmpl)]
            for form, (filler, emo) in CLAUSES.items():
                variants.append(("filler", form, None, stem + filler))
                for word in S.EMOTION_INTENSITY:
                    variants.append(("emotion", form, word, stem + emo.replace("{emotion}", word)))
            for cond, form, word, t in variants:
                rows.append(dict(
                    base_id=b, template=tmpl, person=p.surface, kind=p.kind,
                    counterpart=counterpart, anchor=anchor, pair_id=pair_id,
                    condition=cond, form=form, emotion_word=word,
                    emotion=S.EMOTION_OF[word] if word else None,
                    intensity=S.EMOTION_INTENSITY[word] if word else np.nan,
                    sentence=t.format(person=p.surface),
                    swapped=t.format(person=counterpart),
                ))
    return pd.DataFrame(rows)



### Measurements
def substitution(model: ValenceModel, df: pd.DataFrame) -> pd.DataFrame:
    s = model.scores(df["sentence"].tolist())
    c = model.scores(df["swapped"].tolist())
    out = df.copy()
    out["e"], out["logit_e"], out["maxprob"] = s[:, 0], s[:, 1], s[:, 2]
    out["sub_e"] = s[:, 0] - c[:, 0]
    out["sub_logit"] = s[:, 1] - c[:, 1]
    return out


def occlusion(model: ValenceModel, df: pd.DataFrame) -> pd.DataFrame:
    """Per-position deletion attribution on both scales, for anchor persons and a
    four-word emotion subset. Positions < n_base belong to the base template and
    exist unchanged in every condition; they are the control tokens."""
    sub = df[df["anchor"] & (df["emotion_word"].isna() | df["emotion_word"].isin(OCC_EMOTION_WORDS))]
    plan = []
    for r in sub.itertuples(index=False):
        toks = S.atomic_tokens(r.sentence, r.person)
        variants = [" ".join(toks[:i] + toks[i + 1:]) for i in range(len(toks))]
        plan.append((r, toks, variants))
    # score every sentence and deletion variant in one batched pass
    model.scores(list(sub["sentence"]) + [v for _, _, vs in plan for v in vs])

    rows = []
    for r, toks, variants in plan:
        n_base = len(S.atomic_tokens(r.template.format(person=r.person), r.person))
        base = model.scores([r.sentence])[0]
        dels = model.scores(variants)
        for i, tok in enumerate(toks):
            role = ("identity" if tok == r.person else
                    "control" if i < n_base else
                    "emotion" if S._norm(tok) == r.emotion_word else "clause")
            rows.append(dict(base_id=r.base_id, pair_id=r.pair_id, condition=r.condition,
                             form=r.form, emotion_word=r.emotion_word, position=i,
                             token=S._norm(tok), role=role,
                             occ_e=base[0] - dels[i, 0], occ_logit=base[1] - dels[i, 1]))
    return pd.DataFrame(rows)



### Statistics
def cell_means(df: pd.DataFrame, value: str) -> dict[str, np.ndarray]:
    """Per-condition matrix [base template x person pair] of the mean |value|."""
    d = df.assign(v=df[value].abs())
    out = {}
    for cond in ("bare", "filler", "emotion"):
        m = d[d["condition"] == cond].pivot_table(index="base_id", columns="pair_id",
                                                  values="v", aggfunc="mean")
        out[cond] = m.sort_index().sort_index(axis=1).values
    return out


def two_way_bootstrap(mats: dict[str, np.ndarray], num: str, den: str, rng) -> dict:
    """Ratio and difference of condition means, with a bootstrap that resamples
    base templates and person pairs independently (both are random factors)."""
    a, b = mats[num], mats[den]
    nt, npair = a.shape
    ratios, diffs = np.empty(N_BOOT), np.empty(N_BOOT)
    for i in range(N_BOOT):
        t = rng.integers(0, nt, nt)
        p = rng.integers(0, npair, npair)
        ma, mb = a[np.ix_(t, p)].mean(), b[np.ix_(t, p)].mean()
        ratios[i], diffs[i] = ma / mb, ma - mb
    return {
        "ratio": round(float(a.mean() / b.mean()), 3),
        "ratio_ci": [round(float(x), 3) for x in np.percentile(ratios, [2.5, 97.5])],
        "diff": round(float(a.mean() - b.mean()), 4),
        "diff_ci": [round(float(x), 4) for x in np.percentile(diffs, [2.5, 97.5])],
        "n_templates_higher": int((a.mean(axis=1) > b.mean(axis=1)).sum()),
        "n_templates": int(nt),
    }


def control_specificity(occ: pd.DataFrame, scale: str, rng) -> dict:
    """Identity vs control-token attribution, filler vs emotion condition.
    specificity = (identity drop ratio) / (control drop ratio); > 1 means the
    emotion word takes more attribution away from the identity term than from
    the other base tokens."""
    col = f"occ_{scale}"
    d = occ[occ["condition"].isin(["filler", "emotion"]) & occ["role"].isin(["identity", "control"])]
    g = (d.assign(v=d[col].abs())
          .groupby(["base_id", "role", "condition"])["v"].mean().unstack(["role", "condition"]))
    ids = g.index.values

    def stat(t):
        m = g.loc[t].mean()
        id_ratio = m[("identity", "filler")] / m[("identity", "emotion")]
        ct_ratio = m[("control", "filler")] / m[("control", "emotion")]
        return id_ratio, ct_ratio, id_ratio / ct_ratio

    point = stat(ids)
    boots = np.array([stat(rng.choice(ids, len(ids))) for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots[:, 2], [2.5, 97.5])
    return {"identity_filler_over_emotion": round(float(point[0]), 3),
            "control_filler_over_emotion": round(float(point[1]), 3),
            "specificity": round(float(point[2]), 3),
            "specificity_ci": [round(float(lo), 3), round(float(hi), 3)]}


def ias_by_condition(occ: pd.DataFrame, scale: str) -> dict:
    col = f"occ_{scale}"
    d = occ.assign(v=occ[col].abs())
    key = ["base_id", "pair_id", "condition", "form", "emotion_word"]
    tot = d.groupby(key, dropna=False)["v"].sum()
    ide = d[d["role"] == "identity"].groupby(key, dropna=False)["v"].sum()
    ntok = d.groupby(key, dropna=False)["v"].size()
    share = (ide / tot).rename("ias").reset_index()
    share["uniform"] = (1 / ntok).values
    m = share.groupby("condition")[["ias", "uniform"]].mean()
    return {c: {"ias": round(float(m.loc[c, "ias"]), 4), "uniform": round(float(m.loc[c, "uniform"]), 4)}
            for c in m.index}


def intensity_analysis(res: pd.DataFrame, rng) -> dict:
    """Exploratory: |sub_logit| on NRC intensity within each emotion, template-
    cluster bootstrap. Words missing from NRC are excluded."""
    emo = res[(res["condition"] == "emotion") & res["anchor"]
              & ~res["emotion_word"].isin(S.NRC_MISSING)].copy()
    emo["v"] = emo["sub_logit"].abs()
    out = {}
    for name, g in emo.groupby("emotion"):
        cells = g.groupby(["base_id", "emotion_word"]).agg(v=("v", "mean"), x=("intensity", "first")).reset_index()
        by_t = {t: (c["x"].values, c["v"].values) for t, c in cells.groupby("base_id")}
        ids = list(by_t)
        slope = float(np.polyfit(cells["x"], cells["v"], 1)[0])
        bs = []
        for _ in range(N_BOOT):
            pick = rng.choice(ids, len(ids))
            x = np.concatenate([by_t[t][0] for t in pick])
            v = np.concatenate([by_t[t][1] for t in pick])
            bs.append(np.polyfit(x, v, 1)[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[name] = {"slope": round(slope, 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
                     "n_words": int(cells["emotion_word"].nunique()),
                     "intensity_range": [round(float(cells["x"].min()), 3), round(float(cells["x"].max()), 3)]}
    return out



### Orchestration
def audit_one(name: str, records: pd.DataFrame, rng) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    model = ValenceModel(name)
    res = substitution(model, records)
    occ = occlusion(model, records)
    pairs = res[res["anchor"]]
    row: dict = {"model": name, "status": "ok",
                 "valence_weights": [round(float(w), 2) for w in model.weights]}
    for scale in ("logit", "e"):
        mats = cell_means(pairs, f"sub_{scale}")
        row[f"A_{scale}"] = {
            "mean_abs_sub": {c: round(float(m.mean()), 4) for c, m in mats.items()},
            "filler_vs_emotion": two_way_bootstrap(mats, "filler", "emotion", rng),
            "bare_vs_emotion": two_way_bootstrap(mats, "bare", "emotion", rng),
            "bare_vs_filler": two_way_bootstrap(mats, "bare", "filler", rng),
        }
        row[f"C_{scale}"] = control_specificity(occ, scale, rng)
        row[f"IAS_{scale}"] = ias_by_condition(occ, scale)
    row["saturation"] = {c: round(float((g["maxprob"] > 0.98).mean()), 3)
                         for c, g in res.groupby("condition")}
    row["intensity_exploratory"] = intensity_analysis(res, rng)
    return row, res, occ


def flat(row: dict) -> dict:
    """One-line-per-model view for fleet_table.csv."""
    if row["status"] != "ok":
        return {"model": row["model"], "status": row["status"]}
    out = {"model": row["model"], "status": "ok"}
    for scale in ("logit", "e"):
        fe = row[f"A_{scale}"]["filler_vs_emotion"]
        out[f"{scale}_sub_bare"] = row[f"A_{scale}"]["mean_abs_sub"]["bare"]
        out[f"{scale}_sub_filler"] = row[f"A_{scale}"]["mean_abs_sub"]["filler"]
        out[f"{scale}_sub_emotion"] = row[f"A_{scale}"]["mean_abs_sub"]["emotion"]
        out[f"{scale}_filler_over_emotion"] = fe["ratio"]
        out[f"{scale}_ratio_lo"], out[f"{scale}_ratio_hi"] = fe["ratio_ci"]
        out[f"{scale}_templates_higher"] = f'{fe["n_templates_higher"]}/{fe["n_templates"]}'
        out[f"{scale}_specificity"] = row[f"C_{scale}"]["specificity"]
        out[f"{scale}_spec_lo"], out[f"{scale}_spec_hi"] = row[f"C_{scale}"]["specificity_ci"]
    out.update({f"sat_{k}": v for k, v in row["saturation"].items()})
    return out


def fleet_aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    agg = {"n_ok": len(ok), "n_failed": len(rows) - len(ok)}
    for scale in ("logit", "e"):
        fe = [r[f"A_{scale}"]["filler_vs_emotion"] for r in ok]
        sp = [r[f"C_{scale}"] for r in ok]
        agg[scale] = {
            "filler_over_emotion_median": round(float(np.median([f["ratio"] for f in fe])), 3) if fe else None,
            "n_ratio_gt1": sum(f["ratio"] > 1 for f in fe),
            "n_ratio_ci_excludes_1": sum(f["ratio_ci"][0] > 1 for f in fe),
            "n_ratio_ci_below_1": sum(f["ratio_ci"][1] < 1 for f in fe),
            "specificity_median": round(float(np.median([s["specificity"] for s in sp])), 3) if sp else None,
            "n_specificity_ci_excludes_1": sum(s["specificity_ci"][0] > 1 for s in sp),
        }
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--rerun", action="store_true", help="ignore saved per-model results")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    S.EMOTION_INTENSITY.update(S.load_nrc_intensities())
    records = build_records()
    print(f"Probe: {len(records)} sentences ({len(BASE_TEMPLATES)} base templates) "
          f"x {len(args.models)} models\n")

    rows = []
    for name in args.models:
        tag = name.split("/")[-1]
        saved = os.path.join(RESULTS_DIR, f"{tag}.json")
        if os.path.exists(saved) and not args.rerun:
            with open(saved, encoding="utf-8") as f:
                rows.append(json.load(f))
            print(f"--- {name}: loaded saved result", flush=True)
            continue
        print(f"--- {name}", flush=True)
        try:
            # per-model seed: a resumed run gives the same CIs as an uninterrupted one
            row, res, occ = audit_one(name, records, np.random.default_rng([SEED, zlib.crc32(name.encode())]))
            res.to_csv(os.path.join(RESULTS_DIR, f"{tag}.csv"), index=False)
            occ.to_csv(os.path.join(RESULTS_DIR, f"{tag}_occ.csv"), index=False)
            with open(saved, "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)
            for scale in ("logit", "e"):
                fe = row[f"A_{scale}"]["filler_vs_emotion"]
                m = row[f"A_{scale}"]["mean_abs_sub"]
                print(f"    [{scale:5s}] |sub| bare={m['bare']:.4f} filler={m['filler']:.4f} "
                      f"emotion={m['emotion']:.4f}  filler/emotion={fe['ratio']:.2f} "
                      f"CI{fe['ratio_ci']} ({fe['n_templates_higher']}/{fe['n_templates']} templates)  "
                      f"specificity={row[f'C_{scale}']['specificity']:.2f} CI{row[f'C_{scale}']['specificity_ci']}", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad model must not stop the fleet
            print(f"    SKIPPED: {type(e).__name__}: {e}")
            traceback.print_exc()
            row = {"model": name, "status": f"failed: {type(e).__name__}: {e}"}
        rows.append(row)
        gc.collect()

    pd.DataFrame([flat(r) for r in rows]).to_csv(os.path.join(RESULTS_DIR, "fleet_table.csv"), index=False)
    summary = {"design": {"base_templates": BASE_TEMPLATES, "clauses": CLAUSES,
                          "nrc_missing": sorted(S.NRC_MISSING), "n_boot": N_BOOT, "seed": SEED},
               "fleet": fleet_aggregate(rows), "per_model": rows}
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary["fleet"], indent=2))
    print(f"Results in ./{RESULTS_DIR}/")


if __name__ == "__main__":
    main()
