"""
Prospective test: does a classifier's uncertainty, measured with held-out names, predict
the direction of the emotion-clause effect on UNSEEN template families, including where
the observed direction departs from the model's usual (Study 2) direction?

Stages (each refuses to run out of order; the plan and families are hashed at `freeze`):
    python template_family_test.py freeze        # write plan + families, run leakage checks
    python template_family_test.py uncertainty   # score selection-name sentences only
    python template_family_test.py gate          # count discordant cells (no outcome data)
    python template_family_test.py outcomes      # score test-pair sentences (only after gate)
    python template_family_test.py analyse       # H1, H2, baselines, sensitivity analyses

Outputs: results_template_family/
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
import re
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

import minimal_pair_experiment as M
import salience_experiment as S
import targeted_probe as T

OUT = "results_template_family"
SEED = 20261014
N_CELL_BOOT = 2000
N_CLUSTER_BOOT = 4000
GATE_MIN_DISCORDANT = 40

# emotion words: per emotion, the EEC word with the highest and the lowest NRC Affect
# Intensity ("discouraged" has no NRC score and is excluded); nouns for family F4
HIGH = {"anger": "furious", "fear": "terrified", "joy": "ecstatic", "sadness": "devastated"}
LOW = {"anger": "annoyed", "fear": "anxious", "joy": "relieved", "sadness": "disappointed"}
NOUN = {"furious": "fury", "terrified": "terror", "ecstatic": "ecstasy", "devastated": "devastation",
        "annoyed": "annoyance", "anxious": "anxiety", "relieved": "relief", "disappointed": "disappointment"}

# Each family: (description, emotion frame, filler frame, base fillers). {e} = emotion word,
# {X} = base, {person} = identity slot. Emotion and filler frames differ only in the slot.
FAMILIES = {
    "F1": ("sentence-initial participial adjunct (speaker)",
           "Feeling {e}, I {X}.", "Being early, I {X}.",
           ["called {person} from the car", "wrote a short note to {person}", "waited for {person} at the corner",
            "drove {person} to the airport", "sent a message to {person}", "walked home with {person}",
            "had dinner with {person}", "looked for {person} in the crowd"]),
    "F2": ("sentence-initial when-clause with predicate adjective (speaker)",
           "When I was {e}, I {X}.", "When I was in town, I {X}.",
           ["visited {person}", "phoned {person}", "ran into {person}", "stopped by to see {person}",
            "had coffee with {person}", "asked {person} for directions", "lent {person} my umbrella",
            "shared a taxi with {person}"]),
    "F3": ("exclamative (speaker)",
           "How {e} I was when I {X}!", "How early I was when I {X}!",
           ["met {person} at the door", "finally reached {person}", "saw {person} again",
            "picked up {person} from work", "left the party with {person}", "got the call from {person}",
            "found {person} in the lobby", "drove past {person}"]),
    "F4": ("sentence-initial prepositional phrase with an emotion noun",
           "To my {n}, {X}.", "To my left, {X}.",
           ["{person} sat down at the table", "{person} was already waiting", "I spotted {person} near the window",
            "{person} walked into the room", "the seat beside {person} was empty", "{person} had arrived first",
            "I recognized {person} at once", "{person} took the last chair"]),
    "F5": ("main-clause predicate adjective followed by a temporal clause (speaker)",
           "I was {e} while I {X}.", "I was outside while I {X}.",
           ["waited for {person}", "talked with {person}", "looked after {person}", "listened to {person}",
            "walked beside {person}", "stood next to {person}", "queued behind {person}", "sat across from {person}"]),
    "F6": ("interrogative (speaker)",
           "Why was I so {e} after I {X}?", "Why was I so busy after I {X}?",
           ["spoke with {person}", "visited {person}", "had lunch with {person}", "met {person}",
            "called {person}", "left {person} at the station", "drove {person} home", "said goodbye to {person}"]),
    "F7": ("identity term as experiencer with a predicate adjective",
           "{person} was {e} {X}.", "{person} was there {X}.",
           ["at the meeting", "at the station", "at the party", "in the office", "at the airport",
            "in the garden", "at the concert", "in the kitchen"]),
    "F8": ("sentence-final because-clause with predicate adjective (speaker)",
           "I {X} because I was {e}.", "I {X} because I was nearby.",
           ["called {person}", "visited {person}", "stayed with {person}", "went to see {person}",
            "wrote to {person}", "waited for {person}", "drove over to {person}", "sat with {person}"]),
}
FILLER_WORDS = {"F1": ["being", "early"], "F2": ["in", "town"], "F3": ["early"], "F4": ["left"],
                "F5": ["outside"], "F6": ["busy"], "F7": ["there"], "F8": ["nearby"]}

PLAN = {
    "question": ("Within a sentiment classifier, does uncertainty measured with held-out names predict the "
                 "direction of the emotion-clause effect on identity sensitivity in unseen template "
                 "families, including where the observed direction departs from the model's usual direction?"),
    "status": ("Prospective. Earlier within-model and held-out-person analyses on the Study 2 templates "
               "were exploratory and are not part of this plan."),
    "models": "The 18 classifiers of Study 2 (minimal_pair_experiment.DEFAULT_MODELS). LLMs excluded.",
    "families": "8 families x 8 bases (FAMILIES); emotion words HIGH/LOW per emotion (NRC max/min among EEC words).",
    "persons": ("Predictor: the 20 held-out EEC names (targeted_probe.SELECTION_NAMES). Outcome: the 15 Study 2 "
                "test pairs (5 gender noun phrases, 10 same-gender race name pairs). No overlap."),
    "cell": "model x family x intensity (HIGH or LOW): 18 x 8 x 2 = 288 cells.",
    "predictor": ("U = mean(1 - max_c P(c|x)) over selection-name sentences and bases; prediction = "
                  "sign(log U_filler/U_emotion), emotion pooled over the 4 words of the intensity level."),
    "outcome": ("S = mean primary sensitivity over test pairs and bases (|Delta_l| for binary models, TV for "
                "multi-class models); observed direction = sign(log S_filler/S_emotion). Logit outcome: "
                "|Delta_l| for every model, Delta_l = logit e(x) - logit e(x'), e = expected valence."),
    "clarity": ("A cell's direction is 'clear' if the 95% percentile CI of log S_filler/S_emotion from a two-way "
                "bootstrap over bases and pairs (2000 reps) excludes 0; otherwise 'ambiguous'. Ambiguous cells "
                "are excluded from accuracies and counted; nonsignificance is not read as no effect."),
    "usual_direction": "d_m = sign(log primary ratio) of model m in Study 2 (paper_table.csv), point estimate.",
    "baselines": ("Model-constant: usual direction d_m; label scheme (binary up, multi-class down); neutral "
                  "mass (Study 2 P_neu on filler < 0.5 -> up); fleet majority of d_m. All fixed before outcomes."),
    "H1": ("Generalization. Estimand: directional accuracy of the uncertainty prediction over clear cells "
           "(primary measure). Supported if the two-way cluster-bootstrap (models x families, 4000 reps) 95% "
           "lower bound exceeds 0.5. Secondary: Spearman between log U ratio and log S ratio over all 288 "
           "cells (no thresholding), with the same bootstrap."),
    "H2": ("Beyond the model's usual direction. Discordant cells: prediction differs from d_m. Estimand: "
           "accuracy of the uncertainty prediction over clear discordant cells (> 0.5 means it beats d_m). "
           "Supported if the cluster-bootstrap 95% lower bound exceeds 0.5 on the primary measure AND on the "
           "logit outcome; partially supported if only on the primary measure; not supported if the primary "
           "point estimate is <= 0.5; inconclusive otherwise. Secondary: mean within-model Spearman of log U "
           "vs log S across the model's 16 cells (model-centred), bootstrap over models."),
    "multiplicity": "Holm over the two primary one-sided tests (bootstrap p = share of replicates <= 0.5).",
    "gate": ("Before any outcome sentence is scored: count discordant cells from the uncertainty stage. If "
             "< 40, H2 is declared underpowered and not claimed; H1 is still run. 40 is a minimum, not a "
             "power guarantee."),
    "sensitivity": ("Entropy-based predictor (H(P)/log k); leave-one-family-out point estimates; clarity at "
                    "90% CI. All labelled sensitivity analyses."),
    "decision_table": {
        "H1 supported, H2 supported": "Generalizes to unseen families and adds information beyond the model's usual direction.",
        "H1 supported, H2 not supported/inconclusive": "Generalizes; incremental value beyond stable model characteristics not established.",
        "H1 not supported": "Narrow the predictive claim to the Study 2 templates.",
        "H1 inconclusive": "Insufficient evidence; claim neither generalization nor its absence.",
        "gate failed": "Report H1 only; H2 underpowered, no claim.",
    },
    "exploratory": "Anything not listed above, including LLMs, per-base cells and per-emotion results.",
}


def frames() -> pd.DataFrame:
    rows = []
    for fam, (_, emo_frame, fill_frame, bases) in FAMILIES.items():
        for b, base in enumerate(bases):
            rows.append({"family": fam, "base": b, "condition": "filler", "intensity": "", "word": "",
                         "template": fill_frame.replace("{X}", base)})
            for level, words in (("high", HIGH), ("low", LOW)):
                for w in words.values():
                    t = emo_frame.replace("{X}", base).replace("{e}", w).replace("{n}", NOUN[w])
                    rows.append({"family": fam, "base": b, "condition": "emotion", "intensity": level,
                                 "word": w, "template": t})
    return pd.DataFrame(rows)


def fill(template: str, person: str) -> str:
    s = template.replace("{person}", person)
    return s[0].upper() + s[1:]


def test_pairs() -> pd.DataFrame:
    p = T.test_pairs()
    return p[p.axis != "placebo"].reset_index(drop=True)


def plan_hash() -> str:
    return open(os.path.join(OUT, "plan.sha256"), encoding="utf-8").read().split()[0]


def current_hash() -> str:
    text = json.dumps({"plan": PLAN, "families": FAMILIES, "high": HIGH, "low": LOW, "noun": NOUN,
                       "gate": GATE_MIN_DISCORDANT, "seed": SEED}, indent=2, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def check_frozen() -> None:
    if plan_hash() != current_hash():
        raise SystemExit("plan or families changed since freeze; refusing to run")


def leakage_checks() -> list[str]:
    msgs = []
    pairs = test_pairs()
    test_persons = set(pairs.a) | set(pairs.b)
    assert not test_persons & set(T.SELECTION_NAMES), "selection names overlap test persons"
    msgs.append(f"selection names ({len(T.SELECTION_NAMES)}) and test persons ({len(test_persons)}) disjoint")
    old = [t.lower() for t in S.EEC_NEUTRAL_TEMPLATES]
    old_clauses = [c for pair in M.CLAUSES.values() for c in pair] + [f", and {c}." for c in T.CLAUSES]
    f = frames()
    for t in f.template:
        low = t.lower()
        assert not any(o.replace("{person}", "") in low for o in old), f"overlaps EEC-Neutral: {t}"
        assert not any(c.replace("{emotion}", "") in low for c in old_clauses), f"overlaps a Study 2/3 clause: {t}"
    msgs.append("no family template contains an EEC-Neutral stem or a Study 2/3 clause")
    nrc = set()
    with open(os.path.join("data", "NRC-AffectIntensity-Lexicon.txt"), encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                nrc.add(parts[0])
    bad = {fam: [w for w in ws if w in nrc] for fam, ws in FILLER_WORDS.items()}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"filler words in NRC lexicon: {bad}"
    msgs.append("filler words absent from the NRC Affect Intensity Lexicon")
    for fam, (_, ef, ff, _) in FAMILIES.items():
        n_e = len(re.sub(r"\{[a-zA-Z]\}", "W", ef.replace("{X}", "")).split())
        n_f = len(ff.replace("{X}", "").split())
        assert abs(n_e - n_f) <= 1, f"{fam}: emotion/filler length differ by more than one word"
    msgs.append("emotion and filler frames match in length within one word and in position")
    sel = {fill(t, n) for t in f.template for n in T.SELECTION_NAMES}
    out = {fill(t, p) for t in f.template for p in test_persons}
    assert not sel & out, "uncertainty and outcome sentences overlap"
    msgs.append(f"uncertainty sentences ({len(sel)}) and outcome sentences ({len(out)}) disjoint")
    return msgs


def freeze() -> None:
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "plan.sha256")):
        raise SystemExit("already frozen")
    msgs = leakage_checks()
    with open(os.path.join(OUT, "plan.json"), "w", encoding="utf-8") as f:
        json.dump({"plan": PLAN, "families": FAMILIES, "high": HIGH, "low": LOW, "noun": NOUN,
                   "gate": GATE_MIN_DISCORDANT, "seed": SEED}, f, indent=2, sort_keys=True)
    frames().to_csv(os.path.join(OUT, "templates.csv"), index=False)
    sha = current_hash()
    with open(os.path.join(OUT, "plan.sha256"), "w", encoding="utf-8") as f:
        f.write(f"{sha}  plan.json  frozen {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("\n".join(msgs))
    print(f"frozen, sha256 {sha}; {len(frames())} templates")


@torch.inference_mode()
def score(model_name: str, sentences: list[str], batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities (n, k) and logit of expected valence (n,), computed stably from logits."""
    tok, model = M.load_classifier(model_name)
    model.eval()
    w = torch.tensor(M.valence_weights(model.config.id2label), dtype=torch.float64)
    log_w, log_1mw = torch.log(w), torch.log1p(-w)
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    probs = np.empty((len(sentences), model.config.num_labels))
    lg = np.empty(len(sentences))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([sentences[i] for i in idx], padding=True, truncation=True, max_length=64, return_tensors="pt")
        logits = model(**enc).logits.double()
        probs[idx] = torch.softmax(logits, -1).numpy()
        lg[idx] = (torch.logsumexp(logits + log_w, -1) - torch.logsumexp(logits + log_1mw, -1)).numpy()
    return probs, lg


def uncertainty_stage() -> None:
    check_frozen()
    f = frames()
    for name in M.DEFAULT_MODELS:
        out = os.path.join(OUT, f"unc_{name.split('/')[-1]}.npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        sents = [fill(t, n) for t in f.template for n in T.SELECTION_NAMES]
        probs, _ = score(name, sents)
        np.savez_compressed(out, probs=probs.reshape(len(f), len(T.SELECTION_NAMES), -1))
        print(f"--- {name}: uncertainty scored ({time.time() - t0:.0f}s)", flush=True)
        gc.collect()


def uncertainty_cells(pred: str = "maxprob") -> pd.DataFrame:
    f = frames()
    rows = []
    for name in M.DEFAULT_MODELS:
        tag = name.split("/")[-1]
        p = np.load(os.path.join(OUT, f"unc_{tag}.npz"))["probs"]
        if pred == "maxprob":
            u = (1 - p.max(-1)).mean(1)
        else:
            u = (-(p * np.log(np.clip(p, 1e-300, 1))).sum(-1) / np.log(p.shape[-1])).mean(1)
        g = f.assign(u=u)
        for fam in FAMILIES:
            uf = g[(g.family == fam) & (g.condition == "filler")].u.mean()
            for level in ("high", "low"):
                ue = g[(g.family == fam) & (g.intensity == level)].u.mean()
                rows.append({"model": tag, "family": fam, "intensity": level, "log_u_ratio": float(np.log(uf / ue))})
    return pd.DataFrame(rows)


def study2_baselines() -> pd.DataFrame:
    t = pd.read_csv("results_minimal_pair/paper_table.csv", keep_default_na=False)
    pr = np.where(t.k == 2, t.lg, t.tv)
    d = np.sign(np.log(pr))
    majority = 1.0 if (d > 0).sum() >= (d < 0).sum() else -1.0
    return pd.DataFrame({"model": t.tag, "k": t.k, "usual": d, "label": np.where(t.k == 2, 1.0, -1.0),
                         "neutral_mass": np.where(t.nm_f < 0.5, 1.0, -1.0), "majority": majority})


def gate() -> None:
    check_frozen()
    if any(f.startswith("out_") for f in os.listdir(OUT)):
        raise SystemExit("outcome data already exist; the gate must be computed before outcomes")
    u = uncertainty_cells().merge(study2_baselines(), on="model")
    u["pred"] = np.sign(u.log_u_ratio)
    disc = int((u.pred != u.usual).sum())
    res = {"cells": int(len(u)), "discordant_cells": disc, "threshold": GATE_MIN_DISCORDANT,
           "passed": disc >= GATE_MIN_DISCORDANT,
           "discordant_by_model": u[u.pred != u.usual].groupby("model").size().to_dict(),
           "computed": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(OUT, "gate.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def outcomes_stage() -> None:
    check_frozen()
    if not os.path.exists(os.path.join(OUT, "gate.json")):
        raise SystemExit("run the gate first")
    f, pairs = frames(), test_pairs()
    persons = sorted(set(pairs.a) | set(pairs.b))
    for name in M.DEFAULT_MODELS:
        out = os.path.join(OUT, f"out_{name.split('/')[-1]}.npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        probs, lg = score(name, [fill(t, p) for t in f.template for p in persons])
        np.savez_compressed(out, probs=probs.reshape(len(f), len(persons), -1),
                            logit=lg.reshape(len(f), len(persons)), persons=np.array(persons))
        print(f"--- {name}: outcomes scored ({time.time() - t0:.0f}s)", flush=True)
        gc.collect()


def outcome_cells(measure: str, level_ci: float = 95.0) -> pd.DataFrame:
    """Per cell: log S_filler/S_emotion and its two-way (base x pair) bootstrap CI."""
    f, pairs = frames(), test_pairs()
    rng = np.random.default_rng(SEED)
    rows = []
    for name in M.DEFAULT_MODELS:
        tag = name.split("/")[-1]
        z = np.load(os.path.join(OUT, f"out_{tag}.npz"))
        idx = {p: i for i, p in enumerate(z["persons"].tolist())}
        ia, ib = [idx[a] for a in pairs.a], [idx[b] for b in pairs.b]
        k = z["probs"].shape[-1]
        use_tv = measure == "primary" and k > 2
        if use_tv:
            sens = 0.5 * np.abs(z["probs"][:, ia] - z["probs"][:, ib]).sum(-1)      # (templates, pairs)
        else:
            sens = np.abs(z["logit"][:, ia] - z["logit"][:, ib])
        for fam in FAMILIES:
            m = (f.family == fam).to_numpy()
            fill_rows = np.flatnonzero(m & (f.condition == "filler").to_numpy())  # one per base
            for level in ("high", "low"):
                emo = np.stack([np.flatnonzero(m & (f.intensity == level).to_numpy() & (f.base == b).to_numpy())
                                for b in range(8)])                                # (bases, 4 words)
                fs = sens[fill_rows]                                               # (bases, pairs)
                es = sens[emo].mean(1)                                             # (bases, pairs)
                point = float(np.log(fs.mean() / es.mean()))
                bb = rng.integers(0, 8, (N_CELL_BOOT, 8))
                pp = rng.integers(0, len(pairs), (N_CELL_BOOT, len(pairs)))
                num = np.array([fs[np.ix_(bb[i], pp[i])].mean() for i in range(N_CELL_BOOT)])
                den = np.array([es[np.ix_(bb[i], pp[i])].mean() for i in range(N_CELL_BOOT)])
                lr = np.log(num / den)
                a = (100 - level_ci) / 2
                lo, hi = np.percentile(lr, [a, 100 - a])
                rows.append({"model": tag, "family": fam, "intensity": level, "log_s_ratio": point,
                             "lo": float(lo), "hi": float(hi), "clear": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def cluster_boot(df: pd.DataFrame, stat, rng) -> np.ndarray:
    models, fams = df.model.unique(), df.family.unique()
    out = np.empty(N_CLUSTER_BOOT)
    by = {key: g for key, g in df.groupby(["model", "family"])}
    for i in range(N_CLUSTER_BOOT):
        ms, fs = rng.choice(models, len(models)), rng.choice(fams, len(fams))
        parts = [by[(m, f)] for m in ms for f in fs if (m, f) in by]
        out[i] = stat(pd.concat(parts)) if parts else np.nan
    return out


def accuracy(df: pd.DataFrame, col: str = "pred") -> float:
    c = df[df.clear]
    return float((np.sign(c[col]) == np.sign(c.log_s_ratio)).mean()) if len(c) else np.nan


def analyse() -> None:
    check_frozen()
    rng = np.random.default_rng(SEED + 1)
    base = study2_baselines()
    res = {}
    for measure in ("primary", "logit"):
        oc = outcome_cells(measure)
        oc.to_csv(os.path.join(OUT, f"cells_{measure}.csv"), index=False)
        for pred_kind in ("maxprob", "entropy"):
            d = oc.merge(uncertainty_cells(pred_kind), on=["model", "family", "intensity"]).merge(base, on="model")
            d["pred"] = np.sign(d.log_u_ratio)
            key = f"{measure}_{pred_kind}"
            disc = d[d.pred != d.usual]
            h1 = cluster_boot(d, accuracy, rng)
            h2 = cluster_boot(disc, accuracy, rng) if len(disc) else np.array([np.nan])
            rho = cluster_boot(d, lambda x: stats.spearmanr(x.log_u_ratio, x.log_s_ratio)[0], rng)
            within = [stats.spearmanr(g.log_u_ratio, g.log_s_ratio)[0] for _, g in d.groupby("model")
                      if g.log_s_ratio.std() > 0 and g.log_u_ratio.std() > 0]
            res[key] = {
                "cells": int(len(d)), "clear_cells": int(d.clear.sum()), "ambiguous_cells": int((~d.clear).sum()),
                "H1_accuracy": accuracy(d), "H1_ci": [float(np.nanpercentile(h1, 2.5)), float(np.nanpercentile(h1, 97.5))],
                "H1_boot_p": float(np.nanmean(h1 <= 0.5)),
                "H1_spearman_all_cells": float(stats.spearmanr(d.log_u_ratio, d.log_s_ratio)[0]),
                "H1_spearman_ci": [float(np.nanpercentile(rho, 2.5)), float(np.nanpercentile(rho, 97.5))],
                "baselines_accuracy_clear": {b: accuracy(d, b) for b in ("usual", "label", "neutral_mass", "majority")},
                "discordant_cells": int(len(disc)), "discordant_clear": int(disc.clear.sum()),
                "H2_accuracy": accuracy(disc) if len(disc) else None,
                "H2_ci": [float(np.nanpercentile(h2, 2.5)), float(np.nanpercentile(h2, 97.5))],
                "H2_boot_p": float(np.nanmean(h2 <= 0.5)),
                "H2_within_model_spearman_mean": float(np.mean(within)) if within else None,
                "lofo_H1": {fam: accuracy(d[d.family != fam]) for fam in FAMILIES},
                "lofo_H2": {fam: accuracy(disc[disc.family != fam]) for fam in FAMILIES},
            }
    # clarity at 90% CI, primary measure, max-prob predictor (sensitivity analysis)
    oc90 = outcome_cells("primary", 90.0).merge(uncertainty_cells(), on=["model", "family", "intensity"]).merge(base, on="model")
    oc90["pred"] = np.sign(oc90.log_u_ratio)
    res["sensitivity_ci90"] = {"H1_accuracy": accuracy(oc90), "H2_accuracy": accuracy(oc90[oc90.pred != oc90.usual]),
                               "clear_cells": int(oc90.clear.sum())}
    # decisions exactly as frozen in PLAN (primary measure, max-prob predictor)
    gate_passed = json.load(open(os.path.join(OUT, "gate.json"), encoding="utf-8"))["passed"]
    p, lg = res["primary_maxprob"], res["logit_maxprob"]
    h1 = ("supported" if p["H1_ci"][0] > 0.5 else "not supported" if p["H1_accuracy"] <= 0.5 else "inconclusive")
    if not gate_passed:
        h2 = "underpowered (gate failed); no claim"
    elif p["H2_accuracy"] is None or p["H2_accuracy"] <= 0.5:
        h2 = "not supported"
    elif p["H2_ci"][0] > 0.5 and lg["H2_ci"][0] > 0.5:
        h2 = "supported"
    elif p["H2_ci"][0] > 0.5:
        h2 = "partially supported (primary measure only)"
    else:
        h2 = "inconclusive"
    ps = sorted([("H1", p["H1_boot_p"]), ("H2", p["H2_boot_p"])], key=lambda x: x[1])
    holm = {ps[0][0]: min(1.0, 2 * ps[0][1]), ps[1][0]: min(1.0, max(2 * ps[0][1], ps[1][1]))}
    res["decision"] = {"H1": h1, "H2": h2, "holm_adjusted_boot_p": holm, "gate_passed": gate_passed}
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["freeze", "uncertainty", "gate", "outcomes", "analyse", "check"])
    s = ap.parse_args().stage
    if s == "check":
        print("\n".join(leakage_checks()))
        return
    {"freeze": freeze, "uncertainty": uncertainty_stage, "gate": gate,
     "outcomes": outcomes_stage, "analyse": analyse}[s]()


if __name__ == "__main__":
    main()
