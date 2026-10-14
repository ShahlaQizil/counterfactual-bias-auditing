"""
Fleet runner: reproduce the salience-competition audit across a set of public
HuggingFace sentiment models and report the DISTRIBUTION of the effect, not a
single-model number. This is the multi-system analogue of Kiritchenko &
Mohammad's 219 SemEval submissions.

For each model we compute, on the shared EEC-style probe:
  - H2 gap:  mean identity attribution share (neutral - emotional), + bootstrap CI
  - H1 slope: OLS of comp_ratio on affect intensity (expected ~0; switch not dial)
  - idea 5:  Spearman rho and magnitude ratio between occlusion and substitution
  - saturation: fraction of emotional probes pegged to the softmax rails

Each model is isolated in try/except: a download/label/dependency failure skips
that model and is recorded, so the run always produces a table.

Outputs (./results_salience/):
    fleet_results.csv    one row per model
    fleet_summary.json   aggregate H2-gap distribution across the fleet
    fleet_h2_gap.png     per-model H2 gap with bootstrap CIs (if matplotlib)

Run:
    python run_fleet.py
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import gc
import json
import os
import traceback

import numpy as np
import pandas as pd

import salience_experiment as S

# ~18 public sentiment models spanning architectures (distilbert/bert/roberta/
# xlm-roberta), label conventions (binary, 3-class, 5-star), languages, and
# domains (general/tweets/reviews/financial). Any that fail to download, need
# custom code, or lack an identifiable polarity are skipped and recorded, so the
# run always yields a table -- report the actual n audited, not the list length.
FLEET = [
    # binary English
    "distilbert-base-uncased-finetuned-sst-2-english",
    "siebert/sentiment-roberta-large-english",
    "Seethal/sentiment_analysis_generic_dataset",
    # 3-class general / tweets
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "cardiffnlp/twitter-roberta-base-sentiment",
    "finiteautomata/bertweet-base-sentiment-analysis",
    "j-hartmann/sentiment-roberta-large-english-3-classes",
    "Souvikcmsa/BERT_sentiment_analysis",
    "Souvikcmsa/Roberta_Sentiment_Analysis",
    # multilingual
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    "tabularisai/multilingual-sentiment-analysis",
    # star ratings
    "nlptown/bert-base-multilingual-uncased-sentiment",
    "LiYuan/amazon-review-sentiment-analysis",
    # financial domain
    "ProsusAI/finbert",
    "ahmedrachid/FinancialBERT-Sentiment-Analysis",
    "yiyanghkust/finbert-tone",
    # multilingual distil
    "philschmid/distilbert-base-multilingual-cased-sentiment-2",
]

RESULTS_DIR = S.RESULTS_DIR
SEED = S.SEED


def bootstrap_mean_diff(a, b, n, rng):
    a, b = np.asarray(a), np.asarray(b)
    diffs = np.empty(n)
    for i in range(n):
        diffs[i] = (rng.choice(a, len(a), replace=True).mean()
                    - rng.choice(b, len(b), replace=True).mean())
    return float(diffs.mean()), tuple(float(x) for x in np.percentile(diffs, [2.5, 97.5]))


def audit_one(model_name: str, df: pd.DataFrame, rng) -> dict:
    model = S.SentimentModel(model_name)
    res = S.measure(model, df)

    emo = res[res["has_emotion"]].dropna(subset=["comp_ratio"])
    neu_share = res.loc[~res["has_emotion"], "ias_all"]
    emo_share = res.loc[res["has_emotion"], "ias_all"]

    h2_diff, (h2_lo, h2_hi) = bootstrap_mean_diff(neu_share, emo_share, 1000, rng)
    slope, _, r = S.ols_slope(emo["intensity"], emo["comp_ratio"])
    both = res.dropna(subset=["attr_person_sub"])
    rho = S.spearman(both["attr_person_occ"].abs().values, both["attr_person_sub"].abs().values)
    # Saturation: fraction of emotional probes where the model pegs to ANY one
    # class (max class prob > 0.98) -- class-count-agnostic, unlike thresholding
    # the positive-class probability (which is meaningless for 5-star models).
    emo_sents = res.loc[res["has_emotion"], "sentence"].tolist()
    sat = float((model.max_prob(emo_sents) > 0.98).mean())

    return {
        "model": model_name,
        "status": "ok",
        "mean_ias_neutral": round(float(neu_share.mean()), 4),
        "mean_ias_emotional": round(float(emo_share.mean()), 4),
        "h2_gap": round(h2_diff, 4),
        "h2_gap_lo": round(h2_lo, 4),
        "h2_gap_hi": round(h2_hi, 4),
        "h2_supported": bool(h2_lo > 0),
        "h1_slope": round(slope, 4),
        "h1_r": round(r, 4),
        "idea5_spearman": round(rho, 4),
        "idea5_occ_over_sub": round(
            float(both["attr_person_occ"].abs().mean()
                  / (both["attr_person_sub"].abs().mean() + 1e-12)), 3),
        "emotional_saturation_frac": round(sat, 3),
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    S.EMOTION_INTENSITY.update(S.load_nrc_intensities())

    persons = S.build_person_terms()
    df = S.build_records(persons)
    print(f"Probe: {len(df)} sentences x {len(FLEET)} models\n")

    rows = []
    for name in FLEET:
        print(f"--- {name}")
        try:
            row = audit_one(name, df, rng)
            print(f"    H2 gap={row['h2_gap']:+.3f} CI[{row['h2_gap_lo']:+.3f},"
                  f"{row['h2_gap_hi']:+.3f}]  H1 slope={row['h1_slope']:+.3f}  "
                  f"idea5 occ/sub={row['idea5_occ_over_sub']:.2f}  sat={row['emotional_saturation_frac']:.2f}")
        except Exception as e:  # noqa: BLE001 - resilience is the point
            print(f"    SKIPPED: {type(e).__name__}: {e}")
            traceback.print_exc()
            row = {"model": name, "status": f"failed: {type(e).__name__}: {e}"}
        rows.append(row)
        gc.collect()  # release model weights before loading the next

    fleet = pd.DataFrame(rows)
    fleet.to_csv(os.path.join(RESULTS_DIR, "fleet_results.csv"), index=False)

    ok = fleet[fleet["status"] == "ok"]
    gaps = ok["h2_gap"].astype(float).values if len(ok) else np.array([])
    summary = {
        "fleet_size": len(FLEET),
        "n_ok": int(len(ok)),
        "n_failed": int((fleet["status"] != "ok").sum()),
        "h2_gap_distribution": {
            "mean": round(float(gaps.mean()), 4) if len(gaps) else None,
            "min": round(float(gaps.min()), 4) if len(gaps) else None,
            "max": round(float(gaps.max()), 4) if len(gaps) else None,
            "all_positive": bool((gaps > 0).all()) if len(gaps) else None,
            "n_h2_supported": int(ok["h2_supported"].sum()) if len(ok) else 0,
        },
        "h1_slopes": [round(float(x), 4) for x in ok["h1_slope"]] if len(ok) else [],
        "idea5_occ_over_sub": [round(float(x), 3) for x in ok["idea5_occ_over_sub"]] if len(ok) else [],
        "per_model": rows,
    }
    with open(os.path.join(RESULTS_DIR, "fleet_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Figure: per-model H2 gap with bootstrap CIs.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if len(ok):
            from matplotlib.lines import Line2D
            ok = ok.sort_values("h2_gap").reset_index(drop=True)
            labels = [m.split("/")[-1] for m in ok["model"]]
            y = np.arange(len(ok))
            gap = ok["h2_gap"].astype(float).values
            lo = gap - ok["h2_gap_lo"].astype(float).values
            hi = ok["h2_gap_hi"].astype(float).values - gap
            sig = ok["h2_gap_lo"].astype(float).values > 0
            n_sig = int(sig.sum())
            fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(ok) + 1.2))
            for i in range(len(ok)):
                ax.errorbar(gap[i], y[i], xerr=[[lo[i]], [hi[i]]], fmt="o",
                            color="#b3202e" if sig[i] else "#c9a227",
                            capsize=3, lw=1.4, ms=6)
            ax.axvline(0, color="grey", lw=1, ls="--")
            ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.margins(y=0.03)
            ax.set_xlabel("H2 gap: identity attribution share (neutral $-$ emotional)")
            ax.set_title(f"Identity attribution concentrates on neutral text (n={len(ok)} models)",
                         fontsize=11)
            ax.legend(handles=[
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#b3202e",
                       label=f"95% CI excludes 0 ({n_sig}/{len(ok)})"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#c9a227",
                       label=f"CI touches 0 ({len(ok) - n_sig}/{len(ok)})")],
                fontsize=8, loc="lower right", frameon=True)
            fig.subplots_adjust(left=0.42, right=0.97, top=0.94, bottom=0.09)
            fig.savefig(os.path.join(RESULTS_DIR, "fleet_h2_gap.png"), dpi=150)
            print(f"\nFigure -> {RESULTS_DIR}/fleet_h2_gap.png")
    except Exception as e:  # noqa: BLE001
        print(f"\n(figure skipped: {e})")

    print(f"\nFleet: {len(ok)}/{len(FLEET)} audited. "
          f"H2 gap mean={summary['h2_gap_distribution']['mean']}, "
          f"all positive={summary['h2_gap_distribution']['all_positive']}")
    print(f"Results in ./{RESULTS_DIR}/")


if __name__ == "__main__":
    main()
