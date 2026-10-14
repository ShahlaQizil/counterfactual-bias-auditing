"""
Headline figure for paper_v2: identity sensitivity vs predictive uncertainty.

x: uncertainty ratio (filler / emotion), mean 1 - max_c P(c|x)
y: identity-sensitivity ratio (filler / emotion), primary measure
   (|Delta_logit| for binary models, TV for multi-class), 95% two-way bootstrap CI
Colour and marker shape encode the label scheme (so the figure survives
greyscale printing); hollow markers are not significant after Holm correction.

Inputs:  results_minimal_pair/robust_stats.csv, results_minimal_pair/paper_table.csv
Outputs: The paper/figures_tables/fig_uncertainty.pdf (+ .png preview), and a copy of the
         PDF in results_minimal_pair/ for the code release

Run:
    python make_figure.py
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# reference categorical palette, slots 1-3 (validated all-pairs in the dataviz reference)
SCHEMES = {2: ("binary", "#2a78d6", "o"), 3: ("3-class", "#eb6834", "s"), 5: ("5-class", "#1baf7a", "^")}
INK, MUTED = "#0b0b0b", "#52514e"
LABEL = {
    "distilbert-base-uncased-finetuned-sst-2-english": "distilbert-sst-2",
    "Roberta_Sentiment_Analysis": "Roberta_Sentiment",
    "sentiment-roberta-large-english": "roberta-large-2cls",
    "sentiment-roberta-large-english-3-classes": "roberta-large-3cls",
    "finbert-tone": "finbert-tone",
    "FinancialBERT-Sentiment-Analysis": "FinancialBERT",
    "finbert": "finbert",
    "twitter-xlm-roberta-base-sentiment": "twitter-xlm-roberta",
}


def main() -> None:
    r = pd.read_csv("results_minimal_pair/robust_stats.csv", keep_default_na=False)
    prim = r[((r["k"] == 2) & (r["measure"] == "logit")) | ((r["k"] > 2) & (r["measure"] == "tv"))]
    u = pd.read_csv("results_minimal_pair/paper_table.csv", keep_default_na=False)[["tag", "unc_f", "unc_e"]]
    d = prim.merge(u, left_on="model", right_on="tag")
    assert len(d) == 18
    d["x"] = d["unc_f"] / d["unc_e"]
    rho = spearmanr(np.log(d["x"]), np.log(d["ratio"])).statistic

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.axvline(1, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    for k, (name, color, marker) in SCHEMES.items():
        g = d[d["k"] == k]
        ax.errorbar(g["x"], g["ratio"], yerr=[g["ratio"] - g["lo"], g["hi"] - g["ratio"]],
                    fmt="none", ecolor=color, elinewidth=1, alpha=0.6, zorder=2)
        sig = g["category"] != "null"
        ax.scatter(g.loc[sig, "x"], g.loc[sig, "ratio"], s=34, marker=marker, color=color,
                   edgecolor="white", linewidth=1, zorder=3, label=name)
        ax.scatter(g.loc[~sig, "x"], g.loc[~sig, "ratio"], s=34, marker=marker, facecolor="white",
                   edgecolor=color, linewidth=1.4, zorder=3)
    # per-label offsets (points) keep the top-right labels apart and inside the axes
    offset = {"distilbert-sst-2": (-6, 5, "right"), "roberta-large-2cls": (-6, 4, "right")}
    for row in d.itertuples():
        if row.model in LABEL:
            dx, dy, ha = offset.get(LABEL[row.model], (5, 2, "left"))
            ax.annotate(LABEL[row.model], (row.x, row.ratio), xytext=(dx, dy), ha=ha,
                        textcoords="offset points", fontsize=6.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Uncertainty ratio, filler / emotion")
    ax.set_ylabel("Identity-sensitivity ratio, filler / emotion")
    ax.text(0.02, 0.97, f"Spearman $\\rho$ = {rho:.2f}, n = 18", transform=ax.transAxes,
            va="top", fontsize=7.5, color=INK)
    ax.text(0.98, 0.03, "hollow: n.s. after Holm", transform=ax.transAxes, ha="right",
            fontsize=6.5, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.07), fontsize=7)
    fig.tight_layout()
    # no creation date / timezone in the PDF metadata (anonymous submission)
    _os.makedirs(project.PAPER_TABLES, exist_ok=True)
    for out in (_os.path.join(project.PAPER_TABLES, "fig_uncertainty.pdf"), "results_minimal_pair/fig_uncertainty.pdf"):
        fig.savefig(out, metadata={"CreationDate": None})
    fig.savefig(_os.path.join(project.PAPER_TABLES, "fig_uncertainty.png"), dpi=200)
    print(f"rho={rho:.3f} -> {project.PAPER_TABLES}")


if __name__ == "__main__":
    main()
