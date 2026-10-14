"""
Generate the per-model LaTeX table rows for paper_v2.tex from the result files,
so the paper's numbers cannot drift from the released results.

Inputs:  results_minimal_pair/paper_table.csv, results_eec/eec_audit.csv
Outputs: The paper/figures_tables/table_minimal_pair.tex, table_eec.tex, table_llm.tex,
         table_natural.tex (from results_natural/natural_summary.csv)

Run:
    python make_paper_tables.py
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import json

import numpy as np
import pandas as pd

SHORT = {
    "finbert-tone": "finbert-tone",
    "FinancialBERT-Sentiment-Analysis": "FinancialBERT",
    "sentiment-roberta-large-english-3-classes": "roberta-large-3cls",
    "finbert": "finbert (ProsusAI)",
    "BERT_sentiment_analysis": r"BERT\_sentiment",
    "sentiment_analysis_generic_dataset": r"generic\_dataset",
    "amazon-review-sentiment-analysis": "amazon-review",
    "multilingual-sentiment-analysis": "multilingual-5cls",
    "bertweet-base-sentiment-analysis": "bertweet",
    "twitter-roberta-base-sentiment-latest": "twitter-roberta-latest",
    "distilbert-base-multilingual-cased-sentiment-2": "distil-multi-2",
    "distilbert-base-multilingual-cased-sentiments-student": "distil-multi-student",
    "bert-base-multilingual-uncased-sentiment": "bert-multi-5star",
    "twitter-roberta-base-sentiment": "twitter-roberta",
    "twitter-xlm-roberta-base-sentiment": "twitter-xlm-roberta",
    "sentiment-roberta-large-english": "roberta-large-2cls",
    "Roberta_Sentiment_Analysis": r"Roberta\_Sentiment",
    "distilbert-base-uncased-finetuned-sst-2-english": "distilbert-sst-2",
}
MARK = {"eff": r"$\uparrow$", "rev": r"$\downarrow$", "null": r"$\circ$"}


def fmt(x: float) -> str:
    if x < 0.1:
        return f"{x:.3f}"
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def minimal_pair_rows() -> list[str]:
    # keep_default_na=False: the category label "null" must not be read as missing
    t = pd.read_csv("results_minimal_pair/paper_table.csv", keep_default_na=False)
    binary = t["k"] == 2
    # primary measure: logit(e) for binary models, TV distance for multi-class models
    t["pr"] = np.where(binary, t["lg"], t["tv"])
    t["plo"] = np.where(binary, t["lg_lo"], t["tv_lo"])
    t["phi"] = np.where(binary, t["lg_hi"], t["tv_hi"])
    # category from the Holm-corrected bootstrap p-value (robust_stats.py), not the raw CI
    r = pd.read_csv("results_minimal_pair/robust_stats.csv", keep_default_na=False)
    prim = r[((r["k"] == 2) & (r["measure"] == "logit")) | ((r["k"] > 2) & (r["measure"] == "tv"))]
    holm_cat = dict(zip(prim["model"], prim["category"].map(
        {"effect": "eff", "reversed": "rev", "null": "null"})))
    t["pcat"] = t["tag"].map(holm_cat)
    assert t["pcat"].notna().all()
    lg_rows = r[r["measure"] == "logit"]
    t["lcat"] = t["tag"].map(dict(zip(lg_rows["model"], lg_rows["category"].map(
        {"effect": "eff", "reversed": "rev", "null": "null"}))))
    t = t.sort_values("pr")
    rows = []
    for r in t.itertuples():
        nm = "--" if r.k == 2 else f"{r.nm_f:.2f}"
        rows.append(
            f"{SHORT[r.tag]} & {r.k} & {nm} & {r.unc_f:.3f} & {r.unc_e:.3f} & "
            f"{fmt(r.pr)} [{fmt(r.plo)}, {fmt(r.phi)}] & {MARK[r.pcat]} & "
            f"{fmt(r.lg)}{MARK[r.lcat]} & {fmt(r.tvb)} & {r.spec:.2f} \\\\")
    print("primary categories:", t["pcat"].value_counts().to_dict())
    return rows, list(t["tag"])


def eec_rows(order: list[str]) -> list[str]:
    e = pd.read_csv("results_eec/eec_audit.csv")
    e = e[e["scale"] == "logit"].assign(tag=lambda d: d["model"].str.split("/").str[-1])
    rows = []
    for tag in order:
        g = e[e["tag"] == tag]
        cells = []
        for axis in ("gender", "race"):
            a = g[g["axis"] == axis].set_index("subset")
            for sub in ("full", "emotional", "neutral", "eec_neutral"):
                star = "^{*}" if a.at[sub, "flagged"] else ""
                cells.append(f"${a.at[sub, 'd_z']:+.2f}{star}$")
            cells.append(fmt(a.at["eec_neutral", "mean_abs_delta"] / a.at["emotional", "mean_abs_delta"]))
        rows.append(f"{SHORT[tag]} & " + " & ".join(cells) + " \\\\")
    return rows


def llm_rows() -> list[str]:
    """One row per LLM variant (size x label scheme x prompt wording), from results_llm/summary.csv."""
    s = pd.read_csv("results_llm/summary.csv", keep_default_na=False)
    s["size"] = s["model"].str.extract(r"(0\.5B|1\.5B)")[0]
    s["scheme"] = s["model"].str.extract(r"@(\dcls)")[0].map({"2cls": "binary", "3cls": "3-class"})
    s["wording"] = s["model"].str.extract(r"-(p\d)$")[0].fillna("p1")
    s = s.sort_values(["size", "scheme", "wording"])
    mark = {"effect": r"$\uparrow$", "reversed": r"$\downarrow$", "null": r"$\circ$"}
    rows = []
    for r in s.itertuples():
        unc_e = "$<$0.001" if r.unc_emotion < 0.001 else f"{r.unc_emotion:.3f}"
        nm = "--" if r.k == 2 else f"{r.neutral_mass_filler:.2f}"
        rows.append(
            f"Qwen2.5-{r.size} & {r.scheme} & {r.wording} & {fmt(r.ratio)} [{fmt(r.lo)}, {fmt(r.hi)}] & "
            f"{mark[r.category]} & {fmt(r.logit_ratio)} & {r.unc_filler:.3f} & {unc_e} & {nm} & "
            f"{100 * r.flip_filler:.1f}\\% / {100 * r.flip_emotion:.1f}\\% \\\\")
    return rows


def natural_rows(order: list[str]) -> list[str]:
    """Natural-sentence check (natural_sentence_check.py): classifiers in Table 2 order, then the LLMs."""
    n = pd.read_csv("results_natural/natural_summary.csv", keep_default_na=False)
    n["tag"] = n["model"].str.split("/").str[-1]
    llm = n[n["tag"].str.contains("@")].sort_values("tag")
    n = pd.concat([n.set_index("tag").loc[order].reset_index(), llm])
    verdict = {"F>M": "F$>$M", "M>F": "M$>$F", "none": "--"}
    rows = []
    for r in n.itertuples():
        if "@" in r.tag:
            size, scheme = r.tag.split("@")
            name = f"{size.replace('-Instruct', '')} {'binary' if scheme == '2cls' else '3-class'}"
        else:
            name = SHORT[r.tag]
        lo, hi = json.loads(r.sens_ratio_ci)
        cat = "null" if r.p_sens_ratio_holm >= 0.05 else "eff" if r.sens_ratio > 1 else "rev"
        rows.append(
            f"{name} & {r.rho_within:.2f} & {r.tv_low:.3f} & {r.tv_high:.3f} & "
            f"{r.sens_ratio:.2f} [{lo:.2f}, {hi:.2f}] & {MARK[cat]} & {r.unc_ratio:.2f} & "
            f"{verdict[r.verdict_low]} & {verdict[r.verdict_high]} \\\\")
    return rows


def model_rows(order: list[str]) -> list[str]:
    """Full Hugging Face identifiers: classifiers in Table 2 order, then the LLMs."""
    t = pd.read_csv("results_minimal_pair/paper_table.csv", keep_default_na=False).set_index("tag")
    rows = [f"{SHORT[tag]} & \\texttt{{{t.at[tag, 'model'].replace('_', chr(92) + '_')}}} & {t.at[tag, 'k']} \\\\"
            for tag in order]
    for size in ("0.5B", "1.5B"):
        rows.append(f"Qwen2.5-{size} & \\texttt{{Qwen/Qwen2.5-{size}-Instruct}} & 2 / 3 \\\\")
    return rows


def family_rows() -> list[str]:
    """Unseen template families (template_family_test.py): structure and one example pair."""
    plan = json.load(open("results_template_family/plan.json", encoding="utf-8"))
    rows = []
    for fam, (desc, emo, fill, bases) in plan["families"].items():
        def ex(frame: str) -> str:
            s = frame.replace("{X}", bases[0]).replace("{e}", "furious").replace("{n}", "fury")
            s = s.replace("{person}", "my sister")
            return s[0].upper() + s[1:]
        rows.append(f"{fam} & {desc} & \\emph{{{ex(emo)}}} / \\emph{{{ex(fill)}}} \\\\")
    return rows


def main() -> None:
    _os.makedirs(project.PAPER_TABLES, exist_ok=True)
    with open(_os.path.join(project.PAPER_TABLES, "table_llm.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(llm_rows()) + "\n")
    mp, order = minimal_pair_rows()
    with open(_os.path.join(project.PAPER_TABLES, "table_minimal_pair.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(mp) + "\n")
    with open(_os.path.join(project.PAPER_TABLES, "table_eec.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(eec_rows(order)) + "\n")
    with open(_os.path.join(project.PAPER_TABLES, "table_natural.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(natural_rows(order)) + "\n")
    with open(_os.path.join(project.PAPER_TABLES, "table_models.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(model_rows(order)) + "\n")
    with open(_os.path.join(project.PAPER_TABLES, "table_families.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(family_rows()) + "\n")
    print("\n".join(mp))


if __name__ == "__main__":
    main()
