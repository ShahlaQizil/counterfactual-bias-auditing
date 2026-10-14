"""
Consistency audit for paper_v2.tex: recompute every headline number from the
result files and check that the paper states it. Prints PASS/FAIL per claim;
exit code 1 if anything fails. Rerun after any edit to the paper or results.

Generated tables (\\input files) are checked by regenerating them with
make_paper_tables.py and comparing.

Run:
    python audit_numbers.py
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import glob
import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_TEX_PATH = _os.path.join(project.PAPER, "paper_v2.tex")
if not _os.path.exists(_TEX_PATH):
    print("audit_numbers.py needs the manuscript source (paper_v2.tex), which is not part of this folder.")
    sys.exit(0)
TEX = open(_TEX_PATH, encoding="utf-8").read()
FLAT = re.sub(r"\s+", " ", TEX)
results: list[tuple[bool, str]] = []


def check(label: str, *needles: str) -> None:
    ok = all(n in FLAT for n in needles)
    results.append((ok, f"{label}: {' | '.join(needles)}"))


def f2(x: float) -> str:
    return f"{x:.2f}"


# --- Study 2 (classifiers): Holm categories, primary measure -------------------
r = pd.read_csv("results_minimal_pair/robust_stats.csv", keep_default_na=False)
prim = r[((r.k == 2) & (r.measure == "logit")) | ((r.k > 2) & (r.measure == "tv"))]
c = prim.category.value_counts()
check("classifier Holm counts", f"above 1 for {c['effect']} models",
      f"not distinguishable from 1 for {c['null']}", f"significantly below 1 for {c['reversed']}")
lg = r[r.measure == "logit"]
multi_rev_logit = ((lg.k > 2) & (lg.category == "reversed")).sum()
tv_rev = ((r.measure == "tv") & (r.k > 2) & (r.category == "reversed")).sum()
check("measure flip counts", f"{multi_rev_logit} of the 15 multi-class models", f"under TV only {tv_rev} do")

t = pd.read_csv("results_minimal_pair/paper_table.csv", keep_default_na=False)
check("specificity median", f"Median control-token specificity is {t.spec.median():.2f}")
spec_cat = t.spec_cat.value_counts()
check("specificity counts", f"above 1 in {spec_cat['eff']}", f"below 1 in {spec_cat['rev']}",
      f"not distinguishable from 1 in {spec_cat['null']}")
check("IAS", f"mean {t.ias_f.mean():.3f} vs.\\ {t.ias_e.mean():.3f}", f"in {(t.ias_f > t.ias_e).sum()} of 18 models")
rho = spearmanr(np.log(t.tv), np.log(t.unc_f / t.unc_e)).statistic
rho_b = spearmanr(np.log(t.tvb), np.log(t.unc_b / t.unc_e)).statistic
check("uncertainty rho (TV)", f"Spearman $\\rho = {rho:.2f}$")
check("uncertainty rho (bare)", f"($\\rho={rho_b:.2f}$)")
bin_lg = t[t.k == 2].lg
check("binary logit range", f"ratios {bin_lg.min():.1f}--{bin_lg.max():.1f} on the logit scale")
tvb = t.tvb_cat.value_counts()
check("bare vs emotion counts", f"({tvb['eff']} above 1, {tvb['rev']} below")

# --- Label flips -----------------------------------------------------------------
f = pd.read_csv("results_probs/label_flips.csv")
check("flip direction counts", f"emotional than on filler sentences in {(f.mp_emotion > f.mp_filler).sum()} models",
      f"less often in {(f.mp_filler > f.mp_emotion).sum()}")

# --- Study 1: EEC ---------------------------------------------------------------
e = pd.read_csv("results_eec/eec_audit.csv")
el = e[e.scale == "logit"]
for axis, label in (("gender", "Gender"), ("race", "Race")):
    w = el[el.axis == axis].pivot_table(index="model", columns="subset", values="flagged", aggfunc="first").astype(bool)
    print(f"[info] EEC {axis}: full {w.full.sum()}, neutral-4 {w.neutral.sum()}, EEC-N {w.eec_neutral.sum()}")
eg = el[el.axis == "gender"].pivot_table(index="model", columns="subset", values="flagged", aggfunc="first").astype(bool)
er = el[el.axis == "race"].pivot_table(index="model", columns="subset", values="flagged", aggfunc="first").astype(bool)
# Table 1 now has original and robust columns; checked in the cluster-robust block below
results.append((int(eg.full.sum()) == 13 and int(er.full.sum()) == 11 and int(eg.eec_neutral.sum()) == 10,
                "EEC original-protocol counts as published (13 / 11; EEC-N gender 10)"))
check("contributions EEC-N recovers", "a larger neutral set recovers biases the original four templates miss")

# --- Study 3 (targeted) ------------------------------------------------------------
tg = pd.concat([pd.read_csv(x) for x in glob.glob("results_targeted/*.csv") if "summary" not in x])
fl = tg.groupby(["set", "axis"]).flagged.sum()
check("targeted table row", f"Targeted (max.\\ uncertainty) & \\textbf{{{fl['targeted', 'gender']}}} & "
      f"\\textbf{{{fl['targeted', 'race']}}} & {fl['targeted', 'placebo']}")
check("random table row", f"Random (same pool) & {fl['random', 'gender']} & {fl['random', 'race']} & {fl['random', 'placebo']}")
check("neutral table row", f"EEC-Neutral & {fl['neutral', 'gender']} & {fl['neutral', 'race']} & {fl['neutral', 'placebo']}")
check("abstract targeted", f"directional gender disparity in {fl['targeted', 'gender']} of 18 classifiers",
      "at most 6 for standard template sets")

# false-reassurance paragraph: biases significant on full EEC but not on EEC-Neutral, and the
# sensitivity ranges of the three near-certain classifiers
missed_g = (eg.full & ~eg.eec_neutral).sum()
missed_r = (er.full & ~er.eec_neutral).sum()
check("false reassurance counts", f"In our EEC replication, {missed_g} gender and {missed_r} race biases")
three = ["FinancialBERT-Sentiment-Analysis", "finbert-tone", "sentiment-roberta-large-english-3-classes"]
el3 = el.assign(tag=el.model.str.split("/").str[-1])
el3 = el3[el3.tag.isin(three)]
neu_max = el3[el3.subset == "eec_neutral"].mean_abs_delta.max()
emo = el3[el3.subset == "emotional"].mean_abs_delta
check("false reassurance ranges", f"$|\\Delta_\\ell|\\leq {neu_max + 0.005:.2f}$" if neu_max < 0.025 else f"$|\\Delta_\\ell|\\leq {neu_max:.2f}$",
      f"({emo.min():.2f}--{emo.max():.2f})")
llm_t = pd.concat([pd.read_csv(x) for x in glob.glob("results_llm/*_targeted.csv")])
pooled = pd.concat([tg, llm_t]).groupby(["set", "axis"]).flagged.sum()
check("pooled targeted gender", f"gender biases ({pooled['targeted', 'gender']}, against {pooled['neutral', 'gender']} for EEC-Neutral")
check("pooled targeted race", f"race ({pooled['targeted', 'race']}, against {pooled['random', 'race']} for random")

# --- LLMs ---------------------------------------------------------------------------
s = pd.read_csv("results_llm/summary.csv", keep_default_na=False)
n_eff = (s.category == "effect").sum()
check("LLM robustness count", f"In {n_eff} of the {len(s)} variants")
s["unc_ratio"] = (s.unc_filler + 1e-6) / (s.unc_emotion + 1e-6)
sig = s.category != "null"
agree = (np.sign(np.log(s.ratio)) == np.sign(np.log(s.unc_ratio))) & sig
check("LLM uncertainty agreement", f"In {agree.sum()} of the {sig.sum()} variants with a significant effect")
rho_llm = spearmanr(np.log(s.ratio), np.log(s.unc_ratio))
check("LLM rho", f"$\\rho={rho_llm.statistic:.2f}$ ($p={rho_llm.pvalue:.2f}$)")
eff = s[s.category == "effect"].ratio
check("LLM ratio range (significant variants)", f"among these {len(eff)}, ratios range from {eff.min():.1f} to {eff.max():.1f}")
orig = s[~s.model.str.contains("-p")]
u = t[["tag", "unc_f", "unc_e"]]
pf = prim.merge(u, left_on="model", right_on="tag")
allr = np.r_[pf.ratio, orig.ratio]
allu = np.r_[pf.unc_f / pf.unc_e, orig.unc_ratio]
rho22 = spearmanr(np.log(allr), np.log(allu)).statistic
check("pooled 22 rho (body, abstract)", f"$\\rho={rho22:.2f}$",
      f"four original-prompt LLM configurations, the sensitivity ratio tracks the uncertainty ratio "
      f"(Spearman $\\rho{{=}}{rho22:.2f}$)")
check("abstract LLM rho", f"weaker ($\\rho{{=}}{rho_llm.statistic:.2f}$)")

# --- Study 3 inline numbers ------------------------------------------------------------
from scipy.stats import wilcoxon  # noqa: E402

tg["tag"] = tg.model.str.split("/").str[-1]
efull = el[el.subset == "full"].assign(tag=lambda d: d.model.str.split("/").str[-1])
n_both = n_agree = 0
for axis in ("gender", "race"):
    tt = tg[(tg.axis == axis) & (tg.set == "targeted")].set_index("tag")
    ee = efull[efull.axis == axis].set_index("tag").reindex(tt.index)
    both = tt.flagged & ee.flagged.astype(bool)
    n_both += both.sum()
    n_agree += (both & (np.sign(tt.d_z) == np.sign(ee.d_z))).sum()
check("targeted direction agreement", f"agree in direction in {n_agree} of {n_both} cases")
wz = {}
for axis in ("gender", "race"):
    w = tg[tg.axis == axis].pivot(index="tag", columns="set", values="d_z").abs()
    for other in ("random", "neutral", "emotional"):
        wz[axis, other] = wilcoxon(w.targeted, w[other]).pvalue
check("Wilcoxon random", f"for race (Wilcoxon signed-rank, $p={wz['race', 'random']:.3f}$; "
      f"gender $p={wz['gender', 'random']:.3f}$)")
check("Wilcoxon neutral race", f"race ($p={wz['race', 'neutral']:.3f}$)")
check("Wilcoxon emotional gender", f"gender ($p={wz['gender', 'emotional']:.3f}$)")

# Study 3 model-level paired comparisons (exact McNemar on discordant models, Holm over six)
from scipy.stats import binomtest  # noqa: E402

mc = []
for axis in ("gender", "race"):
    w = tg[tg.axis == axis].pivot(index="tag", columns="set", values="flagged").astype(bool)
    for other in ("random", "neutral", "emotional"):
        b_, c_ = int((w.targeted & ~w[other]).sum()), int((~w.targeted & w[other]).sum())
        mc.append((axis, other, b_, c_, binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0))
p_raw = np.array([m[4] for m in mc])
order_ = np.argsort(p_raw)
p_holm = np.empty_like(p_raw)
run = 0.0
for rank, i in enumerate(order_):
    run = max(run, (len(p_raw) - rank) * p_raw[i])
    p_holm[i] = min(1.0, run)
mcd = {(m[0], m[1]): (m[2], m[3], m[4], p_holm[i]) for i, m in enumerate(mc)}
ge = mcd["gender", "emotional"]
check("McNemar gender vs emotional", f"({ge[0]} vs.\\ {ge[1]} discordant models",
      f"$p_{{\\text{{Holm}}}}={ge[3]:.3f}$")
gr, gn = mcd["gender", "random"], mcd["gender", "neutral"]
check("McNemar gender vs random/neutral", f"({gr[0]} vs.\\ {gr[1]} and {gn[0]} vs.\\ {gn[1]})",
      f"($p={gr[2]:.3f}$ and ${gn[2]:.3f}$ before")
others_sig = [k for k, v in mcd.items() if v[3] < 0.05 and k != ("gender", "emotional")]
results.append((not others_sig, f"only gender-vs-emotional significant after Holm (others: {others_sig})"))

# correlation robustness: leave-one-out range and model bootstrap (seed and order as in the analysis)
xt, yt = np.log(t.tv.values), np.log((t.unc_f / t.unc_e).values)
prim_tab = prim.merge(t[["tag", "unc_f", "unc_e"]], left_on="model", right_on="tag")
xp, yp = np.log(prim_tab.ratio.values), np.log((prim_tab.unc_f / prim_tab.unc_e).values)
loo = lambda x, y: [spearmanr(np.delete(x, i), np.delete(y, i)).statistic for i in range(len(x))]  # noqa: E731
lt, lp = loo(xt, yt), loo(xp, yp)
rng_ = np.random.default_rng(20260713)


def boot(x, y):
    v = []
    for _ in range(10000):
        i = rng_.integers(0, len(x), len(x))
        v.append(spearmanr(x[i], y[i]).statistic)
    return np.nanpercentile(v, [2.5, 97.5])


bt, bp = boot(xt, yt), boot(xp, yp)
check("correlation LOO", f"it ranges from {min(lt):.2f} to {max(lt):.2f} ({min(lp):.2f} to {max(lp):.2f} on the primary")
check("correlation bootstrap", f"interval of {bt[0]:.2f}--{bt[1]:.2f}", f"({bp[0]:.2f}--{bp[1]:.2f})")

# the released analysis script must agree with the independent computation above
saved_pt = pd.read_csv("results_stats/paired_tests.csv")
saved_cr = json.load(open("results_stats/correlation_robustness.json"))
same_pt = all(abs(row.p_mcnemar_holm - mcd[row.axis, row.comparison.split(" vs ")[1]][3]) < 1e-9
              for row in saved_pt.itertuples())
same_cr = (abs(saved_cr["tv"]["loo_min"] - min(lt)) < 1e-9 and abs(saved_cr["tv"]["boot_lo"] - bt[0]) < 1e-9
           and abs(saved_cr["primary"]["boot_hi"] - bp[1]) < 1e-9)
results.append((same_pt and same_cr, "results_stats/ outputs match the audit's independent computation"))

# --- label-flip maxima ------------------------------------------------------------------
fi = f.set_index("model")
check("flip maxima", f"{100 * fi.loc['Roberta_Sentiment_Analysis', 'mp_filler']:.1f}\\% of filler pairs",
      f"{100 * fi.loc['bert-base-multilingual-uncased-sentiment', 'mp_filler']:.1f}\\% for bert-multi-5star")

# --- LLM regime numbers --------------------------------------------------------------------
sm = s.set_index("model")
p1 = sm.loc["Qwen/Qwen2.5-1.5B-Instruct@3cls"]
p3 = sm.loc["Qwen/Qwen2.5-1.5B-Instruct@3cls-p3"]
check("1.5B 3cls p1 regime", f"{100 * p1.neutral_mass_filler:.0f}\\% of probability",
      f"uncertainty {p1.unc_emotion:.3f} vs.\\ {p1.unc_filler:.3f}")
check("1.5B 3cls p3 regime", f"{100 * p3.neutral_mass_filler:.1f}\\% neutral",
      f"uncertainty {p3.unc_filler:.3f} against {p3.unc_emotion:.3f}",
      f"({p3.ratio:.2f} [{p3.lo:.2f}, {p3.hi:.2f}]")

# --- coupling check: association on the logit scale (not compressed near 0 and 1) ----------
_pt = pd.read_csv("results_minimal_pair/paper_table.csv", keep_default_na=False)
rho_lg = spearmanr(np.log(_pt.lg), np.log(_pt.unc_f / _pt.unc_e))[0]
check("logit-scale association", f"(Spearman $\\rho={rho_lg:.2f}$ across the 18 models, using $|\\Delta_\\ell|$ for every model)")

# --- EEC biases missed by EEC-Neutral: effect sizes ---------------------------------------
_e = pd.read_csv("results_eec/eec_audit.csv")
_g = _e[(_e.scale == "logit") & (_e.axis == "gender")].pivot(index="model", columns="subset", values=["d_z", "flagged"])
_m = _g[(_g["flagged"]["full"] == True) & (_g["flagged"]["eec_neutral"] == False)]  # noqa: E712
_dn, _de = _m["d_z"]["eec_neutral"].abs(), _m["d_z"]["emotional"].abs()
results.append((bool((_dn < _de).all()) and len(_m) == 4, "missed gender biases: |d_z| smaller on EEC-Neutral in all four"))

# --- robustness_checks.py: uncertainty definitions and equal-size EEC sets -----------------
_rc = json.load(open("results_stats/robustness_checks.json", encoding="utf-8"))
_ud, _eq = _rc["uncertainty_definitions"], _rc["eec_equal_size"]
results.append((_ud["maxprob_ratio_max_rel_deviation_from_paper"] < 1e-4, "robustness: max-prob ratios reproduce the paper (float32 cache)"))
check("uncertainty definitions (templates)",
      f"the correlation is {_ud['rho_tv_entropy']:.2f} under both definitions on TV" if f"{_ud['rho_tv_entropy']:.2f}" == f"{_ud['rho_tv_margin']:.2f}" else "MISMATCH",
      f"({_ud['rho_logit_entropy']:.2f} and {_ud['rho_logit_margin']:.2f} on $|\\Delta_\\ell|$)",
      f"within-model correlation is {_ud['within_median_entropy']:.2f} and {_ud['within_median_margin']:.2f}")
results.append((_ud["direction_agree_entropy"] == 15 and _ud["direction_agree_margin"] == 18, "robustness: three direction changes under entropy, none under margin"))
check("uncertainty definitions (natural)",
      f"the medians are {_ud['natural_within_median_entropy']:.2f} and {_ud['natural_within_median_margin']:.2f}")
results.append((_ud["natural_within_positive_entropy"] == _ud["natural_within_positive_margin"] == 22, "robustness: natural within-model positive in all 22 under both definitions"))
_g, _r = _eq["gender"], _eq["race"]
_gr = sorted(_g["missed_rates"])
_other = sorted(_gr[:2] + list(_r["missed_rates"]))
check("equal-size EEC (gender misses)", f"(both gender, in {100 * _gr[2]:.0f}\\% and {100 * _gr[3]:.0f}\\% of draws)",
      f"at most {100 * _other[-1]:.0f}\\% of draws")
results.append((_g["missed_detected_by_equal_size_emotional_majority"] == 2 and _r["missed_detected_by_equal_size_emotional_majority"] == 0,
                "equal-size EEC: two of nine misses reflect template type"))
check("equal-size EEC (other direction)", f"flags {_g['eec_neutral_only_vs_equal_size']} gender and {_r['eec_neutral_only_vs_equal_size']} race biases")
check("equal-size EEC (table rows)",
      f"Expected flags, emotional set of EEC-Neutral size & {_g['equal_size_emotional_expected_flags']} & -- & {_r['equal_size_emotional_expected_flags']:.1f} & --",
      f"usually found by equal-sized emotional set & {_g['missed_detected_by_equal_size_emotional_majority']} & -- & {_r['missed_detected_by_equal_size_emotional_majority']} & --",
      f"usually missed by equal-sized emotional set & {_g['eec_neutral_only_vs_equal_size']} & -- & {_r['eec_neutral_only_vs_equal_size']} & --")

# --- heldout_check.py: uncertainty on one half of the person pairs, sensitivity on the other ---
_ho = json.load(open("results_stats/heldout_prediction.json", encoding="utf-8"))
check("held-out prediction",
      f"(median $\\rho={_ho['rho_tv_median']:.2f}$, 5th percentile {_ho['rho_tv_p05']:.2f}; on $|\\Delta_\\ell|$, {_ho['rho_logit_median']:.2f})",
      f"for all {_ho['n_significant_models']} models with a significant effect in every split ({_ho['agree_sig_logit_min']} of {_ho['n_significant_models']} on $|\\Delta_\\ell|$)",
      f"right direction on the other half for all {_ho['n_significant_models']} models")
results.append((_ho["agree_sig_tv_all_splits"] and _ho["agree_sig_logit_min"] == _ho["agree_sig_logit_median"],
                "held-out: TV direction right in every split; logit agreement constant across splits"))

# --- template_family_test.py: prospective test on unseen template families -----------------
_tf = json.load(open("results_template_family/results.json", encoding="utf-8"))
_gt = json.load(open("results_template_family/gate.json", encoding="utf-8"))
_p, _l, _e = _tf["primary_maxprob"], _tf["logit_maxprob"], _tf["primary_entropy"]
_pc = lambda x: f"{100 * x:.0f}\\%"
results.append((_tf["decision"]["H1"] == "supported" and _tf["decision"]["H2"].startswith("partially"),
                "template families: frozen decisions are H1 supported, H2 partially supported"))
check("families cells", f"Of the {_p['cells']} cells", f"the direction is clear in {_p['clear_cells']} cells. The other {_p['ambiguous_cells']}")
check("families H1", f"gives the right direction in {_pc(_p['H1_accuracy'])} of clear cells",
      f"models and families, {_p['H1_ci'][0]:.2f}--{_p['H1_ci'][1]:.2f})",
      f"across all {_p['cells']} cells at $\\rho={_p['H1_spearman_all_cells']:.2f}$ ({_p['H1_spearman_ci'][0]:.2f}--{_p['H1_spearman_ci'][1]:.2f})",
      f"between {min(_p['lofo_H1'].values()):.2f} and {max(_p['lofo_H1'].values()):.2f}")
_b = _p["baselines_accuracy_clear"]
check("families baselines", f"Study~2 direction is right in {_b['usual']:.2f}", f"label scheme in {_b['label']:.2f}",
      f"neutral-class probability in {_b['neutral_mass']:.2f}")
results.append((_gt["discordant_cells"] == _p["discordant_cells"], "families: discordant count in results equals the gate count"))
check("families H2", f"In {_gt['discordant_cells']} cells, counted before any outcome was computed",
      f"In the {_p['discordant_clear']} of these with a clear outcome it was right in {_pc(_p['H2_accuracy'])} ({_p['H2_ci'][0]:.2f}--{_p['H2_ci'][1]:.2f})",
      f"correlate at {_p['H2_within_model_spearman_mean']:.2f} on average")
check("families logit", f"advantage is not established ({_l['H2_accuracy']:.2f}, {_l['H2_ci'][0]:.2f}--{_l['H2_ci'][1]:.2f})",
      f"({_l['baselines_accuracy_clear']['label']:.2f} against {_l['H1_accuracy']:.2f} over clear cells)")
results.append((abs(_e["H1_accuracy"] - _p["H1_accuracy"]) < 0.01 and abs(_e["H2_accuracy"] - _p["H2_accuracy"]) < 0.01,
                "families: entropy-based uncertainty gives the same results"))
check("families abstract/contributions", f"93\\% of cells with a clear effect" if round(100 * _p["H1_accuracy"]) == 93 else "MISMATCH")

# --- template_family_exploratory.py (exploratory follow-up) ---------------------------------
_ex = json.load(open("results_template_family/exploratory.json", encoding="utf-8"))
check("families exploratory",
      f"right direction in {_pc(_ex['acc_all_cells_primary'])} ({_pc(_ex['acc_all_cells_logit'])} on the logit outcome)",
      f"in {_pc(_ex['multiclass_clear_logit_cells_down'])} of clear cells",
      f"$\\rho={_ex['multiclass_spearman_pneu_vs_disagreement']:.2f}$ with the share")
results.append((_ex["binary_primary_logit_direction_agree"] == 1.0, "families exploratory: binary outcomes agree in every cell"))

# --- template_family_logspace.py (exploratory, measure that does not saturate) --------------
_ls = json.load(open("results_template_family/logspace.json", encoding="utf-8"))
_mc, _bi = _ls["multiclass"], _ls["binary"]
check("families log-space",
      f"(all {_bi['clear']} clear cells right)" if _bi["acc"] == 1.0 else "MISMATCH",
      f"right direction in {_pc(_mc['acc'])} of clear cells ({_mc['acc_ci'][0]:.2f}--{_mc['acc_ci'][1]:.2f})",
      f"usual direction ({_pc(_mc['baselines']['usual'])})",
      f"chance level ({_pc(_mc['acc_discordant'])}, {_mc['acc_discordant_ci'][0]:.2f}--{_mc['acc_discordant_ci'][1]:.2f})")

# --- cluster_robust.py: cluster-robust directional tests (Studies 1 and 3) ------------------
_cr = json.load(open("results_stats/cluster_robust.json", encoding="utf-8"))
results.append((_cr["reproduces_published_flags"] == {"study1_mismatches": 0, "study3_mismatches": 0},
                "cluster-robust: original-protocol flags reproduce the published ones"))
_o, _t, _w = _cr["original"]["study1"], _cr["cluster_template"]["study1"], _cr["cluster_twoway"]["study1"]
_wd, _od = _cr["cluster_twoway"]["study1_derived"], _cr["original"]["study1_derived"]
check("Table 1 robust rows",
      f"Significant on full EEC & {_o['full_gender']} & {_w['full_gender']} & {_o['full_race']} & {_w['full_race']}",
      f"(44 / 4 pairs) & {_o['neutral_gender']} & {_w['neutral_gender']} & {_o['neutral_race']} & {_w['neutral_race']}",
      f"(176 / 16 pairs) & {_o['eec_neutral_gender']} & {_w['eec_neutral_gender']} & {_o['eec_neutral_race']} & {_w['eec_neutral_race']}",
      f"missed by original neutral templates & {_od['gender']['eecn_not_neutral4']} & {_wd['gender']['eecn_not_neutral4']} & {_od['race']['eecn_not_neutral4']} & {_wd['race']['eecn_not_neutral4']}",
      f"Significant on full EEC, not on EEC-Neutral & {_od['gender']['full_not_eecn']} & {_wd['gender']['full_not_eecn']} & {_od['race']['full_not_eecn']} & {_wd['race']['full_not_eecn']}",
      f"Significant on EEC-Neutral, not on full EEC & {_od['gender']['eecn_not_full']} & {_wd['gender']['eecn_not_full']} & {_od['race']['eecn_not_full']} & {_wd['race']['eecn_not_full']}")
check("Study 1 robust text", f"falls from {_o['full_gender']} to {_w['full_gender']} ({_t['full_gender']} if only templates",
      f"less affected ({_o['full_race']} to {_w['full_race']})",
      f"{_wd['race']['full_not_eecn']} race biases are significant only on the full EEC and {_wd['race']['eecn_not_full']} only on EEC-Neutral")
results.append((not _cr["twitter_roberta_race"]["emotional"]["flag_template"], "twitter-roberta race reversal not robust"))
_s3t, _s3w = _cr["cluster_template"]["study3"], _cr["cluster_twoway"]["study3"]
check("Study 3 robust text", f"lowers the targeted counts to {_s3t['targeted_gender']} gender and {_s3t['targeted_race']} race",
      f"placebo swaps in {_s3t['targeted_placebo']} models for the targeted set, {_s3t['neutral_placebo']} for EEC-Neutral and {_s3t['emotional_placebo']} for emotional")
results.append((sum(_s3w.values()) == 0, "Study 3: no detections at all under the two-way robust test"))

# --- temperature_check.py and bias_injection.py (exploratory) -------------------------------
_tc2 = json.load(open("results_stats/temperature_check.json", encoding="utf-8"))
_pt1, _pt8, _pt16 = _tc2["per_T"]["1"], _tc2["per_T"]["8"], _tc2["per_T"]["16"]
check("temperature", f"For {_pt1['tv_direction_agrees_with_log_space']} of the 18 models",
      f"For the other {18 - _pt1['tv_direction_agrees_with_log_space']}",
      f"($\\rho={_pt8['rho_tv_vs_u_ratio']:.2f}$ at $T=8$")
results.append((_pt16["tv_direction_agrees_with_log_space"] == 18 and _pt16["tv_direction_changed_vs_T1"] == 18 - _pt1["tv_direction_agrees_with_log_space"],
                "temperature: all 18 match logit space at T=16; only the initially disagreeing models flip"))
_bi = json.load(open("results_stats/bias_injection.json", encoding="utf-8"))["counts"]
_bj = json.load(open("results_stats/bias_injection.json", encoding="utf-8"))
check("bias injection",
      f"detected on emotional templates in {_bi['prob|emotional|0.5']['robust']} of the 18 models, against {_bi['logit|emotional|0.5']['robust']} on the logit scale",
      f"in {_bi['prob|emotional|1.0']['robust']} against {_bi['logit|emotional|1.0']['robust']}",
      f"($\\rho={_bj['rho_injected_effect_vs_uncertainty_delta0.5']:.2f}$ at")

# --- targeting_control.py: cross-model control for Study 3 ------------------------------------
_tc = json.load(open("results_targeting_control/targeting_control.json", encoding="utf-8"))
results.append((_tc["study3_targeted_flags_reproduced"], "targeting control: Study 3 targeted flags reproduced"))
_f, _mc, _mix = _tc["flags"], _tc["mcnemar_targeted_vs_consensus"], _tc["targeted_clause_mix_total"]
check("targeting control flags",
      f"These consensus sets flag a gender disparity in {_f['consensus']['gender']} models",
      f"and a race disparity in {_f['consensus']['race']} (targeted sets: {_f['targeted']['gender']} and {_f['targeted']['race']}",
      f"flags on average {_tc['donor_expected_flags']['gender']} gender",
      f"and {_tc['donor_expected_flags']['race']} race disparities")
results.append((_f["consensus"]["placebo"] == _f["targeted"]["placebo"] == 0, "targeting control: no placebo detections"))
check("targeting control McNemar",
      f"({_mc['gender']['targeted_only']} vs.\\ {_mc['gender']['consensus_only']}), but not significantly (exact McNemar, $p={_mc['gender']['p']:.2f}$)",
      f"no difference ({_mc['race']['targeted_only']} vs.\\ {_mc['race']['consensus_only']})")
check("targeting control specificity",
      f"(mean Jaccard index {_tc['mean_pairwise_jaccard_targeted_sets']:.2f})",
      f"(median {_tc['median_unc_consensus']:.2f} against {_tc['median_unc_targeted']:.2f})",
      f"({_mix['mild negative']}, {_mix['mild positive']} and {_mix['factual']} of {sum(_mix.values())} frames)")

# --- natural-sentence check (natural_sentence_check.py) ---------------------------------------
nr = json.load(open("results_natural/natural_results.json", encoding="utf-8"))
ns = pd.read_csv("results_natural/natural_summary.csv", keep_default_na=False)
fb = ns.set_index(ns.model.str.split("/").str[-1]).loc["FinancialBERT-Sentiment-Analysis"]
in_band = int(((ns.unc_ratio > 0.8) & (ns.unc_ratio < 1.2)).sum())
either = int(((ns.verdict_low != "none") | (ns.verdict_high != "none")).sum())
check("natural H1", f"In all {nr['n_systems']} systems",
      f"(median $\\rho={nr['H1_median_rho']:.2f}$, range {ns.rho_within.min():.2f}--{ns.rho_within.max():.2f})")
results.append((nr["H1_positive"] == nr["H1_significant_holm"] == nr["n_systems"] == 22,
                "natural H1: all 22 positive and Holm-significant"))
check("natural uncertainty band", f"lies between 0.8 and 1.2 for {in_band} of the")
check("natural FinancialBERT", f"(uncertainty ratio {fb.unc_ratio:.2f})", f"less sensitive there ({fb.sens_ratio:.2f})")
results.append((nr["H2_n_sig_ratio"] == 1, "natural: exactly one Holm-significant sensitivity ratio"))
check("natural H2", f"correlate at $\\rho={nr['H2_rho']:.2f}$ ($p={nr['H2_p']:.2f}$;",
      f"{nr['H2_rho_classifiers_only']:.2f}, $p={nr['H2_p_classifiers_only']:.2f}$, for the classifiers alone")
check("natural H3", f"detected in {nr['H3_detected_low']} systems on low-affect",
      f"in {nr['H3_detected_high']} on high-affect sentences, and in {either} on at least one",
      f"For {nr['H3_verdicts_differ']} of the 22 systems")
results.append((nr["H3_sign_conflicts"] == 0, "natural: no direction reversal between strata"))
llm_dir = ns[ns.model.str.contains("@")]
results.append((((llm_dir.verdict_low == "M>F") & (llm_dir.verdict_high == "M>F")).all(),
                "natural: all four LLM configurations M>F in both strata"))
check("natural sample", "kept 1,000 sentences")
results.append((nr["n_sentences"] == 1000 and nr["n_low"] == nr["n_high"] == 500, "natural: 1,000 sentences, 500 per stratum"))

# --- generated tables up to date -----------------------------------------------------
before = {p: open(p, encoding="utf-8").read() for p in
          tuple(_os.path.join(project.PAPER_TABLES, f) for f in
                ("table_minimal_pair.tex", "table_eec.tex", "table_llm.tex", "table_natural.tex", "table_models.tex", "table_families.tex"))}
subprocess.run([sys.executable, project.script("make_paper_tables.py")], check=True, capture_output=True)
for p, old in before.items():
    results.append((open(p, encoding="utf-8").read() == old, f"generated table up to date: {p}"))

for ok, msg in results:
    print(("PASS " if ok else "FAIL ") + msg)
n_fail = sum(not ok for ok, _ in results)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
