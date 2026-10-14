# Where Counterfactual Bias Tests Can See — code and results

Anonymous release accompanying a double-blind submission.

## Data
The Equity Evaluation Corpus may not be redistributed. Download it from
http://saifmohammad.com/WebPages/Biases-SA.html and place
`Equity-Evaluation-Corpus.csv` under `data/Equity-Evaluation-Corpus/`.
The NRC Affect Intensity Lexicon goes to `data/NRC-AffectIntensity-Lexicon.txt`
(http://saifmohammad.com/WebDocs/NRC-AffectIntensity-Lexicon.txt).
The natural sentences are not redistributed either: `natural_sentence_check.py build`
downloads 4,000 reviews from the public IMDB corpus and rebuilds them deterministically.

## Reproducing
Classifiers (CPU; hours for the two roberta-large models):
    python minimal_pair_experiment.py      # Study 2 (+ control tokens)
    python distribution_shift_check.py     # TV distance
    python eec_reproduction.py             # Study 1
    python prob_cache.py && python targeted_probe.py && python label_flips.py   # Study 3, flips
    python robust_stats.py                 # all measures, Holm correction
    python robustness_checks.py            # uncertainty definitions; equal-size EEC sets
    python heldout_check.py                # uncertainty on held-out person pairs
    python cluster_robust.py               # cluster-robust directional tests (Studies 1 and 3)
    python temperature_check.py            # exploratory: direction under temperature scaling
    python bias_injection.py               # exploratory: detection of an injected bias of known size
LLMs as zero-shot classifiers:
    python llm_audit.py                    # original prompts, all three studies
    python llm_audit.py --robustness       # prompt variants (frozen in prompts_frozen.json)
Natural sentences (analysis plan frozen in results_natural/analysis_plan.json):
    python natural_sentence_check.py build && python natural_sentence_check.py score
    python natural_sentence_check.py analyse
Cross-model targeting control for Study 3 (plan frozen in results_targeting_control/):
    python targeting_control.py score && python targeting_control.py analyse
Prospective test on unseen template families (plan frozen in results_template_family/):
    python template_family_test.py uncertainty && python template_family_test.py gate
    python template_family_test.py outcomes && python template_family_test.py analyse
    python template_family_exploratory.py    # exploratory follow-up, not in the frozen plan
    python template_family_logspace.py       # exploratory: a sensitivity measure that does not saturate
Paper artefacts:
    python make_paper_tables.py && python make_figure.py
    python audit_numbers.py                # checks numbers in the paper against these results

Every script resumes from saved per-model results. Seeds are fixed (see SEED).
