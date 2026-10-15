# Where Counterfactual Bias Tests Can See — code and results

Code, frozen analysis plans, templates and per-model results for the paper
"Where Counterfactual Bias Tests Can See: Predictive Uncertainty and Template
Choice in Sentiment Audits". The manuscript source (`paper_v2.tex`) and its
generated tables (`paper_tables/`) are included so that every reported number
can be checked against these results (see the end of this file).

## Requirements
Python 3.13 with the package versions used for the paper:

    pip install numpy==2.4.1 pandas==3.0.0 scipy==1.17.0 matplotlib==3.10.8 \
                torch==2.11.0 transformers==5.8.0 sentencepiece==0.2.2

The classifiers and LLMs are public Hugging Face checkpoints (identifiers in the
paper's appendix and in `minimal_pair_experiment.py` and `llm_classifier.py`);
they are downloaded on first use. Everything runs on CPU. Checking the paper's
numbers from the saved results needs only numpy, pandas and scipy.

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
    python audit_numbers.py                # checks numbers in paper_v2.tex against these results

Every script resumes from saved per-model results. Seeds are fixed (see SEED).

## Scope and limitations
The results concern 18 public sentiment classifiers and two instruction-tuned
LLMs from one family, English synthetic templates (plus one natural-sentence
check on IMDB reviews), binary gender expressions and name-based race proxies.
Exploratory analyses added after a frozen plan are marked as such in the code
and in the paper. See the paper's Limitations section.

## Licence
Code and result files: MIT (see `LICENSE`). Third-party data (the Equity
Evaluation Corpus, the NRC Affect Intensity Lexicon, IMDB reviews) are not
included and remain under their own terms.
