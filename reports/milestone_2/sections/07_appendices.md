# 7. Technical Appendices

<!-- Owner: Nandor (assembly). Outline: code, additional visualizations, detailed model outputs, data dictionary. -->
<!-- DRAFT (AI first pass, Nov 15). The list is complete; file names marked [CONFIRM] depend on what Daniel and Anthony finish. -->
<!-- The appendices do not count toward the 15-25 page limit. -->

## Appendix A. Code

The complete code is in the Jupyter notebooks below. Each one runs from top to bottom with Restart & Run All and is submitted as both `.ipynb` and PDF.

| Notebook | What it does | Owner |
|---|---|---|
| `notebooks/eda/01_initial_eda.ipynb` | Exploratory data analysis, column roles, leakage evidence | Anthony |
| `notebooks/preprocessing/01_data_cleaning.ipynb` [CONFIRM] | Cleaning and feature engineering | Daniel |
| `notebooks/modeling/01_model_comparison.ipynb` | Ten algorithms, tuning, comparison, final test | Daniel |
| `notebooks/evaluation/01_model_evaluation.ipynb` [CONFIRM] | Metrics, interpretation, leakage and temporal checks | Anthony |
| `notebooks/evaluation/02_fairness_business.ipynb` | Fairness by age and income, business value | Nandor |

How to reproduce the results: install the packages in `requirements.txt`, place `ihp_dr4673_lee.csv` in `data/raw/`, and run the notebooks in the order above. Every split and model uses the random seed 42.

## Appendix B. Data dictionary

The full data dictionary is `docs/data_dictionary.md`. It lists all 100 columns with their type, share of missing values, FEMA's description, and the role we gave each one (target, Tier 1 feature, Tier 2 feature, leakage, or ID/constant).

## Appendix C. Detailed model outputs

- Full grid-search tables for every model and both tiers (from the modeling notebook) [CONFIRM: which tables Daniel wants shown]
- Fairness tables for every age and income group, both tiers (from the fairness notebook)
- Triage table: share of applications in each confidence pile and how often the model is right (from the fairness notebook)

## Appendix D. Additional figures

Figures that support the report but did not fit in the main text.

- From the modeling notebook: `m2_model_*.png` [CONFIRM: final list]
- From the evaluation notebook: `m2_eval_*.png` [CONFIRM: final list]
- From the fairness notebook:
  - `m2_fair_error_rate_by_group.png`: error rate for each age and income group against the 5-point limit
  - `m2_fair_error_type_by_group.png`: missed-eligible and wrongly-eligible rates for each group
  - `m2_fair_false_positives_by_reason.png`: how often the model predicted eligible, by FEMA's ineligibility reason
  - `m2_fair_unfinished_by_group.png`: share of unfinished applications by age and income
  - `m2_fair_threshold_tradeoff.png`: the two error rates as the decision cut-off moves

## Appendix E. AI usage log

The team's combined log is `docs/AI_USAGE.md`. It records, by date, what the AI tools did and what each team member wrote, changed, or checked.

## References

FEMA. *OpenFEMA Dataset: Individuals and Households Program - Valid Registrations v2.* https://www.fema.gov/openfema-data-page/individuals-and-households-program-valid-registrations-v2

FEMA. *Serious Needs Assistance* (fact sheet). https://www.fema.gov/fact-sheet/serious-needs-assistance-0

[CONFIRM: add any source Daniel or Anthony cite in Sections 3-5]
