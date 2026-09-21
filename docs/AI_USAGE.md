# AI Usage Log

CAI 4105 requires that AI tools "may be used with explicit documentation of usage." This file is that documentation.
**Keep it updated: add a row every time anyone on the team uses an AI tool for project work.**

## Tools used

| Tool | Who | Where |
|---|---|---|
| Claude Code (Anthropic, model: Claude Sonnet 5) | [team member name] | terminal, inside this repository |

## Log

| Date | Who | What the AI did | What the team did / verified |
|---|---|---|---|
| 2026-09-21 | [team member name] | Cloned the repo and ran `fetch_fema.py`. Fixed the script after a 503 error by adding a server-side county filter (about 20 minutes instead of about 100) and longer retry backoff. | Confirmed the county string `Lee (County)` matches the API and that 194,482 rows came back. |
| 2026-09-21 | [team member name] | Wrote `src/utils/columns.py` (which columns are features, which are leakage), `src/features/prepare.py` (cleaning + train/val/test split), `src/utils/plotting.py`, `src/models/baseline.py`. | [Review each file; run `python -m src.features.prepare`; be able to explain every choice in class.] |
| 2026-09-21 | [team member name] | Wrote and ran `notebooks/eda/01_initial_eda.ipynb` (EDA, leakage check, missing-value and quality analysis, preliminary baselines). Looked up FEMA's official column definitions from the OpenFEMA API for `docs/data_dictionary.md`. | [Re-run "Restart & Run All"; check the numbers in the text against the outputs; challenge any conclusion you don't understand.] |
| 2026-09-21 | [team member name] | Drafted `reports/milestone_1/report_draft.md`. | [Rewrite in the team's own words; confirm every number; fill in team-specific sections.] |

## What the AI produced vs. what needs human judgment

- **AI-drafted:** code, first-pass analysis, chart designs, report draft.
- **Team decisions (not AI decisions):** the target variable, which risks to accept, the success criteria, the modeling plan, the final wording of the report, and who does what.
- **Things the AI could get wrong** and the team must check: claims about how FEMA's program works (e.g. the flat $700 payment is *inferred from the data*, not confirmed from FEMA documentation), and any number typed into prose rather than printed by the notebook.

## Sources

- FEMA. *OpenFEMA Dataset: Individuals and Households Program - Valid Registrations v2.* https://www.fema.gov/openfema-data-page/individuals-and-households-program-valid-registrations-v2
