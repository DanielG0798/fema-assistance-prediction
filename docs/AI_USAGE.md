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
| 2026-09-27 | [team member name] | Restructured `report_draft.md` for team use (section-owner table, shared ML glossary, owner notes, "ML takeaway" lines in §3, embedded figures) without changing any numbers. Generated `Milestone1_Responsibilities_and_Draft.docx` (responsibilities + draft) for Google Docs. | [Assign owners and deadlines; rewrite each section in own words; confirm the "likely reason it is blank" column in §3.6.] |
| 2026-09-28 | Mateo | Compared the team plan with the GitHub issues and the codebase. Looked up the $700 payment (FEMA Serious Needs Assistance) and added the citation to §2.5 and the References. | [Mateo: open the FEMA fact sheet and confirm it supports the §2.5 wording; rewrite §2 in own words; set the §2.2 targets.] |
| 2026-10-04 | Anthony | Claude edited Section 3 (3.1–3.9) of `report_draft.md` for grammar, structure and plainer wording, and drafted some sentences. Updated the section to match the new EDA notebook (Cramér's V leakage check, 25 / 39 features, `utilitiesOut` held out) and re-checked its counts and percentages against the team data file. | Anthony wrote the section. AI-drafted sentences are marked with TODO comments in the draft. [Anthony: reword those sentences; be able to explain each subsection.] |
| 2026-10-05 | Anthony | Claude showed each AI-drafted sentence in Section 3 with a plain-language explanation, checked the reworded versions for correct terms and numbers, and suggested small fixes (for example restoring "stratified", the 50.7 / 49.3 mix, and the Figure 3.2 reference). | Anthony reworded all eight AI-drafted sentences in own words; the TODO comments are removed. |

## What the AI produced vs. what needs human judgment

- **AI-drafted:** code, first-pass analysis, chart designs, report draft.
- **Team decisions (not AI decisions):** the target variable, which risks to accept, the success criteria, the modeling plan, the final wording of the report, and who does what.
- **Things the AI could get wrong** and the team must check: claims about how FEMA's program works (e.g. the flat $700 payment is *inferred from the data*, not confirmed from FEMA documentation), and any number typed into prose rather than printed by the notebook.

## Sources

- FEMA. *OpenFEMA Dataset: Individuals and Households Program - Valid Registrations v2.* https://www.fema.gov/openfema-data-page/individuals-and-households-program-valid-registrations-v2
