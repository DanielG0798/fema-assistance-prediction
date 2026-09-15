# FEMA Assistance Prediction

Machine Learning course project (CAI 4105, Fall 2026) predicting outcomes from applicant-level federal disaster assistance data sourced from FEMA's OpenFEMA platform. The dataset is filtered to Hurricane Ian and Lee County, FL.

## Project Goal

Build an end-to-end ML pipeline following the CRISP-DM methodology to understand, prepare, model, and evaluate patterns in FEMA disaster assistance applications for Hurricane Ian applicants in Lee County, Florida.

## Dataset

- **Source:** FEMA OpenFEMA API — `DisasterAssistanceApplicants` or `RegistrationIntakeIndividualsHouseholds`
- **Scope:** Hurricane Ian (DR-4673-FL), Lee County, FL
- **Target variable:** *To be defined by the team* (e.g., approval status, approved assistance amount, days to decision)
- **Minimum requirements met:**
  - ≥ 10 features (excluding target)
  - ≥ 1,000 observations
  - Mixed numerical and categorical variables

## Repository Structure

```
├── data/
│   ├── raw/              # Original, unmodified data downloaded from FEMA
│   └── processed/        # Cleaned, transformed datasets ready for modeling
├── notebooks/
│   ├── eda/              # Exploratory Data Analysis notebooks
│   ├── preprocessing/    # Data cleaning and feature engineering notebooks
│   ├── modeling/         # Model training, tuning, and evaluation notebooks
│   └── reports/          # Draft content and figures for reports
├── src/
│   ├── data/             # Scripts for downloading and loading data
│   ├── features/         # Feature engineering and transformation scripts
│   ├── models/           # Model training, evaluation, and prediction scripts
│   └── utils/            # Shared helper functions and constants
├── reports/
│   ├── figures/          # Generated charts, plots, and visualizations
│   ├── milestone_1/      # Initial report deliverables
│   └── milestone_2/      # Final report deliverables
├── docs/                 # Meeting notes, data dictionary, references
└── README.md             # This file
```

## Milestones & Deliverables

| Milestone | Due Date | Deliverables |
|-----------|----------|--------------|
| Dataset proposal | Sept 9, 2026 | Dataset name submission |
| Milestone 1 | Sept 30, 2026 | Initial report PDF, EDA notebook, dataset |
| Milestone 2 | Nov 21, 2026 | Final report PDF, complete notebooks, presentation |
| Presentation | Nov 23 / Dec 2, 2026 | 15-minute final presentation + Q&A |

## Getting Started

1. Clone the repository.
2. Place raw FEMA data files in `data/raw/`.
3. Run notebooks in order:
   1. `notebooks/eda/`
   2. `notebooks/preprocessing/`
   3. `notebooks/modeling/`

## Team

- Team Member 1 — *TBD*
- Team Member 2 — *TBD*
- Team Member 3 — *TBD*

## References

- [FEMA OpenFEMA API](https://www.fema.gov/about/openfema/data-sets)
- CRISP-DM methodology
