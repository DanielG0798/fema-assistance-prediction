<!--
DRAFT for the team to rewrite in its own words. Course rules: original work, AI use documented (docs/AI_USAGE.md).
Rubric (18 pts): Business Understanding 5 | Data Understanding 7 | Data Preparation Plan 3 | Presentation & Communication 3.
Length: 10-15 pages PDF excluding appendices. Page budgets are in each heading.
Legend:  [TEAM: ...] = a decision or fact only the team can supply.   (fig: name) = insert reports/figures/name.png here.
Every number below is printed by notebooks/eda/01_initial_eda.ipynb or src/models/baseline.py. Re-check after any re-run.
-->

# Predicting FEMA Disaster-Assistance Eligibility: Hurricane Ian, Lee County, FL
**CAI 4105 Machine Learning, Fall 2026, Project Milestone 1: Initial Report**
[TEAM: team name] | [TEAM: member names] | Due September 30, 2026

---

## 1. Executive Summary *(1 page)*

**Overview.** In September 2022, Hurricane Ian devastated southwest Florida, and Lee County alone produced 194,482 valid applications to FEMA's Individuals and Households Program (IHP), which gives money to disaster survivors for housing and other urgent needs. For each household FEMA must decide whether to award aid. We build a machine-learning pipeline that predicts **whether an applicant will be awarded IHP aid** (`ihpEligible`).

**Dataset.** Public OpenFEMA data (*IHP Valid Registrations v2*), filtered to disaster DR-4673 and Lee County: 194,482 applications, 100 columns, a mix of numeric, yes/no, and categorical variables. The outcome is almost perfectly balanced (50.7% awarded, 49.3% not).

**What we found so far.**
1. **Leakage is the main risk.** 43 of the 100 columns are *results* of the aid decision (dollar amounts, sub-program eligibility, denial reasons). One of them (`ihpAmount`) is identical to the target. A model given these columns would look perfect and be useless, so we separate them from the start.
2. **We defined two honest prediction moments:** *Tier 1*, using only what the applicant reports at registration (26 features), and *Tier 2*, adding what is learned after FEMA inspection and verification (40 features).
3. **The problem is learnable.** Untuned baselines already reach a ROC-AUC of about **0.90 (Tier 1)** and **0.94 (Tier 2)**, against 0.50 for guessing. A large share of eligibility follows fairly mechanical FEMA rules (for example, 46% of all awards are exactly $700), so the model partly learns the screening process itself.
4. **Data quality is good but imperfect:** a few columns are mostly blank, damage amounts are heavily skewed, about 3% of records have a census block outside Lee County, and city names are typed by hand.

**Approach.** Clean and split the data once (stratified 60/20/20), compare interpretable and flexible models (logistic regression, decision tree, random forest, gradient boosting) using stratified cross-validation, judge them on ROC-AUC, F1, and error rates across age and income groups, and keep a human decision-maker in the loop. The remaining work is scheduled in Section 6.

---

## 2. Business Understanding *(2-3 pages)*

### 2.1 Problem definition and motivation
After a major hurricane, tens of thousands of households apply for federal aid within days. In Lee County, applications peaked on September 30, 2022 (15,240 in one day), and 94% arrived by the end of October (fig: applications_over_time). Each application passes through referral, documentation checks, sometimes an inspection, and a decision. About half end without an award, and the reasons vary: 30% of unsuccessful applicants had insurance, 19% had no eligible damage or needs, 13% did not respond or withdrew, and 35% were never referred to the program (fig: ineligible_reasons).

**The question our model answers:** *Given what we know about an applicant, will FEMA award them IHP aid?*

**Why it matters.** Survivors wait for answers while caseworkers are overloaded. A reliable prediction could help FEMA (a) route likely-eligible applications to a fast track, (b) flag likely-ineligible applications early so staff can tell people what is missing (for example insurance documents), (c) decide where to send inspectors first, and (d) plan staffing and budget for the next disaster.

**What it is not.** The model is decision *support*. It must never automatically deny anyone aid. A person makes every final decision.

### 2.2 Business objectives and success criteria
| Objective | How we will measure it | Proposed target [TEAM: confirm] |
|---|---|---|
| Predict eligibility at registration time (Tier 1) | ROC-AUC and F1 on the held-out test set | Beat the untuned baseline: ROC-AUC >= 0.90, F1 >= 0.83 |
| Predict eligibility after inspection (Tier 2) | same | ROC-AUC >= 0.94 |
| Beat naive approaches | compare to majority-class guessing and logistic regression | Clear win over both (majority-class accuracy is 50.7%; untuned logistic regression reaches AUC 0.84 at Tier 1) |
| Be fair across groups | error rates by age band and income band | no group's error rate more than 5 percentage points above the overall rate |
| Be explainable | feature-importance and partial-dependence plots a caseworker could read | top drivers match FEMA's stated eligibility rules |

*Note.* These targets come from preliminary, untuned baselines (Section 5.3). They are goals for tuned models, not promises.

### 2.3 Stakeholder analysis
| Stakeholder | Interest | Impact of a wrong prediction |
|---|---|---|
| Applicants (survivors) | Fast, fair, understandable decisions | A wrongly flagged "ineligible" household could give up on a claim they deserve |
| FEMA caseworkers and program managers | Throughput, accuracy, defensible decisions | Wasted inspections, or eligible people missed |
| Lee County emergency management, state of Florida | Recovery speed, resource planning | Misjudged demand |
| Congress, auditors, taxpayers | Proper use of federal funds | Improper payments or unexplained denials |
| Equity and legal-aid organizations | No group treated worse | Systematic bias against age, income, or place |
| Our team / instructor | Sound methodology, honest reporting | n/a |

### 2.4 Expected impact and value proposition
A tuned, audited model would give FEMA an early, explainable signal at the moment of registration, when the information is cheapest. Its value is speed and prioritization, not replacing judgment. Because the outcome depends partly on FEMA's own screening rules, the model also doubles as a *consistency check*: applications where the model strongly disagrees with the outcome are worth a second look.

### 2.5 Risks and assumptions
- **Selection:** only *valid* registrations are in the data, so we say nothing about invalid applications.
- **Scope:** one storm and one county; results may not transfer to other disasters.
- **Label meaning:** "not eligible" mixes many reasons (insurance, missing documents, withdrawal, never referred).
- **Fairness:** age, income, and location can correlate with vulnerable groups, so we will audit error rates by group.
- **Assumption to verify:** the flat $700 payment we see appears to be an emergency-needs payment; we infer this from the data and will confirm it in FEMA program documentation. [TEAM: confirm]

---

## 3. Data Understanding *(3-4 pages)*

Full analysis: `notebooks/eda/01_initial_eda.ipynb` (also `reports/milestone_1/eda_notebook.html`). Column-by-column definitions: `docs/data_dictionary.md`.

### 3.1 Dataset description, source, and collection
- **Source:** FEMA OpenFEMA, *Individuals and Households Program - Valid Registrations (v2)*, a public government dataset from FEMA's National Emergency Management Information System (NEMIS).
- **Collection method:** downloaded on 2026-09-21 through the public OpenFEMA API (no key needed) with our script `src/data/fetch_fema.py`, filtered to disaster **DR-4673** and county **Lee (County)**. FEMA refreshes the data weekly, so the row count can change slightly over time.
- **Size:** 194,482 rows x 100 columns. Applications dated 2022-09-27 to 2023-01-12. One row is one household's application.
- **Requirement check:** at least 10 features (we have 26 at Tier 1 and 40 at Tier 2), at least 1,000 rows, and both numeric and categorical variables.
- **Caveats from FEMA:** raw operational data "subject to a small percentage of human error"; only valid registrants are included.

### 3.2 The target variable
`ihpEligible` is True when the applicant received a housing and/or other-needs award. It is balanced: 98,648 eligible (50.7%) and 95,834 not (49.3%) (fig: target_balance). Plain accuracy is therefore meaningful, and no special class-imbalance handling is needed.

### 3.3 The key finding: column roles and leakage
We sorted all 100 columns into roles (`src/utils/columns.py`):

| Role | Columns | Meaning |
|---|---|---|
| Target | 1 | `ihpEligible` |
| Tier 1, application-time | 27 | self-reported at registration |
| Tier 2, later-stage | 14 | learned after inspection or verification |
| Leakage | 43 | results of the decision (never used) |
| ID / constant | 15 | identifiers or identical on every row |

To test leakage we scored each column *alone* with a small model (fig: leakage_auc). Outcome columns are near-perfect predictors (`ihpAmount` AUC 1.000, `onaEligible` 0.97, `ineligibleReason` 0.83), while the best honest column reaches only about 0.70. We also tested a borderline field, `currentLocation`: it contains values that are *consequences* of aid ("FEMA-provided unit" is 99% eligible, "new temporary rental" 87%), so we moved it from Tier 1 to Tier 2. This cost only 0.013 AUC and made the model safer.

### 3.4 Summary statistics and distributions
Applicants are mostly older, small households, and homeowners: 57% are 50 or older, most households have one or two people, 65% own the damaged home, and 77% registered online or by mobile app (fig: applicant_profile). Damage measures such as `rpfvl` (FEMA-verified real-property loss) are zero for 83% of applicants and extremely right-skewed for the rest. Two categorical variables have very many values: ZIP code (about 560) and census block group (about 3,000).

### 3.5 Relationships with the target
Eligibility rates by feature (fig: eligibility_by_feature):
- **Primary residence** is close to a hard rule: non-primary homes are about 0.4% eligible.
- **Emergency needs reported:** 34% eligible if not reported, 66% if reported.
- **Homeowners insurance** lowers eligibility (47% vs 55% without), consistent with FEMA covering what insurance does not. Flood insurance barely matters.
- **Income:** eligibility falls as reported income rises (61% under $15k to 45% above $175k), except the "$0" group, which is lowest (42%).
- **Owner vs renter** makes almost no difference (51% vs 50%).
- **Inspection:** applicants whose inspection was completed are 73% eligible vs 38% otherwise; unverified occupancy is under 1% eligible.
- **Geography and timing:** eligibility varies from 43% to 62% across the 22 largest ZIP codes (fig: eligibility_by_zip) and changes with the week of application (fig: eligibility_by_week).

### 3.6 Missing values
Nineteen columns have missing values, eleven of them candidate features (fig: missing_values). Six have more than 10% missing: `renterDamageLevel` (96%), `highWaterLocation` (88%), `shelterNeed` (82%), `habitabilityRepairsRequired` (73%), `foodNeed` (52%), `selfAssessmentInformation` (15%). The blanks carry meaning: some questions apply only to renters or flooded homes, and `foodNeed` / `shelterNeed` are `True` on nearly every row where they exist, so a blank effectively means "need not reported".

### 3.7 Correlation analysis
No single numeric feature has a strong linear relationship with eligibility (the strongest is about 0.36) (fig: correlation_with_target). Several feature pairs are near-duplicates (fig: correlation_matrix): inspection issued vs completed (1.00), real-property loss vs flood-damage amount (0.97), reported damage vs home damage (0.83).

### 3.8 Data quality assessment
| Check | Result |
|---|---|
| Constant columns | 14 columns have one value on every row (expected: one disaster, one county) |
| Duplicates | 407 rows (0.21%) identical on all non-ID columns; probably different households with the same answers, so kept |
| Location errors | 6,357 rows (3.3%) have a census block outside Lee County (2,707 are `NO_INTERSECT`; most others are neighboring counties); only 17 rows have a non-Florida ZIP |
| Hand-typed city names | many spellings (for example `FT MYERS`, `FT MYERS BCH`), so we drop the city column |
| Implausible values | a water depth of 960 inches (80 ft) |
| Dates | only 2 applications are dated before the disaster declaration |

### 3.9 Challenges and limitations
Leakage; one disaster and one county; valid registrants only; a label that bundles many denial reasons; heavy skew; structural missing values; high-cardinality geography; near-duplicate features; and outcomes that partly follow FEMA's own rules (a high score is not proof that the model understands need).

---

## 4. Data Preparation Plan *(2-3 pages)*

The cleaning code already exists and is shared: `src/features/prepare.py`.

### 4.1 Data cleaning strategy
| Issue | Decision |
|---|---|
| Leakage, IDs, constants (58 columns) | Drop (list in `src/utils/columns.py`) |
| `damagedCity` (hand-typed) | Drop; use ZIP and census block instead |
| Yes/No columns stored as text with blanks | Convert to 0/1, keep blanks as truly missing (not 0) |
| Ordered ranges (`applicantAge`, household counts) | Convert to ordered numbers (`>5` becomes 6) |
| Income bracket | Keep as unordered category (the "$0" group breaks the ordering) |
| Missing values | Do **not** fill in during cleaning. Use missing-indicators plus median (linear models) or native handling (boosting), fit on training data only |
| Implausible values (`waterLevel` = 960 in) | Cap at a high percentile |
| Out-of-county blocks | Keep; treat `NO_INTERSECT` as its own category |
| Duplicate-looking rows | Keep (see above) |

### 4.2 Feature engineering
- `daysSinceLandfall` from the application date (already built).
- Log transform (`log1p`) of the skewed dollar and damage columns, plus "has damage" yes/no flags.
- Missing-value indicators for the columns with meaningful blanks.
- ZIP and census block: frequency encoding first (already used in the baselines), with target encoding tested *inside* cross-validation folds.
- Group rare categories together; drop one of each near-duplicate pair for linear models.
- Possible interaction features for emergency needs x residence status, which the boosting baseline suggests matter.

### 4.3 Data transformation
Scaling and one-hot encoding are applied only for models that need them (logistic regression, and any distance-based method). Tree-based models use the unscaled data. **All transformations that learn from data (imputation, scaling, encoding) are fit on the training split only**, inside a scikit-learn pipeline, to avoid a second kind of leakage.

### 4.4 Train / validation / test strategy
- **Stratified split, 60% train / 20% validation / 20% test** (fixed random seed 42, so everyone gets the same split); each part keeps the same 50.7% / 49.3% mix. Implemented in `split_data()`.
- **Test set is touched once**, at the very end of Milestone 2.
- Cross-validation (Section 5.3) uses the 80% development set (train + validation); the 20% test set stays sealed.
- **Robustness check:** because applications arrive over time, we will also train on early applicants and test on later ones to see whether performance holds up.
- We cannot link applications from the same household, so we note this as a limitation.

---

## 5. Modeling Approach *(2-3 pages)*

### 5.1 Algorithm selection and justification
| Model | Why |
|---|---|
| Logistic regression | Simple, fast, easy to explain; the baseline to beat |
| Decision tree | Human-readable rules, and FEMA's process is rule-like |
| Random forest | Robust to skew, mixed types, and outliers; reduces a single tree's overfitting |
| Gradient boosting | Best preliminary results; captures feature interactions |
| [TEAM: add or remove models based on what the course has covered, for example k-NN or a neural network] | |

**Preliminary baselines (untuned, validation split, `src/models/baseline.py`):**

| Feature set | Model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|---|
| Tier 1 (26 features) | Gradient boosting | 0.900 | 0.824 | 0.837 |
| Tier 1 | Logistic regression | 0.843 | 0.764 | 0.771 |
| Tier 1 + 2 (40 features) | Gradient boosting | 0.938 | 0.870 | 0.882 |
| Tier 1 + 2 | Logistic regression | 0.898 | 0.818 | 0.825 |
| any | Always guess majority class | 0.500 | 0.507 | 0.000 |

### 5.2 Evaluation metrics
**ROC-AUC** (main metric: ranks applicants regardless of cutoff; suitable because the classes are balanced), **F1** with **precision** and **recall** (a false "ineligible" and a false "eligible" have different real-world costs, so we will report both and choose a cutoff explicitly), **accuracy** (fair here because of the balance), a **confusion matrix**, and **error rates by age and income group** for fairness.

### 5.3 Cross-validation and tuning
Five-fold stratified cross-validation on the development set; randomized hyperparameter search inside the folds; the same folds for every model so comparisons are fair; the test set used once for the final score.

### 5.4 Expected challenges and mitigation
| Challenge | Mitigation |
|---|---|
| Hidden leakage as scores get high | Keep the roles file as the single source of truth; repeat the "drop a feature group and see what moves" ablation for every final feature set (it already caught `currentLocation`) |
| Very high-cardinality geography | Frequency / target encoding fit inside folds; compare with and without |
| Skewed and mostly-zero damage columns | Log transform plus flags; prefer tree models |
| Model just re-learns FEMA's rules | Say so honestly; report where the model disagrees with outcomes as the interesting cases |
| Fairness across age / income / place | Report group error rates; investigate any gap over 5 points |
| Results specific to one storm | State the scope; run the time-based robustness check |

---

## 6. Project Timeline *(1 page)*

| Dates | Work | Owner |
|---|---|---|
| Sep 22-25 | Team meeting: agree on the target, success criteria, and roles; each member re-runs the EDA notebook and reads the data dictionary | All |
| Sep 26-29 | Finalize and proofread the report; export to PDF; assemble `TeamName_Milestone1_YYYYMMDD.zip` | [TEAM] |
| **Sep 30** | **Milestone 1 due** | All |
| Oct 1-14 | Finish the preprocessing pipeline (feature engineering, encoders) and the `notebooks/preprocessing` notebook | [TEAM] |
| Oct 15-31 | Train all models, cross-validate, tune hyperparameters | [TEAM] |
| Nov 1-10 | Evaluation: model comparison, interpretation (feature importance), leakage ablations, fairness audit | [TEAM] |
| Nov 11-18 | Write the final report; build the presentation; rehearse | All |
| Nov 19 | Internal freeze: code runs top-to-bottom, PDF exported | All |
| **Nov 21** | **Milestone 2 due** | All |
| **Nov 23 / Dec 2** | **Presentation (15 min + Q&A)** | All |

**Team responsibilities.** [TEAM: name] data and EDA | [TEAM: name] modeling | [TEAM: name] evaluation and report. Everyone contributes to every phase and attends the presentation.

**Risks and contingencies.**
| Risk | Contingency |
|---|---|
| A teammate is unavailable | Everything is in Git with documented code, so work can be picked up; weekly check-ins |
| Model scores lower than the baseline suggests after removing leaky features | Report honestly; the leakage-removal story is itself a finding |
| Merge conflicts in shared notebooks | One person per notebook at a time; small commits; pull before starting |
| Running out of time before Nov 21 | Freeze the model set by Oct 31; drop optional models first |

---

## Appendices *(not counted in page limit)*
- A. Data dictionary (`docs/data_dictionary.md`)
- B. EDA notebook export (`reports/milestone_1/eda_notebook.html`)
- C. AI usage log (`docs/AI_USAGE.md`)
- D. Additional figures (`reports/figures/`)

## References
FEMA. *OpenFEMA Dataset: Individuals and Households Program - Valid Registrations v2.* https://www.fema.gov/openfema-data-page/individuals-and-households-program-valid-registrations-v2

## AI-use disclosure
Portions of the code, analysis, and this draft were produced with Claude Code (Anthropic) and reviewed by the team. See `docs/AI_USAGE.md` for the full log.
