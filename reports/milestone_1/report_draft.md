<!--
Numbers in this draft are printed by notebooks/eda/01_initial_eda.ipynb or src/models/baseline.py.
If either is re-run, re-check every number before export.
-->

# Predicting FEMA Disaster-Assistance Eligibility: Hurricane Ian, Lee County, FL
**CAI 4105 Machine Learning, Fall 2026, Project Milestone 1: Initial Report**
[TEAM: team name] | [TEAM: member names] | Due **Wednesday, September 30, 2026**

---

> ## 🧭 Team guide — delete this box before exporting the PDF
>
> **How this draft works.** One shared file, one owner per section. Edit only your own section, rewrite it in your own words, and keep the numbers exactly as they are unless the notebook says otherwise.
>
> | § | Section | Owner | Pages | Rubric points | Status |
> |---|---|---|---|---|---|
> | 1 | Executive Summary | [TEAM: name] | 1 | Presentation & Communication (3, whole report) | Draft |
> | 2 | Business Understanding | [TEAM: name] | 2–3 | **5** | Draft |
> | 3 | Data Understanding | [TEAM: name] | 3–4 | **7** (largest) | Draft → revising |
> | 4 | Data Preparation Plan | [TEAM: name] | 2–3 | **3** | Draft |
> | 5 | Modeling Approach | [TEAM: name] | 2–3 | (supports the plan) | Draft |
> | 6 | Project Timeline | [TEAM: name] | 1 | (supports the plan) | Draft |
>
> Length: **10–15 pages** of PDF, not counting appendices.
>
> **Markers used in this file**
> - `[TEAM: ...]`: a decision or fact only the team can supply. None may remain in the PDF.
> - `> ✏️ Owner note`: guidance for the section owner. Delete before export.
> - **ML takeaway:** a one-line link from a finding to a modeling decision. Keep these; graders look for that link.
> - Figures are embedded from `reports/figures/`. Number them in the order they appear in the PDF.
>
> **Shared vocabulary.** Use these terms the same way in every section so the report reads as one voice.
>
> | Term | What it means in *our* project |
> |---|---|
> | **Target / label** | `ihpEligible`: True if the household was awarded IHP aid. The thing we predict. |
> | **Feature** | An input column the model is allowed to see (26 at Tier 1, 40 at Tier 2). |
> | **Binary classification** | Our task type: predict one of two classes (eligible / not eligible). |
> | **Data leakage** | Letting the model see information that would not exist at prediction time (e.g. the award amount). Produces fake-perfect scores. |
> | **Tier 1 / Tier 2** | Our two *prediction moments*: at registration (Tier 1) and after inspection/verification (Tier 2). |
> | **Class balance** | The share of each class: 50.7% eligible vs 49.3% not, so the data is balanced. |
> | **Stratified split** | A train/validation/test split that keeps the 50.7 / 49.3 mix in every part. |
> | **Baseline** | A simple, untuned model used as the bar to beat. |
> | **Cross-validation (k-fold)** | Train on k−1 parts, score on the held-out part, and rotate. Gives a stable estimate of performance. |
> | **ROC-AUC** | How well the model *ranks* eligible above ineligible applicants. 0.5 is guessing, 1.0 is perfect. Our main metric. |
> | **Precision / Recall / F1** | Precision: of those predicted eligible, how many are. Recall: of the truly eligible, how many we caught. F1: the balance of the two. |
> | **Cardinality** | The number of distinct values in a categorical column (ZIP has about 560, which is *high cardinality*). |
> | **Imputation** | Filling in missing values, always fit on training data only. |
> | **Encoding** | Turning categories into numbers (one-hot, frequency, target encoding). |
> | **Skew** | A lopsided distribution; our damage amounts are mostly zero with a long right tail. |
> | **Multicollinearity** | Two features carrying nearly the same information (correlation near 1). |
> | **Overfitting** | Memorizing the training data instead of learning patterns that generalize. |

---

## 1. Executive Summary *(1 page)*

> ✏️ **Owner note.** Write this section **last**: it summarizes sections 2–6. If a number changes anywhere, update it here too.

**Overview.** In September 2022, Hurricane Ian devastated southwest Florida. Lee County alone produced 194,482 valid applications to FEMA's Individuals and Households Program (IHP), which gives money to disaster survivors for housing and other urgent needs. For each household, FEMA must decide whether to award aid. We frame this as a **binary classification** problem and build a machine-learning pipeline that predicts **whether an applicant will be awarded IHP aid** (`ihpEligible`).

**Dataset.** Public OpenFEMA data (*IHP Valid Registrations v2*), filtered to disaster DR-4673 and Lee County: 194,482 applications and 100 columns, mixing numeric, yes/no, and categorical variables. The target is almost perfectly **balanced** (50.7% awarded, 49.3% not).

**What we found so far.**
1. **Data leakage is the main risk.** 43 of the 100 columns are *results* of the aid decision (dollar amounts, sub-program eligibility, denial reasons). One of them (`ihpAmount`) is identical to the target. A model given these columns would look perfect and be useless, so we separate them from the start.
2. **We defined two honest prediction moments:** *Tier 1* uses only what the applicant reports at registration (26 features). *Tier 2* adds what FEMA learns after inspection and verification (40 features).
3. **The problem is learnable.** Untuned **baselines** already reach a ROC-AUC of about **0.90 (Tier 1)** and **0.94 (Tier 2)**, against 0.50 for guessing. Much of the eligibility decision follows fairly mechanical FEMA rules (for example, 46% of all awards are exactly $700), so the model partly learns the screening process itself.
4. **Data quality is good but imperfect.** A few columns are mostly blank, damage amounts are heavily **skewed**, about 3% of records have a census block outside Lee County, and city names are typed by hand.

**Approach.** Clean the data and split it once (**stratified** 60/20/20). Compare interpretable and flexible models (logistic regression, decision tree, random forest, gradient boosting) with stratified **cross-validation**. Judge them on ROC-AUC, F1, and error rates across age and income groups, and keep a human decision-maker in the loop. Section 6 schedules the remaining work.

---

## 2. Business Understanding *(2–3 pages · 5 pts)*

> ✏️ **Owner note.** Graders want: a clear problem statement, *measurable* success criteria, the stakeholders, and the risks. The `[TEAM: confirm]` items are yours to decide.

### 2.1 Problem definition and motivation
After a major hurricane, tens of thousands of households apply for federal aid within days. In Lee County, applications peaked on September 30, 2022 (15,240 in one day), and 94% had arrived by the end of October (Figure 2.1). Each application passes through referral, documentation checks, sometimes an inspection, and a decision. About half end without an award, for varied reasons: 30% of unsuccessful applicants had insurance, 19% had no eligible damage or needs, 13% did not respond or withdrew, and 35% were never referred to the program (Figure 2.2).

![Figure 2.1: Applications over time](../figures/applications_over_time.png)
*Figure 2.1. Daily applications; the peak is Sep 30, 2022.*

![Figure 2.2: Reasons for ineligibility](../figures/ineligible_reasons.png)
*Figure 2.2. Why applicants were not awarded aid.*

**The question our model answers:** *Given what we know about an applicant, will FEMA award them IHP aid?*

**Why it matters.** Survivors wait for answers while caseworkers are overloaded. A reliable prediction could help FEMA:
- (a) route likely-eligible applications to a fast track;
- (b) flag likely-ineligible applications early, so staff can tell people what is missing (for example, insurance documents);
- (c) decide where to send inspectors first;
- (d) plan staffing and budget for the next disaster.

**What it is not.** The model is decision *support*. It must never automatically deny anyone aid. A person makes every final decision.

### 2.2 Business objectives and success criteria
| Objective | How we will measure it | Proposed target [TEAM: confirm] |
|---|---|---|
| Predict eligibility at registration time (Tier 1) | ROC-AUC and F1 on the held-out test set | Beat the untuned baseline: ROC-AUC ≥ 0.90, F1 ≥ 0.83 |
| Predict eligibility after inspection (Tier 2) | same | ROC-AUC ≥ 0.94 |
| Beat naive approaches | Compare to majority-class guessing and logistic regression | A clear win over both (majority-class accuracy is 50.7%; untuned logistic regression reaches AUC 0.84 at Tier 1) |
| Be fair across groups | Error rates by age band and income band | No group's error rate more than 5 percentage points above the overall rate |
| Be explainable | Feature-importance and partial-dependence plots a caseworker could read | Top drivers match FEMA's stated eligibility rules |

*Note.* These targets come from preliminary, untuned baselines (Section 5.1). They are goals for tuned models, not promises.

### 2.3 Stakeholder analysis
| Stakeholder | Interest | Cost of a wrong prediction |
|---|---|---|
| Applicants (survivors) | Fast, fair, understandable decisions | A household wrongly flagged "ineligible" (a **false negative**) could give up on a claim it deserves |
| FEMA caseworkers and program managers | Throughput, accuracy, defensible decisions | Wasted inspections (**false positives**), or eligible people missed |
| Lee County emergency management, State of Florida | Recovery speed, resource planning | Misjudged demand |
| Congress, auditors, taxpayers | Proper use of federal funds | Improper payments or unexplained denials |
| Equity and legal-aid organizations | No group treated worse | Systematic bias by age, income, or place |
| Our team / instructor | Sound methodology, honest reporting | n/a |

### 2.4 Expected impact and value proposition
A tuned, audited model would give FEMA an early, explainable signal at registration, when information is cheapest to act on. Its value is speed and prioritization, not replacing judgment. Because the outcome depends partly on FEMA's own screening rules, the model also works as a *consistency check*: applications where the model strongly disagrees with the outcome deserve a second look.

### 2.5 Risks and assumptions
- **Selection bias:** only *valid* registrations are in the data, so we can say nothing about invalid applications.
- **Scope / generalization:** one storm and one county, so results may not transfer to other disasters.
- **Label noise:** "not eligible" mixes many reasons (insurance, missing documents, withdrawal, never referred).
- **Fairness:** age, income, and location can correlate with vulnerable groups, so we will audit error rates by group.
- **Assumption to verify:** the flat $700 payment appears to be an emergency-needs payment. We infer this from the data and will confirm it in FEMA program documentation. [TEAM: confirm]

---

## 3. Data Understanding *(3–4 pages · 7 pts)*

> ✏️ **Owner note.** This is the highest-weighted section. The story runs in the order a grader checks: **what the data is → what we predict → what we must not use → what it looks like → how its parts relate → what's wrong with it → what that means for modeling.** Every subsection ends with an **ML takeaway**, the bridge into Section 4.

Full analysis: `notebooks/eda/01_initial_eda.ipynb` (exported as `reports/milestone_1/eda_notebook.html`). Column-by-column definitions: `docs/data_dictionary.md` (Appendix A).

**At a glance**

| | |
|---|---|
| Observations (rows) | **194,482** applications, one household each |
| Columns | **100** raw → **26** usable features at Tier 1, **40** at Tier 2 |
| Target | `ihpEligible`, balanced **50.7% / 49.3%** |
| Leakage columns removed | **43** |
| Columns with >10% missing | **6** |
| Best honest single feature | ROC-AUC ≈ **0.70** (vs 1.000 for the leaky `ihpAmount`) |

### 3.1 Dataset description, source, and collection
- **Source:** FEMA OpenFEMA, *Individuals and Households Program – Valid Registrations (v2)*, a public government dataset drawn from FEMA's National Emergency Management Information System (NEMIS).
- **Collection method:** downloaded on 2026-09-21 through the public OpenFEMA API (no key needed) with our script `src/data/fetch_fema.py`, filtered to disaster **DR-4673** and county **Lee (County)**. FEMA refreshes the data weekly, so the row count can drift slightly over time.
- **Size and grain:** 194,482 rows × 100 columns. Applications are dated 2022-09-27 to 2023-01-12. One row is one household's application (the **unit of observation**).
- **Requirement check:** ✅ ≥ 10 features (26 at Tier 1, 40 at Tier 2) · ✅ ≥ 1,000 rows · ✅ both numeric and categorical variables.
- **Caveats from FEMA:** raw operational data, "subject to a small percentage of human error"; only valid registrants are included.

**ML takeaway:** the dataset is large enough for a three-way split and k-fold cross-validation without starving any fold, but it covers a single disaster, which limits **generalization**.

### 3.2 The target variable
`ihpEligible` is True when the applicant received a housing and/or other-needs award. The classes are balanced: 98,648 eligible (50.7%) and 95,834 not (49.3%) (Figure 3.1).

![Figure 3.1: Target balance](../figures/target_balance.png)
*Figure 3.1. Class balance of `ihpEligible`.*

**ML takeaway:** because the classes are balanced, plain **accuracy** is meaningful and no resampling (e.g. SMOTE) or class weighting is needed. We still **stratify** every split to keep the ratio fixed.

### 3.3 Feature roles and data leakage (our key finding)
We sorted all 100 columns into roles, recorded in `src/utils/columns.py`, the single source of truth for the whole team:

| Role | Columns | Meaning |
|---|---|---|
| Target | 1 | `ihpEligible` |
| Tier 1: application-time | 27 | Self-reported at registration |
| Tier 2: later-stage | 14 | Learned after inspection or verification |
| Leakage | 43 | Results of the decision; never used |
| ID / constant | 15 | Identifiers, or identical on every row |

*(Tier 1 has 27 raw columns but 26 model features: cleaning drops `damagedCity` and replaces `appliedDate` with `daysSinceLandfall`.)*

**How we tested for leakage.** We scored each column *on its own* with a small model, which is a **univariate leakage test** (Figure 3.2). Outcome columns are near-perfect predictors (`ihpAmount` AUC 1.000, `onaEligible` 0.97, `ineligibleReason` 0.83), while the best honest column reaches only about 0.70.

**A borderline case.** `currentLocation` contains values that are *consequences* of aid ("FEMA-provided unit" is 99% eligible, "new temporary rental" 87%), so we moved it from Tier 1 to Tier 2. This cost only 0.013 AUC and made the model more honest.

![Figure 3.2: Single-feature ROC-AUC (leakage check)](../figures/leakage_auc.png)
*Figure 3.2. Each column scored alone. Leakage columns cluster near AUC = 1.*

**ML takeaway:** a high score is only meaningful if the features would exist at **prediction time**. The Tier 1 / Tier 2 split turns that rule into two concrete, testable feature sets.

### 3.4 Summary statistics and distributions
Applicants are mostly older, small households, and homeowners: 57% are 50 or older, most households have one or two people, 65% own the damaged home, and 77% registered online or through the mobile app (Figure 3.3).

- **Skewed numeric features:** damage measures such as `rpfvl` (FEMA-verified real-property loss) are zero for 83% of applicants and extremely **right-skewed** for the rest.
- **High-cardinality categoricals:** ZIP code (about 560 values) and census block group (about 3,000 values).

![Figure 3.3: Applicant profile](../figures/applicant_profile.png)
*Figure 3.3. Age, household size, ownership, and registration method.*

**ML takeaway:** skew calls for a **log transform** plus "has damage" flags. High cardinality rules out naive one-hot encoding, so we use **frequency or target encoding** instead (Section 4.2).

### 3.5 Relationships with the target (bivariate analysis)
Eligibility rate by feature (Figure 3.4):
- **Primary residence** acts almost like a hard rule: non-primary homes are about 0.4% eligible.
- **Emergency needs reported:** 34% eligible if not reported, 66% if reported.
- **Homeowners insurance** lowers eligibility (47% vs 55% without), consistent with FEMA covering what insurance does not. Flood insurance barely matters.
- **Income:** eligibility falls as reported income rises (61% under $15k to 45% above $175k). The "$0" group is the exception and is lowest (42%).
- **Owner vs renter** makes almost no difference (51% vs 50%).
- **Inspection (Tier 2):** applicants whose inspection was completed are 73% eligible vs 38% otherwise; unverified occupancy is under 1% eligible.
- **Geography and timing:** eligibility ranges from 43% to 62% across the 22 largest ZIP codes (Figure 3.5) and shifts with the week of application (Figure 3.6).

![Figure 3.4: Eligibility by feature](../figures/eligibility_by_feature.png)
*Figure 3.4. Eligibility rate within each category of key features.*

![Figure 3.5: Eligibility by ZIP](../figures/eligibility_by_zip.png)
*Figure 3.5. Eligibility across the 22 largest ZIP codes.*

![Figure 3.6: Eligibility by week](../figures/eligibility_by_week.png)
*Figure 3.6. Eligibility by week of application.*

**ML takeaway:** several strong signals are rule-like and non-linear (the income "$0" break, the residence cutoff), which favors **tree-based models**. The income bracket must stay an **unordered category**.

### 3.6 Missing values
Nineteen columns have missing values, eleven of them candidate features (Figure 3.7). Six are more than 10% missing:

| Column | Missing | Likely reason it is blank |
|---|---|---|
| `renterDamageLevel` | 96% | FEMA-determined damage level for *renter* dwellings only |
| `highWaterLocation` | 88% | Only applies where there was a high-water (flood) mark |
| `shelterNeed` | 82% | Almost always True when present, so blank ≈ need not reported |
| `habitabilityRepairsRequired` | 73% | Not filled in for most applicants; *whether* it was filled in is itself a signal [TEAM: confirm cause] |
| `foodNeed` | 52% | Almost always True when present, so blank ≈ need not reported |
| `selfAssessmentInformation` | 15% | Applicant's own damage rating, left blank by some |

Several of these blanks are **structural** (the question does not apply) or **informative** (a blank means "not reported"), so the data is *not* missing completely at random.

![Figure 3.7: Missing values](../figures/missing_values.png)
*Figure 3.7. Share of missing values per column.*

**ML takeaway:** filling blanks with a plain mean would erase a real signal. We add **missing-value indicator** features and impute inside the pipeline, fit on training data only.

### 3.7 Correlation analysis
No single numeric feature has a strong **linear** relationship with eligibility; the strongest correlation is about 0.36 (Figure 3.8). Several feature pairs are near-duplicates (Figure 3.9): inspection issued vs completed (1.00), real-property loss vs flood-damage amount (0.97), reported damage vs home damage (0.83).

![Figure 3.8: Correlation with target](../figures/correlation_with_target.png)
*Figure 3.8. Correlation of each numeric feature with the target.*

![Figure 3.9: Correlation matrix](../figures/correlation_matrix.png)
*Figure 3.9. Feature-to-feature correlations.*

**ML takeaway:** weak individual correlations combined with a strong multi-feature baseline (AUC 0.90) mean the signal comes from **feature interactions**. The near-duplicate pairs are **multicollinearity**: we drop one of each pair for linear models, while trees are largely unaffected.

### 3.8 Data quality assessment
| Check | Result | Decision |
|---|---|---|
| Constant columns | 14 columns have one value on every row (expected: one disaster, one county) | Drop |
| Duplicates | 407 rows (0.21%) identical on all non-ID columns; probably different households with the same answers | Keep |
| Location errors | 6,357 rows (3.3%) have a census block outside Lee County (2,707 are `NO_INTERSECT`; most others are neighboring counties); only 17 rows have a non-Florida ZIP | Keep; `NO_INTERSECT` as its own category |
| Hand-typed city names | Many spellings (e.g. `FT MYERS`, `FT MYERS BCH`) | Drop the city column |
| Implausible values | A water depth of 960 inches (80 ft) | Cap (outlier treatment) |
| Dates | Only 2 applications dated before the disaster declaration | Keep |

**ML takeaway:** quality is good overall. Every issue has a specific, documented fix in Section 4.1.

### 3.9 Challenges and limitations
| Challenge | Why it matters for ML |
|---|---|
| Data leakage | Inflated, meaningless scores if not controlled |
| One disaster, one county | Limited generalization to other storms |
| Valid registrants only | Selection bias |
| Label bundles many denial reasons | Label noise; one "no" can mean very different things |
| Heavy skew, structural missingness | Requires transforms and missing indicators |
| High-cardinality geography | Encoding choices can overfit if not done inside CV folds |
| Near-duplicate features | Multicollinearity for linear models |
| Outcomes partly follow FEMA rules | A high score is not proof the model understands *need* |

---

## 4. Data Preparation Plan *(2–3 pages · 3 pts)*

> ✏️ **Owner note.** Each row in 4.1 should answer a problem raised in Section 3. The cleaning code already exists and is shared: `src/features/prepare.py`.

### 4.1 Data cleaning strategy
| Issue (from §3) | Decision |
|---|---|
| Leakage, IDs, constants (58 columns) | Drop (list in `src/utils/columns.py`) |
| `damagedCity` (hand-typed) | Drop; use ZIP and census block instead |
| Yes/No columns stored as text with blanks | Convert to 0/1; keep blanks as truly missing (not 0) |
| Ordered ranges (`applicantAge`, household counts) | **Ordinal encoding** (`>5` becomes 6) |
| Income bracket | Keep as an unordered category (the "$0" group breaks the ordering) |
| Missing values | Do **not** fill in during cleaning. Use missing indicators plus median imputation (linear models) or native handling (boosting), fit on training data only |
| Implausible values (`waterLevel` = 960 in) | Cap at a high percentile |
| Out-of-county blocks | Keep; treat `NO_INTERSECT` as its own category |
| Duplicate-looking rows | Keep (see §3.8) |

### 4.2 Feature engineering
- `daysSinceLandfall` from the application date (already built).
- **Log transform** (`log1p`) of the skewed dollar and damage columns, plus "has damage" yes/no flags.
- **Missing-value indicators** for the columns whose blanks carry meaning.
- ZIP and census block: **frequency encoding** first (already used in the baselines), with **target encoding** tested *inside* cross-validation folds.
- Group rare categories together; drop one of each near-duplicate pair for linear models.
- Possible **interaction features** for emergency needs × residence status, which the boosting baseline suggests matter.

### 4.3 Data transformation
**Scaling** (standardization) and **one-hot encoding** are applied only for models that need them (logistic regression, and any distance-based method). Tree-based models use the unscaled data. **Every transformation that learns from data (imputation, scaling, encoding) is fit on the training split only**, inside a scikit-learn `Pipeline`, to prevent a second kind of leakage (**train–test contamination**).

### 4.4 Train / validation / test strategy
- **Stratified split, 60% train / 20% validation / 20% test**, with fixed random seed 42 so everyone gets the same split. Each part keeps the same 50.7% / 49.3% mix. Implemented in `split_data()`.
- **The test set is touched once**, at the very end of Milestone 2.
- Cross-validation (Section 5.3) uses the 80% development set (train + validation); the 20% test set stays sealed.
- **Robustness check (temporal validation):** because applications arrive over time, we will also train on early applicants and test on later ones to see whether performance holds up.
- We cannot link applications from the same household, so we note this as a limitation.

---

## 5. Modeling Approach *(2–3 pages)*

> ✏️ **Owner note.** Justify each model by a property of *our* data from Section 3 (skew, interactions, rule-like signals), not by general reputation.

### 5.1 Algorithm selection and justification
| Model | Why it fits this data |
|---|---|
| Logistic regression | Simple, fast, interpretable coefficients; the baseline to beat |
| Decision tree | Human-readable rules, and FEMA's process is rule-like |
| Random forest | Robust to skew, mixed types, and outliers; bagging reduces a single tree's **variance** (overfitting) |
| Gradient boosting | Best preliminary results; captures the feature interactions found in §3.7 |
| [TEAM: add or remove models based on what the course has covered, e.g. k-NN or a neural network] | |

**Preliminary baselines** (untuned, validation split, `src/models/baseline.py`):

| Feature set | Model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|---|
| Tier 1 (26 features) | Gradient boosting | 0.900 | 0.824 | 0.837 |
| Tier 1 | Logistic regression | 0.843 | 0.764 | 0.771 |
| Tier 1 + 2 (40 features) | Gradient boosting | 0.938 | 0.870 | 0.882 |
| Tier 1 + 2 | Logistic regression | 0.898 | 0.818 | 0.825 |
| any | Always guess majority class | 0.500 | 0.507 | 0.000 |

### 5.2 Evaluation metrics
| Metric | Why we use it |
|---|---|
| **ROC-AUC** (primary) | Measures ranking quality regardless of the decision threshold; suitable because the classes are balanced |
| **Precision, Recall, F1** | A false "ineligible" and a false "eligible" have different real-world costs, so we report both and choose the **decision threshold** explicitly |
| **Accuracy** | Fair here because of the class balance |
| **Confusion matrix** | Shows exactly where the errors fall |
| **Group error rates** (age, income) | Fairness audit (target in §2.2) |

### 5.3 Cross-validation and hyperparameter tuning
Five-fold **stratified cross-validation** on the development set. **Randomized hyperparameter search** runs inside the folds. Every model uses the same folds so comparisons are fair. The test set is used once, for the final score.

### 5.4 Expected challenges and mitigation
| Challenge | Mitigation |
|---|---|
| Hidden leakage as scores get high | Keep the roles file as the single source of truth; repeat the "drop a feature group and see what moves" **ablation study** for every final feature set (it already caught `currentLocation`) |
| Very high-cardinality geography | Frequency / target encoding fit inside folds; compare with and without |
| Skewed, mostly-zero damage columns | Log transform plus flags; prefer tree models |
| Model just re-learns FEMA's rules | Say so honestly; report cases where the model disagrees with outcomes as the interesting ones |
| Fairness across age / income / place | Report group error rates; investigate any gap over 5 points |
| Results specific to one storm | State the scope; run the temporal robustness check |

---

## 6. Project Timeline *(1 page)*

> ✏️ **Owner note.** Fill every `[TEAM]` owner cell with a name before export. The internal deadlines for this week are in the team responsibilities doc.

| Dates | Work | Owner |
|---|---|---|
| Sep 22–25 | Team meeting: agree on the target, success criteria, and roles; each member re-runs the EDA notebook and reads the data dictionary | All |
| Sep 26–29 | Finalize and proofread the report; export to PDF; assemble `TeamName_Milestone1_YYYYMMDD.zip` | [TEAM] |
| **Sep 30** | **Milestone 1 due** | All |
| Oct 1–14 | Finish the preprocessing pipeline (feature engineering, encoders) and the `notebooks/preprocessing` notebook | [TEAM] |
| Oct 15–31 | Train all models, cross-validate, tune hyperparameters | [TEAM] |
| Nov 1–10 | Evaluation: model comparison, interpretation (feature importance), leakage ablations, fairness audit | [TEAM] |
| Nov 11–18 | Write the final report; build the presentation; rehearse | All |
| Nov 19 | Internal freeze: code runs top to bottom, PDF exported | All |
| **Nov 21** | **Milestone 2 due** | All |
| **Nov 23 / Dec 2** | **Presentation (15 min + Q&A)** | All |

**Team responsibilities.** [TEAM: name] data and EDA | [TEAM: name] modeling | [TEAM: name] evaluation and report. Everyone contributes to every phase and attends the presentation.

**Risks and contingencies**
| Risk | Contingency |
|---|---|
| A teammate is unavailable | Everything is in Git with documented code, so work can be picked up; weekly check-ins |
| Scores drop below the baseline after removing leaky features | Report honestly; the leakage-removal story is itself a finding |
| Merge conflicts in shared notebooks | One person per notebook at a time; small commits; pull before starting |
| Running out of time before Nov 21 | Freeze the model set by Oct 31; drop optional models first |

---

## Appendices *(not counted in the page limit)*
- A. Data dictionary (`docs/data_dictionary.md`)
- B. EDA notebook export (`reports/milestone_1/eda_notebook.html`)
- C. AI usage log (`docs/AI_USAGE.md`)
- D. Additional figures (`reports/figures/`)

## References
FEMA. *OpenFEMA Dataset: Individuals and Households Program – Valid Registrations v2.* https://www.fema.gov/openfema-data-page/individuals-and-households-program-valid-registrations-v2

## AI-use disclosure
Portions of the code, analysis, and this draft were produced with Claude Code (Anthropic) and reviewed by the team. See `docs/AI_USAGE.md` for the full log.
