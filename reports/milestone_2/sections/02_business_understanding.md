# 2. Enhanced Business Understanding

<!-- Owner: Nandor. Outline: refined problem definition; updated success criteria; lessons learned about the domain. 1-2 pages. -->
<!-- DRAFT (AI first pass, Nov 15). Nandor rewrites this in his own words. -->
<!-- Numbers marked [VAL] come from the validation set in notebooks/modeling/01_model_comparison.ipynb and must be replaced with test-set numbers after the model freeze. -->
<!-- Items marked [TODO] wait on the fairness notebook or on a team decision. -->

## 2.1 Refined problem definition

In Milestone 1 our question was: given what we know about an applicant, will FEMA award them Individuals and Households Program (IHP) aid? After building the models, we now split that question into three more exact ones.

**Question 1: at registration (Tier 1).** Using only the 25 answers a household gives when it registers, how likely is an award? This is the version FEMA could use on the day an application arrives, to sort the queue.

**Question 2: after the inspection (Tier 2).** Using those answers plus the 14 fields that are filled in later, such as the inspection results, how likely is an award? This is more accurate, but the answer comes later, so it is less useful for sorting.

**Question 3: who is in the "not awarded" group.** [TODO: confirm with Daniel] About half of the applicants did not receive aid, but they are not one group. Some were turned down on the merits, for example because insurance covered the loss or the damage did not qualify. Others never finished the process: they did not send documents or did not answer FEMA. Our data records both as "not eligible". A model trained on everyone therefore predicts "will this household end up with an award", which is partly a prediction of who completes the paperwork. We report that model, and we also report a second version trained only on applicants who completed the process, which is closer to "does this household qualify".

The purpose has not changed. The model supports a caseworker's decision. It never denies aid by itself, and a person makes every final decision.

## 2.2 Updated success criteria

The Milestone 1 targets were taken from early, untuned models that had not yet been re-run on the verified data. We said then that they were goals, not promises. The table shows each target next to what the tuned models reached.

| Objective | Milestone 1 target | Result | Met? |
|---|---|---|---|
| Predict an award at registration (Tier 1) | ROC-AUC of 0.90 or higher, F1 of 0.91 or higher | ROC-AUC 0.880, F1 0.824 [VAL] | No |
| Predict an award after inspection (Tier 2) | ROC-AUC of 0.94 or higher | ROC-AUC 0.938 [VAL] | Almost |
| Beat the simple approaches | Clearly beat majority-class guessing and logistic regression | Tier 1 accuracy 81.0% against 50.7% for guessing; ROC-AUC 0.880 against 0.820 for logistic regression [VAL] | Yes |
| Be fair to every group | No age or income group has an error rate more than 5 percentage points above the overall rate | Tier 1: largest gap 4.7 points (ages 19-34). Tier 2: ages 19-34 are 5.1 points above; all income groups are within the limit [VAL] | Tier 1 yes, Tier 2 missed by 0.1 point |
| Be explainable | The main drivers match FEMA's own eligibility rules | [TODO: Anthony's feature importance] | [TODO] |

Two of these need an honest explanation.

**The Tier 1 target was too high.** The original 0.90 came from a baseline that was run before the data and the feature lists were verified. On the verified data, ten different algorithms were tuned with 5-fold cross-validation, and the best four finished within 0.016 of each other, between 0.864 and 0.880. When very different methods stop at the same place, the limit is in the information available at registration, not in the choice of algorithm. We therefore restate the Tier 1 criterion as: clearly beat the reference models and state plainly how accurate triage at registration can be. [TODO: team agrees on the new wording]

**Accuracy is not the only measure that matters.** The best Tier 1 model finds 87.4% of the applicants who were awarded aid, but it also predicts "award" for about a quarter of those who were not (specificity 74.4%) [VAL]. For a triage tool, these two mistakes do not cost the same. Section 5 covers what each one costs.

## 2.3 Lessons learned about the domain

**Much of the dataset is the answer, not the question.** 43 of the 100 columns record the result of FEMA's decision, such as award amounts and the reason for ineligibility. Using any of them would give a perfect score on paper and a useless model in practice. Sorting columns by when their values become known was the most important step in the project.

**When the information arrives matters as much as what it is.** An inspection was issued for 36% of applicants. Of those, 73% received an award, compared with 38% of applicants without one. The inspection fields raise the best ROC-AUC from 0.880 to 0.938 [VAL], but FEMA only has them later. The business choice is between an earlier, rougher signal and a later, sharper one.

**"Not eligible" does not mean "did not qualify".** The label mixes several outcomes. Among applicants without an award, 30% had insurance, 19% had no eligible damage or needs, 13% did not respond or withdrew, and 35% were never referred to the program. A household that stopped answering letters is counted the same way as one whose claim was rejected. This is why Question 3 above exists, and it is the main caution for anyone who uses the model's output.

**The process follows rules, and the models show it.** Models built from decision trees did best, and straight-line models all stopped near 0.82 on Tier 1. That fits a program where the outcome depends on combinations of conditions, such as a primary residence together with an emergency need. Part of what the model learns is FEMA's own screening rules, including the flat $700 Serious Needs Assistance payment that is approved soon after registration.

**One storm, one county.** Every row is Hurricane Ian in Lee County. Nothing here shows that the same model would work for a different disaster, a different state, or a year with different program rules.
