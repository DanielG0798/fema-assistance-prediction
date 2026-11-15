# 5. Results and Evaluation

<!-- Owners: Anthony (performance metrics for all models, model interpretation, limitations and assumptions), Nandor (business value assessment, fairness). 3-4 pages. -->

<!-- ===== Anthony's part goes here: 5.1 performance metrics, 5.2 model interpretation, 5.5 limitations and assumptions ===== -->

<!-- ===== Nandor's part starts here. Only Nandor edits below this line. ===== -->
<!-- OUTLINE (AI first pass, Nov 15). Nandor writes the prose after the real notebook run. About 1.5 pages. -->
<!-- [NB] = take the number from notebooks/evaluation/02_fairness_business.ipynb after the real run. Section numbers may shift to fit Anthony's part. -->

## 5.3 Fairness across age and income groups

Source: Part 2 of the fairness notebook. Figure: `m2_fair_error_rate_by_group.png`. Optional second figure: `m2_fair_error_type_by_group.png`.

- What we tested: the Milestone 1 target, that no age or income group has an error rate more than 5 percentage points above the overall rate.
- Overall error rate for each tier. [NB]
- Result by age group: which groups are inside the limit and which are not, with their size. [NB]
- Result by income group. [NB]
- The pattern: error rates fall as age and income rise. [NB: highest and lowest group]
- Which kind of mistake drives the gap: younger and lower-income applicants are more often predicted eligible when they did not receive aid. [NB: the two rates for the highest and lowest income groups]
- What this means for an applicant: the more harmful mistake, missing a household that qualified, is spread more evenly. [NB: range of the missed-eligible rate across groups]
- Caution on small groups: the under-19 group has only a few hundred applicants, so its rate is uncertain. [NB: group size]
- Verdict on the target, stated plainly. [NB]

## 5.4 Business value assessment

Source: Parts 3 and 4 of the fairness notebook. Figures: `m2_fair_false_positives_by_reason.png` and `m2_fair_threshold_tradeoff.png`.

- What the predictions mean per 1,000 applications, for each tier: how many are correctly sorted, how many eligible households are missed, how many are wrongly predicted eligible. [NB: per-1,000 table]
- The two mistakes cost different things: a missed eligible household may wait longer or give up; a wrongly predicted one costs caseworker time. Link to the stakeholder table in Section 2.
- Three piles instead of one cut-off: share of applications the model can sort with high confidence, and how often it is right in each pile. [NB: triage table, one row per tier]
- What the "uncertain" pile means for staffing: this is where caseworkers should look first. [NB: its share]
- Who the false positives are: the share that are unfinished applications (did not respond, or still owed documents). These are households that looked eligible and dropped out, which FEMA can act on. [NB]
- Who does not finish: unfinished applications by age and income. [NB: highest group]
- What moving the cut-off buys: one sentence from the trade-off figure. [NB]
- What the model cannot do: it does not decide, it reflects one storm in one county, and part of what it learns is FEMA's own screening rules.
