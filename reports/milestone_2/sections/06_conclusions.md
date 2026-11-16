# 6. Conclusions and Recommendations

<!-- Owner: Nandor. Outline: key findings summary; practical recommendations; future work; project reflection and lessons learned. 2-3 pages. -->
<!-- OUTLINE (AI first pass, Nov 15). Bullet points only, not prose. Nandor writes the section from these after the model freeze. -->
<!-- [VAL] = validation-set number, replace with the test-set number. [NB] = take the number from the fairness notebook after the real run. [TEAM] = needs a team decision. -->

## 6.1 Key findings

- The best model is gradient boosting on both tiers: ROC-AUC 0.880 at registration and 0.938 after inspection [VAL].
- Ten algorithms were tuned the same way. The top four finished within 0.016 of each other on Tier 1, so the limit is the information on the application form, not the algorithm.
- The model is far better than the simple references: 81.0% accuracy against 50.7% for always guessing "eligible", and 0.880 against 0.820 ROC-AUC for logistic regression [VAL].
- It misses the Milestone 1 targets for Tier 1 (0.90 ROC-AUC, 0.91 F1) and is just under the Tier 2 target (0.94). Those targets came from early baselines on unverified data. [TEAM: agreed wording]
- The mistakes lean one way: the model rarely misses a household that received aid, and more often predicts aid for a household that did not get it. [NB: both rates, both tiers]
- Fairness: [NB: which groups are inside and outside the 5-point limit]. Error rates fall as age and income rise. [NB]
- "Not eligible" is not one group. [NB: share of the model's false positives that are unfinished applications]
- 43 of the 100 columns are results of FEMA's decision. Finding and removing them mattered more than any modeling choice.

## 6.2 Practical recommendations

- Use the model to sort the queue, never to decide. A person makes every final decision.
- Use three piles, not one yes/no cut-off: likely eligible, likely not eligible, uncertain. [NB: share of applications in each pile and how often the model is right, from the triage table]
- Treat a "likely not eligible" prediction as a prompt to tell the household what is usually missing, not as a rejection.
- Follow up with applicants the model rates as likely eligible who stop responding. [NB: how many of these there are]
- Review the groups with the highest error rates by hand until the gap is understood. [NB: which groups]
- Use the registration model (Tier 1) on day one and the inspection model (Tier 2) once an inspection is back.
- Check the error rates by group again every time the model is retrained.

## 6.3 Future work

- Test on another disaster or another county. Everything here is one storm in one county.
- Predict the reason for ineligibility, not only yes or no, so the model can say what is missing.
- Model "completed the process" and "qualified" as two separate steps. [TEAM: depends on Daniel's completed-only comparison]
- Add outside data, such as flood-zone maps or storm-surge depth by address.
- Try fairness-aware methods, such as a separate cut-off per group, and measure what they cost in accuracy.
- Check the model on later applications only (train on early weeks, test on late weeks). [CONFIRM: Anthony's temporal check may already cover this]

## 6.4 Project reflection and lessons learned

- Sorting columns by when their values become known should come before any modeling.
- Our first targets were set from numbers we had not verified. Set targets after the baseline is re-run, or state clearly that they are provisional.
- A single accuracy number hid two things that matter to FEMA: which kind of mistake is made, and for whom.
- Keeping one seed and one split across notebooks let three people work on the same applicants. [TEAM: add what did and did not work in how we split the work]
- What we would do differently: [TEAM: one or two points from each member]
