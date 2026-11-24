# A4/A5 — follow-up to the supervisor's review (24 Nov 2026)
No label was changed by this follow-up. D0, splits.csv and train_labels_v1.csv are unchanged.

## 1. The two cases where the final label differs from the reviewers' agreement
| Case | row_uid | D0 | Reviewer 1 = Reviewer 2 | Final (D1) | Decision |
|---|---|---|---|---|---|
| 93 | train_2691 | No Fit | Potential Fit | No Fit | KEEP_D0 |
| 44 | train_3292 | Potential Fit | No Fit | Potential Fit | KEEP_D0 |
- Both are KEEP_D0: the adjudication kept the original label instead of following the reviewers. A written adjudication reason exists for both (private file, contains CV text).
- Case 93 (summary): the posting is a JDE business-analyst role with JDE as a mandatory skill and 10 years requested; the CV is a cloud data-engineering profile with about 4 years; no JDE evidence → No Fit kept.
- Case 44 (summary): the posting asks for a CPA with public-accounting-firm experience; the CV shows staff-accounting (AP/AR) experience but no CPA or public-firm evidence → transferable alignment, Potential Fit kept.

## 2. Good Fit changes — consistency check
- In the 400-case sample, 198 cases were D0 "Good Fit"; 194 were changed (to Potential Fit 119, to No Fit 75); 4 kept.
- By sampling stratum (D0 Good Fit cases → D1): CLEANLAB 108 of 111 changed; BOUNDARY 49 of 50 changed; **RANDOM 37 of 37 changed** (27 → Potential, 10 → No).
- Overall change rate by stratum: RANDOM 75/150 (50%), CLEANLAB 120/150 (80%), BOUNDARY 60/100 (60%).
- Reading: because the RANDOM stratum is representative of TRAIN, the 37/37 result means the review applied a clearly stricter definition of Good Fit than the original dataset labels. This can be a real label-quality problem in D0, or a rubric that is stricter than the dataset's notion of "Good Fit" — the data alone cannot tell which. It also explains why D1 lowers Good Fit recall on the (unaudited, D0-labelled) validation set (A6/E1).
- Consequence for the results: D1 models are judged against the original (D0) validation labels, so part of D1's lower Good Fit recall reflects this difference in the definition of Good Fit, not only model quality. This is stated as a limitation in the final report.

## 3. Required fields of reviewed_labels.csv (evidence span, reason, confidence)
- The Drive file reviewed_labels.csv carries IDs, labels, both reviewers' confidence and the decision.
- A complete version with both reviewers' evidence spans and reasons and the adjudication reason for every case (reviewed_labels_FULL_PRIVATE.csv, 400 rows) exists. It contains CV text, so it is handed over privately and is not stored in Drive or GitHub.
