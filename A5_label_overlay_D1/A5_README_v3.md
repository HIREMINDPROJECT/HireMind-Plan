# A5 — versioned TRAIN label overlay D1 (train_labels_v1)   STATUS: COMPLETE on the Dr.'s approval (v3 of this note); direct confirmation of the approval requested before freeze
Date 24 Nov 2026. D0 (original labels) and splits/splits.csv (SHA-256 2ff3926c…539f9b) are NOT modified.

## Approval
- The supervisor, email "HireMind – Phase A4 Adjudication and D1 Approval" (screenshot A5_D1_approval_email_supervisor_SCREENSHOT.png, received 24 Nov 03:16 Riyadh): approves "the finalized adjudication decisions … as the source of the updated D1 training labels, including the 255 documented label corrections" and authorises A5, A6, E1, then E4 and the final selection/freeze.
- Provenance notes (recorded as received): the screenshot does not show the sender address or date; the package's own README/summary were written before the approval and state "source_approval_count_documented": 0; the files name the decision source "A4_400_completed_evidence_recommendations.csv / A4_302_case_by_case_decisions_and_reasons.csv"; no per-case adjudicator name is in the files. The approval email is the authority for these labels.

## Names file received afterwards (24 Nov 03:21): A4_302_names_and_decisions.csv/.xlsx
- 302 rows (138 disagreements + 164 agree-but-differ-from-D0), adjudicated_label equals the D1 label in 302/302.
- Column "الاسم": 301 different first names for 302 rows, i.e. one different person per case (only one name occurs twice). Two of them match names of people involved in the project. The decision_rationale column has the same template text as change_log.csv.
- Project coordinator's explanation (24 Nov 03:23): the adjudication was done earlier, over several months of project work, by this group, one case per person; the file only lists who decided each case.
- Recorded as stated. The per-case names are kept privately (the file also contains CV/job excerpts) and are not uploaded.
- The label decisions are approved by the supervisor (email screenshot). Before the final freeze, a direct confirmation from the supervisor is requested (her reply in the email thread, or the original email forwarded with its header), because the screenshot shows neither sender address nor date.

## Verified (by script, 24 Nov)
- train_labels_v1.csv: 4,993 rows = exactly the frozen TRAIN rows; all non-label columns identical to splits.csv; label and label_id consistent. SHA-256 001212500c11a2d471845272718bda0e458968293d551d4979d6331226c178c5.
- 255 labels changed; all 255 are inside the 400-case A4 sample; changed rows = the 255 CHANGE decisions; D1 label = the decision file's label in all 400 A4 cases; the D0 label in the decision file matches splits.csv in all 400.
- Validation (1,162) and test (1,759) are untouched; D1 changes TRAIN labels only.
- Class mix TRAIN: D0 No 2,533 / Potential 1,231 / Good 1,229 → D1 No 2,615 / Potential 1,337 / Good 1,041. Transitions: Good→Potential 119, Good→No 75, Potential→No 31, No→Potential 24, Potential→Good 6.
- How the 400 were decided (from the decision file): where Amina and Reema agreed (262), D1 follows them in 260; on the 138 disagreements D1 equals Amina's label in 72, Reema's in 66.

## Files here (IDs and labels only; no CV/job text)
train_labels_v1.csv, change_log_ids_only.csv (255 rows), A4_400_case_decisions.csv, D1_variant_summary.json, approval screenshot.
Not uploaded: change_log.csv (contains CV/job excerpts, kept private) and splits_D1.csv (a modified copy of the split manifest; the frozen splits.csv remains the only split file).

## Then (A6/E1, done: see A6_D0_vs_D1)
M4 + Knowledge v1 retrained on D1 TRAIN labels, 8 CPU seeds, same pinned code as A1/E0, evaluated on the unchanged validation labels: GitHub run 45296065549 (branch e1-d1).
