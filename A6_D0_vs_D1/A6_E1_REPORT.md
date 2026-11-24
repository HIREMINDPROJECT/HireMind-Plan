# A6 / E1 — Does label quality improve results? (D0 vs D1)   STATUS: DONE · result: MIXED → NOT ELIGIBLE

## Setup (identical except the TRAIN labels)
M4 = BERT-Tiny + Knowledge v1, same pinned code (Senior-2 ec28dfc, HireMind-DL 1f22f34), seeds 42–49, 512 tokens, 3 epochs = 939 updates, same optimizer and steps, best-validation-macro-F1 checkpoint, CPU.
- D0 = original TRAIN labels (E0 run, which reproduced A1 exactly).
- D1 = train_labels_v1.csv: 255 TRAIN labels changed (approved), all inside the 400-case A4 sample. Overlay applied at load time and checked in every run log ("255 labels changed"). splits.csv untouched.
- VALIDATION labels are the original D0 labels in both arms (1,162 pairs, not audited). Test never read.
GitHub run 45296065549 (branch e1-d1). Checkpoint SHA-256 per seed in D1_checkpoint_sha256.json; weights kept as Actions artifacts (90 days).

## Result: mean of 8 seeds (SD)
| | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall | No Fit recall |
|---|---|---|---|---|---|---|
| D0 | 56.79 (1.17) | 48.61 (2.80) | 36.72 (11.28) | 43.69 (4.64) | 29.68 (4.54) | 80.57 (2.94) |
| D1 | 56.12 (0.86) | 47.04 (1.63) | 24.91 (5.17) | 48.65 (5.54) | 35.12 (4.46) | 82.36 (2.65) |
| Paired Δ (D1−D0) | −0.67 (SD 1.38), better in 3/8 | −1.57 (SD 2.18), 1/8 | **−11.81 (SD 8.93), 1/8** | +4.97 (SD 6.49), 5/8 | +5.44 (SD 5.82), 6/8 | |
Per-seed class-wise P/R/F1 and confusion matrices: D0_vs_D1.csv. Paired differences: D0_vs_D1_paired_deltas.csv.

## Reading (no success claimed)
- **Mixed, not an improvement.** Accuracy and macro-F1 do not change beyond noise. Good Fit recall falls clearly (in 7 of 8 seeds); Good Fit precision and Potential Fit recall rise. Under the preset rule (beat M4 on accuracy AND Good Fit recall) D1 is **not eligible**.
- **Why this direction is expected:** the approved corrections move 194 training labels out of Good Fit (Good→Potential 119, Good→No 75) and only 6 into it, so the model learns to say Good Fit less often. It then finds fewer of the pairs that the validation set calls Good Fit, but is right more often when it does.
- **Important limitation:** the validation labels are unaudited D0 labels. The A4 review found that reviewers often disagree with D0 on Good Fit (125 of the 164 reviewer-agreed changes were D0 "Good Fit"). If validation has the same pattern, part of D1's Good Fit "misses" may be pairs whose D0 label is itself wrong. This experiment cannot tell which; a reviewed validation subset would be needed to separate the two.
- Seed spread of Good Fit recall shrinks with D1 (SD 11.3 → 5.2), i.e. D1 training is more stable.

## Consequence for selection
M4 trained on D0 remains the strongest defensible baseline under the preset rule. D1 is kept as the label-cleaning arm of the ablation table.
