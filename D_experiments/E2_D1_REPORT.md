# E2 (plan definition) — M4 + D1 labels + evidence-selected input   STATUS: DONE · NOT ELIGIBLE
Run 45297263624 (branch e2-d1): same code as the D0 E2 run, B4 lex_k evidence selection (extractor v3.1) + Knowledge v1 line, M4 protocol, 8 CPU seeds, TRAIN labels = D1 (train_labels_v1, 255 changes), validation labels D0 (unaudited). Test never read.

## Mean of 8 seeds (SD) and paired difference vs E1 (same D1 labels, plain input)
| | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall |
|---|---|---|---|---|---|
| E2 on D1 | 56.12 (1.13) | 47.58 (1.83) | 24.91 (4.03) | 50.96 (6.50) | 38.27 (8.27) |
| E1 (D1, plain input) | 56.12 | 47.04 | 24.91 | 48.65 | 35.12 |
| E0 (D0, plain input) | 56.79 | 48.61 | 36.72 | 43.68 | 29.68 |
| E2 − E1 (paired) | 0.00 (better in 5/8) | +0.54 (5/8) | 0.00 (4/8) | +2.31 (4/8) | +3.15 (4/8) |
(The accuracy and Good Fit recall means of E2 and E1 coincide by chance: both arms get 5,217 correct and 576 Good Fit correct in total over the 8 seeds; per-seed values differ, see E2_D1_per_seed_vs_E1_E0.csv.)

## Reading
- On D1 labels, evidence selection changes nothing beyond noise: accuracy and Good Fit recall are unchanged; Good Fit precision and Potential recall move up slightly, inside the seed spread. Same conclusion as E2 on D0.
- Against the M4/D0 reference, Good Fit recall stays ~12 points lower, driven by the label change (see A6/E1), not by the input.
- Preset rule: not eligible.
