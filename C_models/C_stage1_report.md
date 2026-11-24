# Phase C, stage 1 — M1 Logistic Regression and M2 Linear SVM (D0 labels), 23 Nov 2026

**Status update (24 Nov 2026):** the D1 versions of M1 and M2 were run in E4; both label versions are in E4_model_comparison_validation.csv and FINAL_RESULTS_REPORT.md.


**Status:** VERIFIED run, validation only, original labels (D0). Provisional for Q2 (the plan needs D1 for the final comparison).
**Rule:** fixed before the run in `C_selection_rule_PRESET.md`. The test set was not read; no hyperparameter was tuned on validation.
**Run:** GitHub Actions CPU runner, the project's GitHub fork, pinned HireMind-DL @1f22f34, scikit-learn 1.6.1; `pip_freeze.txt` is in the run's results branch (`c-stage1-results-45240656490`).

| Model (TF-IDF word 1–2gram, CV + job) | Accuracy | Good Fit recall | Good Fit precision | Macro-F1 | Potential recall | No Fit recall |
|---|---|---|---|---|---|---|
| **M4 Knowledge v1, 8-seed mean (reference)** | **56.79** | **36.85** | **43.66** | **48.59** | **29.42** | — |
| M1 Logistic Regression | 51.55 | 9.00 | 48.15 | 35.34 | 16.67 | 90.50 |
| M2 Linear SVM | 50.86 | 13.84 | 38.84 | 39.34 | 26.19 | 81.87 |
| M1, class-weighted (sensitivity) | 49.74 | 13.49 | 37.86 | 40.44 | 38.10 | 73.75 |
| M2, class-weighted (sensitivity) | 48.80 | 14.53 | 36.52 | 39.58 | 33.67 | 73.58 |

Confusion matrices (true rows × predicted columns, order No Fit / Potential / Good), each row sums to 579 / 294 / 289: M1 `[[524,41,14],[231,49,14],[250,13,26]]`; M2 `[[474,71,34],[188,77,29],[221,28,40]]`.

## Reading
- **Neither model is eligible** under the preset rule: accuracy and Good Fit recall are both below M4 (the shortfall is 5–6 points in accuracy and 22–28 points in Good Fit recall, far beyond the M4 seed SD of 1.17 and 11.24).
- The lexical models mostly predict No Fit. Class weighting raises Potential recall but accuracy falls further.
- Inference time is negligible (~0.01 ms per pair), so speed does not offset the accuracy gap.
- Result: **negative result for M1 and M2 as replacements for M4.** They remain useful as simple lexical baselines in the final table.
- Caveat: the fixed featurisation and defaults were not tuned; a tuned lexical baseline might do somewhat better, but tuning on the validation set would break the protocol. Any tuning must use TRAIN-only grouped cross-validation.
