# Phase C, stage 2 — M3 XGBoost and M6 MiniLM + classifier (D0 labels) — v3.1 (CURRENT), 24 Nov 2026

**Status update (24 Nov 2026):** the D1 version of M3 was run in E4, and DistilBERT (M5) was run with more seeds and with the Knowledge v1 line through the plan's steps; see FINAL_RESULTS_REPORT.md.


**Status:** VERIFIED run, validation only, original labels (D0). Provisional for Q2 (the plan needs D1 for the final comparison). M5 DistilBERT is reported separately when its run ends.
**Supersedes** the first Stage 2 report (extractor v3.0, M3 accuracy 57.53). The v3.0 sentence splitter ignored spaced sentences and mistyped 11% of job requirement tags (see B1 schema.md); M3 uses those types, and M6's sentence units use the same splitter. The fix was made because of that bug, not because of a result; both sets of numbers are shown so the effect is visible.
**Rule:** the preset in `C_selection_rule_PRESET.md` (fixed before any result). The test set was not read; no hyperparameter was tuned on validation.
**Run:** GitHub Actions run 45251626834, branch `c-stage2` (results branch `c-stage2-results-45251626834`), pinned HireMind-DL @1f22f34, extractor v3.1. Library versions in `pip_freeze.txt`.

| Model | Accuracy | Good Fit recall | Good Fit precision | Macro-F1 | Potential recall |
|---|---|---|---|---|---|
| **M4 Knowledge v1, 8-seed mean (reference)** | **56.79** (SD 1.17) | **36.85** (SD 11.24) | **43.66** | **48.59** (SD 2.80) | **29.42** |
| M3 XGBoost, mean of 8 seeds (SD) | 54.83 (0.78) | 14.23 (2.03) | 45.61 (5.31) | 43.49 (1.33) | 36.27 (0.96) |
| M6 MiniLM + LogReg (C=0.1), single deterministic run | 54.65 | 24.22 | 51.85 | 48.71 | 53.40 |
| M6, class-weighted (sensitivity) | 52.24 | 24.91 | 49.66 | 47.38 | 56.12 |

Superseded v3.0 values for comparison (do not cite): M3 accuracy 57.53 / Good Fit recall 25.30 / Macro-F1 48.43; M6 accuracy 54.73 / Good Fit recall 21.80 / Macro-F1 48.21.

## Rule check (accuracy AND Good Fit recall must both beat the M4 mean; no drop over 2.0 points in Good Fit precision, Macro-F1 or Potential recall)
- **M3: not eligible.** Accuracy 54.83 is below M4 (56.79), and Good Fit recall 14.23 is far below 36.85. Macro-F1 is also 5.1 points lower (over the 2.0 limit).
- **M6: not eligible.** Accuracy (54.65 vs 56.79) and Good Fit recall (24.22 vs 36.85) are both below M4.
- Result: **negative result for M3 and M6 as replacements for M4** on D0. M4 stays the strongest defensible baseline so far.

## Things worth knowing
- M6 finds far more Potential Fit (recall 53.40 vs 29.42) but loses Good Fit recall and accuracy: a different trade-off, not a better model by the rule.
- M3 uses only B1 structured and link features (no B3 semantic features). Its top features are count and rate features of the job description; some of its signal may come from the job text rather than CV evidence. That is untested (would need an E5-style ablation).
- M3 is a deterministic feature set with seed-only variation in XGBoost, so its SD understates other sources of uncertainty. M6 is a single fit, so no SD.
- Labels are D0 (original, unreviewed). Conclusions can change when D1 exists.

## Files
`C_stage2_m3_m6_validation.csv` (per seed, mean, SD), `C_stage2_validation_predictions.csv` (row IDs and predicted classes only), `C_stage2_config.json`, `C_stage2_M3_top_features_lastseed.csv`, `pip_freeze.txt`, `c_stage2_m3_m6.py`.
