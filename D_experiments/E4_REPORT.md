# E4 — model comparison and final selection (validation only)   STATUS: DONE
Full table: E4_model_comparison_validation.csv (15 rows: M1, M2, M3 on D0 and D1; M4 on D0 and D1; M4 + evidence on D0 and D1; M5; M6; M8 × 4 variants). Test set never used for selection.

## Preset rule (fixed before any result, C_selection_rule_PRESET.md)
Eligible only if a model beats M4 (D0) on accuracy AND Good Fit recall, with no drop > 2.0 points in GF precision, macro-F1 or Potential recall; single-seed results are compared with M4's seed spread.

## Outcome
- **No model is eligible.** Closest candidates: M4 + evidence input on D0 (GF recall 41.52 but accuracy 56.30 < 56.79, inside noise) and M8 minus_struct (GF recall 48.79 but accuracy 54.13, one seed).
- Per the plan ("if no model meets the rule, report a negative result and keep the strongest defensible baseline"): **the final candidate is M4 — BERT-Tiny + Knowledge v1 trained on D0 labels.**
- Models not trained on D1: M5 and M8 (DistilBERT, several CPU hours per seed); M6 (not part of the team decision); M7 (optional, outside the DistilBERT-only decision).

## Which checkpoint is frozen (rule stated before choosing)
The 8 M4 seeds differ a lot (GF recall 13.5–49.1). To avoid picking the luckiest seed on validation, the frozen checkpoint is the seed **closest to the 8-seed mean** (standardised distance on accuracy, macro-F1 and Good Fit recall): **seed 46** (accuracy 56.97, macro-F1 48.80, GF recall 38.41, GF precision 43.87, Potential recall 27.21; confusion [[471,40,68],[140,80,74],[87,91,111]]). Checkpoint SHA-256 0953ce892ee0f075d37c1e6b4ff0bceb02a7932fddbca0ee5954f5e17fcb0c8e.
The expected performance to report is the 8-seed mean ± SD, not the single seed.
