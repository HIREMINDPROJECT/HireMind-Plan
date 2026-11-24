# E3 — fixed classifier, lexical vs semantic features (DistilBERT only)   STATUS: DONE on D0, validation only

Question (plan): does semantic matching improve requirement matching over lexical matching?
Setup: the same XGBoost (300 trees, depth 4, lr 0.05) and 8 seeds (42–49) on three feature sets. TRAIN rows fit it, VALIDATION rows score it. Test set not read. D0 labels. B1 extractor v3.1.
- lexical_only = the M3 feature set (B1 exact link counts and rates).
- lexical_plus_semantic = lexical + B3 DistilBERT aggregates (mean/min similarity, "missing" and semantic-only counts, per requirement type).
- semantic_only = the B3 aggregates alone.
The "missing" threshold is the label-free q10 of exact-found similarity on TRAIN (0.8121).

## Mean of 8 seeds (SD in brackets)
| Feature set | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall |
|---|---|---|---|---|---|
| lexical_only | 54.83 (0.78) | 43.49 (1.33) | 14.23 (2.03) | 45.61 (5.31) | 36.27 (0.96) |
| lexical_plus_semantic | 54.53 (0.66) | 42.84 (1.14) | 15.87 (1.96) | 55.15 (2.32) | 30.48 (0.75) |
| semantic_only | 49.66 (0.60) | 38.78 (0.79) | 15.40 (0.85) | 51.05 (2.09) | 25.38 (1.13) |
For reference, M4 (BERT-Tiny + Knowledge v1) on the same validation set: accuracy 56.79, Good Fit recall 36.85, macro-F1 48.59.

## Reading
- Adding semantic features did NOT improve accuracy (−0.30) or macro-F1 (−0.64), both inside one SD. It moved Good Fit precision up (+9.5 points) and Good Fit recall up slightly (+1.6, inside one SD), while Potential recall fell by 5.8 points (more than 5 SDs of the seed spread).
- Semantic-only is clearly worse than lexical-only on accuracy and macro-F1.
- Net: no evidence in this run that semantic matching improves the classification. The gain is a precision/recall trade, not an overall improvement. This is consistent with B3: raw similarity separates exact-found from not-found requirements poorly (AUC 0.576).
- Seed SDs here come only from XGBoost randomness on fixed features, so they understate real variability. Small differences are not claimed.
- Both feature sets are far below M4 on Good Fit recall (14–16 vs 36.85).

## Coverage (VALIDATION, 11,795 requirements)
exact-found 3,067 (26.0%); exact-negative 0; semantic-only candidates 7,349; "missing" 1,379. Exact + semantic candidates cover 88.3%. The semantic-only candidates are NOT verified matches. Judging how many are valid needs human reading (B2 pack or an HR sample).

## Limits
- This tests whether B3 features help a fixed classifier. It does not measure matching accuracy (needs B2 human gold).
- D0 labels; D1 not available.

## Files
E3_validation.csv (per seed + mean/SD), E3_coverage.json, E3_validation_predictions.csv, E3_lexsem_top_features_lastseed.csv, E3_semantic_pair_features_validation.csv (IDs/scores only), e3_lex_vs_sem.py, pip_freeze.txt. Code: branch `e3-lexsem` of HIREMINDPROJECT/HireMind-Plan, run 45256020209.
