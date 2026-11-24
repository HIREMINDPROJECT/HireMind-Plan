# Phase C — model-selection rule, fixed BEFORE any M1/M2 result was seen (23 Nov 2026)
Baseline to beat (M4, Knowledge v1, 8-seed validation mean): accuracy 56.79%, Good Fit recall 36.85%, Good Fit precision 43.66%, Macro-F1 48.59%.
A model is eligible only if it improves BOTH accuracy and Good Fit recall over the M4 mean, and none of the following falls by more than 2.0 points vs M4: Good Fit precision, Macro-F1, Potential Fit recall (M4 Potential recall from baseline_results_v2.csv).
Single-seed or deterministic models are reported as single results and compared with the M4 mean and its sample SD (accuracy 1.17, Good Fit recall 11.24); differences inside one SD are not claimed as improvements.
No hyperparameter is tuned on the validation set: M1/M2 use fixed library defaults declared in the script. The test set is not read.
All numbers in this stage use ORIGINAL labels (D0). They answer Q2 only provisionally; the plan's Q2 needs D1.
