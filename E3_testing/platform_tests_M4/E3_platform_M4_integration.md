# E3 — the frozen model served by the platform (ISS-11), 24 Nov 2026   STATUS: DONE (ready to merge)

Until now the platform's fit classifier (FR-09) was the CS4085 model M0 (BERT-Tiny, seed 42). This change makes it serve the model frozen in E1: **M4, BERT-Tiny + Knowledge v1, original labels, seed 46**. The platform code itself was not changed in the team repository; the change is a ready patch, tested end to end.

## What the patch changes (`m4_platform_final.patch`, against Senior-2 ec28dfc)
| File | Change |
|---|---|
| `backend/models/fit_model.json` | New model card: M4 seed 46, weights SHA-256 0953ce89…, 512 tokens, input `knowledge_v1`, review threshold 0.30, recorded validation results. The previous card is kept as `fit_model_M0_s42_previous.json`. |
| `backend/app/knowledge.py` | Copy of the team's Knowledge v1 code, using the platform's own skill vocabulary. |
| `backend/app/fit_model.py` | `model_pair()` adds the Knowledge v1 line in front of the job text, exactly as in training, when the card says `knowledge_v1`. The SHA-256 check before loading is unchanged. |
| `backend/models/get_m4_weights.sh` | Downloads the weights from the weights/ folder of HIREMINDPROJECT/HireMind-Plan and checks the SHA-256 (the platform refuses any other file). |
| `backend/tests/test_fit.py` | Model-version test updated; new test that the Knowledge v1 line is added. |
| `evaluation/fr09_fit/choose_threshold_m4.py` | Integration check and review threshold, validation only (the test split is not read). |

## Results (GitHub Actions run 45387929083, branch e3-platform-m4-results-45387929083)
- **The platform's own code reproduces the frozen model exactly on validation:** accuracy 56.97 and Good Fit recall 38.41, with the identical confusion matrix [[471, 40, 68], [140, 80, 74], [87, 91, 111]].
- **HR review flag** (same rule as before: the highest threshold, step 0.025, that still flags at least half of the Good Fit pairs, validation only): **0.30**. It flags 32.4% of pairs and catches **54.0% of Good Fit**, against 38.4% for the predicted label alone; 41.4% of flagged pairs are Good Fit. For comparison, the M0 card flagged 34.5% and caught 56.1% with a much weaker model (validation macro-F1 46.5 vs 48.8).
- **Full test suite with the real model: 106 of 106 pass** (the 105 platform tests plus the new Knowledge-line test). Browser test: 20 screens pass. Non-functional tests on PostgreSQL: 5 of 5 pass (dashboard p95 750 ms; CV analysis p95 0.103 s with a fit prediction for 50 of 50 CVs).
- The no-model test step in this run ran before the threshold was set (the card held a placeholder), so 3 fit tests failed there; the full suite after the threshold step is the valid result.

## How the team applies it
```
cd Senior-2 && git apply m4_platform_final.patch && bash backend/models/get_m4_weights.sh
cd backend && pytest -q          # 106 passed with requirements-model.txt installed
```
Decision support only: the classifier never changes the score and never rejects anyone; HR makes every decision.
