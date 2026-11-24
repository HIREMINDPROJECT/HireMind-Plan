# HireMind — final results (validation, 24 Nov 2026)

All numbers are on the frozen VALIDATION split (1,162 pairs: 579 No Fit, 294 Potential Fit, 289 Good Fit). The test set was never used. Mean ± sample SD over seeds; a single seed has no SD. Selection rule (fixed before the experiments): beat M4 E0 on accuracy AND Good Fit recall, with no drop of more than 2.0 points in Good Fit precision, macro-F1 or Potential Fit recall.

## 1. All models

| Group | Model | Train labels | Seeds | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall | Meets rule |
|---|---|---|---|---|---|---|---|---|---|
| Classical | M1 LogReg TF-IDF | D0 | 1 | 51.55 | 35.34 | 9.00 | 48.15 | 16.67 | no |
| Classical | M2 LinearSVC TF-IDF | D0 | 1 | 50.86 | 39.34 | 13.84 | 38.83 | 26.19 | no |
| Structured | M3 XGBoost (B1 features) | D0 | 8 | 54.83 ± 0.78 | 43.49 ± 1.33 | 14.23 ± 2.03 | 45.61 ± 5.31 | 36.27 ± 0.96 | no |
| Semantic | M6 MiniLM + LogReg | D0 | 1 | 54.65 | 48.71 | 24.22 | 51.85 | 53.40 | no |
| Classical | M1 LogReg TF-IDF | D1 | 1 | 51.64 | 34.83 | 6.23 | 54.55 | 19.39 | no |
| Classical | M2 LinearSVC TF-IDF | D1 | 1 | 50.17 | 37.49 | 10.03 | 35.80 | 26.19 | no |
| Structured | M3 XGBoost (B1 features) | D1 | 8 | 55.56 ± 0.47 | 43.23 ± 0.73 | 11.29 ± 1.12 | 59.94 ± 3.56 | 39.12 ± 1.42 | no |
| BERT-Tiny + Knowledge v1 | M4 E0 | D0 | 8 | 56.79 ± 1.17 | 48.61 ± 2.80 | 36.72 ± 11.28 | 43.68 ± 4.64 | 29.68 ± 4.54 | baseline |
| BERT-Tiny + Knowledge v1 | M4 E1 | D1 | 8 | 56.12 ± 0.86 | 47.04 ± 1.63 | 24.91 ± 5.17 | 48.65 ± 5.54 | 35.12 ± 4.46 | no |
| BERT-Tiny + Knowledge v1 | M4 + evidence (E2) | D0 | 8 | 56.30 ± 1.55 | 48.69 ± 3.14 | 41.52 ± 12.39 | 42.99 ± 3.98 | 27.76 ± 5.51 | no |
| BERT-Tiny + Knowledge v1 | M4 + evidence (E2) | D1 | 8 | 56.12 ± 1.13 | 47.58 ± 1.83 | 24.91 ± 4.03 | 50.96 ± 6.50 | 38.27 ± 8.27 | no |
| DistilBERT | M5 plain (CV, job) | D0 | 1 | 50.60 | 39.32 | 12.46 | 32.43 | 30.95 | no |
| DistilBERT + Knowledge v1 | M5 + Knowledge (E0) | D0 | 4 | 50.30 ± 0.80 | 37.10 ± 3.20 | 17.65 ± 11.31 | 35.25 ± 8.13 | 16.92 ± 2.94 | no |
| DistilBERT + Knowledge v1 | M5 + Knowledge (E1) | D1 | 4 | 50.77 ± 1.61 | 39.54 ± 6.43 | 20.33 ± 16.50 | 37.45 ± 6.16 | 24.74 ± 10.81 | no |
| DistilBERT + Knowledge v1 | M5 + Knowledge + evidence (E2) | D0 | 8 | 54.04 ± 3.44 | 45.90 ± 5.00 | 40.40 ± 15.69 | 44.21 ± 8.84 | 23.17 ± 6.17 | no |
| DistilBERT + Knowledge v1 | M5 + Knowledge + evidence (E2) | D1 | 8 | 54.82 ± 2.50 | 46.19 ± 3.91 | 38.75 ± 10.81 | 45.37 ± 7.79 | 22.19 ± 4.38 | no |
| DistilBERT hybrid (M8, E5) | M8 full | D0 | 1 | 53.96 | 45.48 | 32.87 | 42.22 | 25.17 | no |
| DistilBERT hybrid (M8, E5) | M8 minus_evidence | D0 | 1 | 52.32 | 42.95 | 24.22 | 45.16 | 26.19 | no |
| DistilBERT hybrid (M8, E5) | M8 minus_knowledge | D0 | 1 | 53.01 | 44.59 | 38.41 | 36.75 | 20.41 | no |
| DistilBERT hybrid (M8, E5) | M8 minus_struct | D0 | 1 | 54.13 | 49.02 | 48.79 | 41.23 | 30.61 | no |

## 2. Class-wise metrics (mean over seeds)

| Model | Labels | No Fit P / R / F1 | Potential Fit P / R / F1 | Good Fit P / R / F1 |
|---|---|---|---|---|
| M1 LogReg TF-IDF | D0 | 52.1 / 90.5 / 66.2 | 47.6 / 16.7 / 24.7 | 48.1 / 9.0 / 15.2 |
| M2 LinearSVC TF-IDF | D0 | 53.7 / 81.9 / 64.8 | 43.8 / 26.2 / 32.8 | 38.8 / 13.8 / 20.4 |
| M3 XGBoost (B1 features) | D0 | 60.7 / 84.5 / 70.7 | 40.2 / 36.3 / 38.1 | 45.6 / 14.2 / 21.7 |
| M6 MiniLM + LogReg | D0 | 63.4 / 70.5 / 66.7 | 41.0 / 53.4 / 46.4 | 51.9 / 24.2 / 33.0 |
| M1 LogReg TF-IDF | D1 | 52.8 / 90.7 / 66.8 | 42.2 / 19.4 / 26.6 | 54.5 / 6.2 / 11.2 |
| M2 LinearSVC TF-IDF | D1 | 53.4 / 82.4 / 64.8 | 41.2 / 26.2 / 32.0 | 35.8 / 10.0 / 15.7 |
| M3 XGBoost (B1 features) | D1 | 60.9 / 86.0 / 71.3 | 39.7 / 39.1 / 39.4 | 59.9 / 11.3 / 19.0 |
| M4 E0 | D0 | 67.5 / 80.6 / 73.4 | 38.0 / 29.7 / 33.1 | 43.7 / 36.7 / 39.3 |
| M4 E1 | D1 | 67.1 / 82.4 / 73.9 | 34.4 / 35.1 / 34.6 | 48.7 / 24.9 / 32.6 |
| M4 + evidence (E2) | D0 | 67.2 / 78.2 / 72.1 | 39.1 / 27.8 / 32.3 | 43.0 / 41.5 / 41.7 |
| M4 + evidence (E2) | D1 | 65.4 / 80.8 / 72.2 | 37.1 / 38.3 / 37.4 | 51.0 / 24.9 / 33.1 |
| M5 plain (CV, job) | D0 | 61.5 / 79.6 / 69.4 | 30.2 / 31.0 / 30.6 | 32.4 / 12.5 / 18.0 |
| M5 + Knowledge (E0) | D0 | 56.9 / 83.5 / 67.6 | 29.7 / 16.9 / 21.4 | 35.2 / 17.6 / 22.3 |
| M5 + Knowledge (E1) | D1 | 59.9 / 79.2 / 67.3 | 31.9 / 24.7 / 27.2 | 37.4 / 20.3 / 24.1 |
| M5 + Knowledge + evidence (E2) | D0 | 63.5 / 76.5 / 69.2 | 36.1 / 23.2 / 27.8 | 44.2 / 40.4 / 40.6 |
| M5 + Knowledge + evidence (E2) | D1 | 63.8 / 79.4 / 70.6 | 34.7 / 22.2 / 27.0 | 45.4 / 38.8 / 41.0 |
| M8 full | D0 | 64.8 / 79.1 / 71.2 | 32.2 / 25.2 / 28.2 | 42.2 / 32.9 / 37.0 |
| M8 minus_evidence | D0 | 61.1 / 79.6 / 69.1 | 30.6 / 26.2 / 28.2 | 45.2 / 24.2 / 31.5 |
| M8 minus_knowledge | D0 | 68.1 / 76.9 / 72.2 | 29.0 / 20.4 / 24.0 | 36.8 / 38.4 / 37.6 |
| M8 minus_struct | D0 | 70.7 / 68.7 / 69.7 | 35.0 / 30.6 / 32.7 | 41.2 / 48.8 / 44.7 |

## 3. Confusion matrices (rows = true class, columns = predicted: No / Potential / Good; mean count per seed)

**M1 LogReg TF-IDF (D0, 1 seed)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 524 | 41 | 14 |
| Potential Fit | 231 | 49 | 14 |
| Good Fit | 250 | 13 | 26 |

**M2 LinearSVC TF-IDF (D0, 1 seed)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 474 | 71 | 34 |
| Potential Fit | 188 | 77 | 29 |
| Good Fit | 221 | 28 | 40 |

**M3 XGBoost (B1 features) (D0, 8 seeds)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 489.4 | 67.2 | 22.4 |
| Potential Fit | 160.6 | 106.6 | 26.8 |
| Good Fit | 156.1 | 91.8 | 41.1 |

**M6 MiniLM + LogReg (D0, 1 seed)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 408 | 134 | 37 |
| Potential Fit | 109 | 157 | 28 |
| Good Fit | 127 | 92 | 70 |

**M1 LogReg TF-IDF (D1, 1 seed)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 525 | 48 | 6 |
| Potential Fit | 228 | 57 | 9 |
| Good Fit | 241 | 30 | 18 |

**M2 LinearSVC TF-IDF (D1, 1 seed)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 477 | 74 | 28 |
| Potential Fit | 193 | 77 | 24 |
| Good Fit | 224 | 36 | 29 |

**M3 XGBoost (B1 features) (D1, 8 seeds)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 498.0 | 69.0 | 12.0 |
| Potential Fit | 169.1 | 115.0 | 9.9 |
| Good Fit | 150.9 | 105.5 | 32.6 |

**M4 E0 (D0, 8 seeds)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 466.5 | 45.5 | 67.0 |
| Potential Fit | 140.9 | 87.2 | 65.9 |
| Good Fit | 85.4 | 97.5 | 106.1 |

**M4 E1 (D1, 8 seeds)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 476.9 | 62.6 | 39.5 |
| Potential Fit | 152.5 | 103.2 | 38.2 |
| Good Fit | 82.0 | 135.0 | 72.0 |

**M4 + evidence (E2) (D0, 8 seeds)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 452.6 | 50.5 | 75.9 |
| Potential Fit | 130.8 | 81.6 | 81.6 |
| Good Fit | 91.9 | 77.1 | 120.0 |

**M4 + evidence (E2) (D1, 8 seeds)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 467.6 | 73.2 | 38.1 |
| Potential Fit | 147.4 | 112.5 | 34.1 |
| Good Fit | 100.8 | 116.2 | 72.0 |

**M5 plain (CV, job) (D0, 1 seed)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 461 | 79 | 39 |
| Potential Fit | 167 | 91 | 36 |
| Good Fit | 122 | 131 | 36 |

**M5 + Knowledge (E0) (D0, 4 seeds)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 483.8 | 52.0 | 43.2 |
| Potential Fit | 196.8 | 49.8 | 47.5 |
| Good Fit | 171.8 | 66.2 | 51.0 |

**M5 + Knowledge (E1) (D1, 4 seeds)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 458.5 | 67.2 | 53.2 |
| Potential Fit | 177.8 | 72.8 | 43.5 |
| Good Fit | 145.2 | 85.0 | 58.8 |

**M5 + Knowledge + evidence (E2) (D0, 8 seeds)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 443.1 | 47.5 | 88.4 |
| Potential Fit | 159.9 | 68.1 | 66.0 |
| Good Fit | 97.5 | 74.8 | 116.8 |

**M5 + Knowledge + evidence (E2) (D1, 8 seeds)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 459.8 | 42.2 | 77.0 |
| Potential Fit | 167.4 | 65.2 | 61.4 |
| Good Fit | 96.0 | 81.0 | 112.0 |

**M8 full (D0, 1 seed)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 458 | 51 | 70 |
| Potential Fit | 160 | 74 | 60 |
| Good Fit | 89 | 105 | 95 |

**M8 minus_evidence (D0, 1 seed)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 461 | 68 | 50 |
| Potential Fit | 182 | 77 | 35 |
| Good Fit | 112 | 107 | 70 |

**M8 minus_knowledge (D0, 1 seed)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 445 | 37 | 97 |
| Potential Fit | 140 | 60 | 94 |
| Good Fit | 68 | 110 | 111 |

**M8 minus_struct (D0, 1 seed)**  
| true \ predicted | No Fit | Potential Fit | Good Fit |
|---|---|---|---|
| No Fit | 398 | 74 | 107 |
| Potential Fit | 110 | 90 | 94 |
| Good Fit | 55 | 93 | 141 |

## 3b. Colab GPU runs (a separate setting)

Same scripts and inputs, run on a Colab T4 GPU. GPU results can differ slightly from CPU results for the same seed, so they are shown apart and never averaged with the CPU rows.

| Model | Labels | Seeds | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall | Meets rule |
|---|---|---|---|---|---|---|---|---|
| M8 minus_struct (hybrid) | D0 | 3 (43, 44, 45) | 52.93 ± 0.82 | 42.98 ± 1.57 | 27.68 ± 7.64 | 39.28 ± 3.05 | 22.34 ± 2.32 | no |
| M8 full (hybrid) | D0 | 3 (43, 44, 45) | 53.07 ± 2.04 | 43.20 ± 3.22 | 28.14 ± 14.58 | 43.86 ± 8.37 | 23.70 ± 4.25 | no |
| M8 minus_knowledge (hybrid) | D0 | 3 (43, 44, 45) | 54.10 ± 2.94 | 45.81 ± 2.75 | 44.64 ± 4.99 | 42.99 ± 6.65 | 18.82 ± 3.41 | no |

M8 minus_evidence did not complete on the GPU (all three seeds stopped at start-up without a result); its only result is the CPU seed 42 row in section 1.

## 3c. Single runs that pass the rule, and the closest setting

Out of 91 individual runs (every seed of every model), **5 single runs pass the full selection rule** on their own, and 5 more beat the baseline on accuracy and Good Fit recall but drop more than 2 points on one guard metric. The full list is in D_experiments/ALL_RUNS_per_seed_validation.csv.

| Run | Seed | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall |
|---|---|---|---|---|---|---|
| *Baseline: BERT-Tiny + Knowledge, D0, mean of 8 seeds* | — | 56.79 | 48.61 | 36.72 | 43.68 | 29.68 |
| DistilBERT + Knowledge + evidence, D0 | 44 | 60.93 | 56.57 | 66.09 | 56.34 | 32.65 |
| BERT-Tiny + Knowledge + evidence, D0 | 46 | 59.29 | 53.58 | 50.87 | 48.04 | 33.33 |
| DistilBERT + Knowledge + evidence, D1 | 48 | 57.75 | 50.14 | 37.37 | 57.45 | 29.59 |
| BERT-Tiny + Knowledge + evidence, D0 | 47 | 57.40 | 52.52 | 66.44 | 45.07 | 27.89 |
| BERT-Tiny + Knowledge + evidence, D0 | 42 | 56.97 | 50.70 | 43.25 | 46.13 | 32.31 |

- **Why a single run is not enough.** The rule compares the mean of a setting with the baseline mean, because seeds move results a lot: 3 of the baseline's own 8 seeds also beat its mean on accuracy and Good Fit recall (seeds 43, 46, 49). Picking the best seed after seeing validation results would overstate what a setting delivers, and no unused test set is left to confirm a single run.
- **The closest setting on average is BERT-Tiny + Knowledge + evidence (D0).** It keeps every guard (Good Fit precision, macro-F1 and Potential recall all within 2 points), raises Good Fit recall by 4.80 points, and misses only on accuracy, by 0.49 points, less than the seed-to-seed SD of the baseline (1.17). Compared seed by seed, it has higher Good Fit recall on 6 of 8 seeds. Three of its eight seeds pass the rule on their own.
- **On the frozen seed itself (46),** which was chosen as the seed closest to the baseline mean and not from the evidence results, adding evidence gives accuracy 59.29 vs 56.97, Good Fit recall 50.87 vs 38.41, Good Fit precision 48.04 vs 43.87 and Potential recall 33.33 vs 27.21: better on every metric.
- **Recommendation:** keep the frozen baseline as the rule requires, and treat BERT-Tiny + Knowledge + evidence as the first candidate to confirm on new data or a fresh test set; DistilBERT + Knowledge + evidence (seed 44: Good Fit recall 66.09) is the second.

## 4. Notes

- Seeds: BERT-Tiny experiments use seeds 42–49; DistilBERT experiments use the seeds listed (CPU cost). Different seed counts are shown in the table; SD is the sample SD.
- DistilBERT runs use max length 384 (512 was projected over the CPU time budget); M8 uses 512. BERT-Tiny uses 512.
- Same CVs in TRAIN and VALIDATION: job groups are disjoint, but 1,157 of the 1,162 validation CVs also appear in TRAIN; all validation scores are probably optimistic for unseen CVs.
- A team member also ran DistilBERT on a Colab GPU on 22 Nov (lr 1e-4, 512 tokens, 3 seeds; plain: accuracy 48.48, macro-F1 36.81, Good Fit recall 22.95; + Knowledge v1: 47.90 / 39.47 / 16.61). Those runs use a different device and learning rate, so they are reported here as a separate setting and are not averaged with the CPU runs.
- **M8 more seeds, Colab GPU (separate setting, 24 Nov).** The M8 variants were re-run on a Colab GPU with the same script, seeds 43–45, D0 (mean ± SD over 3 seeds): full accuracy 53.07, Good Fit recall 28.14 ± 14.6; minus_knowledge 54.10, Good Fit recall 44.64 ± 5.0; minus_struct 52.93, Good Fit recall 27.68 ± 7.6. None reaches the baseline accuracy (56.79). The single-seed CPU result of minus_struct (seed 42, Good Fit recall 48.79) did not repeat on other seeds, so the seed-42 M8 rows above should not be read as stable.
- M4 E1 and M4 + evidence (E2) on D1 show the same mean Good Fit recall (24.91) by coincidence: the per-seed counts differ (E1: 74, 93, 59, 77, 58, 90, 52, 73; E2: 74, 80, 63, 60, 82, 91, 59, 67) but both sum to 576 of 8 × 289.
- Per-job-group error analysis of the frozen model: D_error_analysis/ (error_pretags_by_job_group.csv).
- Runs not included: a 3- vs 5-epoch test of the same DistilBERT setting on a Colab GPU (seeds 42–44) finished, but its summary table was not exported to the project drive at submission time, so it is not used for any conclusion here. Its per-epoch log for seed 44 shows the same pattern as on CPU (seed 44 strongest: up to 61.10 accuracy and 66.44 Good Fit recall).
