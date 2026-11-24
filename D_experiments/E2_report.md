# Phase D, experiment E2 — does preserving relevant evidence help? (PROVISIONAL, D0 labels)

**Status update (24 Nov 2026):** E2 on D1 (the plan's version) is done: E2_D1_REPORT.md. Both are summarised with paired tests in E4_ROBUSTNESS_CHECKS.md.


**Status:** VERIFIED run, validation only, original labels (D0). The plan wants E2 on D1; that run came later (E2_D1_REPORT.md). The test set was not read.
**One change vs the reference (E0):** the (CV, job) text is replaced by the B4 evidence-aware selection (`lex_k`, extractor v3.1), then the Knowledge v1 line is prepended exactly as in M4. Model, code, hyperparameters, seeds 42–49, 512 tokens, CPU: all the same. Run: GitHub Actions 45252516209, branch `e2-evidence` (results branch `e2-results-45252516209`). A first E2 run used extractor v3.0 and is discarded (B4 selections differed in 97% of pairs).
**Reference E0:** the pinned M4 re-run (identical to the A1 run on all 8 seeds, metric for metric).

| 8-seed mean (SD) | E0: M4 baseline | E2: M4 + evidence-aware input | Difference |
|---|---|---|---|
| Accuracy | 56.79 (1.17) | 56.30 (1.55) | −0.49 |
| Good Fit recall | 36.72 (11.28) | 41.52 (12.39) | +4.80 |
| Good Fit precision | 43.68 (4.64) | 42.99 (3.98) | −0.69 |
| Macro-F1 | 48.61 (2.80) | 48.69 (3.14) | +0.08 |
| Potential Fit recall | 29.68 (4.54) | 27.76 (5.51) | −1.92 |

Paired by seed: accuracy changes by −0.48 on average (SD 2.08, standard error 0.73; E2 is better on 3 of 8 seeds). Good Fit recall changes by +4.80 (SD 16.25, standard error 5.74; better on 6 of 8 seeds). Per-seed values are in `E2_per_seed_vs_E0.csv`.

## Reading
- **No reliable improvement.** Accuracy is slightly lower, Macro-F1 is unchanged, and the Good Fit recall gain (+4.8) is smaller than one seed SD (about 11–12) and below one standard error of the paired difference (5.7). I do not claim it.
- **By the preset selection rule** (accuracy AND Good Fit recall must both beat the M4 mean), E2 is **not eligible**, because accuracy does not improve.
- B4 keeps about 80% of the relevant CV evidence instead of about 33%, yet the classifier does not improve in any way I can separate from seed noise. So losing evidence to truncation is not shown to be what limits this model. Candidate reasons that I have NOT tested: BERT-Tiny is too small to use the extra evidence; label noise (D0) caps accuracy; the extractor's idea of "relevant" is imperfect (B2 not measured); 3 epochs on 4,993 pairs is little.
- Good Fit recall varies a lot from seed to seed (26 to 66 in E2, 13 to 49 in E0), so any single-seed comparison would mislead. Use means with the SD.
- One further check that would make the comparison fairer, and that I have not run: more seeds, or E2 on D1.

## Files
`E2_per_seed_vs_E0.csv`, `E2_summary.json`, `val_predictions_E2_s42..49.csv` (row IDs, labels and probabilities only), `e2_train_evidence.py`, `pip_freeze.txt`.
