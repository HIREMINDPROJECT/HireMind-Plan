# M5 — DistilBERT (team's chosen encoder), D0, validation only   STATUS: DONE (1 seed) · NOT ELIGIBLE

**Status update (24 Nov 2026):** M8 finished (E5_ABLATION_REPORT.md). DistilBERT was also run with the Knowledge v1 line through E0, E1 and E2 on several seeds; the combined results are in FINAL_RESULTS_REPORT.md.


## Setup (run 45249222375, branch c-m5, CPU)
distilbert-base-uncased @12040acc, 3-class head, plain (CV, job) pair (no Knowledge v1 line), AdamW lr 3e-5, batch 16 (micro-batches of 4), 6% warm-up, 3 epochs = 939 updates, best-validation-macro-F1 epoch (as for M4; optimistic). Seed 42 only. **Max length 384**, chosen by the fixed pre-run rule (512 was projected at 5.39 h, over the 4.4 h CPU budget). Run time 3.8 h; not stopped early. Test set not read. Labels D0.

## Result (best epoch = 3)
| | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall |
|---|---|---|---|---|---|
| **M5 DistilBERT (seed 42)** | **50.60** | **39.32** | **12.46** | 32.43 | 30.95 |
| M4 BERT-Tiny + Knowledge v1 (mean of 8 seeds) | 56.79 | 48.59 | 36.85 | 43.66 | 29.42 |
| M4 range over its 8 seeds | 54.39–58.35 | 42.04–50.67 | 13.49–49.13 | | |
Epochs: 1 → acc 48.54 / macro-F1 28.02; 2 → 50.34 / 36.88; 3 → 50.60 / 39.32. Inference 222 ms per pair on CPU (M4 is far faster).
Confusion (rows true No/Potential/Good, cols predicted): [[461,79,39],[167,91,36],[122,131,36]].

## Preset selection rule
Must beat the M4 mean on accuracy AND Good Fit recall, with no drop > 2 points in GF precision, macro-F1 or Potential recall.
- Accuracy 50.60 < 56.79 and Good Fit recall 12.46 < 36.85 → **NOT ELIGIBLE**. M5 is also below M4's worst seed on accuracy, macro-F1 and Good Fit recall.

## Limits (why this is not the last word on DistilBERT)
- One seed only (8 seeds would take ~30 CPU hours). Seed spread for M4 is large (GF recall 13–49).
- 384 tokens instead of 512 (CPU budget); no Knowledge v1 line; learning rate not tuned (the plan fixes no tuning on test, and none was done on validation either).
- Validation loss was not logged, and the logged training loss is scaled for gradient accumulation, so no over-fitting claim is made. Validation accuracy rose only from 48.5 to 50.6 over the 3 epochs.
- M8 (DistilBERT + evidence + Knowledge + structured features) has finished; see E5_ABLATION_REPORT.md.

## Files
C_stage2_m5_validation.csv, C_stage2_M5_epoch_log.csv, C_stage2_M5_validation_predictions.csv (IDs/labels only), C_stage2_M5_config.json, run_log_tail.txt, pip_freeze.txt, c_stage2_m5_distilbert.py. Weights: Actions artifact m5-distilbert-s42 of run 45249222375 (30-day retention).
