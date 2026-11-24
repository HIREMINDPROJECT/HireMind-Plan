# A1 — fresh CPU reproduction: result (v3, 23 Nov 2026)

**Status: VERIFIED_COMPLETE** (judged with the rules fixed in `run_commands_v2.md` before the run). One caveat is listed at the end.

## What was run
Knowledge v1 (BERT-Tiny, 512 tokens, 3 epochs, 939 updates, AdamW, lr 1e-4, batch 16, D0 labels), 8 seeds 42–49, `--device cpu --save-weights --save-preds`, on GitHub Actions `ubuntu-latest` in the project's GitHub fork (run 45236776424), code pinned to Senior-2 `ec28dfc` and HireMind-DL `1f22f34`, the same package pins as the original workflow (transformers 4.51.3, pandas 2.2.3, scikit-learn 1.6.1, huggingface-hub 0.30.2). Not re-run until it matched; this is the first and only attempt of this configuration in this workflow.

## Environment recorded (closes the open gap)
Python 3.12.16 · **torch 2.15.0+cpu · numpy 2.5.5** · transformers 4.51.3 · pandas 2.2.3 · scikit-learn 1.6.1 · huggingface-hub 0.30.2 (`fresh_cpu_run/pip_freeze.txt`, `lscpu.txt`, `python.txt`). The 17 Nov original run left torch and numpy unpinned, so its exact versions remain unknown; the fresh run is the first with versions recorded.

## Hard checks (all pass)
8 seeds present; each confusion matrix sums to 1,162 with true classes 579 / 294 / 289; logged metrics equal their recomputation from the matrix; knowledge = 1; 3 epochs, 939 updates; each checkpoint's SHA-256 recorded by the runner equals the `weights_sha256` logged by the run, for all 8 seeds (`A1_checkpoint_inventory_v3.csv`).

## Result against the reference run (20261117T122609)
| | Reference mean (SD) | Fresh run mean (SD) | Delta |
|---|---|---|---|
| Accuracy | 56.79 (1.17) | 56.79 (1.17) | 0.00 |
| Macro-F1 | 48.59 (2.80) | 48.61 (2.80) | +0.02 |
| Good Fit recall | 36.85 (11.24) | 36.72 (11.28) | −0.13 |
| Good Fit precision | 43.66 (4.64) | 43.68 (4.64) | +0.02 |
| Potential Fit recall | 29.42 (4.36) | 29.68 (4.54) | +0.26 |

Per seed: **7 of 8 identical** confusion matrices (seeds 42, 43, 44, 46, 47, 48, 49). **Seed 45 DIVERGED:** the fresh run selected step 900 instead of 939 (matrix `[[468,54,57],[128,98,68],[82,112,95]]` vs reference `[[471,48,60],[132,92,70],[82,109,98]]`). This is exactly the checkpoint-selection near-tie already documented in `A1_numerical_differences_v2.md` for run 130749, and the fresh-run matrix is identical to that run's. All differences are far below the seed-to-seed SD.

## Explanation of the remaining differences (the plan's pass condition)
1. Seed 45: validation macro-F1 at steps 900 and 939 are within 0.25 points (48.77 vs 48.55 in the reference, 48.97 vs 48.78 in the re-executions); tiny training-state differences between executions flip which step is kept. This was seen in 2 of the 3 independent re-executions that included seed 45 (and each time ends at step 900).
2. The seed-42 checkpoint file differs byte-for-byte from the earlier `final-results` checkpoint (`6de7eaaa…` vs `cdd6b385…`) although every predicted label matches the reference matrix. The two builds used different PyTorch versions and runner hardware, so equal predictions with unequal bytes is expected. Do not compare checkpoint hashes across environments; compare predictions.

## Caveat (action for the project coordinator)
The checkpoint files themselves (`weights.pt`, about 17.5 MB each) exist only as GitHub Actions artifacts of run 45236776424 and expire after 30 days (about 23 Dec 2026). Download the 8 `a1-seed-*` artifacts from the run's page and store them in the A1 Drive folder. This workspace cannot download GitHub artifacts. Hashes are already recorded here for verification.
