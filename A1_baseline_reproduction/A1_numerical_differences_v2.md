# A1 — Reproducibility evidence and explanation of numerical differences (v2)

**Status:** VERIFIED from archived original runs · no new training was performed for this document
**Date:** 23 Nov 2026 · **Prepared for:** the supervisor's plan, Task A1 pass condition
**Builds on (kept unchanged):** `baseline_results.csv`, `baseline_summary_mean_sd.csv`, `checkpoint_inventory.csv`,
`environment.txt`, `run_commands.md`, `A1_ACCEPTANCE_AUDIT_AR.md` (project coordinator, 23 Nov)

## 1. What the pass condition asks
> "The team can reproduce the same evaluation setup and explain any remaining numerical differences."

## 2. The same setup was executed independently several times
The Knowledge v1 arm (BERT-Tiny, 512 tokens, 3 epochs, 939 updates, AdamW, lr 1e-4, batch 16, D0 labels,
frozen 1,162-row validation set) was run by GitHub Actions on separate CPU runners in five workflow runs on
17 Nov 2026, plus one separate run on the `final-results` branch that saved the seed-42 checkpoint.

| Run | Code commit (Senior-2 `claude/knowledge`) | Seeds | Result vs reference run |
|---|---|---|---|
| 20261117T105915 | 295afad | 42–44 | 3/3 identical |
| 20261117T120019 | 6d99d32 | 42–44 | 3/3 identical |
| **20261117T122609 (reference)** | **a171ea5** | **42–49** | — |
| 20261117T124714 | ec28dfc | 42–49 | 7 identical, 1 float-noise (seed 47) |
| 20261117T130749 | 3043f95 | 42–49 | 7 identical, 1 diverged (seed 45) |
| final-results | 1e2cc5e | 42 | identical; checkpoint SHA-256 verified |

Each run was mapped to its commit by matching the run title with `experiment.json` at that commit, not by timestamp.
Between these commits the Knowledge v1 code path is unchanged: `knowledge.py` only gained a branch behind
`HIREMIND_KNOWLEDGE=v2`, and `train_knowledge.py` only gained output options (`--save-preds`, `--vocab`).

**Totals:** 31 executions of the identical configuration; 29 produce bit-identical confusion matrices.
Full table: `A1_cross_run_reproducibility_v2.csv`.

## 3. Per-pair check
For seed 42, two independent executions (`final-results` and run 124714) saved per-pair validation predictions.
All 1,162 predicted labels are identical; the largest probability difference across all three classes is 0.0011.
Checkpoint `weights.pt` SHA-256 `cdd6b38503d708ecb9acd8a4be387893367013675e445b1693c56f9a6ada4d8b`
equals the value logged by that run.

## 4. The two differences, explained
**Seed 47, run 124714 — FLOAT_NOISE.** One Potential Fit pair moved to No Fit (matrix row 2: 127/89 → 128/88).
Same best step (700). Accuracy −0.09 pts; Good Fit metrics unchanged. Cause: run-to-run floating-point
differences on different CPU runners (BLAS/oneDNN kernel choice, thread scheduling), shown in §3 to be of order 1e-3 in probability;
a pair sitting on a decision boundary can flip.

**Seed 45, run 130749 — DIVERGED (checkpoint selection).** Training drifted slightly (validation loss at step 939:
0.90502 vs 0.90279). The training code keeps the checkpoint with the best validation macro-F1 among steps
100…939. In the reference run step 939 won (48.774 vs 48.548 at step 900); in this run step 900 won
(48.972 vs 48.781). A different checkpoint was therefore selected: 15 cases moved, accuracy unchanged
(56.88%), Good Fit recall −1.04 pts, macro-F1 +0.20 pts. Effect on the 8-seed means: accuracy 0.00, Good Fit recall −0.13 pts.

**Size relative to seed variance.** Both effects are far below the seed-to-seed spread of the reference
(sample SD: accuracy 1.17 pts, Good Fit recall 11.24 pts, macro-F1 2.80 pts).

## 5. Differences that are NOT reproduction noise (must not be mixed with this baseline)
| Source | Why it differs | Treatment |
|---|---|---|
| GPU runs (`device: cuda`, Mervat 20–21 Nov) | Different kernels. Same seed, no-knowledge control, seed 42: CPU Good Fit recall 21.8% vs GPU 43.6% | Never compare a GPU run with this CPU baseline as "reproduction"; device is a controlled factor in A6/Phase D |
| Mervat M4 `control` arm (`knowledge: 0`) | No knowledge line — a different model | Not Knowledge v1; not an A1 checkpoint |
| Mervat M4 `evidence` arm | Different feature extractor (`evidence_features`) | Separate experiment (M4-3S-EV) |
| `knowledge_6ep` (57.57%) | 6 epochs, not 3 | Historical alternative, not the baseline |
| Historical test numbers (49.29%) | Test set, already opened | Reference only; not used in A1 |
| Older 256-token M0 results | max_len 256 | Not this configuration |

## 6. Methodological note to report
The best checkpoint is selected on the same validation set that is reported. Validation numbers are therefore
slightly optimistic and selection is sensitive to near-ties (seed 45). This applies equally to D0 and D1 in A6
provided the same rule is used; it should be stated in the report, not changed silently.

## 7. Remaining gaps (not fabricated)
1. Checkpoints for seeds 43–49 were never saved by the original workflow.
2. Exact torch and numpy versions of the 17 Nov runners were not recorded (installed unpinned).

Both are closed only by one fresh CPU execution that saves them: see `A1_fresh_cpu_run_v2.ipynb`
and `run_commands_v2.md`. Its outcome is judged with `A1_compare_runs.py` against the reference run above.
