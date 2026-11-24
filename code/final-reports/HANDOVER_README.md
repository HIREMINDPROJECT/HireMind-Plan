# HireMind — supervisor's improvement plan (19 Nov 2026): complete work and evidence

This repository holds everything produced for the supervisor's Student Improvement Plan: the final report, every phase deliverable (A1–E3), the code and workflows that produced each number, the result files of every finished run, the frozen model weights, and the Colab notebooks.

It builds on the team repositories, which are not changed: `Hams770/HireMind-DL` (training code, pinned at commit `1f22f34`) and `Hams770/Senior-2` (platform, pinned at `ec28dfc`).

**Start with `FINAL_REPORT.pdf`.** Every number in it comes from `00_overview/FINAL_RESULTS_REPORT.md`.

## Rules followed everywhere
- **Validation only for every choice.** The test set was never used to select a model.
- **The selection rule was fixed before the experiments.** A model must beat M4 (BERT-Tiny + Knowledge v1) on accuracy **and** Good Fit recall, with no drop of more than 2 points in Good Fit precision, macro-F1 or Potential Fit recall.
- **Original data is not changed.** D0 labels and the frozen split (SHA-256 `2ff3926c…539f9b`) are kept; D1 is a separate overlay for TRAIN only.
- **No CV or job text in this repository.** Files that contain such text (reviewer sheets, case files) are kept privately.
- **Mean ± SD over seeds.** The number of seeds is shown next to every result.

## Where each part of the plan is

| Plan item | Folder | Main file |
|---|---|---|
| Final report | (root) | `FINAL_REPORT.pdf` |
| All results, diagram, checklist | `00_overview` | `FINAL_RESULTS_REPORT.md`, `HireMind_pipeline_diagram.pdf`, `PLAN_COMPLIANCE_CHECKLIST_v4.md`, `ABLATION_TABLE_three_factors.md` |
| A1 Reproduce the baseline | `A1_baseline_reproduction` | `A1_fresh_cpu_run_result_v3.md` |
| A2 Data quality | `A2_data_quality` | `A2_data_quality_findings_v3.md` |
| A3 Labeling rubric | `A3_label_rubric` | `labeling_guidelines_v2.pdf` |
| A4 Two-reviewer label review | `A4_review` | `reviewed_labels.csv` |
| A5 Label overlay D1 | `A5_label_overlay_D1` | `train_labels_v1.csv`, `change_log_ids_only.csv` |
| A6 D0 vs D1 | `A6_D0_vs_D1` | `A6_E1_REPORT.md` |
| B1–B4 Structured evidence | `B1_structured_schema` … `B4_truncation` | `schema.md`, `B2_REPORT.md`, `B3_REPORT_v3.1.md`, `B4_REPORT.md` |
| C Models M1–M8 | `C_models` | `C_stage1_report.md`, `C_stage2_report.md`, `C_M5_REPORT.md` |
| D Experiments E0–E5, robustness | `D_experiments` | `E4_REPORT.md`, `E4_ROBUSTNESS_CHECKS.md`, `E5_ABLATION_REPORT.md` |
| D Error analysis | `D_error_analysis` | `D_error_analysis_report.md` |
| E1 Frozen model | `E1_frozen_model` + `weights/` | `model_card.pdf`, `reproducibility.md` |
| E2 HR prototype + 10 cases | `E2_HR_prototype` | `hr_review_prototype.html`, `E2_10_example_cases.md` |
| E3 Testing | `E3_testing` | `test_report.pdf`, `issue_log_v3.csv`, `platform_tests_M4/` |

## Code, runs and weights
- **`code/`** holds the scripts and the GitHub Actions workflows, one folder per plan step. The workflows are kept under `code/` so that nothing runs automatically in this repository.
- **`results/`** holds the files committed by every finished GitHub Actions run (metrics, per-seed results, confusion matrices, epoch logs, package versions). Each folder name contains the run ID. `results/RUNS_INDEX.md` lists the runs included and the failed or cancelled ones left out.
- **`weights/`** holds the 8 M4 checkpoints (BERT-Tiny + Knowledge v1, seeds 42–49) with `SHA256SUMS.txt`. The frozen model is seed 46, SHA-256 `0953ce892ee0f075d37c1e6b4ff0bceb02a7932fddbca0ee5954f5e17fcb0c8e`.
- **Branch and run names in the reports.** The runs were made with GitHub Actions in the project's working copy of this repository. Reports name the branch and the run ID: the code of a branch is in `code/<branch>/`, and the files a run produced are in `results/<results branch>/`. Addresses in logs and workflows are written as this repository.
- **`colab_notebooks/`** holds the notebooks used on Colab: the frozen-model re-load check, the 10 example cases, the human-review sheets, and the DistilBERT, epochs and M8 runs on GPU. They fetch code and weights from this repository, so the repository must be public (or the user signed in) for them to run.

## How to check the frozen model in 5 minutes
1. Open `colab_notebooks/HireMind_human_review_sheets_Colab.ipynb` in Colab (CPU is enough).
2. Run the first three cells.
3. The notebook downloads the seed-46 weights from `weights/`, checks the SHA-256, and re-scores validation. It must print **accuracy 56.97**.

## Putting the frozen model into the platform
```
cd Senior-2 && git apply E3_testing/platform_tests_M4/m4_platform_final.patch && bash backend/models/get_m4_weights.sh
cd backend && pytest -q        # 106 passed (with requirements-model.txt installed)
```

## Still open (results added when they come back)
1. **Blind review of 150 validation pairs** by two independent reviewers. Sheets are prepared privately.
2. **Human tags on 50 mistakes** of the frozen model.
3. **Test set:** a question to the supervisor on whether the frozen model may be scored once on the previously opened test set.
