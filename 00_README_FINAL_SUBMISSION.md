# HireMind — final submission package (24 Nov 2026)
Built from verified results only. No personal names, no CV or job text. Working files and full history stay in the phase folders of the project drive.

**Start here: `FINAL_REPORT.pdf`** (8 pages: the question, what was done in each phase, results, limitations). Every number in it comes from `00_overview/FINAL_RESULTS_REPORT.md`.

**Code, run outputs and model weights:** GitHub repository **HIREMINDPROJECT/HireMind-Plan** (https://github.com/HIREMINDPROJECT/HireMind-Plan). It holds this whole package plus the scripts and workflows, the files every GitHub Actions run produced, and the 8 checkpoints of the frozen model with SHA-256.

| Folder | Plan item | Main file |
|---|---|---|
| (root) | final report | FINAL_REPORT.pdf, FINAL_REPORT.md |
| 00_overview | pipeline diagram, all results, ablation answer, requirement checklist | FINAL_RESULTS_REPORT.md, HireMind_pipeline_diagram.pdf, ABLATION_TABLE_three_factors.md, PLAN_COMPLIANCE_CHECKLIST_v4.md |
| A1_baseline_reproduction | A1 | A1_fresh_cpu_run_result_v3.md |
| A2_data_quality | A2 | data_quality_report_v2.csv, A2_data_quality_findings_v3.md |
| A3_label_rubric | A3 | labeling_guidelines_v2.pdf |
| A4_review | A4 | reviewed_labels.csv, A4_review_summary.json |
| A5_label_overlay_D1 | A5 | train_labels_v1.csv, change_log_ids_only.csv |
| A6_D0_vs_D1 | A6 / E1 | D0_vs_D1.csv, A6_E1_REPORT.md |
| B1–B4 | structured evidence | schema.md, B2_REPORT.md, B3_REPORT_v3.1.md, B4_REPORT.md |
| C_models | M1–M6 | C_stage1_report.md, C_stage2_report.md, C_M5_REPORT.md |
| D_experiments | E2–E5 | E4_REPORT.md, E4_ROBUSTNESS_CHECKS.md (paired seed tests, ensembles, threshold), E5_ABLATION_REPORT.md, Colab GPU results (M8_colab_gpu_seeds43-45.csv) |
| D_error_analysis | error analysis | D_error_analysis_report.md |
| E1_frozen_model | E1 | model_card.pdf, model_config.yaml, reproducibility.md |
| E2_HR_prototype | E2 | hr_review_prototype.html (v3, open in a browser), E2_10_example_cases.md (10 real validation cases) |
| E3_testing | E3 | E3_TEST_COVERAGE_vs_PLAN.md (every test the plan asks for), test_report.pdf, issue_log_v3.csv, E3_HR_feedback_section.md, platform_tests/, platform_tests_M4/ (the platform serving the frozen model) |
| colab_notebooks | Colab runs | the notebooks used on Colab (frozen-model re-load, 10 cases, review sheets, DistilBERT, epochs and M8 on GPU) |
| HireMind_code_bundle.zip | all code and workflows | README_CODE_BUNDLE.md inside |

## Result in one paragraph
No model beat the BERT-Tiny + Knowledge v1 baseline (M4: accuracy 56.79, Good Fit recall 36.72, 8 seeds) on both accuracy and Good Fit recall under the selection rule fixed before the experiments, so M4 trained on the original labels is frozen (seed 46, the seed closest to the 8-seed mean). The main cause of weak Good Fit detection is that the model does not see the evidence: 99.6% of pairs exceed the input limit and truncation keeps about a third of the relevant evidence. Evidence selection is the strongest factor (it raises Good Fit recall for both BERT-Tiny and DistilBERT); the reviewed labels (D1) move the class boundary (Good Fit recall −11.8, precision +5.0) rather than improve the model; a larger classifier alone does not help. DistilBERT + Knowledge v1 was taken through the same plan steps for a fair comparison. The test set was not used for any selection.

## Testing in one paragraph
The HR review prototype passes 33 of 33 browser tests (v3). Five HR participants each reviewed two cases; their four main findings are fixed in v3. The platform's own tests were re-run without changing its code: 105 of 105 tests, browser test on 20 screens, 5 of 5 non-functional tests on PostgreSQL. A ready patch makes the platform serve the frozen model: it reproduces the frozen validation result exactly (56.97) and passes 106 of 106 tests. What is still open with people is listed at the end of PLAN_COMPLIANCE_CHECKLIST_v4.md.
