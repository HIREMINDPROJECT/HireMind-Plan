# HireMind — the supervisor's 19 Nov 2026 plan, requirement by requirement (v4, 24 Nov 2026)
Each line: what the plan asks → status → where. "OPEN" = not yet satisfied. Only verified evidence is listed. Replaces v3.

## Phase A
| Task | Plan asks for | Status | Evidence / gap |
|---|---|---|---|
| A1 | baseline_results.csv, environment.txt, run_commands.md; reproduce, explain differences | DONE | A1_baseline_reproduction (fresh CPU run, 8 seeds, 7/8 identical, seed 45 explained); E0 reproduced it again; 8 checkpoints kept in the weights/ folder of the GitHub repository HIREMINDPROJECT/HireMind-Plan with SHA-256 |
| A2 | data_quality_report.csv + 1–2 pages; count + example per issue; row IDs kept | DONE | A2_data_quality |
| A3 | labeling_guidelines.pdf, ≥10 annotated examples; two reviewers apply it independently | DONE | rubric v2 (12 examples); pilot R1/R2: 17/20 agree, κ 0.714, 3 disagreements; pilot accepted by the supervisor |
| A4 | reviewed_labels.csv (pair_id, original, two reviewer labels, adjudicated label, evidence span, reason, confidence, decision); agreement; confirmed errors vs flagged | DONE | reviewed_labels.csv (IDs, labels, confidence, decision); agreement 65.5%, κ 0.365; 255 confirmed errors vs 1,949 flagged (flags ≠ errors). Evidence spans and reasons for all 400 cases are in a private full version (contains CV text). Follow-up to the review: A4_A5_review_followup.md |
| A5 | train_labels_v1.csv + change_log.csv; every change traceable to review + approval; originals kept | DONE | A5_label_overlay_D1; approved by the supervisor; 255 TRAIN changes, all inside the 400 reviewed cases; D0 and splits.csv unchanged |
| A6 | D0_vs_D1.csv; identical runs; no success claim if mixed | DONE · MIXED | A6_D0_vs_D1: GF recall −11.81 (7/8 seeds lower), GF precision +4.97, Potential recall +5.44, accuracy −0.67 → not eligible |

## Phase B
| Task | Plan asks for | Status | Evidence / gap |
|---|---|---|---|
| B1 | structured_pairs.jsonl + schema.md; claims point to text; unknown ≠ negative | DONE | B1_structured_schema (0 pointer failures; data file in the project drive) |
| B2 | extraction_gold.csv + extraction_evaluation.csv; human benchmark; false positives like "no Python", unverifiable years | DONE · accepted deviation (single annotator) | P 0.51, R 0.38; years F1 0.85, dates 0.54 (format-tolerant); gold kept private (CV text). Gold has 0 explicit negatives: negation is covered only by synthetic unit tests. Follow-up: B2_review_followup.md |
| B3 | match_features + matcher; exact + sentence embeddings; valid and false semantic matches; similarity ≠ proof | DONE · accepted deviation (DistilBERT instead of all-MiniLM-L6-v2) | B3_matcher; similarity separates found/not-found poorly (AUC 0.576). Example matches need CV text, so they stay private |
| B4 | evidence_inputs + truncation_audit.csv; retained relevant evidence, not tokens saved | DONE | B4_truncation |

## Phase C/D — models and experiments (VALIDATION only; test never used for selection)
| Item | Status | Result |
|---|---|---|
| Selection rule fixed before results | DONE | beat M4 on accuracy AND GF recall; no drop > 2 points in GF precision, macro-F1, Potential recall |
| Stage 1: M1, M2, M4 | DONE | M4 = baseline (acc 56.79, GF recall 36.72, 8 seeds) |
| Stage 2: M3, M5, M6 | DONE | all not eligible; M5 = DistilBERT (team decision); M6 not part of the team decision |
| DistilBERT + Knowledge v1 through E0, E1, E2 (same inputs as BERT-Tiny) | DONE | DistilBERT + Knowledge: E0 50.30 ± 0.80 / GF recall 17.65 ± 11.31 (4 seeds); E1 50.77 ± 1.61 / 20.33 ± 16.50 (4); E2 D0 54.04 ± 3.44 / 40.40 ± 15.69 (8); E2 D1 54.82 ± 2.50 / 38.75 ± 10.81 (8). None meets the rule on average (FINAL_RESULTS_REPORT.md); best single run seed 44 on D0 (accuracy 60.93, Good Fit recall 66.09) reported but not selected: the rule compares means over seeds |
| Stage 3: M8 DistilBERT hybrid | DONE | full + 3 ablations: seed 42 on CPU, seeds 43–45 on Colab GPU (separate setting); none eligible |
| Stage 4: M7 | NOT RUN | optional; outside the DistilBERT-only decision |
| E0 | DONE | reproduced A1 exactly |
| E1 (= A6) | DONE | mixed; not eligible |
| E2 (D1 labels + evidence input) | DONE | BERT-Tiny: inside noise vs E1; not eligible. DistilBERT + Knowledge: see FINAL_RESULTS_REPORT.md |
| E3 (semantic features) | DONE | precision/recall trade-off, no overall gain |
| E4 (comparison and selection) | DONE | no setting eligible on average → M4 on D0 kept (as the plan allows); 5 single runs pass the rule and the closest setting (BERT-Tiny + Knowledge + evidence) is named as the first candidate to confirm (FINAL_RESULTS_REPORT.md 3c, ALL_RUNS_per_seed_validation.csv) |
| E5 (ablation) | DONE | M8 seed 42 (CPU) + seeds 43–45 on Colab GPU (separate setting; minus_evidence did not complete on GPU). None eligible (E5_ABLATION_REPORT.md) |
| Seeds mean ± SD; single-seed marked | DONE | seed counts shown per row; paired seed-level tests with 95% CI (E4_ROBUSTNESS_CHECKS.md) |
| Metrics incl. inference time | DONE (proxy) | BERT-Tiny in the platform: CV analysis with fit prediction p95 0.117 s per CV (PT-02); M5 DistilBERT 222 ms per pair on CPU |
| Error analysis by job group and class | DONE | rule-based pre-tags (M4, 8 seeds): 476 consistent mistakes, by class and job group; class-wise metrics and confusion matrices for every model in FINAL_RESULTS_REPORT.md |
| Fairness | NOT MEASURABLE | no consented protected attributes in the data; documented in the model card. The platform's own fairness tests (11) pass |

## Phase E
| Task | Plan asks for | Status | Evidence / gap |
|---|---|---|---|
| E1 | model_card.pdf, model_config.yaml, model_weights, reproducibility.md; another student can reproduce | DONE | E1_frozen_model: M4 BERT-Tiny + Knowledge v1, D0, seed 46 (closest to the 8-seed mean), weights SHA-256 0953ce89…cb0c8e in the weights/ folder of HIREMINDPROJECT/HireMind-Plan. Re-loaded on a fresh Colab machine: identical validation accuracy 56.97 |
| E2 | prototype + 10 documented example cases; HR can inspect/correct evidence and override | DONE | prototype v3 (HR-session fixes); 10 documented cases from real validation pairs scored by the frozen model (E2_10_example_cases.md; case files private) |
| E3 | test_report.pdf, issue_log.csv, HR feedback notes; re-run platform tests; tests for extraction, matching, long CVs, wrong parsing, negation, abstention | DONE | prototype tests v3 33/33; HR sessions 5 participants × 2 cases (confirmed by the supervisor); platform re-run: 105/105 tests, e2e 20 screens, NFR 5/5; issue_log_v3.csv: 10 fixed or closed, 1 partial (audit records are stored by the platform, not the prototype); ready patch makes the platform serve the frozen model (106/106 tests, identical validation result); every test type the plan names is mapped to its evidence in E3_TEST_COVERAGE_vs_PLAN.md |

## Final submission checklist (plan page 7)
| Item | Status |
|---|---|
| Final report | DONE (FINAL_REPORT.pdf in the package root; all numbers in 00_overview/FINAL_RESULTS_REPORT.md) |
| Code, run outputs, weights | DONE (GitHub repository HIREMINDPROJECT/HireMind-Plan; HireMind_code_bundle.zip) |
| One diagram of the final pipeline | DONE (HireMind_pipeline_diagram.png/.pdf) |
| Data audit, rubric, approved label overlay, reviewer agreement | DONE |
| ≥ M1, M2, M4 + two stronger alternatives; explain untrained models | DONE (M3, M5, M6, M8; M7 explained) |
| Ablation table: label cleaning, evidence selection, semantic matching | DONE (ABLATION_TABLE_three_factors.md) |
| Class-wise metrics, per-job-group error analysis, confusion matrices, uncertainty | DONE (FINAL_RESULTS_REPORT.md, D_error_analysis, E4_ROBUSTNESS_CHECKS.md) |
| Reproducible code, fixed manifests, model card, limitations, HR safeguards | DONE (code bundle, model card, reproducibility.md) |

## For the team
1. Apply the ready platform patch (E3_testing/platform_tests_M4) in the team repository; it makes the platform serve the frozen model (106/106 tests).
