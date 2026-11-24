# HireMind — Improving Good Fit detection for CV–job matching: final report on the supervisor's improvement plan

Senior project, Effat University · 24 November 2026 · All results on the frozen validation split; the test set was not used for any choice.

## 1. Summary
The plan asked which of three factors (labels, evidence representation, classifier) explains the weak Good Fit detection, and whether a model can beat the baseline (BERT-Tiny + Knowledge v1, 8 seeds: accuracy 56.79 ± 1.17, Good Fit recall 36.72 ± 11.28) on accuracy and Good Fit recall together.

- **Main cause found: the model does not see the evidence.** 99.6% of CV–job pairs exceed the input limit and plain truncation keeps only about a third of the relevant evidence. Evidence-aware selection keeps about 80–90%.
- **Evidence representation is the strongest factor.** It raises Good Fit recall for both encoders (BERT-Tiny 41.52 ± 12.39 vs 36.72 ± 11.28; DistilBERT + Knowledge 41.23 ± 16.76 with evidence vs 17.65 ± 11.31 without).
- **Labels shift the class boundary rather than improve the model.** The reviewed labels (D1, 255 approved corrections) lower BERT-Tiny Good Fit recall by about 12 points (statistically clear) while raising Good Fit precision and Potential Fit recall.
- **The classifier alone does not help.** DistilBERT without evidence is worse than BERT-Tiny and is unstable across seeds.
- **No model meets the preset rule; the frozen baseline stays.** The highest mean Good Fit recall with at least 4 seeds is M4 + evidence (E2) (D0): 41.52 ± 12.39, at accuracy 56.30 ± 1.55.
- The frozen model is served by the platform through a tested patch, and HR stays in control: advice only, evidence shown for every requirement, and a review flag that catches 54% of Good Fit pairs.

## 2. Problem and starting point
HireMind suggests, for a CV and a job description, whether the candidate is a **Good Fit**, **Potential Fit** or **No Fit**, and shows HR the evidence. The supervisor's plan (19 Nov 2026) set the research target: **better Good Fit recall and overall accuracy together, without losing Good Fit precision, macro-F1 or Potential Fit recall**, and an honest answer to which factor explains and improves the Good Fit problem: **labels, evidence representation, or classifier**.

- **Data:** cnamuangtoun/resume-job-description-fit @08978e21; frozen split TRAIN 4,993 / VALIDATION 1,162 / TEST 1,759 (split file SHA-256 2ff3926c…539f9b). Job groups are disjoint across splits.
- **Starting model (M4):** BERT-Tiny with the Knowledge v1 line (a short summary of which required skills the CV shows and the years on each side) in front of the job text, 512 tokens. Mean of 8 seeds: accuracy 56.79, macro-F1 48.61, Good Fit recall 36.72, Good Fit precision 43.68.
- **Selection rule, fixed before any experiment:** a model is better only if it beats M4 on accuracy **and** Good Fit recall, with no drop of more than 2.0 points in Good Fit precision, macro-F1 or Potential Fit recall.

![HireMind pipeline](HireMind_pipeline_diagram.png)

## 3. Phase A — data first
| Step | What was done | Result |
|---|---|---|
| A1 Reproduce | Fresh CPU re-run of the documented M4 training, 8 seeds, pinned code and packages | 7 of 8 seeds identical to the team report; seed 45 differs for a documented reason. Checkpoints kept with SHA-256 |
| A2 Data quality | Counts and examples per issue, TRAIN only, row IDs kept | 99.6% of pairs exceed 512 tokens; 551 of 636 TRAIN CV IDs also occur in other splits; 50 near-duplicate CV pairs; 30 single-class job groups; no empty texts or conflicting duplicates |
| A3 Rubric | Labeling guidelines v2 with decision rules and 12 worked examples; two-reviewer pilot | Pilot 17/20 agreement, κ 0.714; approved |
| A4 Review | 400 TRAIN cases (random, Cleanlab-flagged, boundary), two independent reviewers, adjudication | Agreement 65.5%, κ 0.365; **255 confirmed label errors** (vs 1,949 automatic flags, which are not errors) |
| A5 Overlay | D1 = D0 with the 255 approved changes, TRAIN only; originals and split unchanged | Approved by the supervisor |
| A6 D0 vs D1 | Identical M4 runs on D0 and D1, 8 seeds | Mixed: Good Fit recall −11.8, Good Fit precision +5.0, Potential recall +5.4 → not a success |

## 4. Phase B — what the model sees
| Step | What was done | Result |
|---|---|---|
| B1 Structure | Extractor v3.1 turns each pair into requirements and CV claims, each pointing to its source characters; "not stated" kept apart from "explicitly not" | All 4,993 TRAIN pairs; 0 pointer failures |
| B2 Benchmark | 30 hand-annotated pairs (single annotator, accepted deviation) | Precision 0.51, recall 0.38; years F1 0.85, dates 0.54 (format-tolerant); main cause: skill-vocabulary gaps |
| B3 Matching | Exact + DistilBERT sentence similarity per requirement (DistilBERT instead of MiniLM, accepted deviation) | Similarity separates found from not-found poorly (AUC 0.58): **similarity is not proof** |
| B4 Truncation | Keep the sentences with requirements and evidence instead of cutting the tail | Relevant CV evidence kept 33% → 80%; job requirements kept 34% → 90%, same 512-token budget |

## 5. Phase C/D — models and controlled experiments
Models trained in the plan's stages: M1 logistic regression and M2 linear SVM (TF-IDF), M3 XGBoost on structured features, M4 BERT-Tiny + Knowledge v1, M5 DistilBERT, M6 MiniLM + logistic regression, M8 DistilBERT hybrid (evidence + Knowledge + structured features) with its ablations. M7 (DeBERTa) was optional and not run on CPU. DistilBERT was additionally taken through the same plan steps as BERT-Tiny (E0, E1, E2), with the Knowledge v1 line, so that the two encoders are compared on identical inputs.

All numbers on validation; mean ± SD over seeds (one seed has no SD). Full class-wise metrics and confusion matrices are in FINAL_RESULTS_REPORT.md.

**Classical, structured and semantic baselines**

| Model | Labels | Seeds | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall | Meets rule |
|---|---|---|---|---|---|---|---|---|
| M1 LogReg TF-IDF | D0 | 1 | 51.55 | 35.34 | 9.00 | 48.15 | 16.67 | no |
| M2 LinearSVC TF-IDF | D0 | 1 | 50.86 | 39.34 | 13.84 | 38.83 | 26.19 | no |
| M3 XGBoost (B1 features) | D0 | 8 | 54.83 ± 0.78 | 43.49 ± 1.33 | 14.23 ± 2.03 | 45.61 ± 5.31 | 36.27 ± 0.96 | no |
| M6 MiniLM + LogReg | D0 | 1 | 54.65 | 48.71 | 24.22 | 51.85 | 53.40 | no |
| M1 LogReg TF-IDF | D1 | 1 | 51.64 | 34.83 | 6.23 | 54.55 | 19.39 | no |
| M2 LinearSVC TF-IDF | D1 | 1 | 50.17 | 37.49 | 10.03 | 35.80 | 26.19 | no |
| M3 XGBoost (B1 features) | D1 | 8 | 55.56 ± 0.47 | 43.23 ± 0.73 | 11.29 ± 1.12 | 59.94 ± 3.56 | 39.12 ± 1.42 | no |

**BERT-Tiny + Knowledge v1 (the plan's main line)**

| Model | Labels | Seeds | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall | Meets rule |
|---|---|---|---|---|---|---|---|---|
| M4 E0 | D0 | 8 | 56.79 ± 1.17 | 48.61 ± 2.80 | 36.72 ± 11.28 | 43.68 ± 4.64 | 29.68 ± 4.54 | baseline |
| M4 E1 | D1 | 8 | 56.12 ± 0.86 | 47.04 ± 1.63 | 24.91 ± 5.17 | 48.65 ± 5.54 | 35.12 ± 4.46 | no |
| M4 + evidence (E2) | D0 | 8 | 56.30 ± 1.55 | 48.69 ± 3.14 | 41.52 ± 12.39 | 42.99 ± 3.98 | 27.76 ± 5.51 | no |
| M4 + evidence (E2) | D1 | 8 | 56.12 ± 1.13 | 47.58 ± 1.83 | 24.91 ± 4.03 | 50.96 ± 6.50 | 38.27 ± 8.27 | no |

**DistilBERT (M5 plain, M5 + Knowledge v1 through the plan, M8 hybrid)**

| Model | Labels | Seeds | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall | Meets rule |
|---|---|---|---|---|---|---|---|---|
| M5 plain (CV, job) | D0 | 1 | 50.60 | 39.32 | 12.46 | 32.43 | 30.95 | no |
| M5 + Knowledge (E0) | D0 | 4 | 50.30 ± 0.80 | 37.10 ± 3.20 | 17.65 ± 11.31 | 35.25 ± 8.13 | 16.92 ± 2.94 | no |
| M5 + Knowledge (E1) | D1 | 4 | 50.77 ± 1.61 | 39.54 ± 6.43 | 20.33 ± 16.50 | 37.45 ± 6.16 | 24.74 ± 10.81 | no |
| M5 + Knowledge + evidence (E2) | D0 | 7 | 53.80 ± 3.64 | 45.58 ± 5.31 | 41.23 ± 16.76 | 43.39 ± 9.21 | 22.25 ± 6.05 | no |
| M5 + Knowledge + evidence (E2) | D1 | 7 | 54.68 ± 2.67 | 46.18 ± 4.22 | 39.05 ± 11.64 | 44.24 ± 7.68 | 22.59 ± 4.57 | no |
| M8 full | D0 | 1 | 53.96 | 45.48 | 32.87 | 42.22 | 25.17 | no |
| M8 minus_evidence | D0 | 1 | 52.32 | 42.95 | 24.22 | 45.16 | 26.19 | no |
| M8 minus_knowledge | D0 | 1 | 53.01 | 44.59 | 38.41 | 36.75 | 20.41 | no |
| M8 minus_struct | D0 | 1 | 54.13 | 49.02 | 48.79 | 41.23 | 30.61 | no |


**Robustness (BERT-Tiny, paired by seed, E4_ROBUSTNESS_CHECKS.md).** The label change D1 has a statistically clear effect (Good Fit recall −11.8, 95% CI [−19.3, −4.3], 1 of 8 seeds better) and moves the balance between classes rather than improving the model. Evidence selection raises Good Fit recall on average (+4.8, 6 of 8 seeds) but the interval includes zero. Ensembling the seeds and tuning a Good Fit threshold (cross-validated by job group) do not turn any candidate into a win.

**Error analysis (M4, 8 seeds).** 476 of 1,162 validation pairs are wrong in most seeds. Potential Fit (70%) and Good Fit (62%) are missed most; mistakes are somewhat more often pairs with ambiguous requirements (+10.5 points). Five job groups hold 30% of the consistent mistakes.

### DistilBERT + Knowledge v1 taken through the plan's steps
To answer the plan's classifier question fairly, DistilBERT received exactly the BERT-Tiny inputs (the Knowledge v1 line, and in E2 the B4 evidence selection) and went through the same steps, on CPU with the same seeds where finished (384 tokens).

| Experiment (validation) | Seeds | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall |
|---|---|---|---|---|---|---|
| BERT-Tiny + Knowledge, D0 (E0, frozen baseline) | 8 | 56.79 ± 1.17 | 48.61 ± 2.80 | 36.72 ± 11.28 | 43.68 ± 4.64 | 29.68 ± 4.54 |
| BERT-Tiny + Knowledge, D1 (E1) | 8 | 56.12 ± 0.86 | 47.04 ± 1.63 | 24.91 ± 5.17 | 48.65 ± 5.54 | 35.12 ± 4.46 |
| BERT-Tiny + Knowledge + evidence, D0 (E2) | 8 | 56.30 ± 1.55 | 48.69 ± 3.14 | 41.52 ± 12.39 | 42.99 ± 3.98 | 27.76 ± 5.51 |
| BERT-Tiny + Knowledge + evidence, D1 (E2) | 8 | 56.12 ± 1.13 | 47.58 ± 1.83 | 24.91 ± 4.03 | 50.96 ± 6.50 | 38.27 ± 8.27 |
| DistilBERT + Knowledge, D0 (E0) | 4 | 50.30 ± 0.80 | 37.10 ± 3.20 | 17.65 ± 11.31 | 35.25 ± 8.13 | 16.92 ± 2.94 |
| DistilBERT + Knowledge, D1 (E1) | 4 | 50.77 ± 1.61 | 39.54 ± 6.43 | 20.33 ± 16.50 | 37.45 ± 6.16 | 24.74 ± 10.81 |
| DistilBERT + Knowledge + evidence, D0 (E2) | 7 | 53.80 ± 3.64 | 45.58 ± 5.31 | 41.23 ± 16.76 | 43.39 ± 9.21 | 22.25 ± 6.05 |
| DistilBERT + Knowledge + evidence, D1 (E2) | 7 | 54.68 ± 2.67 | 46.18 ± 4.22 | 39.05 ± 11.64 | 44.24 ± 7.68 | 22.59 ± 4.57 |

- **Evidence is what DistilBERT needs.** Without evidence selection DistilBERT + Knowledge stays far below BERT-Tiny (Good Fit recall 17.65 ± 11.31 on D0); with it, Good Fit recall rises to 41.23 ± 16.76 (D0) and 39.05 ± 11.64 (D1). The text the model sees matters more than the size of the model.
- **The label effect depends on the model.** For BERT-Tiny, D1 lowers Good Fit recall by about 12 points; for DistilBERT with evidence, D0 and D1 give almost the same means (41.23 and 39.05 Good Fit recall, 7 seeds each). The reviewed labels therefore do not hurt every model; they mainly move where BERT-Tiny draws the Good Fit boundary.
- **Results vary strongly between seeds** for DistilBERT with evidence (single seeds ranged from about 19 to 66 Good Fit recall on D0), so only the means are interpreted, and the number of seeds behind each mean is shown.
- **The Knowledge line explains why BERT-Tiny was already strong:** it is computed from the full texts before truncation, so it carries part of the evidence that the 512-token cut removes.
- **The best single run, and why it is not selected.** DistilBERT + Knowledge + evidence on D0, seed 44, reached accuracy 60.93, macro-F1 56.57, Good Fit recall 66.09, Good Fit precision 56.34 and Potential recall 32.65: on its own it is better than the baseline on every metric. It is not selected because it is the luckiest of seven seeds of the same setting, all scored on the same validation set (the other six reached Good Fit recall 18.7–57.1). Picking the best seed after seeing validation results overstates what the setting delivers; the fixed rule compares means over seeds, and no unused test set is left to confirm a single run. The weights of that run were not kept (the run saved only metrics and predictions). It is reported as the strongest sign that evidence-aware DistilBERT can work, and as the first thing to confirm with more data or a fresh test set.
- **Adding more inputs on top of evidence does not help DistilBERT.** The M8 hybrid (evidence + Knowledge + structured features) and its ablations were re-run on seeds 43–45 (Colab GPU, a separate setting): full 53.07 accuracy / 28.14 ± 14.6 Good Fit recall, without Knowledge 54.10 / 44.64 ± 5.0, without structured features 52.93 / 27.68 ± 7.6. No variant reaches the baseline accuracy, and the strong single-seed result of one variant (seed 42) did not repeat.


## 6. Phase E — frozen model, HR prototype, testing
- **Frozen model (E1):** M4 BERT-Tiny + Knowledge v1, original labels, seed 46 (the seed closest to the 8-seed mean). Model card, configuration, reproducibility guide and weights location are in E1_frozen_model. The published weights were re-loaded on a fresh machine and gave the identical validation accuracy (56.97).
- **HR prototype (E2):** advice only. HR sees the suggestion with its probabilities, an abstain flag, the evidence for every requirement, a "Check these first" list (required items not mentioned, experience gaps), and must give a reason to override. **10 documented example cases from real validation pairs** (E2_10_example_cases.md) show correct, missed and low-confidence cases.
- **Testing (E3):** prototype v3 passes 33/33 browser tests. Five HR participants reviewed two cases each; their four main findings were fixed in v3. The platform's own tests were re-run without changes: 105/105 tests, browser test on 20 screens, 5/5 non-functional tests. A ready patch makes the platform serve the frozen model: the platform's own code reproduces the frozen validation result exactly (56.97), 106/106 tests pass, and the HR review flag (P(Good Fit) ≥ 0.30) catches 54% of Good Fit pairs.

## 7. Limitations and ethics
- **Validation labels are not reviewed.** D1 changes TRAIN only, so D1 models are judged against the original labels; a blind review of 150 validation pairs has been prepared to check this.
- **Same CVs in TRAIN and VALIDATION** (1,157 of 1,162 validation CVs appear in TRAIN with other jobs): validation scores are probably optimistic for unseen CVs.
- **Seed variance is large** (BERT-Tiny Good Fit recall from 13 to 49 across seeds); only means over several seeds are interpreted.
- **Extraction** is rule-based with modest recall; "not mentioned" can be an extraction miss, so the interface says it is not evidence of absence.
- **Fairness** cannot be measured: the data has no consented protected attributes. HR makes every decision; there is no automatic rejection.

## 8. Deviations from the plan
- Accepted by the supervisor: B2 gold annotated by one person instead of two; B3 and M8 use DistilBERT only (team decision) instead of all-MiniLM-L6-v2.
- Compute limits, stated with every number: DistilBERT runs use 384 tokens (512 did not fit the CPU time budget) and, where noted, fewer seeds than BERT-Tiny; M7 (optional in the plan) was not run.

## 9. Where everything is
Every claim above points to a file in this package: phase folders A1–E3, FINAL_RESULTS_REPORT.md (all models, class-wise metrics, confusion matrices), E4_ROBUSTNESS_CHECKS.md, E3_TEST_COVERAGE_vs_PLAN.md, PLAN_COMPLIANCE_CHECKLIST, and HireMind_code_bundle.zip (code and the exact workflows that produced every number).

The GitHub repository **HIREMINDPROJECT/HireMind-Plan** holds the same package together with the code, the files produced by every GitHub Actions run, the Colab notebooks, and the 8 checkpoints of the frozen model with SHA-256.
