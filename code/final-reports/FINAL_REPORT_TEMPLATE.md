# HireMind — Improving Good Fit detection for CV–job matching: final report on the supervisor's improvement plan

Senior project, Effat University · 24 November 2026 · All results on the frozen validation split; the test set was not used for any choice.

## 1. Summary
{{SUMMARY}}

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

{{RESULTS_TABLE}}

**Robustness (BERT-Tiny, paired by seed, E4_ROBUSTNESS_CHECKS.md).** The label change D1 has a statistically clear effect (Good Fit recall −11.8, 95% CI [−19.3, −4.3], 1 of 8 seeds better) and moves the balance between classes rather than improving the model. Evidence selection raises Good Fit recall on average (+4.8, 6 of 8 seeds) but the interval includes zero. Ensembling the seeds and tuning a Good Fit threshold (cross-validated by job group) do not turn any candidate into a win.

**Error analysis (M4, 8 seeds).** 476 of 1,162 validation pairs are wrong in most seeds. Potential Fit (70%) and Good Fit (62%) are missed most; mistakes are somewhat more often pairs with ambiguous requirements (+10.5 points). Five job groups hold 30% of the consistent mistakes.

{{SINGLE_RUNS}}

{{DISTILBERT_SECTION}}

## 6. Phase E — frozen model, HR prototype, testing
- **Frozen model (E1):** {{FROZEN}} Model card, configuration, reproducibility guide and weights location are in E1_frozen_model. The published weights were re-loaded on a fresh machine and gave the identical validation accuracy (56.97).
- **HR prototype (E2):** advice only. HR sees the suggestion with its probabilities, an abstain flag, the evidence for every requirement, a "Check these first" list (required items not mentioned, experience gaps), and must give a reason to override. **10 documented example cases from real validation pairs** (E2_10_example_cases.md) show correct, missed and low-confidence cases.
- **Testing (E3):** prototype v3 passes 33/33 browser tests. Five HR participants reviewed two cases each; their four main findings were fixed in v3. The platform's own tests were re-run without changes: 105/105 tests, browser test on 20 screens, 5/5 non-functional tests. {{PLATFORM_M4}}

## 7. Limitations and ethics
- **Validation labels are not reviewed.** D1 changes TRAIN only, so D1 models are judged against the original labels; part of D1's lower Good Fit recall can come from this difference in the definition of Good Fit.
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
