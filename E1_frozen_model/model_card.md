# Model card — HireMind frozen model (E1) · 24 Nov 2026

**Status: FROZEN.** Chosen by the preset rule in E4: no challenger beat the baseline on both accuracy and Good Fit recall, so the strongest defensible baseline is kept (a negative result, reported as such).

## 1. What it is
- **Model:** M4 — BERT-Tiny (google/bert_uncased_L-2_H-128_A-2 @30b0a37c) fine-tuned for 3 classes (No Fit / Potential Fit / Good Fit) on (CV, job description) pairs, with the Knowledge v1 line prepended to the job text; 512 tokens. Trained on the original TRAIN labels (D0). Frozen checkpoint: seed 46 (the seed closest to the 8-seed mean), SHA-256 0953ce89….
- **Intended use:** decision support for HR screening; shows a suggestion with its uncertainty and the evidence per requirement. **HR makes every decision. No automatic hiring or rejection.**
- **Not for:** final hiring decisions; use without HR review; jobs or CV types far from the training data; any use involving protected characteristics.

## 2. Data
cnamuangtoun/resume-job-description-fit @08978e21; frozen split TRAIN 4,993 / VALIDATION 1,162 / TEST 1,759. Job groups are disjoint across splits; CVs repeat (1,157 of 1,162 validation CVs also appear in TRAIN). The test set was opened earlier by the team and is not used for any selection; no test number is reported.

## 3. Performance (VALIDATION, D0 labels)
| | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall |
|---|---|---|---|---|---|
| 8-seed mean (SD) — expected performance | 56.79 (1.17) | 48.61 (2.80) | 36.72 (11.28) | 43.68 (4.64) | 29.68 (4.54) |
| Frozen seed 46 | 56.97 | 48.80 | 38.41 | 43.87 | 27.21 |

Good Fit recall varies strongly between seeds; report the mean ± SD.

## 4. What was tried (E4/E5, full table in E4_model_comparison_validation.csv)
M1 LR, M2 LinearSVC, M3 XGBoost, M5 DistilBERT, M6 MiniLM, M8 DistilBERT hybrid (4 variants), label cleaning D1 (E1), evidence-selected input (E2, on D0 and D1), semantic features (E3). None met the rule. The highest Good Fit recall (48.79, M8 without structured features, one seed) came with lower accuracy (54.13).

## 5. Known limitations and risks
- **Same CVs in TRAIN and VALIDATION:** job groups are disjoint, but 1,157 of the 1,162 validation CVs also appear in TRAIN (paired with other jobs). Validation scores are therefore probably optimistic for CVs never seen before; the test-set and new-CV performance may be lower.
- **Labels:** reviewed labels (D1) disagree strongly with the original labels on Good Fit (in a random TRAIN sample all 37 original Good Fit cases were relabelled). Training on D1 lowers Good Fit recall by 11.8 points on the original validation labels. The validation labels themselves are not audited.
- **Evidence extraction** (rule-based, single-annotator benchmark of 30 cases): precision 0.51, recall 0.38; skill vocabulary gaps; requirement type right about 6 in 10 times. "Not documented in the CV" can be an extraction miss; the interface says "not evidence of absence".
- **Truncation:** at 512 tokens relevant evidence is dropped in many pairs (B4 audit).
- **Semantic similarity** does not prove a match (AUC 0.58).
- **Errors** concentrate in a few job groups (error analysis).
- **Fairness:** not measurable here (no consented protected attributes); proxies such as names, places and dates may be present in text.

## 6. Human oversight
Advice-only interface (prototype v3): suggestion with uncertainty and abstain flag, evidence per requirement, a "Check these first" list (required items not mentioned, experience gaps), filters, jump-to-source, mandatory reason when HR overrides, manual-review option, audit record without CV text. Prototype tests 33/33 (v3), functional evaluation 10/10, HR sessions with 5 participants recorded in E3. Platform tests re-run: 105/105, browser test 20 screens, non-functional 5/5. A ready patch makes the platform serve this model: its own code reproduces the frozen validation result exactly (56.97), 106/106 tests pass, and its HR review flag (P(Good Fit) ≥ 0.30, chosen on validation) catches 54% of Good Fit pairs while flagging 32% of all pairs.

## 7. Reproducibility
model_config.yaml, reproducibility.md, WEIGHTS_LOCATION.md, code bundle in this package.

## 8. Owners
HireMind team, senior project, Effat University. Work follows the 19 Nov 2026 Student Implementation Plan from the project supervisor.
