# A2 — TRAIN data-quality findings (v3, 23 Nov 2026)

**Status: VERIFIED_COMPLETE for the plan's pass condition** (every suspected issue has a count and example; source row IDs preserved). TRAIN only; nothing was modified or relabelled; the test set was not read.
Supersedes v2 (`A2_data_quality_findings_v2.md`, manifest-only), which is kept unchanged.

## How it was measured
- The pinned dataset (cnamuangtoun/resume-job-description-fit @08978e2, train.csv SHA verified by the loader) was loaded on a GitHub Actions CPU runner in the project's GitHub fork, joined to the frozen split (SHA `2ff3926c…539f9b`) with the loader's own assertions that labels and CV/job text IDs match (4,993 rows).
- Only IDs, counts and scores left the runner; no CV or job text was copied. Code: `A2_text_audit_job_v2.py`; environment: `pip_freeze.txt` of the run (A4 folder).
- Tokens use the pinned BERT-Tiny tokenizer (revision 30b0a37). Pair length = CV + job + 3 special tokens.

## Findings (`data_quality_report_v2.csv`, examples in `A2_text_issue_examples.csv` and `A2_data_quality_issue_examples_v2.csv`)
| Issue | Count | Reading |
|---|---|---|
| Empty CV / job text | 0 / 0 | none |
| Exact duplicate CV–job pairs; conflicting labels for a pair | 0; 0 | none |
| **Pairs over the 512-token input limit** | **4,972 of 4,993 (99.58%)** | Almost every TRAIN pair loses text. This matches the 99.57% the supervisor reports for validation, so the problem is dataset-wide. Median pair 1,550 tokens, p95 2,970, max 5,330 |
| **CV IDs shared with validation/test** | **551 of 636 TRAIN CV IDs** (552 CV IDs occur in more than one split) | The same CV is scored against different jobs across splits. Job groups are disjoint (0 of 349), so pairs and jobs do not leak, but the CV side does. Document as a limitation; do not drop rows |
| Short job text (<50 words) | 193 rows | Little requirement information; candidates for "Insufficient Evidence" review |
| Short CV text (<100 words) | 2 rows | Check manually |
| Near-duplicate CV texts (distinct cv_id, TF-IDF cosine ≥ 0.90) | 50 pairs (10 shown; the top ones have cosine 1.0) | Same CV text under different IDs, so identity-based splitting treats them as different CVs. Lexical check against each CV's 3 nearest neighbours; a lower threshold or deeper search may find more |
| Near-duplicate job texts (distinct job_id, cosine ≥ 0.90) | 2 pairs | Possible job-group duplication across IDs |
| Job groups with <5 TRAIN rows | 20 of 223 (3 with a single row) | Little within-job contrast |
| Single-class job groups | 30 of 223 | Group identity is a possible shortcut |
| Class balance | No Fit 2,533 (50.7%) · Potential 1,231 (24.7%) · Good 1,229 (24.6%) | — |

## Length distributions (`A2_length_quantiles.csv`, per row in `A2_train_length_table.csv`)
CV: median 618 words / 1,016 tokens, p95 1,591 words / 2,332 tokens, max 4,348 tokens. Job text: median 328 words / 443 tokens, p95 810 words / 1,050 tokens, max 1,571 tokens.

## Limits
- Near-duplicate scan is lexical (TF-IDF) with a fixed 0.90 cut-off and 3 neighbours; it is a count of what that method found, not an exhaustive list.
- Short-text thresholds (100 and 50 words) are chosen for review priority, not proof of a data error.
- The 99.58% figure uses the exact pair encoding rule (CV + job + 3 special tokens), which is a lower bound on loss because the model truncates the longer text first.
