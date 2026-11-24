# B4 — Truncation-loss prevention (Phase B, step B4) — v3.1 (CURRENT)

**Status update (24 Nov 2026):** the classifier effect is now measured: E2 (evidence-selected input) ran on D0 and D1 for BERT-Tiny (8 seeds each) and for DistilBERT + Knowledge; see E2_report.md, E2_D1_REPORT.md and FINAL_RESULTS_REPORT.md.

Status: **COMPLETE for the audit (TRAIN, D0 labels); classifier effect measured in E2 (see update).**
Supersedes the first B4 audit (extractor v3.0). Reason: the v3.0 sentence splitter ignored spaced sentences (see B1 schema.md). With v3.1 only 2.6% of pairs keep the same selection, so the v3.0 numbers must not be used.
Run: GitHub Actions 45251634726 on HIREMINDPROJECT/HireMind-Plan (branch b4-build, results branch b4-results-45251634726). No raw text leaves the runner: files hold IDs, counts and character offsets only. Test set never read. Labels are never used to build inputs; they only split the report.

## What was built
`b4_select.py` picks which sentences of the CV and job text fit the 512-token pair budget, instead of cutting the tail (baseline = M4 `longest_first` truncation).
- Job side: sentences with requirement spans (densest first), then opening sentences, then the rest; at least 200 tokens guaranteed.
- CV side: sentences with linked evidence, explicit negatives, degrees and stated years (densest first); then B3 semantic best matches (only in `lexsem` variants, not run); then opening sentences; then the rest.
- Kept sentences stay in original order; text is never rewritten. `lex_k` reserves room for the Knowledge v1 line (used by M4 and E2).

## Result (TRAIN, pairs over 512 tokens: lex 4,972; lex_k 4,979)
| Variant | CV relevant evidence kept, baseline → evidence-aware | Job requirements kept, baseline → evidence-aware |
|---|---|---|
| lex | 33.38% → 82.88% | 38.65% → 90.43% |
| lex_k | 33.18% → 79.53% | 34.26% → 90.44% |

By class (lex), CV evidence: Good Fit 31.21 → 81.68, No Fit 34.46 → 83.85, Potential 34.2 → 82.77. The gain is similar across classes.
Median tokens: 512 (baseline) vs 511 (evidence-aware, lex_k): the same budget, different content.

## Where evidence-aware is worse (honest cases)
- CV relevant evidence: fewer kept in 23 pairs (lex) / 23 (lex_k); more kept in 3,648 / 3,658.
- Job requirements: fewer kept in 10 (lex) / 6 (lex_k).
- Listed per case in `truncation_audit.csv`.

## Limits
- "Relevant" means what the B1 extractor found. B1 accuracy is not yet human-measured (B2 pending), so "relevant" can miss real evidence or include false matches.
- Retaining more relevant spans does not by itself prove a better classifier. E2 tests that (M4 + Knowledge v1 + `lex_k`, 8 seeds, D0 labels, extractor v3.1).
- `lexsem` variants (using B3 matches) not run: B3 similarity barely discriminated (AUC 0.58).

## Files
`truncation_audit.csv` (per case), `evidence_inputs.jsonl.gz` (offsets of kept segments, no text), `B4_audit_summary.json`, `b4_build.py`, `b4_select.py`, `knowledge_v1_ORIGINAL.py`, `structured_extractor.py` (v3.1).
