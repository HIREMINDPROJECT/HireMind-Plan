# B2 — extraction benchmark result (v1, 24 Nov 2026)   STATUS: DONE with a single annotator (deviation from the plan)

## Gold
- Source: two annotation files for 30 cases (789 claims each), annotated **by hand by the Dr.** (stated 24 Nov 01:50). Adopted as gold on the project coordinator's instruction, without a second annotator.
- The two files have identical claim lists (789/789); they differ only in the requirement type of some job rows. Both are scored.
- **Deviation:** the plan asks for two independent annotators with agreement and adjudication. With one annotator, inter-annotator agreement cannot be measured. The numbers below are "B1 extractor vs the Dr.'s annotation", not vs an adjudicated two-person gold.
- Gold files contain CV/job text spans: kept locally, NOT in Drive or GitHub.

## Check of the inputs
- All 789 copied spans are found verbatim in the case texts; texts identical to the team pack; B1 pointer check on the 30 cases: 1,115/1,115 hashes match.

## Result: B1 extractor v3.1 vs gold (distinct claims; exact normalized-value match)
| Field | Source | Gold | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Skill/tool | CV | 369 | 270 | 0.633 | 0.463 | 0.535 |
| Skill/tool | Job | 202 | 136 | 0.632 | 0.426 | 0.509 |
| Education | CV | 51 | 28 | 0.786 | 0.431 | 0.557 |
| Education | Job | 16 | 14 | 0.786 | 0.688 | 0.733 |
| Certification | both | 6 | 6 | 0.167 | 0.167 | 0.167 |
| **All** | both | 779 | 578 | **0.507** | **0.376** | **0.432** |
- Years of experience and employment dates score ~0 under exact matching only because of format ("09/2021toCurrent" vs "09/2021 to current"; "more than 5 years" vs "5 years"). With a format-tolerant rule (file extraction_evaluation_format_tolerant_dates_years.csv): **years F1 0.853** (P 0.879, R 0.829), **dates F1 0.541** (P 0.538, R 0.544).
- Requirement type on the 99 matched job claims: **62.6%** correct (gold file 1) / **60.6%** (gold file 2).
- Provenance: 100% of matched claims point to the same text as the gold. Explicit negatives: 0 in gold, 0 predicted (not testable on this sample).

## Reading
- The extractor is precise more than complete: about half of what it finds is in the gold, and it finds about 38% of the gold claims. Main misses are skills outside the B1 vocabulary (e.g. PowerPoint, VPN, Visual Studio) and soft-skill phrases.
- Requirement typing is weak (about 6 in 10), consistent with 48% of job requirements being UNKNOWN in B1.
- These limits matter for B3/B4/M8: the "not stated" status depends on this recall.
- Small sample (30 cases, single annotator): treat as indicative.
