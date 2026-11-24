# B3 — Semantic matcher, DistilBERT only (v3.1)  | STATUS: COMPLETE (see update)

**Status update (24 Nov 2026):** COMPLETE. The deviation (DistilBERT as the only encoder instead of all-MiniLM-L6-v2) was accepted by the supervisor. The plan's pass condition (valid and false semantic matches shown; similarity is not proof) is met by the numbers below; examples quoting CV text are kept private.


Runs on B1 schema v3.1 (extractor v3.1). Supersedes v3.0 (old splitter bug). Encoder: distilbert-base-uncased @12040acc (the only encoder; team decision). The plan's B3 says all-MiniLM-L6-v2: this is a deviation, **accepted by the supervisor**.

## What was computed (TRAIN pairs only, 4,993 pairs; no dataset labels used)
- 48,898 job requirements; every one has a DistilBERT best-sentence similarity score + character offsets of the best CV sentence (no CV text stored).
- Threshold 0.8121 = q10 of best-sentence similarity among exact-found requirements (label-free). "missing" = below threshold, meaning no supporting evidence found, NOT "lacks the skill".
- missing_indicator_total 5,403; semantic_only_candidate_total 30,200 (exact match absent, similarity present). These candidates are UNVERIFIED.

## Pass criterion: "show valid and false semantic matches; similarity is not proof"
Evidence so far points to **false matches being common**:
| Group | n | mean sim | q10 / q50 / q90 |
|---|---|---|---|
| Exact found | 13,295 | 0.8705 | 0.8121 / 0.8779 / 0.9205 |
| Not found / unknown | 33,269 | 0.8588 | 0.7958 / 0.8658 / 0.9107 |
- AUC (exact-found vs not-found by similarity) = 0.576 over 48,898 (13,295 vs 35,603 not exact-found): barely better than chance.
- 84.8% of not-found requirements score at or above the threshold (vs 90.0% of exact-found). The raw similarity cannot separate a real match from a topical one.
- No exact_negative requirements exist, so no check of "similarity vs explicit negative" is possible (consistent with B1: 40 CV negatives, none linked to a job requirement).
- EXPERIENCE requirements are never exact-found (0 of 5,102) because years are not compared (by design).

## What this does and does not show
- Shows: mean-pooled DistilBERT similarity is dominated by general domain vocabulary; high scores do not mean evidence.
- Does NOT show: how many semantic-only candidates are valid. That needs human judgement (B2 pack or HR review of a sample). Nothing is claimed as accurate.
- Consequence: semantic features should only be used as weak extra signals, never as proof, and never shown to HR as matches without the evidence span + a "unverified" flag (done in the E2 prototype).

## Files
B3_matcher_report.json, match_features_pair.csv (pair aggregates: IDs and scores only), match_features.jsonl.gz (per requirement scores + offsets only), pip_freeze.txt. Code: b3 branch `b3-matcher` (HIREMINDPROJECT/HireMind-Plan), run 45251640548.
