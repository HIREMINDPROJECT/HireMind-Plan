# Phase D — error analysis of M4 validation mistakes (D0 labels) — PRE-TAGS, human confirmation pending

**Status update (24 Nov 2026):** the rule-based pre-tags are complete for all 8 seeds of the frozen model (M4); together with the class-wise metrics and confusion matrices in FINAL_RESULTS_REPORT.md they form the error analysis used in the final report.


**Status:** rule-based pre-tags (complete). The tags below are rule-based flags built from objective facts (B1 extraction, how much the M4 truncation dropped, TF-IDF similarity to training jobs). They are **not** human judgements and they do **not** prove the cause of any mistake. Three of the seven plan tags cannot be assigned without people or data I do not have yet (see the end).
**Data:** the 8 M4 seeds (E0, identical to the A1 reference) on the 1,162 validation pairs, 476 of which (41.0%) are "consistent mistakes" = wrong in more than 4 of 8 seeds. Facts run: GitHub Actions 45253477867 (no text in any output). Rules are fixed in `d_error_pretags.py` and were written before the mistake list was looked at.

## Where the mistakes are (majority vote over the 8 seeds; rows = true class)
| True class | n | predicted No Fit | Potential | Good Fit | consistent mistakes |
|---|---|---|---|---|---|
| No Fit | 579 | 489 | 30 | 60 | 92 (15.9%) |
| Potential Fit | 294 | 154 | 85 | 55 | 206 (70.1%) |
| Good Fit | 289 | 85 | 108 | 96 | 178 (61.6%) |
M4 leans towards No Fit. Potential and Good Fit are missed most of the time, which fits the low Good Fit recall in the main results.

## What the flags say (share of mistakes that carry the flag, versus share of correct pairs)
| Flag | Mistakes | Correct pairs | Reading |
|---|---|---|---|
| relevant evidence dropped (M4 cut ≥50% of relevant CV records or of job requirements) | 81.5% | 77.8% | almost no difference: nearly every pair loses evidence, and the mistakes do not lose more. Consistent with E2 not helping |
| requirement ambiguity (≥50% of requirements without a required/preferred cue, or an unclear link) | 58.0% | 47.5% | mistakes are somewhat more often ambiguous (+10.5 points) |
| unseen-job shift (job in the lowest quartile of similarity to any training job) | 19.5% | 29.0% | the opposite of the expectation: mistakes are not concentrated on dissimilar jobs |
By true class, the same three shares are in `error_pretags_by_class.csv`, next to the base rate of each flag among all pairs of that class. For Good Fit, 82.0% of mistakes are ambiguous against 75.4% of all Good Fit pairs.
Only 5–16% of mistakes carry none of the three flags (the residual "candidate classifier error"), but that number is not informative because the flags cover most pairs.

## By job group
55 job groups. The 5 groups with the most mistakes hold 29.6% of all consistent mistakes. 9 groups with at least 10 pairs have a mistake rate more than 15 points above the overall 41.0% (list in `error_pretags_summary.json`; for example one group has 23 mistakes in 29 pairs). A concentrated group is a warning signal that deserves a look at its labels and its requirement extraction, not proof of wrong labels.

## Limits
- Each flag is a crude proxy. "Relevant evidence" depends on the B1 extractor, whose accuracy is not yet human-measured (B2).
- 1,157 of the 1,162 validation CVs also appear in the training split (CV identity repeats across splits; job groups are disjoint). This is the documented limitation in the plan and it affects how far these results generalise to new people.
- A mistake is counted by majority over seeds, so a pair that flips between seeds is not a "consistent mistake".

## Still needs people or data (kept as `pending_human`)
- **Label disagreement:** needs the A4 double review and adjudication.
- **Extraction error:** needs the B2 human gold annotations.
- **Semantic mismatch:** needs B3 features on the validation split and human reading.
Per-pair rows are in `error_pretags_per_pair.csv` (IDs and counts only).
