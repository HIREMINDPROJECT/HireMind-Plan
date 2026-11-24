# B1 — Structured pair schema (v3)   · status: VERIFIED_COMPLETE for the B1 pass conditions (pointers + unknown≠negative); extraction ACCURACY is not claimed (see B2)

Builds on Mervat's Structured Evidence Schema v2 and evidence_v2.py (team Drive, read-only; nothing was written there).
Reuse was checked first. What was kept, and what had to change, is listed below.

## What a pair record looks like (`structured_pairs.jsonl`, one line per TRAIN pair)
| field | meaning |
|---|---|
| `case_id` | the TRAIN `row_uid` (also `pair_id`, `cv_id`, `job_id`) |
| `records[]` | evidence spans from the job text (`source_type=JOB_DESCRIPTION`, id `…:R###`) and the CV (`CV`, `…:C###`) |
| `requirement_links[]` | one entry per explicit job requirement: what the CV shows for it |
| `cv_sha256`, `jd_sha256`, `cv_chars`, `jd_chars` | identify the exact source strings the offsets point into |

Each record (schema file: `structured_evidence_schema_v3.json`, validated for every record) has
`evidence_category` (SKILL, TOOL_TECHNOLOGY, EXPERIENCE, EMPLOYMENT, EDUCATION, CERTIFICATION), `normalized_value`,
`evidence_status`, `confidence`, `requirement_type` (REQUIRED / PREFERRED / UNKNOWN for job spans; NOT_APPLICABLE for CV spans),
`source_start_char` / `source_end_char` (Python string offsets, end exclusive), `span_sha256`, and for CV spans the `section`.

## Pass condition 1 — every claim points to source text
A claim is the offsets plus the SHA-256 of the exact span. `B1_pointer_verification.json` re-slices the source for every record
and compares hashes. The shareable file does not contain CV or job text, so `verbatim_text` (required in v2) is optional in v3.
It exists only in a private form, which is built locally with `structure_pair(..., with_verbatim=True)`.

## Pass condition 2 — unknown is not negative
Statuses mean different things and are never merged:
- `EXPLICIT`: the source states it.
- `EXPLICIT_NEGATIVE` (new): the CV itself says the person lacks it ("No Java experience", "never used Tableau").
- `UNCLEAR`: mentioned but not usable ("interested in learning Docker").
- Absence: a requirement with no CV span gets `link_status = NOT_STATED_UNKNOWN` and `unknown_is_not_negative = true`. No span record is created and nothing is counted as a lack.
Education links report `LEVEL_MET` / `LEVEL_BELOW_REQUIRED` (degree level only; the field of study is not compared).
Years-of-experience links are `YEARS_STATED_NOT_COMPARED`: the CV states durations, but the extractor cannot tell which field they refer to.

## Requirement type is never guessed as REQUIRED
Order: sentence cue ("preferred", "a plus", "bonus" → PREFERRED; "required", "must", "minimum" → REQUIRED), then the section
heading (Qualifications/Requirements → REQUIRED; Preferred/Bonus → PREFERRED), else `UNKNOWN`. Perks/Benefits/About-us sentences produce no requirements.

## Changes from Mervat's v2 / evidence_v2.py, and why
1. Section detection: this dataset glues headings and sentences together ("SummaryDetail-oriented…", "needed.Computed…") and has few line breaks.
   v2 expects a heading alone on a line, so on this data every CV section would be "other". v3 detects a heading as a prefix of a line.
2. Spans: v2 returned the whole line as evidence, and here a line is a whole section. v3 points to the matched term and stores the sentence range as context.
3. Negation: v2 dropped negated mentions, so "No Java experience" looked the same as silence. v3 records them as `EXPLICIT_NEGATIVE`.
4. Glued items and sentences ("developmentExcel", "experience.Python") are matched under strict rules, which are limited to terms of five or more characters and to non-ambiguous aliases. The cost is that short glued terms are missed.
5. Added: degree level, stated years of experience, employment date ranges (as written, with no computed durations), and certification acronyms.
Kept unchanged: the SKILLS vocabulary and CONTEXT_ONLY list from Senior-2 (`skills_vocab_senior2_ORIGINAL.py`, byte-identical copy).

## Known limits (stated, not hidden)
- Rule-based vocabulary matching only. The vocabulary leans technical and misses most skills in other occupations (nursing, trades, retail), so the absence of a link is not evidence of absence.
- `requirement_type` comes from cue words and headings and can be wrong.
- Accuracy is not measured here. Coverage figures are not accuracy. B2 measures extraction against human annotation, which needs two human annotators.

## Verification result (extractor v3.1; GitHub Actions run 45251606713, branch b1-results-45251606713, pinned HireMind-DL @1f22f34)
- **v3.1 fix (24 Nov 2026).** v3.0 split sentences only where a period was glued to a capital letter. A check on TRAIN found that 4,488 of 4,993 job texts also contain normally spaced sentences, so a cue ("preferred", "required") could retype a whole spaced paragraph. v3.1 also splits after ". ", "? ", "! " and "; ". Effect on TRAIN: 5,434 of 49,166 job requirement tags (11%) in 1,330 pairs changed type; 393 old tags disappeared and 205 new ones appeared; CV records unchanged (147,311). v3.0 files are kept in the folder marked SUPERSEDED. No label was used to make this change.
- 4,993 TRAIN pairs, 196,289 evidence records, 0 pointer failures, 0 schema failures (every record re-sliced from the source and hash-checked, and validated against the v3 schema).
- Unit tests on synthetic text pass (glued headings, glued sentences, spaced sentences, negation, interest-only, perks ignored, MySQL≠SQL, unknown≠negative).
- Coverage: median 9 job requirements per pair; 87 pairs have no detected job requirement and 244 have no skill requirement; every pair has CV evidence.
- Requirement links: FOUND_EXPLICIT 11,642 · LEVEL_MET 1,653 · LEVEL_BELOW_REQUIRED 462 · YEARS_STATED_NOT_COMPARED 1,872 · NOT_STATED_UNKNOWN 33,269 (never treated as negative). Explicit negatives in CVs: 40 records, none linked to a job requirement in TRAIN, so the "no Python experience" case is too rare in this data to test from TRAIN alone.
- Requirement type: 23,414 of 48,978 job tags are UNKNOWN (no cue and no section), 14,388 REQUIRED, 11,176 PREFERRED. UNKNOWN is deliberate: the extractor never defaults to REQUIRED.
- The file is delivered as `structured_pairs.jsonl.gz` (gzip of `structured_pairs.jsonl`, 10.7 MB; SHA-256 of the gz d0eab302…; of the uncompressed file 60cc1dec…). It contains no CV or job text, only offsets, hashes and normalized values.
- Not measured: how often the extractor is right. NOT_STATED_UNKNOWN links mostly mean the vocabulary has no entry or the CV does not mention the term, not that the candidate lacks it.
