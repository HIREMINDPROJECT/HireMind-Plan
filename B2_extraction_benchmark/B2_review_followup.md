# B2 — follow-up to the supervisor's review (24 Nov 2026)
The arithmetic of extraction_evaluation.csv was re-checked and is unchanged. Weaknesses confirmed, with the action taken:

| Finding | Evidence | Action |
|---|---|---|
| Skill vocabulary misses common tools (PowerPoint, VPN, Visual Studio …) | skill/tool recall 0.45 | Documented as the main cause of low recall. Expanding the vocabulary changes B1 and therefore every downstream run (B3, B4, E2, M3, M8); it is listed as future work and NOT done before submission, so the reported experiments stay consistent with one frozen extractor (v3.1). |
| Certifications weak | F1 0.17 on 6 gold claims | Same as above; very small gold count, so the estimate itself is uncertain. |
| Required vs preferred detection | 61–63% on 99 matched job claims | Documented; 48% of job requirements stay UNKNOWN by design (no default to REQUIRED). |
| Negation not tested by the gold | 0 explicit negatives in the gold | Negation is covered by unit tests on synthetic text (test_b1.py: "No Java experience", "Never used Tableau" → EXPLICIT_NEGATIVE). A gold sample containing real negations is still needed. |
| Years/dates: exact match misleading | strict F1 ≈ 0 vs format-tolerant 0.85 / 0.54 | Both numbers are reported; the format-tolerant rule is written in extraction_evaluation_format_tolerant_dates_years.csv. |
| Gold provenance | single annotator | Recorded as an accepted deviation; the gold files are private (contain CV text). |
