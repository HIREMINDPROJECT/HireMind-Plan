# E3 — the tests the plan asks for, and where each one is   (24 Nov 2026)

The plan (E3) asks for tests of extraction, matching, long CVs, wrong parsing, negation and abstention, a re-run of the platform tests, and HR feedback. This page points to the evidence for each one. Every line refers to a file in this package or in the code bundle.

| Plan asks to test | What was tested | Result | Where |
|---|---|---|---|
| **Extraction** | Extractor v3.1 on all 4,993 TRAIN pairs: every claim must point back to its source characters | 0 pointer failures | B1_structured_schema/schema.md, B1_pointer_verification.json |
| | Extractor against 30 hand-annotated pairs | Precision 0.51, recall 0.38; years F1 0.85 | B2_extraction_benchmark/B2_REPORT.md |
| | Unit tests on synthetic text: glued headings and sentences, interest-only mentions, perks ignored | pass | code bundle: b1-structured/b1/test_b1.py |
| **Matching** | Exact + DistilBERT similarity per requirement; does similarity separate found from not-found evidence? | Weakly (AUC 0.58): similarity is not proof, so the interface never treats it as proof | B3_matcher/B3_REPORT_v3.1.md |
| | Look-alike skills must not match (MySQL ≠ SQL) | pass | test_b1.py |
| **Long CVs** | How much relevant evidence survives the 512-token limit | 99.6% of pairs are longer; plain cut keeps 33% of CV evidence, B4 selection keeps 80% | B4_truncation/B4_REPORT.md |
| | Prototype with 300 requirements and a 14,000-character CV | renders in 0.31 s (limit 3 s) | test_results_v3.csv (V18), test_report.md (T20) |
| **Wrong parsing** | Broken or wrong case files | clear message, nothing shown, no crash | test_results_v3.csv (V11, V12) |
| | Overlapping or bad evidence spans, emoji offsets | highlighted correctly, no crash | V13, V14; T17 |
| | Extraction misses shown honestly | "Not mentioned" with "not evidence that the candidate lacks it" | V02, W02 |
| **Negation** | "No Java experience" must be recorded as explicitly negative, not as silence | pass (unit test). The 30-pair gold has no negative phrase, so negation is not measured on real data | test_b1.py; B2_review_followup.md |
| | Unknown must never be shown as negative | pass | T05; schema.md (unknown ≠ negative) |
| **Abstention** | Low-confidence suggestion shown as abstain / manual review | pass | V04; E2_10_example_cases.md (cases 09, 10) |
| | Platform review flag for possible Good Fit pairs | P(Good Fit) ≥ 0.30 flags 32.4% of pairs and catches 54.0% of Good Fit | platform_tests_M4/E3_platform_M4_integration.md |
| **Platform tests re-run** | The platform's own tests, unchanged code | 105/105, browser test 20 screens, non-functional 5/5 | platform_tests/E3_platform_test_report.md |
| | The platform serving the frozen model (ready patch) | identical validation result (56.97), 106/106 tests | platform_tests_M4/ |
| **HR feedback** | 5 HR participants × 2 cases; findings fixed and re-tested | 4 main findings fixed in v3; 33/33 prototype tests | E3_HR_feedback_section.md, E3_prototype_v3_changes_and_retest.md |
| **Issues** | Every issue found, its status | 10 fixed or closed, 1 partial | issue_log_v3.csv |

**Not covered, said plainly:** screen-reader testing and browsers other than Chromium; negation on real (not synthetic) text.
