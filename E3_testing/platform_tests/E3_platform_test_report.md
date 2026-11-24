# E3 — platform tests re-run (24 Nov 2026)   STATUS: DONE · all pass

The plan asks to re-run the platform tests. The platform's own test suites were re-run without changing its code:
- Platform: Senior-2 at commit ec28dfc85e618a737cb192cbf959714cfee89aa5 (read-only clone; this commit contains the current main, 4f0b04e).
- Where: GitHub Actions run 45309386512 (ubuntu-latest, Python 3.12, PostgreSQL 16 service), workflow `e3_platform.yml` on branch e3-platform of the project fork. All reports are on branch `e3-platform-results-45309386512`.
- Data: only the synthetic records the tests seed themselves. No CV from the dataset is used.

## Results
| Suite | What it covers | Result |
|---|---|---|
| Unit and API tests, no model (SQLite) | login and roles, jobs and policies, CV upload and parser, scoring, assessments, feedback, fairness, security, reliability | 102 passed, 1 skipped (the real-model test needs PyTorch) |
| Full suite with the real fit classifier (FR-09) | the same suite plus 3 real-model tests | **105 passed, 0 failed, 0 skipped** |
| Browser test of every screen (e2e) | HR and candidate screens, mobile view, failed-CV screen | **passed, 20 screens, 0 browser errors** |
| Non-functional tests on PostgreSQL | PT-01, PT-02, PT-03, RP-01, RT-01 | **5 / 5 pass** |

Tests per module (full suite): security 27, fairness 11, parser 11, fit service 9, evaluations 8, assessments 7, scoring 7, feedback 6, applications 5, auth 4, reliability 4, jobs 3, real model 3.

## Non-functional results (nfr_report.md)
| Test | Target | Measured |
|---|---|---|
| PT-01 dashboard | p95 ≤ 2 s, 10 users, 200 requests | p95 1,052 ms, 0 errors |
| PT-02 CV analysis | p95 ≤ 120 s over 50 CVs | p95 0.117 s, 50/50 parsed, 50/50 with a fit prediction |
| PT-03 concurrent applications | 100, none lost or duplicated | 100 stored, 100 unique |
| RP-01 repeatability | same inputs and policy → same score | 1 distinct score in 5 runs |
| RT-01 restart | data survives a restart | 150 applications before and after |

## Notes
- A local re-run in this workspace gave the same 102 passed / 1 skipped. The workspace cannot reach huggingface.co or Google Fonts, so the real-model tests and the browser font loads were checked only in GitHub Actions.
- This run tests the platform as the team left it (serving the CS4085 model M0). A second run with a ready patch that makes the platform serve the frozen model (M4 BERT-Tiny + Knowledge v1, seed 46) is in ../platform_tests_M4/E3_platform_M4_integration.md: identical validation result (56.97), 106/106 tests, e2e and NFR pass.
- UT-01 (usability with real users) and AT-01 (extraction F1 on annotated CVs) are listed as "to do" in the platform README; they are not automated tests and were not part of this re-run. The project's HR sessions (E3) and B2 extraction benchmark are reported separately.

Files: outcomes.txt, platform_commit.txt, junit_full.xml, nfr_report.md, e2e.log (screenshots are on the results branch).
