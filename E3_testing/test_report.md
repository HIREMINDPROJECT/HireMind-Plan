# E3 system test report — HR review prototype (standalone)

**Status update (24 Nov 2026):** this is the v1 test report (21/21). Superseded for the final state by prototype v3 (33/33, E3_prototype_v3_changes_and_retest.md), the HR sessions (E3_HR_feedback_section.md) and the platform re-run (platform_tests/E3_platform_test_report.md).

Status (v1, at the time): PARTIAL. Date 24 Nov 2026. Synthetic data only.

## 1. What was tested
`hr_review_prototype.html` (advice-only HR evidence review, runs in the browser, no server) in headless Chromium with Playwright, served from localhost. Test data: the built-in SYNTHETIC demo and small synthetic case files built by the test script. No real CV was used.

## 2. Results: 21 of 21 automated tests pass
| ID | Test | Result | Detail |
|---|---|---|---|
| T01 | page loads with no console errors | PASS |  |
| T02 | advice-only banner visible | PASS |  |
| T03 | built-in demo loads; rows = requirements | PASS | 7 vs 7 |
| T04 | abstain flag shown as low confidence | PASS |  |
| T05 | unknown is shown as 'not negative' | PASS |  |
| T06 | save without decision is blocked | PASS | ['Choose a decision first. The tool never decides for you.'] |
| T07 | override without reason is blocked | PASS | ['Please give a reason when your decision differs from the suggestion.'] |
| T08 | audit record content | PASS | {"overridden": true, "hr_decision": "No Fit", "suggestion": "Potential Fit"} |
| T09 | audit record has no CV/job text | PASS |  |
| T10 | no request leaves localhost | PASS | [] |
| T12 | HTML in CV/job/requirement text is not executed | PASS |  |
| T13 | HTML shown literally | PASS |  |
| T14 | invalid JSON: message shown, app stays hidden | PASS | ['Could not read the file'] |
| T15 | wrong schema: message shown, app stays hidden | PASS | ['Not a valid case file'] |
| T16 | highlight correct with emoji (code-point offsets) | PASS | ['Python'] |
| T17 | bad/overlapping spans do not crash | PASS | [] |
| T18 | HR checks from case 1 do not leak into case 2's audit record | PASS | {} |
| T19 | case without prediction shows notice and saves | PASS |  |
| T20 | 300 requirements, 14k-char CV renders in < 3 s | PASS | 0.34s |
| T21 | demo over file:// shows the help message | PASS | ["Open this page next to demo_case_SYNTHETIC.json (or use 'Load a case JSON')."] |
| T22 | controls have accessible names | PASS |  |

## 3. Issues found (issue_log.csv)
- ISS-01 (Low, OPEN): Errors and validation use blocking browser alert() dialogs; they interrupt screen readers and cannot be styled or logged.
- ISS-02 (Low, OPEN): When two evidence spans overlap
- ISS-03 (Info, OPEN): Audit record is only downloaded as a JSON file; nothing is stored or sent. A reviewer who forgets to save loses the decision.
- ISS-04 (High, OPEN): No HR user has used the prototype. There is no HR feedback yet. Usability claims cannot be made.
- ISS-05 (High, OPEN): The Senior-2 platform repository is not accessible from this workspace
- ISS-06 (Medium, OPEN): The 10 real example cases were not made (needs a frozen candidate model and private case files).

## 4. What was NOT tested (be clear before anyone relies on this)
- **HR users.** No HR or recruiting person used the tool. There is no HR feedback, no usability result, no comprehension check of the wording ("Not stated" is not "lacks").
- **The platform.** The Senior-2 platform repository is not reachable from this workspace, so its code and tests were not run. This report covers only the standalone prototype.
- **Real cases.** Only synthetic data. The 10 real example cases do not exist yet.
- **Accessibility and browsers.** Only Chromium; no screen-reader test; keyboard use only checked through labels.
- **Model quality.** Not in scope here; see the Phase C and D reports.

## 5. Statement
The prototype behaves as designed on these 21 checks: it never accepts or rejects a candidate, it needs a reason for an override, it saves no CV text in the audit record, it sends nothing over the network, and it shows page content as plain text. This is a system test, not a user study.
