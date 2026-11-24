# E3 — HR prototype v2: changes from the interface review and re-test (24 Nov 2026)
Status: system test of the standalone prototype, SYNTHETIC data only. Not a user study. v1 is kept unchanged in E2_hr_prototype.

## Changes made (from E3_interface_usability_review.csv)
| Item | Change in v2 |
|---|---|
| UX-01 (High) | No browser alert(). Messages appear in an on-page status line (aria-live); focus moves to the field that needs input. |
| UX-02 (Medium) | Overlapping evidence spans are merged, so no passage is hidden; each requirement also lists its CV excerpts. |
| UX-03 (High) | Unsaved decision/check/reason: loading another case is stopped once with a message; leaving the page triggers the browser's leave-page warning. Save shows a confirmation. |
| UX-04 (Medium) | Wording changed to "Not documented in CV (not evidence of absence)" in the table and banner. Comprehension not yet tested with people. |
| UX-07 (Low) | Filter the requirement table by evidence status, with a shown/total count. |
| UX-08 (Medium) | Jump-to-source: each requirement has "in job" and CV-excerpt buttons that scroll to and focus the highlighted text. |
| UX-05 | Unchanged by design (advice-only banner and override reason kept; re-tested). |
| UX-06 (Medium) | Partly: the save message says the record has no CV text and must go to the approved location. A storage policy still needs a decision. |
| UX-09, UX-10 (High) | Not doable here: need the platform repository and real HR participants. |

## Re-test: 24 of 24 automated checks pass
| ID | Test | Result | Detail |
|---|---|---|---|
| V01 | page loads with no console errors | PASS |  |
| V02 | advice-only banner; wording 'not evidence' of absence | PASS |  |
| V03 | demo loads; rows = requirements | PASS |  |
| V04 | abstain shown as low confidence | PASS |  |
| V05 | UX-01 no decision: inline message, no browser dialog | PASS | Choose a decision first. The tool never decides for you. |
| V06 | UX-01 override without reason: inline message, focus on reason | PASS |  |
| V07 | audit record content and no CV text | PASS |  |
| V08 | save confirmation shown | PASS |  |
| V09 | no request leaves localhost | PASS |  |
| V10 | HTML in texts not executed, shown literally | PASS |  |
| V11 | UX-01 invalid JSON: inline error, app hidden, no dialog | PASS | Could not read the file (not valid JSON). |
| V12 | wrong schema: inline error | PASS |  |
| V13 | emoji offsets highlight correctly | PASS |  |
| V14 | UX-02 overlapping spans merged: both passages highlighted | PASS | ['Senior Python developer'] |
| V15 | UX-02/08 evidence list shows each excerpt per requirement | PASS | ['CV: “Senior Python”', 'CV: “Python developer”'] |
| V16 | UX-08 jump-to-source focuses the highlight | PASS |  |
| V17 | UX-07 filter by evidence status (300 rows) | PASS | 150 visible |
| V18 | 300-requirement case renders in < 3 s | PASS | 0.31s |
| V19 | UX-03 loading another case with an unsaved decision is stopped once | PASS |  |
| V20 | second load discards and opens the new case | PASS |  |
| V21 | UX-03 leaving the page with unsaved input (after a real user gesture) triggers the browser warning | PASS | ['beforeunload'] |
| V22 | HR checks do not leak between cases | PASS |  |
| V23 | controls have accessible names | PASS |  |
| V24 | no browser alert() used anywhere | PASS |  |

Note on V21: browsers show the leave-page warning only after a real user gesture on the page (a Chromium rule). The test types into the reason field first, as a person would.

## Still open
- HR participant sessions (no feedback exists yet); wording comprehension (UX-04) must be checked with people.
- Platform integration and persistence (UX-03 server side, UX-09).
- Real example cases (UX-10).
