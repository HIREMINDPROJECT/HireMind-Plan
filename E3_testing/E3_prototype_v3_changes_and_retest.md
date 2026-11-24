# E2/E3 — HR prototype v3: fixes from the HR sessions and re-test (24 Nov 2026)   STATUS: DONE

v3 changes only the HR review page (`hr_review_prototype.html`). The model, the extractor and the case format are unchanged, so v2 case files open in v3 as they are.

## What changed, and which HR finding it answers
| Issue | HR finding (E3 sessions) | Change in v3 |
|---|---|---|
| ISS-07 | Experience-year gaps were easy to miss next to matched skills (cases 03, 06) | New "Check these first" box above the requirement table. It lists experience or level below the requirement ("Experience below the requirement: 4 years") and required years the tool did not compare ("compare with the CV dates yourself"). |
| ISS-08 | A required certification that is not documented was easy to miss (case 04) | Required items not mentioned in the CV are listed in the same box; certifications are listed first and labelled "(certification)". The matching table rows are marked with a coloured left border. |
| ISS-09 | When no suggestion exists, the empty state was unclear (case 10) | A separate notice: "No suggestion is available for this case. Decide from the evidence below, or mark it for manual review." with a button that sets the decision to "Needs manual review". |
| ISS-10 | The long status wording was overlooked while scanning (case 08) | Short status "Not mentioned"; the explanation sits underneath: "The CV does not mention it. This is not evidence that the candidate lacks it." |

The box says the listed items "are points to verify, not reasons to reject on their own". Preferred (non-required) items that are not mentioned are not flagged. HR still makes every decision, and an override still needs a reason.

## Re-test (Playwright, Chromium, synthetic data only)
`test_prototype_v3.py` = the 24 v2 tests (regression) + 9 new tests (W01–W09) for the four fixes. Result: **33 / 33 pass** (`test_results_v3.csv`).

| Test | Checks |
|---|---|
| W01 | Demo case: the box lists the 2 items that need checking (required Kubernetes not mentioned; 5 years not compared) |
| W02 | Short status "Not mentioned" with the explanation underneath |
| W03 | Flagged rows are marked in the table |
| W04 | The box states these are not reasons to reject |
| W05 | A required certification is listed first and labelled |
| W06 | Experience below the requirement is shown as an explicit warning |
| W07 | Clicking an item clears the filter, scrolls to its row and focuses the HR check |
| W08 | Preferred items are not flagged; the box is hidden when nothing needs checking |
| W09 | No-suggestion notice with its button; the saved record has decision "Needs manual review", no suggestion and no override |

## Not changed
ISS-03 (the audit record is only downloaded as a file) stays open: where audit records are stored is a platform decision (the platform has its own audit log, FR-22).
