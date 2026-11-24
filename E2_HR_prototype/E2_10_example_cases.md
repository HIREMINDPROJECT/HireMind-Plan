# E2 — 10 documented example cases from real validation pairs (24 Nov 2026)   STATUS: DONE

The plan asks for the prototype plus 10 documented example cases. These 10 are **real VALIDATION pairs**, scored by the **frozen model** (M4 BERT-Tiny + Knowledge v1, seed 46) and structured by the **B1 extractor v3.1**. They were chosen by a fixed, seeded rule (seed 20261124) so that every situation HR meets is shown, not the best-looking cases.

- Built with `HireMind_E2_10_real_cases_Colab.ipynb` (Colab, CPU). The notebook first re-scores the frozen model and stops unless validation accuracy is 56.97; it was.
- The case files (`case_01.json` … `case_10.json`) contain CV and job text, so they are kept **private** and not in this package. They open in `hr_review_prototype.html` with *Load case*.
- A requirement's status comes from the extractor: *found*, *not mentioned in the CV* (not evidence of absence), or *explicitly negative*.

| Case | Situation | True label | Model | P(No / Potential / Good) | Abstain | Requirements (required) | Found | Not mentioned |
|---|---|---|---|---|---|---|---|---|
| 01 | Good Fit, model right | Good Fit | Good Fit | 0.235 / 0.195 / 0.571 | no | 6 (0) | 1 | 5 |
| 02 | Good Fit, model right | Good Fit | Good Fit | 0.184 / 0.240 / 0.576 | no | 6 (0) | 2 | 4 |
| 03 | Potential Fit, model right | Potential Fit | Potential Fit | 0.198 / 0.507 / 0.295 | no | 7 (4) | 5 | 0 |
| 04 | No Fit, model right | No Fit | No Fit | 0.805 / 0.127 / 0.068 | no | 3 (0) | 0 | 3 |
| 05 | No Fit, model right | No Fit | No Fit | 0.744 / 0.194 / 0.061 | no | 11 (3) | 1 | 10 |
| 06 | Good Fit missed | Good Fit | No Fit | 0.757 / 0.127 / 0.116 | no | 6 (0) | 0 | 6 |
| 07 | Good Fit missed | Good Fit | No Fit | 0.702 / 0.223 / 0.075 | no | 25 (5) | 2 | 22 |
| 08 | Model said Good Fit, true No Fit | No Fit | Good Fit | 0.269 / 0.224 / 0.507 | no | 27 (20) | 10 | 15 |
| 09 | Low confidence | Good Fit | Potential Fit | 0.277 / 0.500 / 0.223 | **yes** | 4 (0) | 0 | 4 |
| 10 | Low confidence | No Fit | No Fit | 0.429 / 0.367 / 0.204 | **yes** | 4 (4) | 0 | 2 |

No case had an explicitly negative requirement ("no Python experience").

## What the cases show HR (and why the prototype is advice-only)
- **The model's confidence is modest even when it is right** (Good Fit at 0.57–0.58). The prototype shows the probabilities, not only the label.
- **Both missed Good Fit cases (06, 07) have almost nothing found by the extractor** (0 of 6, 2 of 25). The model leans to No Fit when the evidence lookup comes back empty. In the prototype these requirements appear as *Not mentioned* with the explanation "not evidence that the candidate lacks it", and they are listed under *Check these first*. That is exactly where HR's own reading is needed.
- **The false Good Fit (08)** has many requirements, 10 found and 15 not mentioned. A high match count on some skills can hide missing required ones, which is why the v3 *Check these first* box lists required items that are not mentioned.
- **Low-confidence cases (09, 10)** are flagged *abstain / manual review*. Case 09 is a real Good Fit that the model was unsure about, so the flag sends it to a person instead of a quiet rejection.
- **Requirement type is often unknown** (cases 01, 02, 04, 06, 09 have no requirement marked REQUIRED). The extractor does not guess "required" when the posting does not say it (B2: about half of job requirements stay UNKNOWN by design).

These are 10 illustrations, not a performance estimate; performance is reported on all 1,162 validation pairs.
