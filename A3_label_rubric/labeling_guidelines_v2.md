---
title: "HireMind Labeling Guidelines — v2 (approved after the two-reviewer pilot)"
---

**Status update (24 Nov 2026):** APPROVED and used. The two-reviewer pilot (20 cases, two independent reviewers) reached 17/20 agreement, Cohen's κ 0.714, above the pre-set threshold; the supervisor accepted it and it was used for the 400-case review (A4).


**Status (when written):** draft; to become usable for A4 only after the pilot in section 9 meets its pre-set agreement threshold and the supervisor approves.
**Scope:** TRAIN label review only. This rubric must never be used to change a label silently, and HR remains the final decision-maker; nothing here is an automatic hire or reject rule.
**Replaces:** `A3_labeling_guidelines_DRAFT.pdf` (v1, kept unchanged). v2 adds explicit decision rules and boundary examples; the v1 principles are retained.

## 1. Labels
| Label | Meaning (one line) |
|---|---|
| **Good Fit** | The role-defining skill and the required experience are directly evidenced, and every other core mandatory requirement is directly evidenced (one education, certificate or other must-have may be only partly evidenced). |
| **Potential Fit** | Relevant evidence exists and nothing is contradicted, but at least one core mandatory requirement is partial, transferable, or unverifiable. |
| **No Fit** | The role-defining requirement is contradicted, or absent with no related evidence, or most core requirements are missing. |
| **Insufficient Evidence** (audit-only) | The documents are defective, so the case cannot be judged. Never a training class; never a way to avoid a hard but judgeable case. |

## 2. Step 1 — list the core requirements (from the job text only)
1. Read the job text and list its **mandatory** requirements: anything marked *required, must, minimum, essential, need to*, or listed under a requirements/qualifications heading without a "preferred" marker.
2. List **preferred** items separately: *preferred, plus, nice to have, desirable, advantage, bonus*. Preferred items never change the label.
3. Keep at most **5 core mandatory requirements**, in this priority order: (R1) the role-defining skill or duty (the main tool, language, domain or function the job is about); (R2) required experience (years or level); (R3) required education; (R4) required certificate or licence; (R5) other explicit must-haves. If more than 5 remain, keep the first 5 in that order.
4. If the job gives no requirements list, derive R1 and R2 from the duties, and set confidence to **Low**.

## 3. Step 2 — grade the evidence for each core requirement
Use only the CV text. Quote the exact words (at most 40 words) that support the grade.

| Grade | Meaning | Examples |
|---|---|---|
| **DIRECT** | The CV states the skill, duty, degree or credential explicitly, in a real context (job, project, course with output). | "Built SQL reporting queries" for "SQL required". |
| **PARTIAL** | Related, in progress, inferred from the same ecosystem, shorter than required, or not verifiable. | "PyTorch models" for "Python required"; "currently learning Spark"; a skills-list mention only; undated experience for "3 years". |
| **NOT_EVIDENCED** | A complete CV does not mention it. Absence is recorded, not treated as proof of lacking the skill. | No mention of a required certificate. |
| **CONTRADICTED** | The CV explicitly denies it or shows it lapsed. | "No Python experience"; "certification expired 2024". |

Rules that apply to the grading:

- **Negation.** "No X", "never used X", "not familiar with X" → CONTRADICTED. "Learning X", "beginner in X" → PARTIAL.
- **Skills list only.** A bare mention in a skills list, with no role or project using it, is PARTIAL for a role-defining skill (R1) and DIRECT for a minor one (R5).
- **Years of experience.** Count only dated roles; overlapping dates count once. Present-tense roles ("2023–present") count to the latest date visible in the CV. If dates are missing, the grade is PARTIAL (unverifiable). Required years met or exceeded → DIRECT; 70–99% of the required years → PARTIAL; under 70% → NOT_EVIDENCED.
- **Education.** Required level met or higher, in the stated or an explicitly "related" field → DIRECT. Right level, other field → PARTIAL. Degree in progress or expected → PARTIAL. Lower level → NOT_EVIDENCED.
- **Certificates and licences.** Listed, not expired → DIRECT. Listed as pending, expired, or "equivalent" without detail → PARTIAL when pending or equivalent; CONTRADICTED when expired.
- **Recency.** A skill used only in roles that ended more than 7 years ago, followed by unrelated roles, is PARTIAL.
- **Transferable skills.** Count as PARTIAL only when the CV shows the same function in another setting (for example complaint handling in retail for a service-desk job). Never DIRECT.
- **Do not infer** degrees, years, certificates or skills the CV does not state. Do not use name, gender, age, nationality, photo, religion or any protected characteristic.

## 4. Step 3 — decide the label
Apply the first rule that fits, in this order.

| # | Condition on the core mandatory requirements | Label |
|---|---|---|
| 1 | The CV or the job text is empty, unreadable, cut off before the relevant part, or the job text has no identifiable role, **or** the CV contradicts itself on R1 | **Insufficient Evidence** |
| 2 | R1 is CONTRADICTED, **or** R1 is NOT_EVIDENCED and no other requirement is DIRECT, **or** at least half of the core requirements are NOT_EVIDENCED or CONTRADICTED | **No Fit** |
| 3 | R1 is DIRECT, R2 (if the job has one) is DIRECT, and every other core requirement is DIRECT **or** exactly one of R3–R5 is PARTIAL; none is NOT_EVIDENCED or CONTRADICTED | **Good Fit** |
| 4 | Anything else (R1 DIRECT or PARTIAL; no R1 contradiction; fewer than half of the core requirements missing) | **Potential Fit** |

**Good vs Potential (the hard boundary).** Good Fit needs R1 and R2 DIRECT. If R1 or R2 is only PARTIAL (for example years that cannot be verified), the answer is Potential Fit however strong the other requirements are. Two PARTIAL grades among R3–R5 also give Potential Fit.
**Potential vs No Fit.** A CONTRADICTED or NOT_EVIDENCED requirement other than R1 gives Potential Fit as long as R1 is not contradicted and fewer than half of the core requirements are missing (rule 2); otherwise No Fit.

## 5. Step 4 — handle uncertainty
- If you cannot choose between two **adjacent** labels, record your best label, confidence **Low**, and tick *needs discussion*. Do not switch to Insufficient Evidence to avoid deciding.
- Record true ambiguity (a requirement that can be read two ways) in the notes. Ambiguity is flagged for analysis; it is never converted to another label automatically.
- Separately note whether the problem is an **extraction issue** (text cut, garbled, tables lost) or a **requirement ambiguity** (job wording unclear).

## 6. Review record (one row per reviewer per case)
`pair_id, row_uid, reviewer_id, reviewer_label, R1_grade, R2_grade, R3_grade, R4_grade, R5_grade, evidence_span, reason, confidence (High/Medium/Low), needs_discussion (Yes/No), notes`.
The original label and any model prediction are hidden from reviewers. Reviewers do not see each other's sheets before submitting.

## 7. Worked examples (12). Examples 1–10 are carried over from v1; 11–12 test the Good/Potential boundary.
These are **synthetic**, written for the rubric. They are not TRAIN cases and not human-reviewed data.

| # | Job requirement(s) | CV evidence | Grades | Label |
|---|---|---|---|---|
| 1 | Python mandatory | "Developed data-processing scripts using Python." | R1 DIRECT | **Good Fit** (rule 3) |
| 2 | SQL required; Tableau preferred | "Built SQL reports and queried relational databases." | R1 DIRECT; preferred ignored | **Good Fit** |
| 3 | Current nursing licence mandatory (R4) | Hospital administration experience; no licence information | R1 PARTIAL, R4 NOT_EVIDENCED | **Potential Fit** (rule 4). The v1 label "Insufficient Evidence" is replaced because the CV is complete and simply silent. |
| 4 | Python mandatory | "Currently learning analytics; no professional Python experience." | R1 CONTRADICTED | **No Fit** (rule 2) |
| 5 | Customer communication and issue resolution | "Resolved customer complaints and handled service requests in a retail role." | R1 PARTIAL (transferable) | **Potential Fit** |
| 6 | 3 years of cybersecurity experience (R2) | "Worked on vulnerability assessments"; no dates | R1 DIRECT, R2 PARTIAL | **Potential Fit** (rule 4: R2 not verifiable, so not Good) |
| 7 | Bachelor's in computer science or closely related field (R3) | "Bachelor's degree in Computer Science." | R3 DIRECT | **Good Fit** only if R1 is also DIRECT; otherwise Potential Fit |
| 8 | Active professional certification (R4) | "Certification listed as expired in 2024." | R4 CONTRADICTED | **Potential Fit** if R1 is not contradicted and fewer than half of the core requirements are missing; **No Fit** otherwise |
| 9 | Hands-on Linux administration (R1) | Summary says "Linux administrator"; experience section says "no Linux administration duties" | R1 self-contradictory | **Insufficient Evidence** (rule 1) |
| 10 | Experience with cloud platforms (R1) | CV contains only a name and contact details | CV unreadable for the task | **Insufficient Evidence** (rule 1) |
| 11 | R1 data analysis in Python; R2 2 years; R3 BSc | Python dashboards (DIRECT); 2019–2022 analyst (DIRECT); BSc (DIRECT) | all DIRECT | **Good Fit** |
| 12 | R1 data analysis in Python; R2 2 years; R3 BSc | Only "NumPy, SciPy" in skills list, no project; 2019–2022 analyst; BSc | R1 PARTIAL (skills list only) | **Potential Fit** (boundary: R1 PARTIAL cannot be Good) |

Note on examples 7 and 8: they show a single requirement; the label depends on R1, which the table states.

## 8. Changes from v1 (kept transparent)
1. Added the 4-step procedure and an ordered decision table.
2. Defined DIRECT / PARTIAL / NOT_EVIDENCED / CONTRADICTED.
3. Example 3 changed from Insufficient Evidence to Potential Fit; Insufficient Evidence is now limited to defective documents (v1 left the boundary with "missing evidence" open).
4. Example 8 changed from No Fit to a conditional label: a lapsed credential alone is not decisive unless the role-defining skill is also lacking. This is a rubric choice for the supervisor to confirm.

## 9. Pilot (the pass condition: two reviewers can apply the rubric independently)
- 20 TRAIN cases, fixed seed, original labels hidden (`A3/pilot/A3_pilot_20_ADMIN_KEY.csv` is kept away from reviewers).
- Two human reviewers label independently with this document, then compute `compute_agreement.py` (raw agreement, Cohen's κ, linear-weighted κ, label confusion).
- **Threshold, set before the pilot:** unweighted κ ≥ 0.60 and no more than 4 of 20 disagreements. If not met, discuss the disagreements, edit the decision rules (not the labels), and run a new pilot on 20 different cases.
- Status after the pilot: `PILOT_PASSED` or `PILOT_FAILED_REVISE`. Supervisor approval is a separate step.
