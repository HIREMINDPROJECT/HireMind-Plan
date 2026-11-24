# E4 robustness checks: are the differences real? (validation, 24 Nov 2026)

These checks test whether the BERT-Tiny + Knowledge experiments of the plan differ from the baseline (M4 E0) by more than seed noise, and whether two common tricks would change the conclusion. They use only saved scores and predictions; the test set is not used. DistilBERT + Knowledge results are added to the final results report when their runs finish.

## 1. Paired comparison by seed (same seeds 42–49 in both runs)
Difference = experiment minus M4 E0 for the same seed. 95% CI from the t-distribution over 8 paired seeds; p from the Wilcoxon signed-rank test (8 pairs, so p cannot go below about 0.008).

| Experiment | Metric | Mean difference [95% CI] | Seeds better | p |
|---|---|---|---|---|
| **E1: D1 labels** | Accuracy | −0.67 [−1.82, +0.48] | 3/8 | 0.250 |
| | Good Fit recall | **−11.81 [−19.27, −4.34]** | 1/8 | **0.023** |
| | Good Fit precision | +4.97 [−0.46, +10.39] | 5/8 | 0.148 |
| | Macro-F1 | −1.57 [−3.39, +0.25] | 1/8 | 0.078 |
| | Potential recall | +5.44 [+0.58, +10.30] | 6/8 | 0.055 |
| **E2: evidence input, D0** | Accuracy | −0.48 [−2.22, +1.25] | 3/8 | 0.641 |
| | Good Fit recall | +4.80 [−8.78, +18.39] | 6/8 | 0.383 |
| | Good Fit precision | −0.69 [−5.71, +4.32] | 5/8 | 1.000 |
| | Macro-F1 | +0.08 [−3.10, +3.26] | 4/8 | 0.945 |
| | Potential recall | −1.91 [−8.22, +4.39] | 2/8 | 0.461 |
| **E2: evidence input, D1** | Accuracy | −0.67 [−1.41, +0.07] | 2/8 | 0.047 |
| | Good Fit recall | **−11.81 [−19.85, −3.77]** | 1/8 | **0.016** |
| | Good Fit precision | **+7.28 [+2.12, +12.44]** | 7/8 | **0.023** |
| | Macro-F1 | −1.03 [−2.79, +0.72] | 2/8 | 0.250 |
| | Potential recall | **+8.59 [+0.96, +16.22]** | 8/8 | **0.008** |

**Reading.** The label change (D1) has a clear, repeatable effect: it trades Good Fit recall (about −12 points) for Good Fit precision and Potential Fit recall. Evidence selection on D0 moves Good Fit recall up on average (+4.8, 6 of 8 seeds) but the interval includes zero, so it is not distinguishable from seed noise. No result supports a claim of improvement under the preset rule.

## 2. Would averaging the 8 seeds (an ensemble) change the conclusion?
Mean of the 8 models' class probabilities, then the most probable class (checked after the fact; reported as a sensitivity check, not as a new candidate chosen on validation).

| Ensemble of 8 seeds | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall | Meets rule |
|---|---|---|---|---|---|---|
| M4 + evidence (E2), D0 | 56.37 | 47.58 | 35.99 | 41.94 | 25.51 | no |
| M4 + evidence (E2), D1 | 58.00 | 49.40 | 25.95 | 55.97 | 38.78 | no |

Averaging raises accuracy and precision but not Good Fit recall, so the conclusion does not change.

## 3. Can a Good Fit threshold buy recall? (HR operating point)
For E2 (D0), Good Fit's probability was multiplied by a factor chosen to maximise macro-F1 on half of the validation job groups and applied to the other half (2-fold by job group, repeated 20 times, so the factor is never chosen on the pairs it is scored on).

| E2 D0, 8 seeds (mean) | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall |
|---|---|---|---|---|---|
| Plain decision (most probable class) | 56.30 | 48.69 | 41.52 | 42.99 | 27.76 |
| Group-cross-validated Good Fit factor | 55.97 | 48.19 | 47.93 | 41.62 | 23.05 |

A threshold raises Good Fit recall by about 6 points but costs accuracy and Potential Fit recall, which the rule does not allow. It is still useful for HR: the interface can offer a "high recall" view (flag more possible Good Fit candidates for human review) without changing the model, because HR makes the final decision.

## 4. Conclusion
- Labels (D1) are the factor with a statistically clear effect, and it moves the balance between classes rather than improving the model overall.
- Evidence selection and semantic matching change Good Fit recall within seed noise.
- Ensembling and thresholds do not turn any candidate into a win under the preset rule.
