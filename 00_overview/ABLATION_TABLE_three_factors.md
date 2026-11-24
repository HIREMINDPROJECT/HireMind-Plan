# Ablation table — labels vs evidence representation vs semantic matching vs classifier (validation, D0 labels unless stated)
Plan question: which factor explains and improves the Good Fit problem? Reference: M4 BERT-Tiny + Knowledge v1, D0 labels, mean of 8 seeds: accuracy 56.79, macro-F1 48.61, Good Fit recall 36.72, GF precision 43.68, Potential recall 29.68.

| Factor | Comparison | Seeds | Δ accuracy | Δ macro-F1 | Δ GF recall | Δ GF precision | Δ Potential recall | Verdict |
|---|---|---|---|---|---|---|---|---|
| Label cleaning | M4 D1 vs M4 D0 (E1) | 8 paired | −0.67 | −1.57 | **−11.81** (7/8 seeds lower) | +4.97 | +5.44 | Changes the class trade-off; no gain under the rule. Validation labels are unaudited D0. |
| Evidence selection | M4 + B4 input vs M4, D0 (E2) | 8 paired | −0.48 | +0.08 | +4.80 (SD 16.3) | −0.69 | −1.92 | Inside noise |
| Evidence selection | same, D1 labels (E2 vs E1) | 8 paired | 0.00 | +0.54 | 0.00 | +2.31 | +3.15 | Inside noise |
| Evidence selection | M8 full vs minus_evidence (E5) | 1 | +1.64 | +2.53 | +8.65 | −2.94 | −1.02 | Direction only |
| Semantic matching | XGBoost lexical+semantic vs lexical (E3) | 8 (fixed features) | −0.30 | −0.64 | +1.64 | +9.54 | −5.79 | Precision/recall trade-off, no overall gain |
| Classifier | M8 minus_struct (DistilBERT) vs M4 | 1 vs 8 | −2.66 | +0.41 | +12.07 | −2.45 | +0.93 | Higher GF recall at lower accuracy; single seed |
| Classifier | M5 DistilBERT plain vs M4 | 1 vs 8 | −6.19 | −9.29 | −24.26 | −11.25 | +1.27 | Worse |
| Classifier | M3 XGBoost (B1 features) vs M4 | 8 vs 8 | −1.96 | −5.12 | −22.49 | +1.93 | +6.59 | Worse on GF recall |

## Answer to the plan's question (honest, negative result)
- No single factor improves both accuracy and Good Fit recall over M4 under the preset rule.
- The largest effect on Good Fit recall comes from the **label definition**: the reviewed labels (D1) move many TRAIN pairs out of Good Fit, and the model then predicts Good Fit far less often. Because the validation labels were not reviewed, this experiment cannot say whether D1 is "more correct"; it shows the Good Fit problem is strongly tied to how Good Fit is labelled.
- Evidence selection and semantic matching change the precision/recall balance by small amounts inside the seed noise.
- A larger encoder with evidence input (M8 minus_struct) reaches the highest Good Fit recall seen (48.79) but loses accuracy, on one seed: a trade-off to report, not a win.
