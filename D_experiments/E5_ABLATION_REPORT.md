# E5 — M8 (DistilBERT hybrid) minus one component per run   STATUS: DONE · single seed (42) · direction only
Run 45257498571. All variants: distilbert-base-uncased, 512 tokens, AdamW lr 3e-5, batch 16, 3 epochs, best-validation-macro-F1 epoch, D0 labels, validation only, CPU.

| Variant | Evidence input | Knowledge line | Structured features | Accuracy | Macro-F1 | GF recall | GF precision | Potential recall | Best epoch |
|---|---|---|---|---|---|---|---|---|---|
| full | yes | yes | yes | 53.96 | 45.48 | 32.87 | 42.22 | 25.17 | 2 (time guard stopped after 2) |
| minus_evidence | no | yes | yes | 52.32 | 42.95 | 24.22 | 45.16 | 26.19 | 3 |
| minus_knowledge | yes | no | yes | 53.01 | 44.59 | 38.41 | 36.75 | 20.41 | 2 |
| minus_struct | yes | yes | no | 54.13 | 49.02 | 48.79 | 41.23 | 30.61 | 3 |

## Reading (one seed; M4's seed-to-seed SD is 1.2 points accuracy and 11 points GF recall, so nothing here is claimed as a significant effect)
- Removing evidence selection gives the lowest accuracy and GF recall → in this run evidence selection helped the hybrid.
- Removing the Knowledge line raises GF recall but lowers GF precision and Potential recall → a trade-off, not a gain.
- Removing the structured features gives the best variant on every metric except GF precision. The full model was stopped after 2 epochs by the CPU time guard while minus_struct trained 3, so part of this difference may be training length, not the features.
- No variant meets the preset selection rule (all are below M4's accuracy of 56.79).

## Update (24 Nov 2026): seeds 43–45 on a Colab GPU (separate setting)
The same script was re-run on a Colab T4 GPU for seeds 43–45 (file M8_colab_gpu_seeds43-45.csv). GPU results are not averaged with the CPU seed 42.

| Variant | Seeds | Accuracy | Macro-F1 | GF recall | GF precision | Potential recall |
|---|---|---|---|---|---|---|
| full | 3 | 53.07 ± 2.04 | 43.20 ± 3.22 | 28.14 ± 14.58 | 43.86 ± 8.37 | 23.70 ± 4.25 |
| minus_knowledge | 3 | 54.10 ± 2.94 | 45.81 ± 2.75 | 44.64 ± 4.99 | 42.99 ± 6.65 | 18.82 ± 3.41 |
| minus_struct | 3 | 52.93 ± 0.82 | 42.98 ± 1.57 | 27.68 ± 7.64 | 39.28 ± 3.05 | 22.34 ± 2.32 |
| minus_evidence | 0 | did not complete on the GPU (all three seeds stopped at start-up) | | | | |

- The seed-42 lead of minus_struct (GF recall 48.79) did not repeat: 27.68 on average over seeds 43–45. This is why single-seed ablations are read as direction only.
- Without the Knowledge line GF recall is highest and most stable (44.64 ± 4.99), but Potential recall falls to 18.82, so it breaks the rule (no drop over 2 points in Potential recall) and also stays below M4's accuracy.
- No variant meets the preset rule.
