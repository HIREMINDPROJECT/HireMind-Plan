# DistilBERT at 512 tokens: follow-up to the supervisor's plan

The DistilBERT runs in `FINAL_REPORT.pdf` used 384 input tokens, because 512 did not fit the CPU time budget, while the baseline (BERT-Tiny + Knowledge v1) used 512. This folder removes that limitation: every DistilBERT step of the plan was repeated at the full 512-token budget, on GPU, with 5 epochs.

**Start with `DistilBERT_512_results.pdf`** (4 pages: what was done, all results, what did better and why, conclusion).

| File | Content |
|---|---|
| `DistilBERT_512_results.pdf` | The report |
| `distilbert_512_all_runs.csv` | Every training (22): configuration, seed, best epoch, the five metrics, and the preset-rule check |
| `code/c_stage2_gpu.py` | The training script (the plan's M5 script with GPU, fp16 autocast, a full 16-pair batch per step) |
| `code/run_queue.py` | Runs all 22 trainings on the GPUs of one machine |

**Result in one line.** No DistilBERT configuration meets the preset selection rule on average at 512 tokens either; the closest, Knowledge + evidence on D1, is level with the baseline (accuracy 56.26 vs 56.79, Good Fit recall 35.99 vs 36.72). The conclusion of the final report for DistilBERT is unchanged.

**Rules followed.** Validation split only (1,162 pairs); the test set was never used. The selection rule is the one fixed before the experiments. No CV or job text is stored here.

**To re-run** (one machine with a GPU, e.g. Kaggle T4 x2): clone this repository, then in its root
```
git clone -q https://github.com/Hams770/HireMind-DL.git _dl && git -C _dl checkout -q 1f22f34462c400f884bb0a00f061dbea0bd687d5
git clone -q https://github.com/Hams770/Senior-2.git _s2 && git -C _s2 checkout -q ec28dfc85e618a737cb192cbf959714cfee89aa5
pip install transformers==4.51.3 huggingface-hub==0.30.2 sentencepiece protobuf
python 512_token_followup/code/run_queue.py out_512
```
