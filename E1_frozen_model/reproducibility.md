# Reproducibility — frozen model (E1)   STATUS: FINAL (24 Nov 2026)
Goal: another student can run predictions and reproduce the validation metrics of the frozen model.

## What you need
1. Code: `HireMind_code_bundle.zip` in this package (all scripts and workflow files used for every result). The team's training code is the pinned Senior-2 commit ec28dfc85e618a737cb192cbf959714cfee89aa5 (`evaluation/knowledge/train_knowledge.py`) and the HireMind-DL data loader at commit 1f22f34462c400f884bb0a00f061dbea0bd687d5 (team repositories).
2. Data: downloaded automatically from Hugging Face `cnamuangtoun/resume-job-description-fit` @08978e21…; the loader checks train.csv SHA-256 b86fa488… and splits.csv SHA-256 2ff3926c….
3. Weights: `a1_knowledge_M4_seed46_weights.pt` (SHA-256 0953ce89…, 17,558,120 bytes), kept with the project's model files (see WEIGHTS_LOCATION.md).
4. Environment: Python 3.12, CPU, packages as in model_config.yaml (pip freeze files are in each result folder).

## Reproduce the validation metrics (CPU, ~10 minutes per seed)
```
export HIREMIND_MAX_LEN=512
python evaluation/knowledge/train_knowledge.py --dl <HireMind-DL checkout> --seed 46 --variant knowledge --knowledge 1 --device cpu --save-preds --save-weights out/
```
Expected (validation, D0): accuracy 56.97, macro-F1 48.80, Good Fit recall 38.41, confusion [[471,40,68],[140,80,74],[87,91,111]]; checkpoint SHA-256 0953ce89…. Use CPU: GPU runs of the same seed give different numbers.

## Verified so far
- The 8-seed CPU run was reproduced twice with identical metrics (A1 fresh run and E0); all 8 checkpoint hashes match the A1 inventory.
- 7 of 8 seeds match the original team report exactly; seed 45 differs for a documented reason (A1_numerical_differences_v2.md).
- **Independent re-use of the frozen weights (24 Nov 2026):** on a fresh Google Colab machine (CPU), the published seed-46 weights were downloaded from the weights/ folder of HIREMINDPROJECT/HireMind-Plan (SHA-256 check passed), the validation split was rebuilt from the pinned dataset and manifest, the Knowledge v1 line was added with the pinned team code, and the model was scored: **accuracy 56.97, identical to the frozen result.** Notebook: HireMind_human_review_sheets_Colab.ipynb (cells 1–3).

## Load and score the frozen model (no training)
```
git clone https://github.com/Hams770/HireMind-DL.git _dl && git -C _dl checkout 1f22f34462c400f884bb0a00f061dbea0bd687d5
git clone https://github.com/Hams770/Senior-2.git _s2 && git -C _s2 checkout ec28dfc85e618a737cb192cbf959714cfee89aa5
# weights: the weights/ folder of HIREMINDPROJECT/HireMind-Plan, file a1_knowledge_M4_seed46_weights.pt, SHA-256 0953ce892ee0f075d37c1e6b4ff0bceb02a7932fddbca0ee5954f5e17fcb0c8e
```
```python
m = build_model("M0", load_config(), 46); m.load_state_dict(torch.load("a1_knowledge_M4_seed46_weights.pt", map_location="cpu"))
# input: CV + (knowledge_line(features(cv, job)) + "\n" + job), 512 tokens, longest_first -> evaluate(m, ds, "cpu", 64)
```

## Known non-reproducible parts
- The original November run did not pin torch/numpy; the fresh runs record them (torch 2.15.0+cpu, numpy 2.5.5).
