# A1 — run commands v2 (23 Nov 2026)

`run_commands.md` (project coordinator, 23 Nov) is kept. This version adds three corrections:
1. It pins the exact commits instead of the moving branch HEAD.
2. It requires CPU.
3. It saves checkpoints, predictions and the full package list.

## A. Commands that produced the historical reference (recovered, not re-run)
GitHub Actions, one CPU runner per seed, workflow `.github/workflows/knowledge.yml` at Senior-2 `a171ea5`:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.51.3 pandas==2.2.3 scikit-learn==1.6.1 huggingface-hub==0.30.2 numpy
git clone -q --depth 1 https://github.com/Hams770/HireMind-DL.git ../HireMind-DL   # HEAD = 1f22f34
HIREMIND_MAX_LEN=512 python evaluation/knowledge/train_knowledge.py --dl ../HireMind-DL --seed $SEED --variant knowledge --knowledge 1
```
Results: Senior-2 branch `knowledge-results/20261117T122609/knowledge_run_knowledge_s{42..49}.json`.

## B. Fresh CPU reproduction (closes the checkpoint and version gaps)
Recommended: run `A1_fresh_cpu_run_v2.ipynb` on a Colab **CPU** runtime. It runs exactly the following:
```bash
git clone https://github.com/Hams770/Senior-2.git && git -C Senior-2 checkout ec28dfc85e618a737cb192cbf959714cfee89aa5
git clone https://github.com/Hams770/HireMind-DL.git && git -C HireMind-DL checkout 1f22f34462c400f884bb0a00f061dbea0bd687d5
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.51.3 pandas==2.2.3 scikit-learn==1.6.1 huggingface-hub==0.30.2 numpy
pip freeze > env/pip_freeze.txt
export HIREMIND_MAX_LEN=512 HIREMIND_RESULTS=$OUT/results
cd Senior-2
for SEED in 42 43 44 45 46 47 48 49; do
  python evaluation/knowledge/train_knowledge.py --dl ../HireMind-DL --seed $SEED \
    --variant knowledge --knowledge 1 --device cpu \
    --save-weights $OUT/checkpoints --save-preds
done
python A1_compare_runs.py --run $OUT/results --run-preds $OUT/results --weights $OUT/checkpoints \
  --ref ref/20261117T122609 --ref-preds ref/20261117T124714 --out $OUT/comparison
```
`ec28dfc` has the same training code as the reference commit `a171ea5`; it only adds `--save-preds`.
It reproduced 7 of 8 reference seeds exactly on 17 Nov.

## C. Do not
- Do not use `--device auto` or `cuda`: GPU results differ materially from this CPU baseline.
- Do not use the `m4_knowledge` package's `--variant final` or `control` arms as A1.
  - `final` is computationally the same arm under a different name, but its outputs would not match the checker naming.
  - `control` has no knowledge line.
- Do not read the test split; do not change labels or the split manifest.

## D. How the fresh run is judged (fixed before running)
- Hard checks (any failure = run invalid):
  - 8 seeds present
  - each confusion matrix sums to 1,162, with true classes 579/294/289
  - logged metrics equal their recomputation from the matrix
  - knowledge=1, vocab v1, 3 epochs, 939 updates
  - checkpoint SHA-256 equals the logged value
- Per-seed verdict vs reference:
  - IDENTICAL
  - FLOAT_NOISE: at most 4 cases moved, same best step
  - DIVERGED: must be explained, as was done for seed 45 in `A1_numerical_differences_v2.md`
- The fresh run is reported as it comes out; it is not rerun until it matches.
- If its 8-seed means differ from 56.79% accuracy and 36.85% Good Fit recall, report the delta next to the reference seed SD (1.17 and 11.24 pts).
