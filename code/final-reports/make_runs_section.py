"""Every single run (seed) of every model, checked against the preset rule.
Writes D_experiments/ALL_RUNS_per_seed_validation.csv and runs_section.md (used by the results and final reports).
usage: python make_runs_section.py <fork_clone> <package_dir>"""
import contextlib, io, runpy, sys
from pathlib import Path
import pandas as pd

here = Path(__file__).parent
fork, pkg = sys.argv[1], Path(sys.argv[2])
sys.argv = ["b", fork, str(pkg), str(here / "_tmp_results.md")]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(here / "build_results.py"))
metrics, MODELS = g["metrics"], g["MODELS"]
B = dict(acc=56.79, mf1=48.61, gfr=36.72, gfp=43.68, pr=29.68)   # M4 E0, mean of 8 seeds
SEEDS = {("M5 + Knowledge + evidence (E2)", "D0"): [42, 43, 44, 45, 46, 47, 49],
         ("M5 + Knowledge + evidence (E2)", "D1"): [42, 43, 44, 45, 47, 48, 49]}
rows = []
for grp, name, lab, dev, cms, note in MODELS:
    seeds = SEEDS.get((name, lab)) if len(SEEDS.get((name, lab), [])) == len(cms) else (list(range(42, 42 + len(cms))) if len(cms) > 1 else [42])
    assert len(seeds) == len(cms), (name, lab)
    for s, cm in zip(seeds, cms):
        m = metrics(cm)
        rows.append(dict(model=name, labels=lab, device="cpu", seed=s, accuracy=m["acc"], macro_f1=m["mf1"],
                         good_fit_recall=m["r"][2], good_fit_precision=m["p"][2], potential_recall=m["r"][1]))
gpu = pkg / "D_experiments/M8_colab_gpu_seeds43-45.csv"
if gpu.exists():
    for _, r in pd.read_csv(gpu).iterrows():
        rows.append(dict(model=f"M8 {r.variant}", labels="D0", device="colab-gpu", seed=int(r.seed), accuracy=r.accuracy, macro_f1=r.macro_f1,
                         good_fit_recall=r.good_fit_recall, good_fit_precision=r.good_fit_precision, potential_recall=r.potential_fit_recall))
d = pd.DataFrame(rows)
d["beats_acc_and_gf_recall"] = (d.accuracy > B["acc"]) & (d.good_fit_recall > B["gfr"])
d["passes_full_rule"] = d.beats_acc_and_gf_recall & (d.good_fit_precision >= B["gfp"] - 2) & (d.macro_f1 >= B["mf1"] - 2) & (d.potential_recall >= B["pr"] - 2)
d.round(2).to_csv(pkg / "D_experiments/ALL_RUNS_per_seed_validation.csv", index=False)

NICE = {"M4 + evidence (E2)": "BERT-Tiny + Knowledge + evidence", "M5 + Knowledge + evidence (E2)": "DistilBERT + Knowledge + evidence",
        "M4 E0": "BERT-Tiny + Knowledge (the baseline itself)"}
def nice(r): return f"{NICE.get(r.model, r.model)}, {r.labels}" + (", GPU" if r.device != "cpu" else "")
P = d[d.passes_full_rule].sort_values(["accuracy"], ascending=False)
N = d[d.beats_acc_and_gf_recall & ~d.passes_full_rule]
own = d[(d.model == "M4 E0") & d.beats_acc_and_gf_recall]
t = ["| Run | Seed | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall |", "|---|---|---|---|---|---|---|",
     f"| *Baseline: BERT-Tiny + Knowledge, D0, mean of 8 seeds* | — | {B['acc']} | {B['mf1']} | {B['gfr']} | {B['gfp']} | {B['pr']} |"]
t += [f"| {nice(r)} | {r.seed} | {r.accuracy:.2f} | {r.macro_f1:.2f} | {r.good_fit_recall:.2f} | {r.good_fit_precision:.2f} | {r.potential_recall:.2f} |" for r in P.itertuples()]
e2 = d[(d.model == "M4 + evidence (E2)") & (d.labels == "D0")]
e0 = d[(d.model == "M4 E0") & (d.labels == "D0")]
pair = e2.set_index("seed").good_fit_recall - e0.set_index("seed").good_fit_recall
s46 = e2[e2.seed == 46].iloc[0]; f46 = e0[e0.seed == 46].iloc[0]
text = f"""### Single runs that pass the rule, and the closest setting
Out of {len(d)} individual runs (every seed of every model), **{len(P)} single runs pass the full selection rule** on their own, and {len(N)} more beat the baseline on accuracy and Good Fit recall but drop more than 2 points on one guard metric. The full list is in D_experiments/ALL_RUNS_per_seed_validation.csv.

{chr(10).join(t)}

- **Why a single run is not enough.** The rule compares the mean of a setting with the baseline mean, because seeds move results a lot: {len(own)} of the baseline's own 8 seeds also beat its mean on accuracy and Good Fit recall (seeds {', '.join(str(s) for s in own.seed)}). Picking the best seed after seeing validation results would overstate what a setting delivers, and no unused test set is left to confirm a single run.
- **The closest setting on average is BERT-Tiny + Knowledge + evidence (D0).** It keeps every guard (Good Fit precision, macro-F1 and Potential recall all within 2 points), raises Good Fit recall by {e2.good_fit_recall.mean() - B['gfr']:.2f} points, and misses only on accuracy, by {B['acc'] - e2.accuracy.mean():.2f} points, less than the seed-to-seed SD of the baseline (1.17). Compared seed by seed, it has higher Good Fit recall on {int((pair > 0).sum())} of 8 seeds. Three of its eight seeds pass the rule on their own.
- **On the frozen seed itself (46),** which was chosen as the seed closest to the baseline mean and not from the evidence results, adding evidence gives accuracy {s46.accuracy:.2f} vs {f46.accuracy:.2f}, Good Fit recall {s46.good_fit_recall:.2f} vs {f46.good_fit_recall:.2f}, Good Fit precision {s46.good_fit_precision:.2f} vs {f46.good_fit_precision:.2f} and Potential recall {s46.potential_recall:.2f} vs {f46.potential_recall:.2f}: better on every metric.
- **Recommendation:** keep the frozen baseline as the rule requires, and treat BERT-Tiny + Knowledge + evidence as the first candidate to confirm on new data or a fresh test set; DistilBERT + Knowledge + evidence (seed 44: Good Fit recall 66.09) is the second.
"""
(here / "runs_section.md").write_text(text)
print(f"{len(d)} runs; pass full rule: {len(P)}; near: {len(N)}")
print(P[["model", "labels", "seed", "accuracy", "good_fit_recall"]].round(2).to_string())
