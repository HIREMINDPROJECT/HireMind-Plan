"""Builds FINAL_RESULTS_REPORT.md: every model of the plan with mean ± SD over seeds, class-wise metrics, confusion matrices
(mean counts per seed), and the BERT-Tiny vs DistilBERT (+ Knowledge v1) comparison through the plan's steps.
Reads only scores and confusion matrices (no text). Run from anywhere: python build_results.py <hiremind-dl clone> <final dir> <out.md>"""
import csv, json, subprocess, sys, statistics as st
from pathlib import Path
REPO, FINAL, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PKG = FINAL
CL = ["No Fit", "Potential Fit", "Good Fit"]

def git_show(ref, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{path}"], capture_output=True, text=True).stdout

def git_ls(ref):
    return subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", ref], capture_output=True, text=True).stdout.split()

def branches(prefix):
    out = subprocess.run(["git", "-C", str(REPO), "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin"], capture_output=True, text=True).stdout.split()
    return [b for b in out if b.startswith("origin/" + prefix)]

def metrics(cm):
    tp = [cm[i][i] for i in range(3)]; col = [sum(cm[r][c] for r in range(3)) for c in range(3)]; row = [sum(cm[r]) for r in range(3)]
    p = [100 * tp[i] / col[i] if col[i] else 0.0 for i in range(3)]; r = [100 * tp[i] / row[i] if row[i] else 0.0 for i in range(3)]
    f = [2 * p[i] * r[i] / (p[i] + r[i]) if p[i] + r[i] else 0.0 for i in range(3)]
    return dict(acc=100 * sum(tp) / sum(row), mf1=sum(f) / 3, p=p, r=r, f=f)

MODELS = []  # (group, name, labels, device, [cm per seed], note)
def add(group, name, labels, device, cms, note=""):
    if cms: MODELS.append((group, name, labels, device, cms, note))

def rows_csv(path):
    return list(csv.DictReader(open(path)))

# ---- classical and structured (deterministic or 8 seeds)
for r in rows_csv(FINAL / "C_models/C_stage1_m1_m2_validation.csv"):
    if r["role"] == "primary":
        add("Classical", r["model"].split("_")[0] + " " + ("LogReg TF-IDF" if r["model"].startswith("M1") else "LinearSVC TF-IDF"), "D0", "cpu", [json.loads(r["confusion_true_rows_pred_cols"])], "deterministic")
m3 = [json.loads(r["confusion_true_rows_pred_cols"]) for r in rows_csv(FINAL / "C_models/C_stage2_m3_m6_validation.csv") if r["model"].startswith("M3_XGBoost_s") and r["model"][-2:].isdigit()]
add("Structured", "M3 XGBoost (B1 features)", "D0", "cpu", m3)
for r in rows_csv(FINAL / "C_models/C_stage2_m3_m6_validation.csv"):
    if r["model"] == "M6_MiniLM_LogReg_C0.1":
        add("Semantic", "M6 MiniLM + LogReg", "D0", "cpu", [json.loads(r["confusion_true_rows_pred_cols"])], "deterministic; not part of the team decision")
# ---- the same classical / structured models on the reviewed labels D1 (E4)
_e4 = "origin/e4d1-results-45299027830"; _p = "e4d1_results/run_45299027830/"
for r in csv.DictReader(git_show(_e4, _p + "E4_D1_stage1_m1_m2_validation.csv").splitlines()):
    if r["role"] == "primary":
        add("Classical", r["model"].split("_")[0] + " " + ("LogReg TF-IDF" if r["model"].startswith("M1") else "LinearSVC TF-IDF"), "D1", "cpu", [json.loads(r["confusion_true_rows_pred_cols"])], "deterministic")
add("Structured", "M3 XGBoost (B1 features)", "D1", "cpu", [json.loads(r["confusion_true_rows_pred_cols"]) for r in csv.DictReader(git_show(_e4, _p + "E4_D1_stage2_m3_validation.csv").splitlines()) if r["model"][-2:].isdigit()])
# ---- BERT-Tiny + Knowledge v1 through the plan
d0d1 = rows_csv(FINAL / "A6_D0_vs_D1/D0_vs_D1.csv")
for lab, step in (("D0", "E0"), ("D1", "E1")):
    add("BERT-Tiny + Knowledge v1", f"M4 {step}", lab, "cpu", [json.loads(r["confusion_true_rows_pred_cols"]) for r in d0d1 if r["labels"] == lab and r["seed"].isdigit()])
e2d0 = []
for b in branches("e2-results-"):
    for f in git_ls(b):
        if f.endswith(".json") and "/E2_run_s" in f: e2d0.append(json.loads(git_show(b, f))["confusion"])
add("BERT-Tiny + Knowledge v1", "M4 + evidence (E2)", "D0", "cpu", e2d0)
add("BERT-Tiny + Knowledge v1", "M4 + evidence (E2)", "D1", "cpu", [json.loads(r["E2_D1_confusion"]) for r in rows_csv(FINAL / "D_experiments/E2_D1_per_seed_vs_E1_E0.csv") if r["seed"].isdigit()])

# ---- DistilBERT (+ Knowledge v1) through the plan, CPU, from the results branches
def m5_runs(prefix):
    found = {}
    for b in branches(prefix):
        for f in git_ls(b):
            if f.endswith("C_stage2_m5_validation.csv"):
                r = list(csv.DictReader(git_show(b, f).splitlines()))[0]
                found[r["model"]] = json.loads(r["confusion_true_rows_pred_cols"])
    return found
plain = {}
for r in rows_csv(FINAL / "C_models/C_stage2_m5_validation.csv"): plain[r["model"]] = json.loads(r["confusion_true_rows_pred_cols"])
add("DistilBERT", "M5 plain (CV, job)", "D0", "cpu", list(plain.values()), "plain-input seeds 43-45 were stopped on purpose")
k = m5_runs("c-m5-knowledge-results-45360641467")
add("DistilBERT + Knowledge v1", "M5 + Knowledge (E0)", "D0", "cpu", list(k.values()))
pl = {**m5_runs("c-m5-plan-results-45361800058"), **m5_runs("c-m5-plan-results-45365434364")}  # run 45361593555 failed before training (missing labels file) and is ignored
for suffix, name, lab in (("_knowledge_D1", "M5 + Knowledge (E1)", "D1"), ("_knowledge_evidence", "M5 + Knowledge + evidence (E2)", "D0"), ("_knowledge_evidence_D1", "M5 + Knowledge + evidence (E2)", "D1")):
    add("DistilBERT + Knowledge v1", name, lab, "cpu", [v for m, v in pl.items() if m.endswith(suffix)])
# ---- M8 (E5) seed 42 CPU + any later seeds
for v in ("full", "minus_evidence", "minus_knowledge", "minus_struct"):
    cms = []
    for b in branches(f"m8-results-45257498571-{v}") + branches("m8-seeds-results-"):
        if v not in b: continue
        for f in git_ls(b):
            if f.endswith(".json") and "/M8_" in f: cms.append(json.loads(git_show(b, f))["confusion"])
    add("DistilBERT hybrid (M8, E5)", f"M8 {v}", "D0", "cpu", cms)
colab = Path(sys.argv[4]) if len(sys.argv) > 4 else None
if colab and colab.exists():  # optional: Colab GPU M8 summary rows (one CSV with columns variant, seed, accuracy, ... and no confusion)
    pass

# ---- table helpers
def mean_sd(vals):
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else None)
def fmt(ms):
    m, s = ms
    return f"{m:.2f}" + (f" ± {s:.2f}" if s is not None else "")
M4 = None
lines = ["# HireMind — final results (validation, 24 Nov 2026)", "",
         "All numbers are on the frozen VALIDATION split (1,162 pairs: 579 No Fit, 294 Potential Fit, 289 Good Fit). The test set was never used. "
         "Mean ± sample SD over seeds; a single seed has no SD. Selection rule (fixed before the experiments): beat M4 E0 on accuracy AND Good Fit recall, "
         "with no drop of more than 2.0 points in Good Fit precision, macro-F1 or Potential Fit recall.", "",
         "## 1. All models", "",
         "| Group | Model | Train labels | Seeds | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall | Meets rule |",
         "|---|---|---|---|---|---|---|---|---|---|"]
summ = []
for g, n, lab, dev, cms, note in MODELS:
    ms = [metrics(c) for c in cms]
    s = dict(acc=mean_sd([m["acc"] for m in ms]), mf1=mean_sd([m["mf1"] for m in ms]), gfr=mean_sd([m["r"][2] for m in ms]),
             gfp=mean_sd([m["p"][2] for m in ms]), pr=mean_sd([m["r"][1] for m in ms]))
    summ.append((g, n, lab, dev, cms, note, ms, s))
    if n == "M4 E0": M4 = s
for g, n, lab, dev, cms, note, ms, s in summ:
    ok = (s["acc"][0] > M4["acc"][0] and s["gfr"][0] > M4["gfr"][0] and s["gfp"][0] >= M4["gfp"][0] - 2 and s["mf1"][0] >= M4["mf1"][0] - 2 and s["pr"][0] >= M4["pr"][0] - 2)
    lines.append(f"| {g} | {n} | {lab} | {len(cms)} | {fmt(s['acc'])} | {fmt(s['mf1'])} | {fmt(s['gfr'])} | {fmt(s['gfp'])} | {fmt(s['pr'])} | " + ("baseline" if n == "M4 E0" else ("**yes**" if ok else "no")) + " |")
lines += ["", "## 2. Class-wise metrics (mean over seeds)", "",
          "| Model | Labels | No Fit P / R / F1 | Potential Fit P / R / F1 | Good Fit P / R / F1 |", "|---|---|---|---|---|"]
for g, n, lab, dev, cms, note, ms, s in summ:
    cell = lambda i: " / ".join(f"{st.mean([m[k][i] for m in ms]):.1f}" for k in ("p", "r", "f"))
    lines.append(f"| {n} | {lab} | {cell(0)} | {cell(1)} | {cell(2)} |")
lines += ["", "## 3. Confusion matrices (rows = true class, columns = predicted: No / Potential / Good; mean count per seed)", ""]
for g, n, lab, dev, cms, note, ms, s in summ:
    avg = [[sum(c[r][k] for c in cms) / len(cms) for k in range(3)] for r in range(3)]
    lines.append(f"**{n} ({lab}, {len(cms)} seed{'s' if len(cms) > 1 else ''})**  ")
    lines.append("| true \\ predicted | No Fit | Potential Fit | Good Fit |\n|---|---|---|---|")
    for r in range(3): lines.append(f"| {CL[r]} | " + " | ".join(f"{x:.1f}" if len(cms) > 1 else f"{x:.0f}" for x in avg[r]) + " |")
    lines.append("")
# ---- Colab GPU runs (separate setting; never averaged with CPU runs)
import pandas as _pd
_gpu = []
_m8 = PKG / "D_experiments" / "M8_colab_gpu_seeds43-45.csv"
if _m8.exists():
    d = _pd.read_csv(_m8)
    for v, g in d.groupby("variant", sort=False):
        _gpu.append((f"M8 {v} (hybrid)", "D0", g))
_ep = PKG / "D_experiments" / "DistilBERT_epochs_gpu_all_runs.csv"
if _ep.exists():
    d = _pd.read_csv(_ep).rename(columns={"potential_recall": "potential_fit_recall"})
    for e, g in d.groupby("epochs"):
        _gpu.append((f"M5 + Knowledge + evidence, {int(e)} epochs", "D0", g))
if _gpu:
    lines += ["## 3b. Colab GPU runs (a separate setting)", "",
              "Same scripts and inputs, run on a Colab T4 GPU. GPU results can differ slightly from CPU results for the same seed, so they are shown apart and never averaged with the CPU rows.", "",
              "| Model | Labels | Seeds | Accuracy | Macro-F1 | Good Fit recall | Good Fit precision | Potential recall | Meets rule |", "|---|---|---|---|---|---|---|---|---|"]
    for name, lab, g in _gpu:
        cols = ["accuracy", "macro_f1", "good_fit_recall", "good_fit_precision", "potential_fit_recall"]
        ms = {c: (g[c].mean(), g[c].std() if len(g) > 1 else None) for c in cols}
        ok = (ms["accuracy"][0] > 56.79 and ms["good_fit_recall"][0] > 36.72 and ms["good_fit_precision"][0] >= 43.68 - 2 and ms["macro_f1"][0] >= 48.61 - 2 and ms["potential_fit_recall"][0] >= 29.68 - 2)
        seeds = ", ".join(str(int(x)) for x in g.seed)
        lines.append(f"| {name} | {lab} | {len(g)} ({seeds}) | " + " | ".join(fmt(ms[c]) for c in cols) + f" | {'yes' if ok else 'no'} |")
    if (PKG / "D_experiments" / "M8_colab_gpu_seeds43-45.csv").exists():
        lines += ["", "M8 minus_evidence did not complete on the GPU (all three seeds stopped at start-up without a result); its only result is the CPU seed 42 row in section 1."]
    lines.append("")
_rs = Path(__file__).parent / "runs_section.md"
if _rs.exists():
    lines += ["## 3c. " + _rs.read_text().split("\n", 1)[0].lstrip("# ").strip(), ""] + _rs.read_text().split("\n", 1)[1].strip().splitlines() + [""]
lines += ["## 4. Notes", "",
          "- Seeds: BERT-Tiny experiments use seeds 42–49; DistilBERT experiments use the seeds listed (CPU cost). Different seed counts are shown in the table; SD is the sample SD.",
          "- DistilBERT runs use max length 384 (512 was projected over the CPU time budget); M8 uses 512. BERT-Tiny uses 512.",
          "- Same CVs in TRAIN and VALIDATION: job groups are disjoint, but 1,157 of the 1,162 validation CVs also appear in TRAIN; all validation scores are probably optimistic for unseen CVs.",
          "- A team member also ran DistilBERT on a Colab GPU on 22 Nov (lr 1e-4, 512 tokens, 3 seeds; plain: accuracy 48.48, macro-F1 36.81, Good Fit recall 22.95; + Knowledge v1: 47.90 / 39.47 / 16.61). "
          "Those runs use a different device and learning rate, so they are reported here as a separate setting and are not averaged with the CPU runs.",
          "- **M8 more seeds, Colab GPU (separate setting, 24 Nov).** The M8 variants were re-run on a Colab GPU with the same script, seeds 43–45, D0 (mean ± SD over 3 seeds): "
          "full accuracy 53.07, Good Fit recall 28.14 ± 14.6; minus_knowledge 54.10, Good Fit recall 44.64 ± 5.0; minus_struct 52.93, Good Fit recall 27.68 ± 7.6. "
          "None reaches the baseline accuracy (56.79). The single-seed CPU result of minus_struct (seed 42, Good Fit recall 48.79) did not repeat on other seeds, so the seed-42 M8 rows above should not be read as stable.",
          "- M4 E1 and M4 + evidence (E2) on D1 show the same mean Good Fit recall (24.91) by coincidence: the per-seed counts differ (E1: 74, 93, 59, 77, 58, 90, 52, 73; E2: 74, 80, 63, 60, 82, 91, 59, 67) but both sum to 576 of 8 × 289.",
          "- Per-job-group error analysis of the frozen model: D_error_analysis/ (error_pretags_by_job_group.csv).",
          "- 3- vs 5-epoch test (section 3b, Colab GPU, seeds 42–44): per-run results in D_experiments/DistilBERT_epochs_gpu_all_runs.csv and the 3-seed ensembles in D_experiments/DistilBERT_epochs_gpu_ensemble.csv. "
          "The 5-epoch mean beats the baseline on accuracy and Good Fit recall but fails the Potential recall guard; the 5-epoch ensemble (accuracy 60.15, Good Fit recall 58.13, Potential recall 19.39) fails the same guard."]
OUT.write_text("\n".join(lines) + "\n")
print("written", OUT, "models:", len(summ))
for g, n, lab, dev, cms, note, ms, s in summ: print(f"{n:38s} {lab} n={len(cms)} acc={s['acc'][0]:.2f} gfr={s['gfr'][0]:.2f}")
