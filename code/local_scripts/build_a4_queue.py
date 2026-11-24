"""A4: rebuild the 400-case MIXED queue (random 150 + Cleanlab 150 + hard Good/Potential boundary 100). TRAIN only, fixed seed.
Usage: python build_a4_queue.py A4_oof_cleanlab_scores_lexical.csv OUTDIR
Input = IDs/scores only (no text). Output: sampling manifest (ADMIN), reviewer case list (no labels, no strata), templates, coverage report."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
SEED = 1123; N_RANDOM, N_CLEAN, N_BOUND = 150, 150, 100; CAP_CLEAN, CAP_BOUND = 4, 3
s = pd.read_csv(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
assert len(s) == 4993 and s.row_uid.is_unique
rng = np.random.default_rng(SEED); chosen = {}
# 1 RANDOM: proportional to the original class mix, simple random within class
share = s.original_label.value_counts(normalize=True)
quota = (share * N_RANDOM).round().astype(int); quota.iloc[0] += N_RANDOM - quota.sum()
for lab, n in quota.items():
    ids = s[s.original_label == lab].row_uid.to_numpy(); 
    for u in rng.choice(ids, size=n, replace=False): chosen[u] = "RANDOM"
# 2 CLEANLAB: lowest label-quality score among flagged, at most CAP_CLEAN per job group, not already chosen
cl = s[s.cleanlab_flag & ~s.row_uid.isin(chosen)].sort_values(["label_quality_score", "row_uid"]); cnt = {}; k = 0
for _, r in cl.iterrows():
    if cnt.get(r.job_group, 0) >= CAP_CLEAN: continue
    chosen[r.row_uid] = "CLEANLAB"; cnt[r.job_group] = cnt.get(r.job_group, 0) + 1; k += 1
    if k == N_CLEAN: break
# 3 BOUNDARY: original Good/Potential where the out-of-fold model also puts Good/Potential on top, smallest |p_good - p_potential|; 50/50 by original label; cap per group
cnt = {}
for lab, n in [("Good Fit", N_BOUND // 2), ("Potential Fit", N_BOUND // 2)]:
    b = s[s.boundary_candidate & (s.original_label == lab) & ~s.row_uid.isin(chosen)].sort_values(["gp_margin", "row_uid"]); k = 0
    for _, r in b.iterrows():
        if cnt.get(r.job_group, 0) >= CAP_BOUND: continue
        chosen[r.row_uid] = "BOUNDARY"; cnt[r.job_group] = cnt.get(r.job_group, 0) + 1; k += 1
        if k == n: break
    assert k == n, f"not enough boundary candidates for {lab}"
m = s[s.row_uid.isin(chosen)].copy(); m["stratum"] = m.row_uid.map(chosen)
assert len(m) == 400 and m.stratum.value_counts().to_dict() == {"RANDOM": 150, "CLEANLAB": 150, "BOUNDARY": 100}
m = m.sample(frac=1, random_state=SEED).reset_index(drop=True); m.insert(0, "case_no", range(1, 401))
m["pair_id"] = m.cv_id + "_" + m.job_id; m["sampling_seed"] = SEED
m.rename(columns={"original_label": "ORIGINAL_LABEL_ADMIN_ONLY"}).to_csv(out / "A4_sampling_manifest_ADMIN_ONLY.csv", index=False)
m[["case_no", "row_uid", "pair_id"]].to_csv(out / "A4_reviewer_case_list.csv", index=False)
cols = ["case_no", "row_uid", "pair_id", "reviewer_id", "reviewer_label", "R1_grade", "R2_grade", "R3_grade", "R4_grade", "R5_grade", "evidence_span", "reason", "confidence", "needs_discussion", "notes"]
t = m[["case_no", "row_uid", "pair_id"]].copy()
for c in cols[3:]: t[c] = ""
for r in ("R1", "R2"): t[cols].to_csv(out / f"A4_reviewer_sheet_{r}_TEMPLATE.csv", index=False)
rl = m[["case_no", "pair_id", "row_uid", "original_label", "stratum", "cleanlab_flag"]].copy()
for c in ["reviewer1_label", "reviewer2_label", "adjudicated_label", "evidence_span", "reason", "confidence", "decision"]: rl[c] = ""
rl.to_csv(out / "reviewed_labels_TEMPLATE_ADMIN_ONLY.csv", index=False)  # required columns of the plan + stratum/flag for the analysis
# coverage report
G = s.job_group.nunique(); cov = {}
for st, g in m.groupby("stratum"): cov[st] = dict(cases=len(g), by_original_label=g.original_label.value_counts().to_dict(), job_groups=int(g.job_group.nunique()), max_per_group=int(g.job_group.value_counts().max()))
rep = dict(seed=SEED, strata=cov, total_cases=len(m), job_groups_covered=int(m.job_group.nunique()), job_groups_in_train=int(G), distinct_cvs=int(m.cv_id.nunique()),
           original_label_mix=m.original_label.value_counts().to_dict(), train_label_mix=s.original_label.value_counts().to_dict(),
           largest_group_share_of_queue=round(float(m.job_group.value_counts().iloc[0] / len(m)), 4), caps=dict(cleanlab_per_group=CAP_CLEAN, boundary_per_group=CAP_BOUND),
           cleanlab_flagged_in_queue=int(m.cleanlab_flag.sum()), note="Cleanlab/boundary scores come from a lexical TF-IDF out-of-fold model (A4_oof_summary.json). Flags are review priority, not proof of error.")
json.dump(rep, open(out / "A4_coverage_report.json", "w"), indent=1); print(json.dumps(rep, indent=1))
