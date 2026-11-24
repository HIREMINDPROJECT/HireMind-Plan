"""A4 results: reviewer agreement, adjudication outcome, confirmed errors counted SEPARATELY from merely flagged cases.
Usage: python A4_summarize.py reviewed_labels.csv out.json
Required columns: case_no,pair_id,row_uid,original_label,stratum,cleanlab_flag,reviewer1_label,reviewer2_label,adjudicated_label,evidence_span,reason,confidence,decision
decision values: CONFIRMED_ERROR (adjudicated label differs from original, evidence recorded) | KEEP_ORIGINAL | TRUE_AMBIGUITY | INSUFFICIENT_EVIDENCE
Rows with empty reviewer labels (not yet reviewed) are reported as PENDING and never counted as agreement or error."""
import json, sys
import numpy as np, pandas as pd
L = ["No Fit", "Potential Fit", "Good Fit"]
d = pd.read_csv(sys.argv[1], keep_default_na=False)
norm = lambda s: s.astype(str).str.strip().str.lower().map({"no fit": "No Fit", "potential fit": "Potential Fit", "good fit": "Good Fit", "insufficient evidence": "Insufficient Evidence", "": ""})
for c in ["original_label", "reviewer1_label", "reviewer2_label", "adjudicated_label"]: d[c] = norm(d[c])
done = d[(d.reviewer1_label != "") & (d.reviewer2_label != "")]
res = dict(cases=len(d), double_reviewed=len(done), pending=len(d) - len(done))
def kappa(x, y, linear=False):
    idx = {l: i for i, l in enumerate(L)}; O = np.zeros((3, 3))
    for u, v in zip(x, y): O[idx[u], idx[v]] += 1
    O /= O.sum(); E = np.outer(O.sum(1), O.sum(0)); W = np.abs(np.subtract.outer(range(3), range(3))) / 2 if linear else 1 - np.eye(3)
    return float(1 - (W * O).sum() / (W * E).sum())
if len(done):
    k = done[done.reviewer1_label.isin(L) & done.reviewer2_label.isin(L)]
    res.update(raw_agreement=round(float((k.reviewer1_label == k.reviewer2_label).mean()), 4), cohen_kappa=round(kappa(k.reviewer1_label, k.reviewer2_label), 4),
               linear_weighted_kappa=round(kappa(k.reviewer1_label, k.reviewer2_label, True), 4), disagreements_needing_adjudication=int((done.reviewer1_label != done.reviewer2_label).sum()))
    adj = done[done.adjudicated_label != ""]
    conf = adj[adj.decision == "CONFIRMED_ERROR"]
    res.update(adjudicated=len(adj), confirmed_label_errors=len(conf), true_ambiguity=int((adj.decision == "TRUE_AMBIGUITY").sum()), insufficient_evidence=int((adj.decision == "INSUFFICIENT_EVIDENCE").sum()),
               flagged_by_cleanlab_in_queue=int(d.cleanlab_flag.astype(str).str.lower().eq("true").sum()),
               flagged_and_confirmed=int(conf.cleanlab_flag.astype(str).str.lower().eq("true").sum()),
               flagged_not_confirmed=int(adj[adj.cleanlab_flag.astype(str).str.lower().eq("true") & (adj.decision != "CONFIRMED_ERROR")].shape[0]),
               confirmed_rate_by_stratum={st: f"{int((g.decision == 'CONFIRMED_ERROR').sum())}/{len(g)}" for st, g in adj.groupby("stratum")},
               note="The RANDOM stratum confirmed rate is the unbiased estimate of label-error rate; CLEANLAB/BOUNDARY rates are enriched samples.")
print(json.dumps(res, indent=1)); 
if len(sys.argv) > 2: json.dump(res, open(sys.argv[2], "w"), indent=1)
