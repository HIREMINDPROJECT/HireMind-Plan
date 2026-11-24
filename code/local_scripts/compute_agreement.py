"""Inter-reviewer agreement for A3 pilot / A4. Usage: python compute_agreement.py reviewer1.csv reviewer2.csv [out.json]
Each CSV needs row_uid and reviewer_label (Good Fit / Potential Fit / No Fit / Insufficient Evidence). Insufficient Evidence is reported but excluded from kappa."""
import json, sys
import numpy as np, pandas as pd
L = ["No Fit", "Potential Fit", "Good Fit"]
norm = lambda s: s.astype(str).str.strip().str.lower().str.replace(" fit", "", regex=False).map({"no": "No Fit", "potential": "Potential Fit", "good": "Good Fit", "insufficient evidence": "Insufficient Evidence"})
a, b = (pd.read_csv(f, keep_default_na=False)[["row_uid", "reviewer_label"]] for f in sys.argv[1:3])
d = a.merge(b, on="row_uid", suffixes=("_1", "_2"))
assert len(d) == len(a) == len(b), "case lists differ"
d["l1"], d["l2"] = norm(d.reviewer_label_1), norm(d.reviewer_label_2)
assert d.l1.notna().all() and d.l2.notna().all(), "unknown label value"
k = d[d.l1.isin(L) & d.l2.isin(L)]
def kappa(x, y, w=None):
    n = len(x); idx = {l: i for i, l in enumerate(L)}
    O = np.zeros((3, 3)); 
    for u, v in zip(x, y): O[idx[u], idx[v]] += 1
    O /= n; E = np.outer(O.sum(1), O.sum(0))
    W = np.zeros((3, 3)) if w is None else np.abs(np.subtract.outer(range(3), range(3))) / 2
    if w is None: W = 1 - np.eye(3)
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float("nan")
res = dict(cases=len(d), compared=len(k), insufficient_by_either=int(len(d) - len(k)), raw_agreement=round(float((k.l1 == k.l2).mean()), 4), disagreements=int((k.l1 != k.l2).sum()),
           cohen_kappa=round(kappa(k.l1, k.l2), 4), linear_weighted_kappa=round(kappa(k.l1, k.l2, "linear"), 4),
           confusion=pd.crosstab(k.l1, k.l2).reindex(index=L, columns=L, fill_value=0).values.tolist(), disagreement_row_uids=d[d.l1 != d.l2].row_uid.tolist())
res["pilot_status"] = "PILOT_PASSED" if res["cohen_kappa"] >= 0.60 and res["disagreements"] <= 4 and len(d) == 20 else "PILOT_FAILED_REVISE_OR_NOT_A_20_CASE_PILOT"
print(json.dumps(res, indent=1)); 
if len(sys.argv) > 3: json.dump(res, open(sys.argv[3], "w"), indent=1)
