"""A2 near-duplicate scan. Run where the pinned dataset is reachable (Colab/GitHub).
Usage: python A2_near_duplicate_scan.py train.csv splits.csv out.csv   (train.csv SHA must be b86fa488...ec40)"""
import sys, hashlib, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
tp, sp, out = sys.argv[1:4]
assert hashlib.sha256(open(tp,"rb").read()).hexdigest().startswith("b86fa488"), "wrong train.csv"
t = pd.read_csv(tp); s = pd.read_csv(sp); s = s[s.split=="train"]
t = t.iloc[s.orig_index.values].reset_index(drop=True); t["row_uid"] = s.row_uid.values; t["cv_id"] = s.cv_id.values
cols = [c for c in t.columns if "resume" in c.lower()]; jcols = [c for c in t.columns if "job" in c.lower() and "text" in c.lower()]
rows = []
for name, col, key in [("NEAR_DUP_CV_TEXT", cols[0], "cv_id")]+([("NEAR_DUP_JOB_TEXT", jcols[0], None)] if jcols else []):
    u = t.drop_duplicates(key) if key else t
    X = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(u[col].fillna(""))
    d, i = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X).kneighbors(X)
    for a in np.where(1 - d[:,1] >= 0.95)[0]:
        b = i[a,1]
        if a < b: rows.append(dict(issue=name, row_uid=u.iloc[a].row_uid, other_row_uid=u.iloc[b].row_uid, cosine=round(1-d[a,1],4)))
pd.DataFrame(rows).to_csv(out, index=False); print(len(rows), "near-duplicate pairs (cosine>=0.95) ->", out)
