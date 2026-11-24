"""A2 text-level audit + A4 support scores. TRAIN only. Writes IDs / scores / counts ONLY (no CV or job text).
Run from a checkout of HireMind-DL @1f22f34:  python a2_a4_text_jobs.py HireMind-DL OUT"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "src"))
from hiremind_dl import load_config  # noqa: E402
from hiremind_dl.data import download_raw, load_manifest, read_raw, split_frame  # noqa: E402

cfg = load_config()
man = load_manifest(cfg)
raw = {"train": read_raw(download_raw(cfg, "train"))}
tr = split_frame(raw, man, "train", expected_rows=4993)  # asserts labels and ids match the frozen manifest
print("TRAIN rows", len(tr), tr.label.value_counts().to_dict())
cv, jd = tr.resume_text.fillna(""), tr.job_description_text.fillna("")

# ---------------- A2: text-level checks ----------------
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2", revision="30b0a37ccaaa32f332884b96992754e246e48c5f")
cache = {}


def ntok(t):
    h = hashlib.md5(t.encode()).hexdigest()
    if h not in cache:
        cache[h] = len(tok(t, add_special_tokens=False)["input_ids"])
    return cache[h]


d = pd.DataFrame({"row_uid": tr.row_uid, "cv_id": tr.cv_id, "job_id": tr.job_id, "job_group": tr.job_group, "label": tr.label,
                  "cv_chars": cv.str.len(), "jd_chars": jd.str.len(), "cv_words": cv.str.split().str.len(), "jd_words": jd.str.split().str.len(),
                  "cv_tokens": [ntok(t) for t in cv], "jd_tokens": [ntok(t) for t in jd]})
d["pair_tokens_with_special"] = d.cv_tokens + d.jd_tokens + 3
d["exceeds_512"] = d.pair_tokens_with_special > 512
d.to_csv(out / "A2_train_length_table.csv", index=False)
q = [0, .01, .05, .25, .5, .75, .95, .99, 1]
rows = []
for col in ["cv_words", "jd_words", "cv_tokens", "jd_tokens", "pair_tokens_with_special"]:
    for lab, g in [("ALL", d)] + list(d.groupby("label")):
        rows.append(dict(field=col, label=lab, n=len(g), **{f"q{int(p * 100)}": float(g[col].quantile(p)) for p in q}))
pd.DataFrame(rows).to_csv(out / "A2_length_quantiles.csv", index=False)

ex = []


def add(issue, sub, detail):
    for _, r in sub.iterrows():
        ex.append(dict(issue=issue, row_uid=r.row_uid, cv_id=r.cv_id, job_id=r.job_id, job_group=r.job_group, label=r.label, detail=detail(r)))


add("EMPTY_CV_TEXT", d[d.cv_chars == 0], lambda r: "cv_chars=0")
add("EMPTY_JOB_TEXT", d[d.jd_chars == 0], lambda r: "jd_chars=0")
add("SHORT_CV_TEXT", d.nsmallest(5, "cv_words"), lambda r: f"cv_words={int(r.cv_words)}")
add("SHORT_JOB_TEXT", d.nsmallest(5, "jd_words"), lambda r: f"jd_words={int(r.jd_words)}")
add("LONG_CV_TEXT", d.nlargest(3, "cv_tokens"), lambda r: f"cv_tokens={int(r.cv_tokens)}")
n_exceed = int(d.exceeds_512.sum())
if n_exceed:
    add("PAIR_EXCEEDS_512_TOKENS", d[d.exceeds_512].sample(n=min(5, n_exceed), random_state=42),
        lambda r: f"pair_tokens={int(r.pair_tokens_with_special)} (limit 512); {n_exceed} of {len(d)} TRAIN pairs exceed")

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

nd = {}
for name, col, key in [("CV", "resume_text", "cv_id"), ("JOB", "job_description_text", "job_id")]:
    u = tr.drop_duplicates(key).reset_index(drop=True)
    X = TfidfVectorizer(min_df=1, sublinear_tf=True).fit_transform(u[col].fillna(""))
    dist, idx = NearestNeighbors(n_neighbors=min(4, len(u)), metric="cosine").fit(X).kneighbors(X)
    pairs = set()
    for a in range(len(u)):
        for k in range(1, idx.shape[1]):
            b, c = int(idx[a, k]), 1 - float(dist[a, k])
            if c >= 0.90 and a != b:
                pairs.add((min(a, b), max(a, b), round(c, 4)))
    nd[name] = len(pairs)
    for a, b, c in sorted(pairs, key=lambda x: -x[2])[:10]:
        ex.append(dict(issue=f"NEAR_DUPLICATE_{name}_TEXT", row_uid=tr.row_uid[tr[key] == u[key][a]].iloc[0], cv_id=u.cv_id[a], job_id=u.job_id[a],
                       job_group=u.job_group[a], label="", detail=f"{key} {u[key][a]} ~ {u[key][b]} cosine={c} (distinct ids, near-identical text)"))
pd.DataFrame(ex).to_csv(out / "A2_text_issue_examples.csv", index=False)
a2 = dict(rows=len(d), empty_cv=int((d.cv_chars == 0).sum()), empty_job=int((d.jd_chars == 0).sum()), cv_words_lt_100=int((d.cv_words < 100).sum()),
          jd_words_lt_50=int((d.jd_words < 50).sum()), pairs_exceeding_512_tokens=n_exceed, pct_exceeding_512=round(100 * d.exceeds_512.mean(), 2),
          near_duplicate_pairs_cosine_ge_0_90=nd, distinct_cv_texts=int(tr.cv_id.nunique()), distinct_job_texts=int(tr.job_id.nunique()),
          exact_duplicate_pairs=int(tr.duplicated(["cv_id", "job_id"]).sum()))
json.dump(a2, open(out / "A2_text_checks.json", "w"), indent=1)
print(json.dumps(a2, indent=1))

# ---------------- A4 support: grouped OOF + Cleanlab (lexical TF-IDF LR; NOT the BERT model) ----------------
from cleanlab.filter import find_label_issues  # noqa: E402
from cleanlab.rank import get_label_quality_scores  # noqa: E402
from scipy.sparse import hstack  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

y = tr.label_id.to_numpy()
assert tr.groupby("label").label_id.first().to_dict() == {"Good Fit": 2, "No Fit": 0, "Potential Fit": 1}
oof = np.zeros((len(tr), 3))
for f, (a, b) in enumerate(GroupKFold(5).split(tr, y, tr.job_group)):
    v1 = TfidfVectorizer(min_df=2, max_features=50000, sublinear_tf=True).fit(tr.resume_text.iloc[a])
    v2 = TfidfVectorizer(min_df=2, max_features=20000, sublinear_tf=True).fit(tr.job_description_text.iloc[a])

    def mk(ix):
        return hstack([v1.transform(tr.resume_text.iloc[ix]), v2.transform(tr.job_description_text.iloc[ix])]).tocsr()

    m = LogisticRegression(max_iter=400, C=1.0).fit(mk(a), y[a])
    oof[b] = m.predict_proba(mk(b))
    print("fold", f, "acc", round(float((m.predict(mk(b)) == y[b]).mean()), 4))
score = get_label_quality_scores(y, oof)
flag = np.zeros(len(y), bool)
flag[find_label_issues(y, oof)] = True
s = pd.DataFrame({"row_uid": tr.row_uid, "cv_id": tr.cv_id, "job_id": tr.job_id, "job_group": tr.job_group, "original_label": tr.label, "original_label_id": y,
                  "p_no_fit": oof[:, 0], "p_potential_fit": oof[:, 1], "p_good_fit": oof[:, 2], "oof_pred_id": oof.argmax(1),
                  "label_quality_score": score, "cleanlab_flag": flag})
s["gp_margin"] = (s.p_good_fit - s.p_potential_fit).abs()
s["boundary_candidate"] = s.original_label_id.isin([1, 2]) & s.oof_pred_id.isin([1, 2])
s.to_csv(out / "A4_oof_cleanlab_scores_lexical.csv", index=False)
json.dump(dict(method="5-fold GroupKFold by job_group; TF-IDF(CV)+TF-IDF(job) multinomial LR; cleanlab find_label_issues / get_label_quality_scores",
               oof_accuracy=round(float((oof.argmax(1) == y).mean()), 4), flagged=int(flag.sum()),
               flagged_by_label=s[s.cleanlab_flag].original_label.value_counts().to_dict(), boundary_candidates=int(s.boundary_candidate.sum()),
               note="Flags are review priority only, not proof of label error. Lexical model; differs from the team's earlier OOF."),
          open(out / "A4_oof_summary.json", "w"), indent=1)
print(open(out / "A4_oof_summary.json").read())
