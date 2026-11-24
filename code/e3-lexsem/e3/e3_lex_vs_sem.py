"""E3 (D0 labels, validation only): FIXED downstream classifier (XGBoost, same settings and seeds as M3), compare
  (a) LEXICAL matching only   = B1 structured / exact-match features (the M3 feature set)
  (b) LEXICAL + SEMANTIC      = (a) + B3 DistilBERT similarity features (sem means/mins, missing and semantic-only counts)
Also reports requirement COVERAGE: exact-found requirements vs exact + semantic-only candidates (a candidate is NOT a verified match).
Threshold rule for 'missing' is label-free (q10 of best-sentence similarity among exact-found requirements, TRAIN). Test set never read. No text in outputs.
Usage: python e3_lex_vs_sem.py <HireMind-DL checkout> <OUT dir>"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
here = Path(__file__).parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(root / "src"))
from hiremind_dl import load_config  # noqa: E402
from hiremind_dl.data import download_raw, load_manifest, read_raw, split_frame  # noqa: E402
from structured_extractor import structure_pair  # noqa: E402
import matcher as M  # noqa: E402

cfg = load_config(); man = load_manifest(cfg)
raw = {"train": read_raw(download_raw(cfg, "train"))}
tr = split_frame(raw, man, "train", 4993); va = split_frame(raw, man, "validation", 1162)
assert np.bincount(va.label_id).tolist() == [579, 294, 289]
ytr, yva = tr.label_id.to_numpy(), va.label_id.to_numpy()
txt = lambda s: s if isinstance(s, str) else ""  # noqa: E731
CATS = {"skill": ("SKILL", "TOOL_TECHNOLOGY"), "edu": ("EDUCATION",), "years": ("EXPERIENCE",), "cert": ("CERTIFICATION",)}


def lexical(p, cv, jd):
    f = {"cv_chars": len(cv), "jd_chars": len(jd), "n_jd_records": sum(r["source_type"] == "JOB_DESCRIPTION" for r in p["records"]), "n_cv_records": sum(r["source_type"] == "CV" for r in p["records"])}
    L = p["requirement_links"]
    for cn, cats in CATS.items():
        for rt in ("REQUIRED", "PREFERRED", "UNKNOWN"):
            s = [l for l in L if l["category"] in cats and l["requirement_type"] == rt]; n = len(s)
            found = sum(l["link_status"] in ("FOUND_EXPLICIT", "LEVEL_MET") for l in s)
            f[f"{cn}_{rt.lower()}_n"] = n; f[f"{cn}_{rt.lower()}_found"] = found
            f[f"{cn}_{rt.lower()}_unknown"] = sum(l["link_status"] == "NOT_STATED_UNKNOWN" for l in s)
            f[f"{cn}_{rt.lower()}_neg"] = sum(l["link_status"] == "FOUND_EXPLICIT_NEGATIVE" for l in s)
            f[f"{cn}_{rt.lower()}_rate"] = found / n if n else np.nan
    f["edu_below"] = sum(l["link_status"] == "LEVEL_BELOW_REQUIRED" for l in L)
    jy = [r["years_experience"] for r in p["records"] if r["source_type"] == "JOB_DESCRIPTION" and r["evidence_category"] == "EXPERIENCE"]
    cy = [r["years_experience"] for r in p["records"] if r["source_type"] == "CV" and r["evidence_category"] == "EXPERIENCE"]
    f["job_years_max"] = max(jy) if jy else np.nan; f["cv_years_max"] = max(cy) if cy else np.nan
    f["cv_years_ge_job"] = (1.0 if max(cy) >= max(jy) else 0.0) if jy and cy else np.nan
    f["cv_n_skill_records"] = sum(r["source_type"] == "CV" and r["evidence_category"] in ("SKILL", "TOOL_TECHNOLOGY") for r in p["records"])
    f["cv_n_employment_ranges"] = sum(r["source_type"] == "CV" and r["evidence_category"] == "EMPLOYMENT" for r in p["records"])
    f["cv_has_degree"] = float(any(r["source_type"] == "CV" and r["evidence_category"] == "EDUCATION" for r in p["records"]))
    f["n_requirements"] = len(L)
    return f


t0 = time.time()
emb = M.Embedder(M.ENCODERS["distilbert"])
rows = {}
for name, df in (("train", tr), ("validation", va)):
    L, reqs_all = [], []
    for r in df.itertuples():
        cv, jd = txt(r.resume_text), txt(r.job_description_text)
        p = structure_pair(r.row_uid, cv, jd)
        L.append(lexical(p, cv, jd)); reqs_all.append(M.match_pair(p, cv, jd, emb))
    rows[name] = (pd.DataFrame(L), reqs_all)
    print(name, "features+matching done", round(time.time() - t0), "s", flush=True)
ref = [r["sem_sentence_max"] for reqs in rows["train"][1] for r in reqs if r["exact_found"] and r["sem_sentence_max"] is not None]
thr = round(float(np.quantile(ref, M.THRESH_Q)), 4)
SEMCOLS = None
feat = {}
for name, (Fl, reqs_all) in rows.items():
    S = []
    for reqs in reqs_all:
        M.finalize(reqs, thr); a = M.aggregates(reqs)
        a.pop("education_level_met"); a["has_no_requirements"] = float(a["has_no_requirements"])
        S.append(a)
    Sd = pd.DataFrame(S).astype(float)
    if SEMCOLS is None: SEMCOLS = list(Sd.columns)
    feat[name] = (Fl, Sd[SEMCOLS])
Ftr_l, Ftr_s = feat["train"]; Fva_l, Fva_s = feat["validation"]; Fva_l = Fva_l[Ftr_l.columns]
# ---------- coverage (requirement level, no labels) ----------
cov = {}
for name, (Fl, reqs_all) in rows.items():
    allr = [r for reqs in reqs_all for r in reqs]
    ef = sum(r["exact_found"] for r in allr); so = sum(r["semantic_only_candidate"] for r in allr); mi = sum(r["missing_indicator"] for r in allr)
    cov[name] = dict(requirements=len(allr), exact_found=ef, exact_negative=sum(r["exact_negative"] for r in allr), semantic_only_candidates=so, missing_indicator=mi,
                     exact_found_pct=round(100 * ef / len(allr), 2), exact_plus_semantic_candidates_pct=round(100 * (ef + so) / len(allr), 2))
cov["threshold"] = thr
cov["note"] = "semantic-only candidates are NOT verified matches: B3 showed raw DistilBERT similarity barely separates real from false matches; human reading is needed before calling any of them evidence."
json.dump(cov, open(out / "E3_coverage.json", "w"), indent=1)
# ---------- downstream: fixed XGBoost ----------
from xgboost import XGBClassifier  # noqa: E402


def metrics(y, p):
    cm = np.zeros((3, 3), int)
    for a, b in zip(y, p): cm[a, b] += 1
    tp = np.diag(cm).astype(float); pr = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0); rc = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)
    f = np.divide(2 * pr * rc, pr + rc, out=np.zeros(3), where=(pr + rc) > 0)
    return dict(accuracy_pct=100 * tp.sum() / cm.sum(), macro_f1_pct=100 * f.mean(), potential_fit_recall_pct=100 * rc[1], good_fit_precision_pct=100 * pr[2], good_fit_recall_pct=100 * rc[2]), cm


res, preds = [], pd.DataFrame({"row_uid": va.row_uid, "true_id": yva})
sets = {"lexical_only": (Ftr_l, Fva_l), "lexical_plus_semantic": (pd.concat([Ftr_l, Ftr_s.add_prefix("sem_")], axis=1), pd.concat([Fva_l, Fva_s.add_prefix("sem_")], axis=1)), "semantic_only": (Ftr_s.add_prefix("sem_"), Fva_s.add_prefix("sem_"))}
imp_last = None
for name, (A, B) in sets.items():
    per = []
    for seed in range(42, 50):
        clf = XGBClassifier(objective="multi:softprob", n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=2, tree_method="hist", eval_metric="mlogloss")
        clf.fit(A, ytr); p = clf.predict(B); m, cm = metrics(yva, p); per.append(m); preds[f"{name}_s{seed}"] = p
        res.append(dict(feature_set=name, seed=seed, **{k: round(v, 4) for k, v in m.items()}, confusion=json.dumps(cm.tolist())))
        if name == "lexical_plus_semantic" and seed == 49: imp_last = pd.Series(clf.feature_importances_, index=A.columns).sort_values(ascending=False).head(15)
    for stat, fn in (("mean8", np.mean), ("sd8", lambda v: np.std(v, ddof=1))):
        res.append(dict(feature_set=name, seed=stat, **{k: round(float(fn([m[k] for m in per])), 4) for k in per[0]}, confusion=""))
pd.DataFrame(res).to_csv(out / "E3_validation.csv", index=False)
preds.to_csv(out / "E3_validation_predictions.csv", index=False)
imp_last.round(4).to_csv(out / "E3_lexsem_top_features_lastseed.csv", header=["importance"])
pd.concat([Fva_s.assign(row_uid=va.row_uid.values), ], axis=1).to_csv(out / "E3_semantic_pair_features_validation.csv", index=False)
print(pd.DataFrame(res).query("seed in ['mean8','sd8']").drop(columns="confusion").to_string())
