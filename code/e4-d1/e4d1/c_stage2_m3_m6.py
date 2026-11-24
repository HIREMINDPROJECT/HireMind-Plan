"""Phase C stage 2: M3 XGBoost (structured evidence + exact-match features from B1) and M6 MiniLM embeddings + classifier.
D0 labels, TRAIN fit, VALIDATION evaluation only. The test set is never read. No text leaves the runner.
Usage: python c_stage2_m3_m6.py <HireMind-DL checkout> <OUT dir>
Everything below is fixed before results are seen (no tuning on validation)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
here = Path(__file__).parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(root / "src"))
from hiremind_dl import load_config  # noqa: E402
from hiremind_dl.data import download_raw, load_manifest, read_raw, split_frame  # noqa: E402
from structured_extractor import structure_pair  # noqa: E402

cfg = load_config()
man = load_manifest(cfg)
raw = {"train": read_raw(download_raw(cfg, "train"))}
tr = split_frame(raw, man, "train", 4993)
va = split_frame(raw, man, "validation", 1162)

import os as _os
_ov = pd.read_csv(_os.environ["HIREMIND_TRAIN_LABELS"]).set_index("row_uid"); assert sorted(_ov.index) == sorted(tr.row_uid)
_n = int((_ov.loc[tr.row_uid, "label_id"].to_numpy() != tr.label_id.to_numpy()).sum()); assert _n == 255, _n
tr["label_id"] = _ov.loc[tr.row_uid, "label_id"].to_numpy(); print("TRAIN label overlay applied:", _n, flush=True)
assert np.bincount(va.label_id).tolist() == [579, 294, 289]
ytr, yva = tr.label_id.to_numpy(), va.label_id.to_numpy()


def metrics(y, p):
    cm = np.zeros((3, 3), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    tp = np.diag(cm).astype(float)
    pr = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    rc = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)
    f = np.divide(2 * pr * rc, pr + rc, out=np.zeros(3), where=(pr + rc) > 0)
    m = {"accuracy_pct": 100 * tp.sum() / cm.sum(), "macro_f1_pct": 100 * f.mean()}
    for i, c in enumerate(["no_fit", "potential_fit", "good_fit"]):
        m[f"{c}_precision_pct"], m[f"{c}_recall_pct"], m[f"{c}_f1_pct"] = 100 * pr[i], 100 * rc[i], 100 * f[i]
    return m, cm


def txt(s):
    return s if isinstance(s, str) else ""


# ------------------------------- M3: structured features from B1 -------------------------------
CATS = {"skill": ("SKILL", "TOOL_TECHNOLOGY"), "edu": ("EDUCATION",), "years": ("EXPERIENCE",), "cert": ("CERTIFICATION",)}


def feats(cv, jd, cid):
    p = structure_pair(cid, cv, jd)
    f = {"cv_chars": len(cv), "jd_chars": len(jd), "n_jd_records": sum(r["source_type"] == "JOB_DESCRIPTION" for r in p["records"]),
         "n_cv_records": sum(r["source_type"] == "CV" for r in p["records"])}
    L = p["requirement_links"]
    for cn, cats in CATS.items():
        for rt in ("REQUIRED", "PREFERRED", "UNKNOWN"):
            s = [l for l in L if l["category"] in cats and l["requirement_type"] == rt]
            n = len(s)
            found = sum(l["link_status"] in ("FOUND_EXPLICIT", "LEVEL_MET") for l in s)
            f[f"{cn}_{rt.lower()}_n"] = n
            f[f"{cn}_{rt.lower()}_found"] = found
            f[f"{cn}_{rt.lower()}_unknown"] = sum(l["link_status"] == "NOT_STATED_UNKNOWN" for l in s)
            f[f"{cn}_{rt.lower()}_neg"] = sum(l["link_status"] == "FOUND_EXPLICIT_NEGATIVE" for l in s)
            f[f"{cn}_{rt.lower()}_rate"] = found / n if n else np.nan
    f["edu_below"] = sum(l["link_status"] == "LEVEL_BELOW_REQUIRED" for l in L)
    jy = [r["years_experience"] for r in p["records"] if r["source_type"] == "JOB_DESCRIPTION" and r["evidence_category"] == "EXPERIENCE"]
    cy = [r["years_experience"] for r in p["records"] if r["source_type"] == "CV" and r["evidence_category"] == "EXPERIENCE"]
    f["job_years_max"] = max(jy) if jy else np.nan
    f["cv_years_max"] = max(cy) if cy else np.nan
    f["cv_years_ge_job"] = (1.0 if max(cy) >= max(jy) else 0.0) if jy and cy else np.nan
    f["cv_n_skill_records"] = sum(r["source_type"] == "CV" and r["evidence_category"] in ("SKILL", "TOOL_TECHNOLOGY") for r in p["records"])
    f["cv_n_employment_ranges"] = sum(r["source_type"] == "CV" and r["evidence_category"] == "EMPLOYMENT" for r in p["records"])
    f["cv_has_degree"] = float(any(r["source_type"] == "CV" and r["evidence_category"] == "EDUCATION" for r in p["records"]))
    f["n_requirements"] = len(L)
    return f


t0 = time.time()
Ftr = pd.DataFrame([feats(txt(a), txt(b), c) for a, b, c in zip(tr.resume_text, tr.job_description_text, tr.row_uid)])
Fva = pd.DataFrame([feats(txt(a), txt(b), c) for a, b, c in zip(va.resume_text, va.job_description_text, va.row_uid)])
Fva = Fva[Ftr.columns]
print("M3 features", Ftr.shape, "in", round(time.time() - t0, 1), "s")
rows, preds = [], pd.DataFrame({"row_uid": va.row_uid, "true_id": yva})
from xgboost import XGBClassifier  # noqa: E402

per_seed = []
for seed in range(42, 50):
    clf = XGBClassifier(objective="multi:softprob", n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, n_jobs=2, tree_method="hist", eval_metric="mlogloss")
    t1 = time.time(); clf.fit(Ftr, ytr); tfit = time.time() - t1
    t1 = time.time(); p = clf.predict(Fva); tinf = time.time() - t1
    m, cm = metrics(yva, p)
    per_seed.append(m); preds[f"M3_xgb_s{seed}"] = p
    rows.append(dict(model=f"M3_XGBoost_s{seed}", role="primary (seed)", labels="D1 train / D0 validation", train_rows=len(tr), validation_rows=len(va), fit_seconds=round(tfit, 2),
                     inference_ms_per_pair=round(1000 * tinf / len(va), 4), **{k: round(v, 4) for k, v in m.items()}, confusion_true_rows_pred_cols=json.dumps(cm.tolist(), separators=(",", ":"))))
mean = {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}
sd = {k: float(np.std([m[k] for m in per_seed], ddof=1)) for k in per_seed[0]}
rows.append(dict(model="M3_XGBoost_mean8", role="primary (mean of 8 seeds 42-49)", labels="D1 train / D0 validation", train_rows=len(tr), validation_rows=len(va), **{k: round(v, 4) for k, v in mean.items()}))
rows.append(dict(model="M3_XGBoost_sd8", role="sample SD of 8 seeds", labels="D1 train / D0 validation", **{k: round(v, 4) for k, v in sd.items()}))
imp = pd.Series(clf.feature_importances_, index=Ftr.columns).sort_values(ascending=False)
imp.head(15).round(4).to_csv(out / "E4_D1_M3_top_features_lastseed.csv", header=["importance"])
print("M3 mean", {k: round(v, 2) for k, v in mean.items() if k in ("accuracy_pct", "macro_f1_pct", "good_fit_recall_pct", "good_fit_precision_pct", "potential_fit_recall_pct")})

# M6 (MiniLM) not run: not part of the team decision (DistilBERT only).
MID, rev = "not run", "-"

pd.DataFrame(rows).to_csv(out / "E4_D1_stage2_m3_validation.csv", index=False)
preds.to_csv(out / "E4_D1_stage2_validation_predictions.csv", index=False)
json.dump(dict(m3="XGBoost multi:softprob n_estimators=300 max_depth=4 lr=0.05 subsample=0.8 colsample=0.8, seeds 42-49; features = B1 structured evidence + exact-match link counts (no B3 semantic features in this run)",
               m6=f"{MID} @ {rev}: sentence-unit embeddings mean-pooled per document; features [cv, job, cv*job, |cv-job|, cos]; StandardScaler + LogisticRegression C=0.1 (fixed)",
               hyperparameters_tuned_on_validation=False, test_set_read=False, labels="D1 train / D0 validation"), open(out / "E4_D1_stage2_config.json", "w"), indent=1)
