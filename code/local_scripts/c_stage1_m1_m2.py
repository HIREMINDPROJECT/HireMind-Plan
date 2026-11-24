"""Phase C stage 1: M1 Logistic Regression and M2 Linear SVM on TF-IDF of CV + job text. D0 labels, TRAIN fit, VALIDATION evaluation only.
Writes metrics/confusions/predictions (IDs only). No text leaves the runner. Usage: python c_stage1_m1_m2.py HireMind-DL OUT"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
root, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "src"))
from hiremind_dl import load_config
from hiremind_dl.data import download_raw, load_manifest, read_raw, split_frame
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
cfg = load_config(); man = load_manifest(cfg); raw = {"train": read_raw(download_raw(cfg, "train"))}
tr = split_frame(raw, man, "train", 4993); va = split_frame(raw, man, "validation", 1162)
assert np.bincount(va.label_id).tolist() == [579, 294, 289]
CL = ["No Fit", "Potential Fit", "Good Fit"]
def metrics(y, p):
    cm = np.zeros((3, 3), int)
    for a, b in zip(y, p): cm[a, b] += 1
    tp = np.diag(cm).astype(float); pr = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0); rc = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)
    f = np.divide(2 * pr * rc, pr + rc, out=np.zeros(3), where=(pr + rc) > 0)
    m = {"accuracy_pct": 100 * tp.sum() / cm.sum(), "macro_f1_pct": 100 * f.mean()}
    for i, c in enumerate(["no_fit", "potential_fit", "good_fit"]): m[f"{c}_precision_pct"], m[f"{c}_recall_pct"], m[f"{c}_f1_pct"] = 100 * pr[i], 100 * rc[i], 100 * f[i]
    return m, cm
# fixed featurisation (declared before results): word 1-2gram TF-IDF, CV and job separately, fit on TRAIN only
v1 = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000, sublinear_tf=True, lowercase=True)
v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True, lowercase=True)
Xtr = hstack([v1.fit_transform(tr.resume_text), v2.fit_transform(tr.job_description_text)]).tocsr()
Xva = hstack([v1.transform(va.resume_text), v2.transform(va.job_description_text)]).tocsr()
ytr, yva = tr.label_id.to_numpy(), va.label_id.to_numpy()
cfgs = [("M1_LogReg_C1", LogisticRegression(C=1.0, max_iter=1000), "primary"), ("M2_LinearSVC_C1", LinearSVC(C=1.0, random_state=42), "primary"),
        ("M1_LogReg_C1_balanced", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"), "sensitivity (class weights)"),
        ("M2_LinearSVC_C1_balanced", LinearSVC(C=1.0, random_state=42, class_weight="balanced"), "sensitivity (class weights)")]
rows, preds = [], pd.DataFrame({"row_uid": va.row_uid, "true_id": yva})
for name, clf, role in cfgs:
    t0 = time.time(); clf.fit(Xtr, ytr); tfit = time.time() - t0
    t0 = time.time(); p = clf.predict(Xva); tinf = time.time() - t0
    m, cm = metrics(yva, p); preds[name] = p
    rows.append(dict(model=name, role=role, labels="D0 original", train_rows=len(tr), validation_rows=len(va), fit_seconds=round(tfit, 2), inference_ms_per_pair=round(1000 * tinf / len(va), 4),
                     **{k: round(v, 4) for k, v in m.items()}, confusion_true_rows_pred_cols=json.dumps(cm.tolist(), separators=(",", ":"))))
    print(name, {k: round(v, 2) for k, v in m.items() if k in ("accuracy_pct", "macro_f1_pct", "good_fit_recall_pct", "good_fit_precision_pct", "potential_fit_recall_pct")}, cm.tolist())
pd.DataFrame(rows).to_csv(out / "C_stage1_m1_m2_validation.csv", index=False); preds.to_csv(out / "C_stage1_validation_predictions.csv", index=False)
json.dump(dict(features="TF-IDF word 1-2gram, CV max 100k + job max 40k, min_df=3, sublinear_tf; fit on TRAIN only", hyperparameters_tuned_on_validation=False, test_set_read=False), open(out / "C_stage1_config.json", "w"), indent=1)
