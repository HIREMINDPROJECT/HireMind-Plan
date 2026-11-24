#!/usr/bin/env python3
"""HireMind A1 — compare an 8-seed M4 Knowledge v1 run against the historical reference.

No training, no data edits. Reads run JSONs (knowledge_run_<variant>_s<seed>.json), optional
per-pair validation predictions (val_predictions_<variant>_s<seed>.csv) and optional checkpoints
(<weights>/M0_L512_<variant>_s<seed>/weights.pt).

Usage
  python A1_compare_runs.py --run DIR [--run-preds DIR] [--weights DIR] \
      --ref DIR --ref-preds DIR [--variant knowledge] --out DIR

Outputs (in --out)
  A1_per_seed_comparison.csv   one row per seed: recomputed metrics for run and reference, deltas,
                               confusion-matrix cell differences, prediction flips, checkpoint SHA-256
  A1_run_metrics_mean_sd.csv   mean and sample SD of every metric for the run
  A1_comparison_summary.json   verdict counts and aggregate deltas

Verdict per seed (fixed before running):
  IDENTICAL       confusion matrix equal to the reference
  FLOAT_NOISE     confusion matrices differ by <= 4 cases in total (|cell differences| summed / 2)
                  and the same checkpoint step was selected
  DIVERGED        anything else (e.g. a different best checkpoint step); must be explained
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["no_fit", "potential_fit", "good_fit"]
VAL_ROWS = 1162
VAL_SUPPORT = [579, 294, 289]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(3), where=(prec + rec) > 0)
    m = {"accuracy_pct": 100 * tp.sum() / cm.sum(), "macro_f1_pct": 100 * f1.mean()}
    for i, c in enumerate(CLASSES):
        m[f"{c}_precision_pct"] = 100 * prec[i]
        m[f"{c}_recall_pct"] = 100 * rec[i]
        m[f"{c}_f1_pct"] = 100 * f1[i]
    return m


def load_run(d: Path, variant: str, seed: int) -> dict | None:
    f = d / f"knowledge_run_{variant}_s{seed}.json"
    return json.loads(f.read_text()) if f.is_file() else None


def check_cm(j: dict, tag: str, errors: list) -> np.ndarray | None:
    cm = np.array(j.get("confusion", []), dtype=int)
    if cm.shape != (3, 3):
        errors.append(f"{tag}: confusion matrix is not 3x3")
        return None
    if cm.sum() != VAL_ROWS:
        errors.append(f"{tag}: confusion sum {cm.sum()} != {VAL_ROWS}")
    if cm.sum(1).tolist() != VAL_SUPPORT:
        errors.append(f"{tag}: true-class counts {cm.sum(1).tolist()} != frozen validation {VAL_SUPPORT}")
    rec = metrics(cm)
    if "accuracy" in j and abs(rec["accuracy_pct"] - float(j["accuracy"])) > 0.01:
        errors.append(f"{tag}: logged accuracy {j['accuracy']} disagrees with its confusion matrix")
    if "macro_f1" in j and abs(rec["macro_f1_pct"] - float(j["macro_f1"])) > 0.01:
        errors.append(f"{tag}: logged macro-F1 {j['macro_f1']} disagrees with its confusion matrix")
    return cm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--run-preds", type=Path)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--ref-preds", type=Path)
    ap.add_argument("--variant", default="knowledge")
    ap.add_argument("--ref-variant", default="knowledge")
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)

    errors, rows = [], []
    for s in seeds:
        j, r = load_run(a.run, a.variant, s), load_run(a.ref, a.ref_variant, s)
        if j is None:
            errors.append(f"seed {s}: run JSON missing in {a.run}")
            continue
        if r is None:
            errors.append(f"seed {s}: reference JSON missing in {a.ref}")
            continue
        for k, v in [("seed", s), ("epochs", 3), ("updates", 939)]:
            if j.get(k) != v:
                errors.append(f"seed {s}: run {k}={j.get(k)} expected {v}")
        if j.get("knowledge") not in (1, True):
            errors.append(f"seed {s}: run is not a Knowledge v1 arm (knowledge={j.get('knowledge')})")
        if j.get("vocab", "v1") != "v1":
            errors.append(f"seed {s}: vocab {j.get('vocab')} != v1")
        if j.get("feature_mode", "keyword") != "keyword":
            errors.append(f"seed {s}: feature_mode {j.get('feature_mode')} != keyword")
        cm, cr = check_cm(j, f"run s{s}", errors), check_cm(r, f"ref s{s}", errors)
        if cm is None or cr is None:
            continue
        mr, mf = metrics(cm), metrics(cr)
        cell_diff = int(np.abs(cm - cr).sum())
        row = {"seed": s, "run_device": j.get("device"), "ref_device": r.get("device"),
               "run_best_step": j.get("best_step"), "ref_best_step": r.get("best_step"),
               "run_seconds": j.get("seconds"),
               "run_confusion": json.dumps(cm.tolist(), separators=(",", ":")),
               "ref_confusion": json.dumps(cr.tolist(), separators=(",", ":")),
               "cm_cases_moved": cell_diff // 2}
        for k in mr:
            row[f"run_{k}"] = mr[k]
            row[f"ref_{k}"] = mf[k]
            row[f"delta_{k}"] = mr[k] - mf[k]
        # per-pair predictions
        row["prediction_flips"] = ""
        row["max_abs_prob_diff"] = ""
        if a.run_preds and a.ref_preds:
            fp = a.run_preds / f"val_predictions_{a.variant}_s{s}.csv"
            rp = a.ref_preds / f"val_predictions_{a.ref_variant}_s{s}.csv"
            if fp.is_file() and rp.is_file():
                x, y = pd.read_csv(fp), pd.read_csv(rp)
                m = x.merge(y, on="row_uid", suffixes=("_run", "_ref"))
                if len(m) != VAL_ROWS or (m.label_run != m.label_ref).any():
                    errors.append(f"seed {s}: prediction files do not align on the 1,162 validation row_uids/labels")
                row["prediction_flips"] = int((m.pred_run != m.pred_ref).sum())
                row["max_abs_prob_diff"] = float(np.abs(m[[f"{c}_run" for c in ("p_no", "p_potential", "p_good")]].values
                                                        - m[[f"{c}_ref" for c in ("p_no", "p_potential", "p_good")]].values).max())
                pcm = pd.crosstab(x.label, x.pred).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).values
                if not np.array_equal(pcm, cm):
                    errors.append(f"seed {s}: run prediction file does not reproduce the run JSON confusion matrix")
        # checkpoint
        row["checkpoint_sha256"] = ""
        row["checkpoint_sha256_matches_log"] = ""
        if a.weights:
            w = a.weights / f"M0_L512_{a.variant}_s{s}" / "weights.pt"
            if w.is_file():
                h = sha256(w)
                row["checkpoint_sha256"] = h
                row["checkpoint_sha256_matches_log"] = (h == j.get("weights_sha256"))
                if j.get("weights_sha256") and h != j["weights_sha256"]:
                    errors.append(f"seed {s}: checkpoint SHA-256 differs from the run log")
            else:
                errors.append(f"seed {s}: checkpoint missing at {w}")
        if cell_diff == 0:
            v = "IDENTICAL"
        elif cell_diff // 2 <= 4 and j.get("best_step") == r.get("best_step"):
            v = "FLOAT_NOISE"
        else:
            v = "DIVERGED"
        row["verdict"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(a.out / "A1_per_seed_comparison.csv", index=False)
    summary = {"seeds_compared": len(df), "errors": errors}
    if len(df):
        keys = [k for k in metrics(np.eye(3, dtype=int)).keys()]
        ms = pd.DataFrame({k: df[f"run_{k}"] for k in keys})
        rs = pd.DataFrame({k: df[f"ref_{k}"] for k in keys})
        out = pd.DataFrame({"metric": keys, "run_mean_pct": ms.mean().values, "run_sample_sd_pct": ms.std(ddof=1).values,
                            "ref_mean_pct": rs.mean().values, "ref_sample_sd_pct": rs.std(ddof=1).values})
        out["delta_mean_pct"] = out.run_mean_pct - out.ref_mean_pct
        out.to_csv(a.out / "A1_run_metrics_mean_sd.csv", index=False)
        summary["verdicts"] = df.verdict.value_counts().to_dict()
        summary["delta_mean_accuracy_pct"] = float(out.set_index("metric").loc["accuracy_pct", "delta_mean_pct"])
        summary["delta_mean_good_fit_recall_pct"] = float(out.set_index("metric").loc["good_fit_recall_pct", "delta_mean_pct"])
        summary["delta_mean_macro_f1_pct"] = float(out.set_index("metric").loc["macro_f1_pct", "delta_mean_pct"])
    summary["status"] = "CHECKS_FAILED" if errors else ("ALL_IDENTICAL" if summary.get("verdicts", {}) == {"IDENTICAL": len(seeds)} else "COMPLETE_DIFFERENCES_TO_EXPLAIN")
    (a.out / "A1_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
