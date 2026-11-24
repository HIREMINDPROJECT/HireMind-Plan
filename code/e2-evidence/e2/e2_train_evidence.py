"""E2 (PROVISIONAL, D0 labels): M4 with the SAME training code, hyper-parameters and Knowledge v1 line as the A1/M4 reference, but the (CV, job) text
is replaced by the evidence-aware selection from B4 (variant lex_k: lexical evidence, room reserved for the Knowledge line). One change vs M4: the input text.
Labels are ORIGINAL (D0). Validation only; the test file is never downloaded. No text leaves the runner.
Usage: python e2_train_evidence.py --dl <HireMind-DL checkout> --seed 42..49 --out <dir>   (HIREMIND_MAX_LEN=512, CPU)"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b4_select import select_pair, assemble  # noqa: E402
from knowledge_v1_ORIGINAL import features as kfeat, knowledge_line  # noqa: E402
from structured_extractor import structure_pair  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dl", required=True)
ap.add_argument("--seed", type=int, required=True, choices=range(42, 50))
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert os.getenv("HIREMIND_MAX_LEN") == "512"
sys.path.insert(0, str(Path(a.dl).resolve() / "src"))
from transformers import AutoTokenizer  # noqa: E402
from hiremind_dl import load_config  # noqa: E402
from hiremind_dl.data import as_dataset, download_raw, load_manifest, read_raw, split_frame, tokenize  # noqa: E402
from hiremind_dl.training import train_run  # noqa: E402

cfg = load_config(Path(a.dl) / "configs" / "experiment.json")
assert cfg["tokenization"]["max_len"] == 512
batches = -(-cfg["dataset"]["rows"]["train"] // cfg["training"]["batch"])
cfg["training"]["epochs"], cfg["training"]["updates"] = 3, 3 * batches
raw = {"train": read_raw(download_raw(cfg, "train"))}  # test file is never downloaded
man = load_manifest(cfg)
tok = AutoTokenizer.from_pretrained(cfg["student"]["name"], revision=cfg["student"]["revision"])
ds, kept_stats = {}, {}
for part in ("train", "validation"):
    d = split_frame(raw, man, part, cfg["dataset"]["rows"][part])
    cvs, jds, ntk = [], [], []
    for r in d.itertuples():
        cv = r.resume_text if isinstance(r.resume_text, str) else ""
        jd = r.job_description_text if isinstance(r.job_description_text, str) else ""
        kline = knowledge_line(kfeat(cv, jd)) + "\n"            # computed from the FULL texts, as in M4
        pair = structure_pair(r.row_uid, cv, jd)
        sel = select_pair(cv, jd, tok, pair, None, reserve=len(tok(kline, add_special_tokens=False)["input_ids"]))
        tcv, tjd = assemble(cv, jd, sel)
        assert len(tok(tcv, kline + tjd)["input_ids"]) <= 512, "evidence input would be truncated"
        cvs.append(tcv)
        jds.append(kline + tjd)
    d = d.copy()
    d["resume_text"], d["job_description_text"] = cvs, jds
    ds[part] = as_dataset(tokenize(d, tok, 512))
    kept_stats[part] = len(d)
dev = "cpu"
torch.set_num_threads(os.cpu_count() or 4)
info, best_state = train_run("M0", cfg["selected"]["M0"]["lr"], a.seed, cfg, ds["train"], ds["validation"], dev)
b = info["best"]
out = {"variant": "E2_evidence_lex_k_D0_extractor_v3.1", "seed": a.seed, "labels": "D0 original", "input": "B4 lex_k selection (extractor v3.1) + Knowledge v1 line", "epochs": 3, "device": dev,
       "best_step": b["step"], "updates": info["updates"], "seconds": round(info["seconds"]), "accuracy": b["accuracy"], "macro_f1": b["macro_f1"],
       "good_fit_recall": b["good_fit_recall"], "good_fit_precision": 100 * b["per_class"]["Good Fit"]["precision"],
       "potential_fit_recall": 100 * b["per_class"]["Potential Fit"]["recall"], "confusion": b["confusion"], "history": info["history"], "torch": torch.__version__}
from hiremind_dl.evaluation import evaluate  # noqa: E402
from hiremind_dl.models import build_model  # noqa: E402
m = build_model("M0", cfg, a.seed)
m.load_state_dict(best_state)
_, logits, pred = evaluate(m.to(dev), ds["validation"], dev, 64)
prob = logits.softmax(-1).numpy()
v = split_frame(raw, man, "validation", cfg["dataset"]["rows"]["validation"])
outd = Path(a.out)
outd.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"row_uid": v.row_uid, "job_id": v.job_id, "job_group": v.job_group, "label": v.label_id, "pred": pred, "p_no": prob[:, 0], "p_potential": prob[:, 1], "p_good": prob[:, 2]}) \
    .to_csv(outd / f"val_predictions_E2_s{a.seed}.csv", index=False)
(outd / f"E2_run_s{a.seed}.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k != "history"}, indent=2))
