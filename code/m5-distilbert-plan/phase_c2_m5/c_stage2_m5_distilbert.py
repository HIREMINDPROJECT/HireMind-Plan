"""Phase C stage 2: M5 DistilBERT (distilbert-base-uncased, 3-class head), fine-tuned on TRAIN with ORIGINAL labels (D0), evaluated on VALIDATION only.
Protocol copied from M4 so the comparison is fair: AdamW, batch 16, 3 epochs, linear decay, best-validation-macro-F1 checkpoint (first best wins ties).
The best-checkpoint rule uses the same validation set that is reported, exactly as M4 does, so both numbers are optimistic in the same way.
Input path = plain (CV, job) pair, longest_first truncation. This differs from M4 (which adds the Knowledge v1 line) and is documented as a different input path.
Memory/time rule (fixed before running, set after the first attempt was killed, probably out of memory at 16x512): the effective batch stays 16 but is built from micro-batches of 4 (gradient accumulation; same update). Max length = the largest of 512, 384, 256, 192, 128 whose projected full-run time (6 timed micro-steps at 256 tokens, scaled linearly with length) is <= 4.4 h. Training also stops after an epoch if the next one would pass 5.3 h of wall time (reported).
One seed (42): CPU cost makes 8 seeds impossible. The test set is never read. No text leaves the runner.
Usage: python c_stage2_m5_distilbert.py <HireMind-DL checkout> <OUT dir> [MAXLEN_OVERRIDE]"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "src"))
from hiremind_dl import load_config  # noqa: E402
from hiremind_dl.data import download_raw, load_manifest, read_raw, split_frame  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

MODEL = "distilbert-base-uncased"
SEED = int(os.environ.get("M5_SEED", "42"))  # extra seeds 43-45 added 24 Nov 2026; same protocol
EPOCHS, BATCH, LR = 3, 16, 3e-5
MICRO = 4
torch.set_num_threads(4)
torch.manual_seed(SEED)
np.random.seed(SEED)
cfg = load_config()
man = load_manifest(cfg)
raw = {"train": read_raw(download_raw(cfg, "train"))}
tr = split_frame(raw, man, "train", 4993)
va = split_frame(raw, man, "validation", 1162)
assert np.bincount(va.label_id).tolist() == [579, 294, 289]
# ---- Optional inputs, each copied from the BERT-Tiny experiments of the supervisor's plan (all off = plain M5):
# M5_LABELS=<train_labels_v1.csv>  D1 TRAIN-label overlay exactly as E1/E2 (255 changes; validation stays D0)
# M5_KNOWLEDGE=1                   Knowledge v1 line prepended to the job text, computed from the full texts (as M4/E0/E1)
# M5_EVIDENCE=1                    B4 evidence-aware selection (lex_k, extractor v3.1) + Knowledge v1 line (as E2), budget = max length
USE_KNOWLEDGE = os.environ.get("M5_KNOWLEDGE", "0") == "1"
USE_EVIDENCE = os.environ.get("M5_EVIDENCE", "0") == "1"
LABELS = "D1" if os.environ.get("M5_LABELS") else "D0"
if LABELS == "D1":
    _ov = pd.read_csv(os.environ["M5_LABELS"]).set_index("row_uid")
    assert sorted(_ov.index) == sorted(tr.row_uid)
    _n = int((_ov.loc[tr.row_uid, "label_id"].to_numpy() != tr.label_id.to_numpy()).sum()); assert _n == 255, _n
    tr = tr.copy(); tr["label_id"] = _ov.loc[tr.row_uid, "label_id"].to_numpy()
    print("TRAIN label overlay applied:", _n, "labels changed", flush=True)
if USE_KNOWLEDGE and not USE_EVIDENCE:
    sys.path.insert(0, os.environ["M5_KNOWLEDGE_DIR"])
    from knowledge import features, knowledge_line  # noqa: E402
    for _d in (tr, va):
        _d["job_description_text"] = [knowledge_line(features(cv, job)) + "\n" + job for cv, job in zip(_d.resume_text, _d.job_description_text)]
    print("Knowledge v1 line added to", len(tr), "train and", len(va), "validation job texts", flush=True)
TAG = f"s{SEED}" + ("_knowledge" if (USE_KNOWLEDGE or USE_EVIDENCE) else "") + ("_evidence" if USE_EVIDENCE else "") + ("_D1" if LABELS == "D1" else "")
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    from huggingface_hub import model_info
    rev = model_info(MODEL).sha
except Exception as e:  # noqa: BLE001
    rev = f"unresolved: {e.__class__.__name__}"


def enc(df, L):
    e = tok([x if isinstance(x, str) else "" for x in df.resume_text], [x if isinstance(x, str) else "" for x in df.job_description_text],
            truncation="longest_first", max_length=L, padding="max_length", return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


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


model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
ytr, yva = torch.tensor(tr.label_id.tolist()), va.label_id.to_numpy()
# ---- fixed max-length rule (see header)
print("cpus", os.cpu_count(), "threads", torch.get_num_threads(), flush=True)
if len(sys.argv) > 3:
    L, probe = int(sys.argv[3]), None
else:
    xi, xm = enc(tr.iloc[:MICRO], 256)
    opt0 = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()
    for _ in range(6):
        loss = model(input_ids=xi, attention_mask=xm, labels=ytr[:MICRO]).loss
        loss.backward(); opt0.step(); opt0.zero_grad()
    t256 = (time.time() - t0) / 6
    micro_steps = EPOCHS * int(np.ceil(len(tr) / MICRO))
    proj = {l: round(t256 * (l / 256) * micro_steps / 3600 + 0.35 * (l / 256), 2) for l in (512, 384, 256, 192, 128)}
    L = next((l for l in (512, 384, 256, 192, 128) if proj[l] <= 4.4), 128)
    probe = dict(seconds_per_micro_step_256=round(t256, 3), micro_steps=micro_steps, projected_hours_by_len=proj)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)  # reset after the probe
    torch.manual_seed(SEED)
print("max_len", L, probe, flush=True)
if USE_EVIDENCE:
    sys.path.insert(0, os.environ["M5_E2_DIR"])
    import b4_select  # noqa: E402
    from b4_select import select_pair, assemble  # noqa: E402
    from knowledge_v1_ORIGINAL import features as kfeat, knowledge_line as kline_of  # noqa: E402
    from structured_extractor import structure_pair  # noqa: E402
    b4_select.MAXLEN = L  # same selection rule as E2, with the DistilBERT length budget
    for _d in (tr, va):
        cvs, jds = [], []
        for r in _d.itertuples():
            cv = r.resume_text if isinstance(r.resume_text, str) else ""
            jd = r.job_description_text if isinstance(r.job_description_text, str) else ""
            kl = kline_of(kfeat(cv, jd)) + "\n"  # from the FULL texts, as in M4/E2
            sel = select_pair(cv, jd, tok, structure_pair(r.row_uid, cv, jd), None, reserve=len(tok(kl, add_special_tokens=False)["input_ids"]))
            tcv, tjd = assemble(cv, jd, sel)
            assert len(tok(tcv, kl + tjd)["input_ids"]) <= L, "evidence input would be truncated"
            cvs.append(tcv); jds.append(kl + tjd)
        _d["resume_text"], _d["job_description_text"] = cvs, jds
    print("B4 evidence selection + Knowledge v1 applied (budget", L, "tokens)", flush=True)
Xi, Xm = enc(tr, L)
Vi, Vm = enc(va, L)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
steps = EPOCHS * int(np.ceil(len(tr) / BATCH))
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
g = torch.Generator().manual_seed(SEED)
best, best_state, log = -1.0, None, []
t_train = time.time()
step = 0
stopped_early = False
for ep in range(EPOCHS):
    t_ep = time.time()
    model.train()
    perm = torch.randperm(len(tr), generator=g)
    for off in range(0, len(tr), BATCH):
        ix = perm[off:off + BATCH]
        for mo in range(0, len(ix), MICRO):
            mi = ix[mo:mo + MICRO]
            loss = model(input_ids=Xi[mi], attention_mask=Xm[mi], labels=ytr[mi]).loss * (len(mi) / len(ix))
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad()
        step += 1
        if step % 50 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f} elapsed_h {(time.time() - t_train) / 3600:.2f}", flush=True)
    model.eval()
    t0 = time.time()
    preds = []
    with torch.no_grad():
        for off in range(0, len(va), 32):
            preds.append(model(input_ids=Vi[off:off + 32], attention_mask=Vm[off:off + 32]).logits.argmax(-1))
    p = torch.cat(preds).numpy()
    inf_ms = 1000 * (time.time() - t0) / len(va)
    m, cm = metrics(yva, p)
    log.append(dict(epoch=ep + 1, step=step, inference_ms_per_pair=round(inf_ms, 2), **{k: round(v, 4) for k, v in m.items()}))
    print("epoch", ep + 1, {k: round(v, 2) for k, v in m.items() if k in ("accuracy_pct", "macro_f1_pct", "good_fit_recall_pct")}, flush=True)
    if m["macro_f1_pct"] > best:
        best = m["macro_f1_pct"]
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_row = dict(epoch=ep + 1, step=step, metrics=m, cm=cm, p=p, inf_ms=inf_ms)
    ep_h = (time.time() - t_ep) / 3600
    if (time.time() - t_train) / 3600 + ep_h > 5.3 and ep + 1 < EPOCHS:
        stopped_early = True
        print('wall-time guard: stopping after epoch', ep + 1, flush=True)
        break
torch.save(best_state, out / f"M5_distilbert_{TAG}_best.pt")
m, cm = best_row["metrics"], best_row["cm"]
pd.DataFrame(log).to_csv(out / "C_stage2_M5_epoch_log.csv", index=False)
pd.DataFrame({"row_uid": va.row_uid, "true_id": yva, f"M5_distilbert_{TAG}": best_row["p"]}).to_csv(out / "C_stage2_M5_validation_predictions.csv", index=False)
pd.DataFrame([dict(model=f"M5_DistilBERT_{TAG}", role="primary (single seed)", labels=("D1 TRAIN overlay, D0 validation" if LABELS == "D1" else "D0 original"), max_len=L, train_rows=len(tr), validation_rows=len(va), best_epoch=best_row["epoch"],
                   fit_seconds=round(time.time() - t_train, 1), inference_ms_per_pair=round(best_row["inf_ms"], 2), **{k: round(v, 4) for k, v in m.items()},
                   confusion_true_rows_pred_cols=json.dumps(cm.tolist(), separators=(",", ":")))]).to_csv(out / "C_stage2_m5_validation.csv", index=False)
json.dump(dict(model=MODEL, revision=rev, seed=SEED, epochs=EPOCHS, batch=BATCH, lr=LR, max_len=L, probe=probe, checkpoint_rule="best validation macro-F1 (same as M4; optimistic)",
               input_path=("B4 evidence selection + Knowledge v1 line, as E2" if USE_EVIDENCE else "(CV, Knowledge v1 line + job) pair, as M4" if USE_KNOWLEDGE else "plain (CV, job) pair; no Knowledge v1 line"), micro_batch=MICRO, stopped_early_by_wall_guard=stopped_early, test_set_read=False, labels=LABELS, torch=torch.__version__), open(out / "C_stage2_M5_config.json", "w"), indent=1)
