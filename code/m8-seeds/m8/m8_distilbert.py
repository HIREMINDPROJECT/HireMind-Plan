"""M8 evidence-aware hybrid, DISTILBERT ONLY (team decision). D0 labels, validation only. One seed (42): CPU cost makes 8 seeds impossible.
Components (each can be switched off for E5): evidence = B4 evidence-aware selection (extractor v3.1) instead of plain longest_first truncation;
knowledge = Knowledge v1 line prepended to the job text; struct = B1 structured link features (M3 feature set, standardised on TRAIN) fused with the DistilBERT [CLS] head.
Variants: full | minus_evidence | minus_knowledge | minus_struct. B3 semantic features are NOT in M8 (see B3/E3).
Training protocol = the M5 script exactly (so M5 vs M8 differs only in the added components): distilbert-base-uncased, AdamW lr 3e-5, 6% warm-up, clip 1.0, batch 16 built from micro-batches of 4,
3 epochs, evaluate after each epoch, keep the best-validation-macro-F1 epoch, wall-time guard 5.3 h, MAX LENGTH is passed in (same value for every variant and for the M5 comparison).
Test set never read. No text leaves the runner.
Usage: python m8_distilbert.py <HireMind-DL checkout> <OUT dir> <variant> <MAXLEN> | python m8_distilbert.py --probe <HireMind-DL checkout> <OUT dir>"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROBE = sys.argv[1] == "--probe"
args = sys.argv[2:] if PROBE else sys.argv[1:]
root, out = Path(args[0]), Path(args[1]); out.mkdir(parents=True, exist_ok=True)
variant = "full" if PROBE else args[2]
assert variant in ("full", "minus_evidence", "minus_knowledge", "minus_struct")
L = 0 if PROBE else int(args[3])
USE_EV, USE_K, USE_S = variant != "minus_evidence", variant != "minus_knowledge", variant != "minus_struct"
sys.path.insert(0, str(root / "src"))
from hiremind_dl import load_config  # noqa: E402
from hiremind_dl.data import download_raw, load_manifest, read_raw, split_frame  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402
import b4_select  # noqa: E402
from b4_select import assemble, select_pair  # noqa: E402
from knowledge_v1_ORIGINAL import features as kfeat, knowledge_line  # noqa: E402
from structured_extractor import structure_pair  # noqa: E402

MODEL = "distilbert-base-uncased"
SEED = int(os.environ.get("M8_SEED", "42"))  # extra seeds 43-45 added 24 Nov 2026; same protocol
EPOCHS, BATCH, LR, MICRO = 3, 16, 3e-5, 4
if os.getenv("M8_SMOKE") == "1": EPOCHS = 1
torch.set_num_threads(4); torch.manual_seed(SEED); np.random.seed(SEED)
cfg = load_config(); man = load_manifest(cfg)
raw = {"train": read_raw(download_raw(cfg, "train"))}
tr = split_frame(raw, man, "train", 4993); va = split_frame(raw, man, "validation", 1162)
SMOKE = os.getenv("M8_SMOKE") == "1"  # tiny wiring test only; results are discarded
if SMOKE:
    tr, va = tr.iloc[:64], va.iloc[:48]
else:
    assert np.bincount(va.label_id).tolist() == [579, 294, 289]
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    from huggingface_hub import model_info
    rev = model_info(MODEL).sha
except Exception as e:  # noqa: BLE001
    rev = f"unresolved: {e.__class__.__name__}"
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


class Hybrid(nn.Module):
    """DistilBERT sequence-classification head of M5 (pre_classifier+ReLU+dropout) with an optional structured-feature branch before the last layer."""
    def __init__(self, base, nf):
        super().__init__()
        self.enc, self.pre, self.drop, hid = base.distilbert, base.pre_classifier, base.dropout, base.config.dim
        self.nf = nf
        self.fb = nn.Sequential(nn.Linear(nf, 32), nn.ReLU(), nn.Dropout(0.1)) if nf else None
        self.out = nn.Linear(hid + (32 if nf else 0), 3)
        nn.init.normal_(self.out.weight, std=0.02); nn.init.zeros_(self.out.bias)

    def forward(self, ids, am, x):
        h = self.enc(input_ids=ids, attention_mask=am).last_hidden_state[:, 0]
        h = self.drop(torch.relu(self.pre(h)))
        if self.nf: h = torch.cat([h, self.fb(x)], dim=-1)
        return self.out(h)


def new_model(nf):
    torch.manual_seed(SEED)
    return Hybrid(AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3), nf)


ytr, yva = torch.tensor(tr.label_id.tolist()), va.label_id.to_numpy()
if PROBE:  # same rule as M5: largest length whose projected full-run time is <= 4.4 h (6 timed micro-steps at 256 tokens, scaled linearly)
    e = tok([x if isinstance(x, str) else "" for x in tr.resume_text[:MICRO]], [x if isinstance(x, str) else "" for x in tr.job_description_text[:MICRO]], truncation="longest_first", max_length=256, padding="max_length", return_tensors="pt")
    model = new_model(0); opt0 = torch.optim.AdamW(model.parameters(), lr=LR); model.train(); t0 = time.time()
    for _ in range(6):
        loss = nn.functional.cross_entropy(model(e["input_ids"], e["attention_mask"], None), ytr[:MICRO]); loss.backward(); opt0.step(); opt0.zero_grad()
    t256 = (time.time() - t0) / 6; micro_steps = EPOCHS * int(np.ceil(len(tr) / MICRO))
    proj = {l: round(t256 * (l / 256) * micro_steps / 3600 + 0.35 * (l / 256), 2) for l in (512, 384, 256, 192, 128)}
    Lp = next((l for l in (512, 384, 256, 192, 128) if proj[l] <= 4.4), 128)
    (out / "probe.json").write_text(json.dumps(dict(seconds_per_micro_step_256=round(t256, 3), projected_hours_by_len=proj, chosen_max_len=Lp, cpus=os.cpu_count())))
    print("MAXLEN", Lp, proj); sys.exit(0)

b4_select.MAXLEN = L
frames, feats = {"train": tr, "validation": va}, {}
enc_ids = {}
for part, d in frames.items():
    cvs, jds, F = [], [], []
    for r in d.itertuples():
        cv = r.resume_text if isinstance(r.resume_text, str) else ""; jd = r.job_description_text if isinstance(r.job_description_text, str) else ""
        pair = structure_pair(r.row_uid, cv, jd)
        if USE_S: F.append(lexical(pair, cv, jd))
        kline = (knowledge_line(kfeat(cv, jd)) + "\n") if USE_K else ""
        if USE_EV:
            sel = select_pair(cv, jd, tok, pair, None, reserve=len(tok(kline, add_special_tokens=False)["input_ids"]) if kline else 0)
            tcv, tjd = assemble(cv, jd, sel)
        else:
            tcv, tjd = cv, jd
        cvs.append(tcv); jds.append(kline + tjd)
    e = tok(cvs, jds, truncation="longest_first", max_length=L, padding="max_length", return_tensors="pt")
    enc_ids[part] = (e["input_ids"], e["attention_mask"])
    if USE_S: feats[part] = pd.DataFrame(F)
if USE_S:
    cols = feats["train"].columns; mu = feats["train"].mean(); sd = feats["train"].std().replace(0, 1).fillna(1)
    for part in feats:
        Z = ((feats[part][cols] - mu) / sd).fillna(0.0).clip(-5, 5)  # missing -> TRAIN mean
        feats[part] = torch.tensor(Z.to_numpy(), dtype=torch.float32)
    NF = feats["train"].shape[1]
else:
    NF = 0
Xi, Xm = enc_ids["train"]; Vi, Vm = enc_ids["validation"]
Xf = feats["train"] if USE_S else None; Vf = feats["validation"] if USE_S else None
model = new_model(NF)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
steps = EPOCHS * int(np.ceil(len(tr) / BATCH))
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
g = torch.Generator().manual_seed(SEED)
best, best_state, log, step, stopped_early = -1.0, None, [], 0, False
t_train = time.time()
for ep in range(EPOCHS):
    t_ep = time.time(); model.train(); perm = torch.randperm(len(tr), generator=g)
    for off in range(0, len(tr), BATCH):
        ix = perm[off:off + BATCH]
        for mo in range(0, len(ix), MICRO):
            mi = ix[mo:mo + MICRO]
            loss = nn.functional.cross_entropy(model(Xi[mi], Xm[mi], Xf[mi] if USE_S else None), ytr[mi]) * (len(mi) / len(ix)); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); step += 1
        if step % 50 == 0: print(f"step {step}/{steps} loss {loss.item():.4f} elapsed_h {(time.time() - t_train) / 3600:.2f}", flush=True)
    model.eval(); preds, probs = [], []
    with torch.no_grad():
        for off in range(0, len(va), 32):
            lg = model(Vi[off:off + 32], Vm[off:off + 32], Vf[off:off + 32] if USE_S else None); preds.append(lg.argmax(-1)); probs.append(lg.softmax(-1))
    p = torch.cat(preds).numpy(); pr_ = torch.cat(probs).numpy(); m, cm = metrics(yva, p)
    log.append(dict(epoch=ep + 1, step=step, **{k: round(v, 4) for k, v in m.items()}))
    print("epoch", ep + 1, {k: round(v, 2) for k, v in m.items() if k in ("accuracy_pct", "macro_f1_pct", "good_fit_recall_pct")}, flush=True)
    if m["macro_f1_pct"] > best:
        best = m["macro_f1_pct"]; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_row = dict(epoch=ep + 1, step=step, metrics=m, cm=cm, p=p, prob=pr_)
    if (time.time() - t_train) / 3600 + (time.time() - t_ep) / 3600 > 5.3 and ep + 1 < EPOCHS:
        stopped_early = True; print("wall-time guard: stopping after epoch", ep + 1, flush=True); break
torch.save(best_state, out / f"M8_distilbert_{variant}_s42_best.pt")
m, cm = best_row["metrics"], best_row["cm"]
pd.DataFrame(log).to_csv(out / f"M8_{variant}_epoch_log.csv", index=False)
pd.DataFrame({"row_uid": va.row_uid, "job_id": va.job_id, "job_group": va.job_group, "label": yva, "pred": best_row["p"], "p_no": best_row["prob"][:, 0], "p_potential": best_row["prob"][:, 1], "p_good": best_row["prob"][:, 2]}).to_csv(out / f"val_predictions_M8_{variant}_s42.csv", index=False)
res = dict(model="M8_DistilBERT", variant=variant, components=dict(evidence=USE_EV, knowledge=USE_K, struct=USE_S), seed=SEED, labels="D0 original", extractor="v3.1", encoder=MODEL, encoder_revision=rev, max_len=L,
           best_epoch=best_row["epoch"], accuracy=m["accuracy_pct"], macro_f1=m["macro_f1_pct"], good_fit_recall=m["good_fit_recall_pct"], good_fit_precision=m["good_fit_precision_pct"], potential_fit_recall=m["potential_fit_recall_pct"],
           confusion=cm.tolist(), n_struct_features=NF, stopped_early_by_wall_guard=stopped_early, checkpoint_rule="best validation macro-F1 over 3 epoch-end evaluations (optimistic; same as M5)", test_set_read=False, torch=torch.__version__)
(out / f"M8_{variant}_s42.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: x for k, x in res.items() if k != "confusion"}))
