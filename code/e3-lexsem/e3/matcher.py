"""B3 requirement-evidence matcher: exact (from B1) + semantic (DistilBERT, per team decision).
Usage: python matcher.py distilbert <HireMind-DL checkout> <structured_pairs.jsonl> <OUT dir>
Writes match_features.jsonl (per pair: requirement-level features + pair aggregates), match_features_pair.csv, B3_matcher_report.json.
Output has IDs, offsets and scores only; no CV or job text.
Design rules (fixed before looking at any result):
  * Exact evidence comes from B1 and keeps its meaning (FOUND_EXPLICIT / FOUND_EXPLICIT_NEGATIVE / NOT_STATED_UNKNOWN ...).
  * Semantic similarity is kept as a number, plus the offset of the best CV unit. It is never turned into a hidden decision.
  * missing_indicator is its own field: true only when there is no exact evidence AND best semantic score < SEM_PRESET. "Missing" means
    "no supporting evidence found", never "candidate lacks it". SEM_PRESET is a preset, not tuned on labels.
  * No automatic rejection: these are features for a model and for HR review."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from structured_extractor import sentence_starts  # noqa: E402

ENCODERS = {"distilbert": "distilbert-base-uncased"}  # team decision: DistilBERT only (the plan text names all-MiniLM-L6-v2; deviation recorded in B3 report)
# Semantic threshold rule (label-free, fixed before results): per encoder, the 10th percentile of the best-sentence similarity
# among requirements that B1 already found by exact match. Raw DistilBERT is not a sentence encoder and its cosines are compressed
# into a high range, so a fixed 0.50 would be meaningless; the rule adapts to the encoder.
THRESH_Q = 0.10
MAX_UNIT_CHARS = 400


def units(text):
    """Sentence-like units [(start, end)] of the original text; long runs are cut at whitespace so the encoder does not truncate them silently."""
    st = sentence_starts(text)
    out = []
    for i, s in enumerate(st):
        e = st[i + 1] if i + 1 < len(st) else len(text)
        while s < e and text[s].isspace():
            s += 1
        if e - s < 3:
            continue
        while e - s > MAX_UNIT_CHARS:
            cut = text.rfind(" ", s, s + MAX_UNIT_CHARS)
            cut = cut if cut > s + 100 else s + MAX_UNIT_CHARS
            out.append((s, cut))
            s = cut + 1
        if e - s >= 3:
            out.append((s, e))
    return out


class Embedder:
    def __init__(self, model_id):
        from sentence_transformers import SentenceTransformer
        self.model_id = model_id
        self.m = SentenceTransformer(model_id, device="cpu")  # plain transformer ids get mean pooling
        self.cache = {}

    def enc(self, texts):
        need = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if need:
            v = self.m.encode(need, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            for t, e in zip(need, v):
                self.cache[t] = e
        return np.stack([self.cache[t] for t in texts])


def match_pair(pair, cv_text, jd_text, emb):
    cvu = units(cv_text)
    cv_vec = emb.enc([cv_text[a:b] for a, b in cvu]) if cvu else np.zeros((0, 384))
    reqs = []
    for L in pair["requirement_links"]:
        rec = next(r for r in pair["records"] if r["record_id"] == L["requirement_id"])
        sent = jd_text[rec["context_start_char"]:rec["context_end_char"]].strip()[:MAX_UNIT_CHARS] if rec.get("context_start_char") is not None else ""
        term = L["normalized_value"].replace("_", " ").lower()
        qs = [t for t in (term, sent) if t]
        feat = dict(requirement_id=L["requirement_id"], category=L["category"], requirement_type=L["requirement_type"],
                    normalized_value=L["normalized_value"], exact_status=L["link_status"], exact_found=L["link_status"] in ("FOUND_EXPLICIT", "LEVEL_MET"),
                    exact_negative=L["link_status"] == "FOUND_EXPLICIT_NEGATIVE", sem_term_max=None, sem_sentence_max=None, best_unit_start=None, best_unit_end=None)
        if len(cv_vec) and qs:
            qv = emb.enc(qs)
            sims = qv @ cv_vec.T
            if term:
                j = int(sims[0].argmax())
                feat["sem_term_max"] = round(float(sims[0, j]), 4)
            if sent:
                j = int(sims[-1].argmax())
                feat["sem_sentence_max"] = round(float(sims[-1, j]), 4)
                feat["best_unit_start"], feat["best_unit_end"] = cvu[j]
        reqs.append(feat)
    return reqs


def finalize(reqs, thr):
    for feat in reqs:
        best = feat["sem_sentence_max"] if feat["sem_sentence_max"] is not None else (feat["sem_term_max"] or 0.0)
        feat["missing_indicator"] = bool(not feat["exact_found"] and not feat["exact_negative"] and best < thr)
        feat["semantic_only_candidate"] = bool(not feat["exact_found"] and not feat["exact_negative"] and best >= thr)


def aggregates(reqs):
    def sel(t):
        return [r for r in reqs if r["requirement_type"] == t]
    out = {}
    for t in ("REQUIRED", "PREFERRED", "UNKNOWN"):
        s = sel(t)
        out[f"n_{t.lower()}"] = len(s)
        out[f"exact_rate_{t.lower()}"] = round(sum(r["exact_found"] for r in s) / len(s), 4) if s else None
        sims = [r["sem_sentence_max"] for r in s if r["sem_sentence_max"] is not None]
        out[f"sem_mean_{t.lower()}"] = round(float(np.mean(sims)), 4) if sims else None
        out[f"sem_min_{t.lower()}"] = round(float(np.min(sims)), 4) if sims else None
        out[f"n_missing_{t.lower()}"] = sum(r["missing_indicator"] for r in s)
        out[f"n_semantic_only_{t.lower()}"] = sum(r["semantic_only_candidate"] for r in s)
        out[f"n_exact_negative_{t.lower()}"] = sum(r["exact_negative"] for r in s)
    out["education_level_met"] = next((r["exact_status"] for r in reqs if r["category"] == "EDUCATION"), None)
    out["has_no_requirements"] = len(reqs) == 0
    return out


if __name__ == "__main__":
    enc_name, root, spj, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    MODEL_ID = ENCODERS[enc_name]
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "src"))
    from hiremind_dl import load_config  # noqa: E402
    from hiremind_dl.data import download_raw, load_manifest, read_raw, split_frame  # noqa: E402
    cfg = load_config()
    tr = split_frame({"train": read_raw(download_raw(cfg, "train"))}, load_manifest(cfg), "train", expected_rows=4993)
    txt = {r.row_uid: (r.resume_text if isinstance(r.resume_text, str) else "", r.job_description_text if isinstance(r.job_description_text, str) else "")
           for r in tr.itertuples()}
    emb = Embedder(MODEL_ID)
    try:
        from huggingface_hub import model_info
        rev = model_info(MODEL_ID).sha
    except Exception as e:  # noqa: BLE001
        rev = f"unresolved: {e.__class__.__name__}"
    pairs = []
    for line in open(spj):
        p = json.loads(line)
        cv, jd = txt[p["case_id"]]
        assert len(cv) == p["cv_chars"] and len(jd) == p["jd_chars"], "source text differs from the one B1 pointed into"
        pairs.append((p["case_id"], match_pair(p, cv, jd, emb)))
    allr = [r for _, reqs in pairs for r in reqs]
    ref = [r["sem_sentence_max"] for r in allr if r["exact_found"] and r["sem_sentence_max"] is not None]
    thr = round(float(np.quantile(ref, THRESH_Q)), 4)
    rows = []
    with open(out / "match_features.jsonl", "w") as fo:
        for cid, reqs in pairs:
            finalize(reqs, thr)
            agg = aggregates(reqs)
            fo.write(json.dumps(dict(case_id=cid, encoder=enc_name, requirements=reqs, aggregates=agg), separators=(",", ":")) + "\n")
            rows.append(dict(case_id=cid, **agg))
    with open(out / "match_features_pair.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sims = [r["sem_sentence_max"] for r in allr if r["sem_sentence_max"] is not None]
    by = {}
    for k, f in [("exact_found", lambda r: r["exact_found"]), ("not_found_unknown", lambda r: r["exact_status"] == "NOT_STATED_UNKNOWN"),
                 ("exact_negative", lambda r: r["exact_negative"])]:
        v = [r["sem_sentence_max"] for r in allr if f(r) and r["sem_sentence_max"] is not None]
        by[k] = dict(n=len(v), mean=round(float(np.mean(v)), 4) if v else None, q10_q50_q90=[round(float(x), 4) for x in np.quantile(v, [.1, .5, .9])] if v else None)
    json.dump(dict(encoder=enc_name, model=MODEL_ID, model_revision=rev, threshold_rule=f"q{int(THRESH_Q*100)} of best-sentence similarity among B1 exact-found requirements (no dataset labels used)",
                   threshold=thr, pairs=len(rows), requirements=len(allr), requirements_with_semantic_score=len(sims), similarity_by_exact_status=by,
                   missing_indicator_total=sum(r["missing_indicator"] for r in allr), semantic_only_total=sum(r["semantic_only_candidate"] for r in allr),
                   note="Scores and offsets only. 'missing' = no supporting evidence found, not 'lacks the skill'."),
              open(out / "B3_matcher_report.json", "w"), indent=1)
    print(open(out / "B3_matcher_report.json").read())
