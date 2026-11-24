"""Phase D error-analysis FACTS for the 1,162 VALIDATION pairs (objective, label-free; no text in outputs).
Usage: python d_facts.py <HireMind-DL checkout> <OUT dir>
Per pair: job group, true label, token counts, how much relevant B1 evidence the M4 baseline truncation drops vs the B4 lex_k selection,
requirement-type / link-status counts, nearest-TRAIN-job TF-IDF cosine (unseen-job shift), CV seen in TRAIN (CV identity repeats across splits).
These are facts used to PRE-TAG mistakes; they are not human judgements."""
import csv, hashlib, sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
here = Path(__file__).parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(root / "src"))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from b4_select import MAXLEN, SPECIAL, covered, select_pair
from knowledge_v1_ORIGINAL import features as kfeat, knowledge_line
from structured_extractor import structure_pair
from hiremind_dl import load_config
from hiremind_dl.data import download_raw, load_manifest, read_raw, split_frame
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2", revision="30b0a37ccaaa32f332884b96992754e246e48c5f")
cfg = load_config(); raw = {"train": read_raw(download_raw(cfg, "train"))}; man = load_manifest(cfg)
tr = split_frame(raw, man, "train", cfg["dataset"]["rows"]["train"]); va = split_frame(raw, man, "validation", cfg["dataset"]["rows"]["validation"])
H = lambda s: hashlib.sha256((s or "").encode()).hexdigest()
tr_jd = list(dict.fromkeys(x if isinstance(x, str) else "" for x in tr.job_description_text))
tr_cv_hash = {H(x) for x in tr.resume_text if isinstance(x, str)}
vec = TfidfVectorizer(stop_words="english", min_df=2, max_features=50000, sublinear_tf=True).fit(tr_jd)
Xtr = vec.transform(tr_jd)
rows = []
for r in va.itertuples():
    cv = r.resume_text if isinstance(r.resume_text, str) else ""; jd = r.job_description_text if isinstance(r.job_description_text, str) else ""
    pair = structure_pair(r.row_uid, cv, jd)
    kline = knowledge_line(kfeat(cv, jd)) + "\n"; kt = len(tok(kline, add_special_tokens=False)["input_ids"])
    enc = tok(cv, kline + jd, truncation="longest_first", max_length=MAXLEN, return_offsets_mapping=True)
    sid, off = enc.sequence_ids(), enc["offset_mapping"]
    cv_cut = max([e for (s, e), q in zip(off, sid) if q == 0] or [0]); jd_cut = max(0, max([e for (s, e), q in zip(off, sid) if q == 1] or [0]) - len(kline))
    sel = select_pair(cv, jd, tok, pair, None, reserve=kt)
    hit = lambda spans, rel: sum(1 for s, e in rel if covered(spans, s, e))
    links = pair["requirement_links"]
    cnt = lambda k, v: sum(1 for l in links if l[k] == v)
    sim = float((Xtr @ vec.transform([jd]).T).max()) if jd.strip() else 0.0
    rows.append(dict(row_uid=r.row_uid, job_group=r.job_group, label=r.label_id, cv_tokens=sel["cv_all_tokens"], jd_tokens=sel["jd_all_tokens"],
        exceeds_512=int(sel["cv_all_tokens"] + sel["jd_all_tokens"] + SPECIAL + kt > MAXLEN),
        cv_rel=len(sel["cv_rel"]), cv_rel_kept_baseline=hit([(0, cv_cut)], sel["cv_rel"]), cv_rel_kept_lex_k=hit(sel["cv_segments"], sel["cv_rel"]),
        jd_req=len(sel["jd_rel"]), jd_req_kept_baseline=hit([(0, jd_cut)], sel["jd_rel"]), jd_req_kept_lex_k=hit(sel["jd_segments"], sel["jd_rel"]),
        n_links=len(links), n_found=cnt("link_status", "FOUND_EXPLICIT"), n_negative=cnt("link_status", "FOUND_EXPLICIT_NEGATIVE"), n_unclear=cnt("link_status", "FOUND_UNCLEAR"),
        n_unknown=cnt("link_status", "NOT_STATED_UNKNOWN"), n_type_unknown=cnt("requirement_type", "UNKNOWN"),
        job_nn_train_cos=round(sim, 4), cv_in_train=int(H(cv) in tr_cv_hash)))
with open(out / "validation_pair_facts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "validation pairs;", sum(x["exceeds_512"] for x in rows), "exceed 512; cv_in_train", sum(x["cv_in_train"] for x in rows), "; median nn cos", float(np.median([x["job_nn_train_cos"] for x in rows])))
