"""B4 builder. Usage: python b4_build.py <HireMind-DL checkout> <OUT dir> [match_features.jsonl(.gz)]
For the TRAIN split writes (no source text in any output):
  evidence_inputs.jsonl   per pair and variant: segment OFFSETS into the original CV/job text + token count
  truncation_audit.csv    per pair and variant: relevant evidence retained under the BASELINE truncation vs the evidence-aware input
  B4_audit_summary.json   aggregates (overall and by class)
Variants: lex (no Knowledge line), lex_k (room reserved for the Knowledge v1 line used by M4); lexsem / lexsem_k when B3 features are given.
Baseline = exactly M4's pair encoding: (CV, [Knowledge line + job]) with longest_first truncation at 512 tokens. For 'lex' variants the baseline is the plain pair."""
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
mfp = Path(sys.argv[3]) if len(sys.argv) > 3 else None
out.mkdir(parents=True, exist_ok=True)
here = Path(__file__).parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(root / "src"))
from b4_select import MAXLEN, SPECIAL, assemble, covered, ntoks, select_pair  # noqa: E402
from knowledge_v1_ORIGINAL import features as kfeat, knowledge_line  # noqa: E402
from structured_extractor import structure_pair  # noqa: E402
from hiremind_dl import load_config  # noqa: E402
from hiremind_dl.data import download_raw, load_manifest, read_raw, split_frame  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2", revision="30b0a37ccaaa32f332884b96992754e246e48c5f")
cfg = load_config()
tr = split_frame({"train": read_raw(download_raw(cfg, "train"))}, load_manifest(cfg), "train", expected_rows=4993)
mf = {}
if mfp:
    for line in (gzip.open(mfp, "rt") if str(mfp).endswith(".gz") else open(mfp)):
        m = json.loads(line)
        mf[m["case_id"]] = m
variants = [("lex", False, False), ("lex_k", False, True)] + ([("lexsem", True, False), ("lexsem_k", True, True)] if mf else [])
rows = []
with open(out / "evidence_inputs.jsonl", "w") as fo:
    for r in tr.itertuples():
        cid = r.row_uid
        cv = r.resume_text if isinstance(r.resume_text, str) else ""
        jd = r.job_description_text if isinstance(r.job_description_text, str) else ""
        pair = structure_pair(cid, cv, jd)
        kline = knowledge_line(kfeat(cv, jd)) + "\n"
        k_tokens = len(tok(kline, add_special_tokens=False)["input_ids"])
        base_cache = {}
        for name, use_sem, use_k in variants:
            # ---- baseline for this variant family
            if use_k not in base_cache:
                jd_in = (kline + jd) if use_k else jd
                enc = tok(cv, jd_in, truncation="longest_first", max_length=MAXLEN, return_offsets_mapping=True)
                sid, off = enc.sequence_ids(), enc["offset_mapping"]
                cv_cut = max([e for (s, e), q in zip(off, sid) if q == 0] or [0])
                jd_cut_in = max([e for (s, e), q in zip(off, sid) if q == 1] or [0])
                jd_cut = max(0, jd_cut_in - (len(kline) if use_k else 0))  # offset back into the job text itself
                base_cache[use_k] = (cv_cut, jd_cut, len(enc["input_ids"]))
            cv_cut, jd_cut, base_n = base_cache[use_k]
            sel = select_pair(cv, jd, tok, pair, mf.get(cid) if use_sem else None, reserve=k_tokens if use_k else 0)
            tcv, tjd = assemble(cv, jd, sel)
            n_ev = len(tok(tcv, (kline + tjd) if use_k else tjd)["input_ids"])
            assert n_ev <= MAXLEN, f"{cid}/{name}: {n_ev} tokens"
            fo.write(json.dumps(dict(case_id=cid, variant=name, join=" ", knowledge_line_prefix=bool(use_k), cv_segments=sel["cv_segments"], jd_segments=sel["jd_segments"], n_tokens=n_ev), separators=(",", ":")) + "\n")
            base_cv, base_jd = [(0, cv_cut)], [(0, jd_cut)]
            hit = lambda spans, rel: sum(1 for s, e in rel if covered(spans, s, e))  # noqa: E731
            rows.append(dict(case_id=cid, label=r.label, variant=name, cv_tokens_all=sel["cv_all_tokens"], jd_tokens_all=sel["jd_all_tokens"],
                             exceeds_512=int(sel["cv_all_tokens"] + sel["jd_all_tokens"] + SPECIAL + (k_tokens if use_k else 0) > MAXLEN),
                             cv_relevant_records=len(sel["cv_rel"]), cv_relevant_kept_baseline=hit(base_cv, sel["cv_rel"]), cv_relevant_kept_evidence=hit(sel["cv_segments"], sel["cv_rel"]),
                             jd_requirement_records=len(sel["jd_rel"]), jd_req_kept_baseline=hit(base_jd, sel["jd_rel"]), jd_req_kept_evidence=hit(sel["jd_segments"], sel["jd_rel"]),
                             baseline_tokens=base_n, evidence_tokens=n_ev))
with open(out / "truncation_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def share(rs, kept, total):
    t = sum(r[total] for r in rs)
    return round(100 * sum(r[kept] for r in rs) / t, 2) if t else None


summ = {}
for name, _, _ in variants:
    L = [r for r in rows if r["variant"] == name and r["exceeds_512"]]
    summ[name] = dict(
        pairs_exceeding_512=len(L),
        cv_relevant_evidence_retained_pct=dict(baseline=share(L, "cv_relevant_kept_baseline", "cv_relevant_records"), evidence_aware=share(L, "cv_relevant_kept_evidence", "cv_relevant_records")),
        jd_requirements_retained_pct=dict(baseline=share(L, "jd_req_kept_baseline", "jd_requirement_records"), evidence_aware=share(L, "jd_req_kept_evidence", "jd_requirement_records")),
        by_label={lab: dict(n=len(g), cv_rel_baseline=share(g, "cv_relevant_kept_baseline", "cv_relevant_records"), cv_rel_evidence=share(g, "cv_relevant_kept_evidence", "cv_relevant_records"),
                            jd_req_baseline=share(g, "jd_req_kept_baseline", "jd_requirement_records"), jd_req_evidence=share(g, "jd_req_kept_evidence", "jd_requirement_records"))
                  for lab, g in ((l, [r for r in L if r["label"] == l]) for l in sorted({r["label"] for r in L}))},
        pairs_evidence_aware_keeps_fewer_cv_relevant=sum(1 for r in L if r["cv_relevant_kept_evidence"] < r["cv_relevant_kept_baseline"]),
        pairs_evidence_aware_keeps_more_cv_relevant=sum(1 for r in L if r["cv_relevant_kept_evidence"] > r["cv_relevant_kept_baseline"]),
        pairs_evidence_aware_keeps_fewer_jd_req=sum(1 for r in L if r["jd_req_kept_evidence"] < r["jd_req_kept_baseline"]),
        median_tokens=dict(baseline=float(np.median([r["baseline_tokens"] for r in rows if r["variant"] == name])), evidence_aware=float(np.median([r["evidence_tokens"] for r in rows if r["variant"] == name]))))
summ["note"] = ("Relevant = CV records linked to a job requirement, explicit negatives, degrees and stated years, plus job requirement records. 'Retained' = every character of the span is in the kept text. "
                "Labels split the report only; they never build inputs. Retaining more relevant evidence does not by itself show the classifier improves: that is tested in E2.")
json.dump(summ, open(out / "B4_audit_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
