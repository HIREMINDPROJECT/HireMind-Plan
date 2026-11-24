"""B2 evaluation. Usage:
  python b2_evaluate.py gold <restricted csv> <annotator1.csv> <annotator2.csv> <OUT dir>     -> agreement + gold template for adjudication
  python b2_evaluate.py eval <restricted csv> <extraction_gold.csv> <structured_pairs.jsonl[.gz]> <OUT dir>   -> extraction_evaluation.csv
extraction_gold.csv columns: case_id, source_type, field_type, span_text_copied_exactly, normalized_value, status, requirement_type  (adjudicated rows).
A predicted claim matches a gold claim when case, source_type, field_type and normalized value (case-insensitive) are equal. Claims are compared as distinct (case, source, field, normalized value); a term mentioned twice counts once. Provenance (is the predicted span text one the annotators also copied?) is reported separately."""
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

mode, src = sys.argv[1], Path(sys.argv[2])
raw = pd.read_csv(src).drop_duplicates("row_uid").set_index("row_uid")
SRC = {"CV": "FULL_CV_TEXT_PRIVATE", "JOB_DESCRIPTION": "FULL_JOB_DESCRIPTION_PRIVATE"}
FIELD = {"SKILL": ("SKILL", "TOOL_TECHNOLOGY"), "SKILL_TOOL": ("SKILL", "TOOL_TECHNOLOGY"), "EDUCATION": ("EDUCATION",), "YEARS_EXPERIENCE": ("EXPERIENCE",),
         "CERTIFICATION": ("CERTIFICATION",), "EMPLOYMENT_DATES": ("EMPLOYMENT",)}


def col(df, prefix):
    return next(c for c in df.columns if c.startswith(prefix))


def locate(text, span):
    span = str(span).strip()
    i = text.find(span)
    return (i, i + len(span)) if i >= 0 and span else None


def norm(df):
    c = {k: col(df, k) for k in ["case_id", "source_type", "field_type", "span_text", "normalized", "status", "requirement_type"]}
    rows = []
    for _, r in df.iterrows():
        if not str(r[c["span_text"]]).strip() or str(r[c["span_text"]]) == "nan":
            continue
        cid, st = str(r[c["case_id"]]), str(r[c["source_type"]]).strip()
        loc = locate(raw.loc[cid, SRC[st]], r[c["span_text"]])
        rows.append(dict(case_id=cid, source_type=st, field_type=str(r[c["field_type"]]).strip(), span=loc, text=str(r[c["span_text"]]).strip().lower(), normalized=str(r[c["normalized"]]).strip().lower(),
                         status=str(r[c["status"]]).strip(), requirement_type=str(r[c["requirement_type"]]).strip() if str(r[c["requirement_type"]]) != "nan" else ""))
    return rows


def key(x):
    return (x["case_id"], x["source_type"], x["field_type"], x["normalized"])


def match(a, b):
    """Claim-level match: same case, source, field type and normalized value. A claim can occur many times in a text, so
    positions are not compared here; provenance is checked separately (provenance_rate)."""
    return key(a) == key(b)


def distinct(rows):
    seen = {}
    for r in rows:
        seen.setdefault(key(r), r)
    return list(seen.values())


def prf(pred, gold):
    pred, gold = distinct(pred), distinct(gold)
    gk, pk = {key(g) for g in gold}, {key(p) for p in pred}
    tp = len(gk & pk)
    pr, rc = (tp / len(pk) if pk else None), (tp / len(gk) if gk else None)
    f1 = 2 * pr * rc / (pr + rc) if pr and rc else (0.0 if pk and gk else None)
    return len(gk), len(pk), pr, rc, f1


out = Path(sys.argv[-1])
out.mkdir(parents=True, exist_ok=True)
if mode == "gold":
    a1, a2 = norm(pd.read_csv(sys.argv[3])), norm(pd.read_csv(sys.argv[4]))
    n, p1, p2, _, f1 = (lambda t: t)(prf(a2, a1))
    unmatched = [x for x in a1 if not any(match(x, y) for y in a2)] + [y for y in a2 if not any(match(y, x) for x in a1)]
    json.dump(dict(annotator1_claims=len(a1), annotator2_claims=len(a2), agreement_f1_between_annotators=f1,
                   claims_needing_adjudication=len(unmatched), status="PENDING_ADJUDICATION" if unmatched else "READY_FOR_GOLD"), open(out / "B2_annotator_agreement.json", "w"), indent=1)
    with open(out / "extraction_gold_TO_ADJUDICATE.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "source_type", "field_type", "normalized", "status", "requirement_type", "span_start", "span_end", "only_annotated_by"])
        for x in unmatched:
            w.writerow([x["case_id"], x["source_type"], x["field_type"], x["normalized"], x["status"], x["requirement_type"], *(x["span"] or ("", "")),
                        "1" if x in a1 else "2"])
    print(open(out / "B2_annotator_agreement.json").read())
else:
    gold = norm(pd.read_csv(sys.argv[3]))
    pairs = [json.loads(l) for l in (gzip.open(sys.argv[4], "rt") if sys.argv[4].endswith(".gz") else open(sys.argv[4]))]
    case_ids = {g["case_id"] for g in gold}
    pred = []
    for p in pairs:
        if p["case_id"] not in case_ids:
            continue
        for r in p["records"]:
            for ft, cats in FIELD.items():
                if r["evidence_category"] in cats and ft in ("SKILL_TOOL", "EDUCATION", "YEARS_EXPERIENCE", "CERTIFICATION", "EMPLOYMENT_DATES"):
                    nv = r["normalized_value"].lower()
                    pred.append(dict(case_id=p["case_id"], source_type=r["source_type"], field_type=ft, span=(r["source_start_char"], r["source_end_char"]), normalized=nv,
                                     text=raw.loc[p["case_id"], SRC[r["source_type"]]][r["source_start_char"]:r["source_end_char"]].strip().lower(),
                                     status=r["evidence_status"], requirement_type=r["requirement_type"] if r["source_type"] == "JOB_DESCRIPTION" else ""))
    rows = []
    for ft in ["SKILL_TOOL", "EDUCATION", "YEARS_EXPERIENCE", "CERTIFICATION", "EMPLOYMENT_DATES", "ALL"]:
        for st in ["CV", "JOB_DESCRIPTION", "BOTH"]:
            g = [x for x in gold if (ft == "ALL" or x["field_type"] == ft) and (st == "BOTH" or x["source_type"] == st)]
            p = [x for x in pred if (ft == "ALL" or x["field_type"] == ft) and (st == "BOTH" or x["source_type"] == st)]
            ng, np_, pr, rc, f1 = prf(p, g)
            rows.append(dict(field_type=ft, source_type=st, gold_claims=ng, predicted_claims=np_, precision=pr, recall=rc, f1=f1))
    # requirement-type accuracy on matched job claims; explicit negatives; unknown handling
    mj = [(g, next(p for p in pred if match(g, p))) for g in distinct(gold) if g["source_type"] == "JOB_DESCRIPTION" and any(match(g, p) for p in pred)]
    gtext = defaultdict(set)
    for g in gold:
        gtext[key(g)].add(g["text"])
    ptext = {}
    for p_ in pred:
        ptext.setdefault(key(p_), p_)
    prov_ok = [any(pp["text"] in t or t in pp["text"] for t in gtext[k]) for k, pp in ptext.items() if k in gtext]
    provenance_rate = round(sum(prov_ok) / len(prov_ok), 4) if prov_ok else None
    rt = sum(1 for g, p in mj if g["requirement_type"] == p["requirement_type"]) / len(mj) if mj else None
    neg_gold = [g for g in distinct(gold) if g["status"] == "EXPLICIT_NEGATIVE"]
    neg_pred = [p for p in distinct(pred) if p["status"] == "EXPLICIT_NEGATIVE"]
    false_neg = [p for p in neg_pred if not any(match(g, p) and g["status"] == "EXPLICIT_NEGATIVE" for g in gold)]
    not_found_in_source = sum(1 for g in gold if g["span"] is None)
    pd.DataFrame(rows).to_csv(out / "extraction_evaluation.csv", index=False)
    json.dump(dict(cases=len(case_ids), provenance_rate_on_matched_claims=provenance_rate, requirement_type_accuracy_on_matched_job_claims=rt, matched_job_claims=len(mj), gold_explicit_negatives=len(neg_gold),
                   predicted_explicit_negatives=len(neg_pred), predicted_negatives_not_confirmed_by_gold=len(false_neg), gold_rows_whose_copied_text_is_not_in_source=not_found_in_source,
                   note="Score of the rule-based extractor against human gold. Valid only once gold is adjudicated from two independent human annotators."),
              open(out / "B2_extra_metrics.json", "w"), indent=1)
    print(pd.DataFrame(rows).to_string())
