"""B1 builder. Usage: python build_structured_pairs.py <HireMind-DL checkout> <OUT dir>
Reads the frozen TRAIN split (asserts ids/labels against the manifest), writes:
  structured_pairs.jsonl           one JSON object per pair; offsets + SHA-256 of each span; NO source text
  B1_coverage_report.json          counts only
  B1_pointer_verification.json     every record re-sliced from the source and hash-checked
  B1_link_status_summary.csv       link-status counts by category and requirement type
Nothing written here contains CV or job text."""
import collections
import csv
import json
import sys
from pathlib import Path

import jsonschema

here = Path(__file__).parent
sys.path.insert(0, str(here))
from structured_extractor import SCHEMA_VERSION, sha, structure_pair  # noqa: E402

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "src"))
from hiremind_dl import load_config  # noqa: E402
from hiremind_dl.data import download_raw, load_manifest, read_raw, split_frame  # noqa: E402

cfg = load_config()
man = load_manifest(cfg)
tr = split_frame({"train": read_raw(download_raw(cfg, "train"))}, man, "train", expected_rows=4993)
schema = json.load(open(here / "structured_evidence_schema_v3.json"))
validator = jsonschema.Draft202012Validator(schema)

n_rec = bad_ptr = bad_schema = 0
err_msgs = collections.Counter()
link_c = collections.Counter()
cat_c = collections.Counter()
per_pair = []
with open(out / "structured_pairs.jsonl", "w") as fh:
    for r in tr.itertuples():
        cv = r.resume_text if isinstance(r.resume_text, str) else ""
        jd = r.job_description_text if isinstance(r.job_description_text, str) else ""
        res = structure_pair(r.row_uid, cv, jd)
        res["cv_id"], res["job_id"] = r.cv_id, r.job_id
        for x in res["records"]:
            n_rec += 1
            src = cv if x["source_type"] == "CV" else jd
            if sha(src[x["source_start_char"]:x["source_end_char"]]) != x["span_sha256"] or not (0 <= x["source_start_char"] < x["source_end_char"] <= len(src)):
                bad_ptr += 1
            e1 = next(validator.iter_errors(x), None)
            if e1 is not None:
                bad_schema += 1
                err_msgs[(".".join(map(str, e1.absolute_path)), e1.validator)] += 1
            cat_c[(x["source_type"], x["evidence_category"], x["evidence_status"])] += 1
        for l in res["requirement_links"]:
            link_c[(l["category"], l["requirement_type"], l["link_status"])] += 1
        nreq = len(res["requirement_links"])
        per_pair.append((nreq, sum(1 for l in res["requirement_links"] if l["category"] in ("SKILL", "TOOL_TECHNOLOGY")),
                         sum(1 for x in res["records"] if x["source_type"] == "CV")))
        fh.write(json.dumps(res, separators=(",", ":")) + "\n")

with open(out / "B1_link_status_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["category", "requirement_type", "link_status", "n"])
    for k, v in sorted(link_c.items()):
        w.writerow([*k, v])
n = len(per_pair)
rep = dict(schema_version=SCHEMA_VERSION, pairs=n, records=n_rec, pairs_with_no_job_requirement=sum(1 for p in per_pair if p[0] == 0),
           pairs_with_no_skill_requirement=sum(1 for p in per_pair if p[1] == 0), pairs_with_no_cv_evidence=sum(1 for p in per_pair if p[2] == 0),
           median_requirements_per_pair=sorted(p[0] for p in per_pair)[n // 2],
           records_by_source_category_status={"|".join(k): v for k, v in sorted(cat_c.items())},
           link_status_totals={s: sum(v for k, v in link_c.items() if k[2] == s) for s in sorted({k[2] for k in link_c})},
           note="Rule-based extractor. Coverage numbers are NOT accuracy: extraction accuracy is measured in B2 against human annotation.")
json.dump(rep, open(out / "B1_coverage_report.json", "w"), indent=1)
ver = dict(records_checked=n_rec, pointer_failures=bad_ptr, schema_failures=bad_schema, passed=(bad_ptr == 0 and bad_schema == 0),
           first_schema_errors=[f"{k[0]}:{k[1]} x{v}" for k, v in err_msgs.most_common(8)],
           check="re-sliced source text at [start,end) and compared SHA-256; validated each record against structured_evidence_schema_v3.json")
json.dump(ver, open(out / "B1_pointer_verification.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
print(json.dumps(ver, indent=1))
assert ver["passed"], "pointer or schema failure"
