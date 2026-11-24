"""B2: build the blank-annotation pack. Usage: python make_b2_annotation_pack.py <restricted review csv> <OUT dir> [N=30]
The restricted csv (team's 400-case review file) has columns row_uid, cv_id, job_id, FULL_CV_TEXT_PRIVATE, FULL_JOB_DESCRIPTION_PRIVATE.
Outputs:
  B2_sample_manifest.csv                  IDs only (safe to upload)
  B2_annotator_sheet_TEMPLATE.csv         blank, IDs only (safe to upload): one copy per annotator
  B2_texts_PRIVATE.txt                    the CV/job texts to annotate  -> LOCAL ONLY, never uploaded
Annotators work BLIND (they do not see any extractor output) and copy the exact text of each claim into span_text."""
import csv
import random
import sys
from pathlib import Path

import pandas as pd

src, out = Path(sys.argv[1]), Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
out.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(src).drop_duplicates("row_uid")
# one case per distinct job text, so the benchmark covers many different jobs, then a seeded random draw
d = d.drop_duplicates("job_id")
rng = random.Random(2026)
ids = sorted(d.row_uid.tolist())
rng.shuffle(ids)
pick = d[d.row_uid.isin(ids[:n])].sort_values("row_uid")
pd.DataFrame({"case_id": pick.row_uid, "cv_id": pick.cv_id, "job_id": pick.job_id}).to_csv(out / "B2_sample_manifest.csv", index=False)
cols = ["annotator_id", "case_id", "source_type(CV|JOB_DESCRIPTION)", "field_type(SKILL_TOOL|EDUCATION|YEARS_EXPERIENCE|CERTIFICATION|EMPLOYMENT_DATES)",
        "span_text_copied_exactly", "normalized_value", "status(EXPLICIT|EXPLICIT_NEGATIVE|UNCLEAR)", "requirement_type(REQUIRED|PREFERRED|UNKNOWN; job text only)", "notes"]
with open(out / "B2_annotator_sheet_TEMPLATE.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for cid in pick.row_uid:
        w.writerow(["", cid, "", "", "", "", "", "", ""])
with open(out / "B2_texts_PRIVATE.txt", "w") as f:
    for r in pick.itertuples():
        f.write(f"===== CASE {r.row_uid} =====\n--- CV ---\n{r.FULL_CV_TEXT_PRIVATE}\n--- JOB DESCRIPTION ---\n{r.FULL_JOB_DESCRIPTION_PRIVATE}\n\n")
print(len(pick), "cases ->", out)
