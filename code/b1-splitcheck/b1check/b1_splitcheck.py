"""Compare the B1 extractor with a sentence-splitter fix (spaced sentences). Counts only; no text in outputs.
Usage: python b1_splitcheck.py <HireMind-DL checkout> <OUT dir>"""
import importlib.util, json, re, sys
from collections import Counter
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
here = Path(__file__).parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(root / "src"))
def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, here / fn); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
old, new = load("se_old", "structured_extractor.py"), load("se_new", "structured_extractor_v31.py")
from hiremind_dl import load_config  # noqa: E402
from hiremind_dl.data import download_raw, load_manifest, read_raw, split_frame  # noqa: E402
cfg = load_config()
tr = split_frame({"train": read_raw(download_raw(cfg, "train"))}, load_manifest(cfg), "train", expected_rows=4993)
SP = re.compile(r"[a-z0-9\)][.!?;][ \t]+[A-Z•]")
GL = re.compile(r"[a-z0-9\)][.!?;][A-Z•]")
st = Counter(); tdist = {"old": Counter(), "new": Counter()}; ldist = {"old": Counter(), "new": Counter()}
for r in tr.itertuples():
    cv = r.resume_text if isinstance(r.resume_text, str) else ""
    jd = r.job_description_text if isinstance(r.job_description_text, str) else ""
    st["pairs"] += 1
    st["jd_with_spaced_boundary"] += bool(SP.search(jd)); st["jd_with_glued_boundary"] += bool(GL.search(jd))
    st["cv_with_spaced_boundary"] += bool(SP.search(cv)); st["cv_with_glued_boundary"] += bool(GL.search(cv))
    st["jd_spaced_boundaries_total"] += len(SP.findall(jd)); st["jd_glued_boundaries_total"] += len(GL.findall(jd))
    a, b = old.structure_pair(r.row_uid, cv, jd), new.structure_pair(r.row_uid, cv, jd)
    ra = {(x["normalized_value"], x["source_start_char"]): x["requirement_type"] for x in a["records"] if x["source_type"] == "JOB_DESCRIPTION"}
    rb = {(x["normalized_value"], x["source_start_char"]): x["requirement_type"] for x in b["records"] if x["source_type"] == "JOB_DESCRIPTION"}
    st["jd_records_old"] += len(ra); st["jd_records_new"] += len(rb)
    both = set(ra) & set(rb)
    ch = sum(ra[k] != rb[k] for k in both)
    st["jd_records_type_changed"] += ch; st["pairs_with_type_change"] += ch > 0
    st["jd_records_only_old"] += len(set(ra) - set(rb)); st["jd_records_only_new"] += len(set(rb) - set(ra))
    for k, v in ra.items(): tdist["old"][v] += 1
    for k, v in rb.items(): tdist["new"][v] += 1
    for l in a["requirement_links"]: ldist["old"][l["link_status"]] += 1
    for l in b["requirement_links"]: ldist["new"][l["link_status"]] += 1
    st["cv_records_old"] += sum(x["source_type"] != "JOB_DESCRIPTION" for x in a["records"]); st["cv_records_new"] += sum(x["source_type"] != "JOB_DESCRIPTION" for x in b["records"])
res = dict(counts=dict(st), jd_requirement_type_old=dict(tdist["old"]), jd_requirement_type_new=dict(tdist["new"]), link_status_old=dict(ldist["old"]), link_status_new=dict(ldist["new"]))
json.dump(res, open(out / "b1_splitcheck.json", "w"), indent=1); print(json.dumps(res, indent=1))
