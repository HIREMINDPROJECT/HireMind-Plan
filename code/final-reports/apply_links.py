"""Point every link in a folder (and in HireMind_code_bundle.zip inside it) at HIREMINDPROJECT/HireMind-Plan.
usage: python apply_links.py <dir>"""
import re, shutil, sys, tempfile, zipfile
from pathlib import Path

PLAN = "https://github.com/HIREMINDPROJECT/HireMind-Plan.git"
GET_PLAN = f"(test -d _plan || git clone -q --depth 1 {PLAN} _plan)"
RULES = [
    # notebooks: code that used to be cloned from branches of the working repository now comes from code/ in this repository
    (re.compile(r"git clone -q -b c-m5-plan https://github\.com/(?:HIREMINDPROJECT|HIREMINDPROJECT)/HireMind-DL\.git _m5 && git -C _m5 checkout -q [0-9a-f]{40}"),
     f"{GET_PLAN} && cp -r _plan/code/m5-distilbert-plan _m5 && cp _plan/A5_label_overlay_D1/train_labels_v1.csv _m5/phase_c2_m5/e2mods/"),
    (re.compile(r"git clone -q -b m8-seeds https://github\.com/(?:HIREMINDPROJECT|HIREMINDPROJECT)/HireMind-DL\.git _m8 && git -C _m8 checkout -q [0-9a-f]{40}"),
     f"{GET_PLAN} && mkdir -p _m8 && cp -r _plan/code/m8-distilbert/m8 _m8/ && cp _plan/code/m8-seeds/m8/m8_distilbert.py _m8/m8/"),
    (re.compile(r"wget -q -O m4_s46\.pt https://github\.com/\S+?a1_knowledge_M4_seed46_weights\.pt \|\| wget -q -O m4_s46\.pt https://github\.com/\S+?a1_knowledge_M4_seed46_weights\.pt"),
     "wget -q -O m4_s46.pt https://github.com/HIREMINDPROJECT/HireMind-Plan/raw/main/weights/a1_knowledge_M4_seed46_weights.pt"),
    (re.compile(r"!rm -rf _dl _s2 _m5 _m8 _plan _plan"), "!rm -rf _dl _s2 _m5 _m8 _plan _plan _plan"),
    # repository and weights addresses
    (re.compile(r"(?:HIREMINDPROJECT|HIREMINDPROJECT)/HireMind-DL", re.I), "HIREMINDPROJECT/HireMind-Plan"),
    (re.compile(r"HIREMINDPROJECT", re.I), "HIREMINDPROJECT"),
    (re.compile(r"HireMind-Plan/raw/main/weights/"), "HireMind-Plan/raw/main/weights/"),
    (re.compile(r"the weights/ folder of the GitHub repository HIREMINDPROJECT/HireMind-Plan"), "the weights/ folder of the GitHub repository HIREMINDPROJECT/HireMind-Plan"),
    (re.compile(r"\brelease a1-checkpoints-v1\b"), "the weights/ folder of HIREMINDPROJECT/HireMind-Plan"),
]
TEXT = {".txt", ".log", ".md", ".yml", ".yaml", ".py", ".json", ".ipynb", ".csv", ".sh", ".patch", ".html", ".cfg", ".toml"}

def fix_tree(root: Path) -> int:
    n = 0
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in TEXT:
            t = f.read_text(errors="surrogateescape"); t2 = t
            for rx, rep in RULES: t2 = rx.sub(rep, t2)
            if t2 != t: f.write_text(t2, errors="surrogateescape"); n += 1
    return n

if __name__ == "__main__":
    root = Path(sys.argv[1])
    n = fix_tree(root)
    z = root / "HireMind_code_bundle.zip"
    if z.exists():
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            with zipfile.ZipFile(z) as zf: zf.extractall(td)
            nz = fix_tree(td)
            if nz:
                z.unlink()
                with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in sorted(td.rglob("*")):
                        zf.write(f, f.relative_to(td).as_posix())
            print("zip files changed:", nz)
    print("files changed:", n)
