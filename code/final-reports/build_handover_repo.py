"""Builds the clean hand-over repository (no personal names, no CV/job text).
usage: python build_handover_repo.py <fork_clone> <package_dir> <weights_dir> <notebooks_dir> <out_dir>
- phase folders + final report: copied from the submission package
- code/: the code bundle (workflows kept under code/, so nothing runs on push)
- results/<results-branch>/: the files each finished GitHub Actions run committed
- weights/: the 8 M4 checkpoints + SHA-256
"""
import hashlib, re, shutil, subprocess, sys, zipfile
from apply_links import fix_tree
from pathlib import Path

fork, pkg, wdir, out = map(Path, sys.argv[1:5])
if out.exists(): shutil.rmtree(out)
out.mkdir(parents=True)
SKIP_RUNS = {"45361593555", "45358860971", "45359437931", "45362218813", "45387638647", "45357880772"}  # failed or cancelled
REDACT = [(re.compile(r"(HIREMINDPROJECT|HIREMINDPROJECT)/HireMind-DL", re.I), "HIREMINDPROJECT/HireMind-Plan"),
          (re.compile(r"HIREMINDPROJECT", re.I), "HIREMINDPROJECT"),
          (re.compile(r"HIREMINDPROJECT/HireMind-Plan/raw/main/weights/"), "HIREMINDPROJECT/HireMind-Plan/raw/main/weights/")]
skipped, kept = [], []

def git(*a): return subprocess.run(["git", "-C", str(fork), *a], capture_output=True, check=True).stdout

# 1) package
for p in pkg.iterdir():
    if p.name == "HireMind_code_bundle.zip": continue
    (shutil.copytree if p.is_dir() else shutil.copy2)(p, out / p.name)
shutil.copy2(Path(__file__).parent / "HANDOVER_README.md", out / "README.md")
# 2) code bundle
with zipfile.ZipFile(pkg / "HireMind_code_bundle.zip") as z: z.extractall(out / "code")
# 3) results branches
for line in git("branch", "-r").decode().split():
    if "result" not in line or not line.startswith("origin/"): continue
    b = line[len("origin/"):]
    run = re.search(r"(\d{11})", b)
    if run and run.group(1) in SKIP_RUNS: skipped.append(b); continue
    files = [f for f in git("show", "--name-only", "--format=", f"origin/{b}").decode().splitlines() if f]
    useful = [f for f in files if not f.startswith(".github/") and Path(f).name not in ("pip_freeze.txt", "run_log_tail.txt")]
    if not useful: skipped.append(b); continue
    for f in files:
        if f.startswith(".github/"): continue
        dst = out / "results" / b / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(git("show", f"origin/{b}:{f}"))
    kept.append(b)
# 4) weights
(out / "weights").mkdir()
sums = []
for w in sorted(wdir.glob("a1_knowledge_M4_seed*_weights.pt")):
    shutil.copy2(w, out / "weights" / w.name)
    sums.append(f"{hashlib.sha256(w.read_bytes()).hexdigest()}  {w.name}")
(out / "weights" / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
# 5) links: point everything at HIREMINDPROJECT/HireMind-Plan (same rules as the Drive package)
n_red = fix_tree(out)
(out / "results" / "RUNS_INDEX.md").write_text(
    "# Results of the GitHub Actions runs\n\nEach folder is named after the results branch the run committed (it contains the run ID). "
    "Failed or cancelled runs are left out.\n\n## Included\n" + "\n".join(f"- {b}" for b in sorted(kept)) +
    "\n\n## Left out (failed, cancelled or no result files)\n" + "\n".join(f"- {b}" for b in sorted(skipped)) + "\n")
print(f"kept {len(kept)} result branches, skipped {len(skipped)}, redacted {n_red} files")
