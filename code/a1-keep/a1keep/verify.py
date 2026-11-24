import csv, hashlib, shutil, glob, sys
ok = True
for r in csv.DictReader(open("a1keep/A1_checkpoint_inventory_v3.csv")):
    s = r["seed"]
    fs = glob.glob(f"ck/a1-seed-{s}/**/weights.pt", recursive=True)
    if len(fs) != 1:
        print("seed", s, "weights.pt not found:", fs); ok = False; continue
    h = hashlib.sha256(open(fs[0], "rb").read()).hexdigest()
    match = h == r["fresh_cpu_checkpoint_sha256"]
    print("seed", s, h, "MATCH" if match else "MISMATCH (expected " + r["fresh_cpu_checkpoint_sha256"] + ")")
    ok = ok and match
    shutil.copy(fs[0], f"rel/a1_knowledge_M4_seed{s}_weights.pt")
open("rel/SHA256_CHECK.txt", "w").write("all 8 match the A1 inventory\n" if ok else "MISMATCH, see log\n")
sys.exit(0 if ok else 1)
