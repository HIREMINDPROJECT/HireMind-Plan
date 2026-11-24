"""Insert a TRAIN-label overlay into Senior-2 train_knowledge.py (validation labels stay D0; split file untouched)."""
import sys
p = sys.argv[1]
s = open(p).read()
old = '        d = split_frame(raw, man, part, cfg["dataset"]["rows"][part])\n'
new = old + ('        if part == "train" and os.environ.get("HIREMIND_TRAIN_LABELS"):\n'
             '            import pandas as _pd\n'
             '            _ov = _pd.read_csv(os.environ["HIREMIND_TRAIN_LABELS"]).set_index("row_uid")\n'
             '            assert sorted(_ov.index) == sorted(d.row_uid), "overlay must cover exactly the TRAIN rows"\n'
             '            _n = int((_ov.loc[d.row_uid, "label_id"].to_numpy() != d.label_id.to_numpy()).sum())\n'
             '            d["label_id"] = _ov.loc[d.row_uid, "label_id"].to_numpy(); d["label"] = _ov.loc[d.row_uid, "label"].to_numpy()\n'
             '            print("TRAIN label overlay applied:", _n, "labels changed", flush=True)\n')
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("patched")
