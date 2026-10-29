#!/bin/bash
# End-to-end smoke test at tiny size (< 2 min): train 40 steps -> tune+eval on 20 maps/set -> demo PNGs.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
T=$(mktemp -d)
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
$PY train.py --minutes 1 --max_steps 40 --n_train 500 --out "$T/ck/unet"
$PY evaluate.py --ckpt "$T/ck/unet" --n_val 20 --n_test 20 --out "$T/res" > "$T/eval.txt"
mkdir -p "$T/media"
$PY demo.py --results "$T/res" --media "$T/media" --n 2
$PY - "$T" <<'PY'
import json, sys, os
t = sys.argv[1]
r = json.load(open(f"{t}/res/eval.json"))["results"]
assert set(r) == {"val", "test_id", "test_bighole", "test_delaunay"}
for s in r.values():
    for m in ("fill_all", "diffusion", "stub_lines", "unet"):
        assert 0 <= s[m]["correct_and_faithful"] <= 1
assert os.path.exists(f"{t}/media/strip_test_id.png")
# ground truth scored against itself must be perfect
import numpy as np
from maps import build_set, VAL0
from metrics import score, aggregate
d = build_set(range(VAL0, VAL0 + 20), "grid", (12, 24), hole_seed=11)
a = aggregate([score(im, h, im) for im, h in zip(d["ims"], d["holes"])])
assert a["correct_and_faithful"] == 1.0 and a["pair_f1"] == 1.0, a
print("SMOKE OK")
PY
rm -rf "$T"
