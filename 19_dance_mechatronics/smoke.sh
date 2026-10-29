#!/bin/bash
# End-to-end smoke test at tiny size (well under 2 min): clip feasibility check, tiny baseline tuning,
# tiny PPO run + kill-free resume, tiny evaluation, render a short demo. Writes only to a temp dir.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
T=$(mktemp -d)
export OMP_NUM_THREADS=1
$PY - <<'PY'
from clip import Clip
f = Clip().feasibility()
assert f["cop_inside_foot"] and f["friction_ok"] and f["tau_ok"] and f["Fz_min"] > 0, f
print("clip feasible:", {k: round(v, 3) for k, v in f.items()})
PY
$PY tune_baseline.py --n_random 8 --gens 1 --pop 4 --workers 2 --train_seeds 2 --out "$T/b.json" > "$T/tune.log"
$PY ppo.py --workers 2 --envs_per_worker 2 --T 32 --max_updates 4 --minibatch 64 --save_every 2 \
    --baseline "$T/b.json" --out "$T/ck" > "$T/ppo.log"
$PY ppo.py --workers 2 --envs_per_worker 2 --T 32 --max_updates 6 --minibatch 64 --save_every 2 \
    --baseline "$T/b.json" --out "$T/ck" --resume > "$T/resume.log"
grep -q "resumed at update 4" "$T/resume.log"
$PY eval.py --ckpt "p=$T/ck.pkl" --conditions nominal push80 --seeds 1 --loops 1 --workers 2 \
    --baseline "$T/b.json" --out "$T/e.json" > "$T/eval.log"
$PY - "$T" <<'PY'
import json, pickle, sys
T = sys.argv[1]
ck = pickle.load(open(f"{T}/ck.pkl", "rb"))
assert ck["update"] == 6 and len(ck["history"]) == 6, ck["update"]
e = json.load(open(f"{T}/e.json"))["table"]
assert set(e) == {"pd_only", "baseline", "p"} and "falls_per_loop" in e["p"]["push80"]
print("tune/train/resume/eval OK")
PY
$PY render_demo.py --ckpt "$T/ck.pkl" --loops 0.3 --baseline "$T/b.json" --out "$T/demo" > "$T/render.log"
test -s "$T/demo.mp4" && test -s "$T/demo.gif"
rm -rf "$T"
echo SMOKE OK
