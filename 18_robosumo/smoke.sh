#!/bin/bash
# End-to-end smoke test at tiny size (< 1 min): train 20 PPO updates, resume to 30, run a tiny
# tournament, render one bout. Writes only to a temp dir.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
T=$(mktemp -d)
export OMP_NUM_THREADS=2
$PY ppo.py --max_updates 20 --snap_every 5 --out "$T/ck" > "$T/train.log"
$PY ppo.py --resume --max_updates 30 --snap_every 5 --out "$T/ck" > "$T/resume.log"
grep -q "resumed at update 20" "$T/resume.log"
$PY tournament.py --ckpt "$T/ck.pkl" --n 4 --n_small 2 --snaps 3 --workers 2 --out "$T/t.json" > "$T/tour.log"
$PY - "$T" <<'PY'
import json, pickle, sys
T = sys.argv[1]
ck = pickle.load(open(f"{T}/ck.pkl", "rb"))
assert ck["update"] == 30 and len(ck["history"]) == 30 and len(ck["archive"]) == 6, (ck["update"], len(ck["archive"]))
t = json.load(open(f"{T}/t.json"))
assert len(t["rr"]) == 15 and all(r["W"] + r["L"] + r["D"] == 4 for r in t["rr"])
print("train/resume/tournament OK")
PY
$PY render_demo.py --ckpt "$T/ck.pkl" --opponents charger --out "$T/demo" > "$T/render.log"
test -s "$T/demo.gif"
$PY - <<'PY'
from sumo_env import SumoEnv, make_script
from review_scripts import REVIEW_CLASSES
from review_eval import bout_with_contact
r = bout_with_contact(SumoEnv(seed=1), REVIEW_CLASSES["corner"](), make_script("charger"))
assert r["outcome"] in (-1, 0, 1) and len(r["touch"]) == r["steps"]
print("review corner/contact OK")
PY
rm -rf "$T"
echo SMOKE OK
