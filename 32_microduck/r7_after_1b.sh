#!/bin/bash
# SUPERSEDED by r7_queue4.sh before launching anything of its own (kept because the ledger refers to it); do not run.
# Round 7: wait for both 1B lanes to reach 16k steps, then select each seed on the 40 rooms (two jobs).
cd "$(dirname "$0")"
step() { ../.venv/bin/python -c "import json;print(json.load(open('$1/latest.json'))['step'])" 2>/dev/null || echo 0; }
until [ "$(step runs/r7_1b_s0)" -ge 16000 ] && [ "$(step runs/r7_1b_s1)" -ge 16000 ] && ! pgrep -f "r7_lane.sh 1b_s" > /dev/null; do sleep 20; done
echo "[after1b] $(date +%H:%M:%S) both trained; selecting"
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select --run runs/r7_1b_s0 > logs/r7_1b_s0_select.log 2>&1 &
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select --run runs/r7_1b_s1 > logs/r7_1b_s1_select.log 2>&1 &
wait
echo "[after1b] $(date +%H:%M:%S) selections done: $(tail -1 logs/r7_1b_s0_select.log) | $(tail -1 logs/r7_1b_s1_select.log)"
