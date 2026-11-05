#!/bin/bash
# SUPERSEDED by r7_queue3.sh before launching anything of its own (kept because the ledger refers to it); do not run.
# Round 7 queue (amendment 2): after the 1B-v0 selections and the v2 belief side files: train 1B-v1 (two seeds,
# two lanes), select both, then lane 1 = 1C belief-heads -> 1C onset-weight, lane 2 = the 1D servo baseline.
cd "$(dirname "$0")"
step() { ../.venv/bin/python -c "import json;print(json.load(open('$1/latest.json'))['step'])" 2>/dev/null || echo 0; }
until ! pgrep -f "r7_eval.py select --run runs/r7_1b" > /dev/null && [ -f results/r7/.belief_v2_ready ]; do sleep 20; done
echo "[queue2] $(date +%H:%M:%S) 1B-v0 selections done and v2 side files ready; training 1B-v1"
./r7_lane.sh 1bv1_s0 > logs/r7_lane_1bv1_s0.log 2>&1 &
./r7_lane.sh 1bv1_s1 > logs/r7_lane_1bv1_s1.log 2>&1 &
wait
echo "[queue2] $(date +%H:%M:%S) 1B-v1 trained ($(step runs/r7_1bv1_s0) / $(step runs/r7_1bv1_s1)); selecting"
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select --run runs/r7_1bv1_s0 > logs/r7_1bv1_s0_select.log 2>&1 &
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select --run runs/r7_1bv1_s1 > logs/r7_1bv1_s1_select.log 2>&1 &
wait
echo "[queue2] $(date +%H:%M:%S) 1B-v1 selections done: $(tail -1 logs/r7_1bv1_s0_select.log) | $(tail -1 logs/r7_1bv1_s1_select.log)"
./r7_lane.sh 1c_bh_s1 1c_ow_s1 > logs/r7_lane3.log 2>&1 &
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_servo_baseline.py select > logs/r7_servo_select.log 2>&1
echo "[queue2] $(date +%H:%M:%S) servo done"
wait
echo "[queue2] $(date +%H:%M:%S) 1C lane done"
