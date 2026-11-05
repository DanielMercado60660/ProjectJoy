#!/bin/bash
# SUPERSEDED by r7_queue2.sh before launching anything of its own (kept because the ledger refers to it); do not run.
# Round 7 queue after the 1B selections: lane 1 = 1C belief-heads run then 1C onset-weight run; lane 2 = the
# 1D servo baseline (40 selection rooms, both latches). Never more than two of my jobs at once.
cd "$(dirname "$0")"
until ! pgrep -f "r7_eval.py select --run runs/r7_1b" > /dev/null; do sleep 20; done
echo "[queue1] $(date +%H:%M:%S) 1B selections finished; starting 1C lane and the 1D servo"
./r7_lane.sh 1c_bh_s1 1c_ow_s1 > logs/r7_lane3.log 2>&1 &
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_servo_baseline.py select > logs/r7_servo_select.log 2>&1
echo "[queue1] $(date +%H:%M:%S) servo done: $(grep -c policy logs/r7_servo_select.log) rows"
wait
echo "[queue1] $(date +%H:%M:%S) 1C lane done"
