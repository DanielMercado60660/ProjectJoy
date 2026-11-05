#!/bin/bash
# SUPERSEDED by r7_queue4.sh before launching anything of its own (kept because the ledger refers to it); do not run.
# Round 7 queue 3: after the 1B-v1 lanes finish: select both seeds (two jobs); then, with never more than two of
# my jobs at once,
#   lane 1: 1C belief-heads (seed 1) -> [2A seed 0 -> 2A shuffled seed 0 if 1B passed the reach gate] -> 1C onset
#   lane 2: 1D servo -> Stage 3 selector arms on the 40 rooms -> [2A seed 1 -> 2A shuffled seed 1 if 1B passed]
# The 1B reach gate (>= 6/18 reached in at least one seed) is read from the select files by r7_gate.py.
cd "$(dirname "$0")"
step() { ../.venv/bin/python -c "import json;print(json.load(open('$1/latest.json'))['step'])" 2>/dev/null || echo 0; }
until [ "$(step runs/r7_1bv1_s0)" -ge 16000 ] && [ "$(step runs/r7_1bv1_s1)" -ge 16000 ] && ! pgrep -f "r7_lane.sh 1bv1" > /dev/null; do sleep 20; done
echo "[queue3] $(date +%H:%M:%S) 1B-v1 trained; selecting both seeds"
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select --run runs/r7_1bv1_s0 > logs/r7_1bv1_s0_select.log 2>&1 &
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select --run runs/r7_1bv1_s1 > logs/r7_1bv1_s1_select.log 2>&1 &
wait
echo "[queue3] $(date +%H:%M:%S) 1B-v1 selections done: $(tail -1 logs/r7_1bv1_s0_select.log) | $(tail -1 logs/r7_1bv1_s1_select.log)"
if ../.venv/bin/python r7_gate.py 1b results/r7/select_r7_1bv1_s0.json results/r7/select_r7_1bv1_s1.json; then
  echo "[queue3] $(date +%H:%M:%S) 1B reach gate PASSED: Stage 2A is funded"; STAGE2=1
else
  echo "[queue3] $(date +%H:%M:%S) 1B reach gate FAILED on both seeds: Stage 2A is cut (kill rule)"; STAGE2=0
fi
(
  ./r7_lane.sh 1c_bh_s1 > logs/r7_lane_1c_bh.log 2>&1
  if [ "$STAGE2" = 1 ]; then ./r7_lane.sh 2a_s0 2a_sh_s0 > logs/r7_lane_2a_lane1.log 2>&1; fi
  ./r7_lane.sh 1c_ow_s1 > logs/r7_lane_1c_ow.log 2>&1
  echo "[queue3] $(date +%H:%M:%S) lane 1 done"
) &
(
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_servo_baseline.py select > logs/r7_servo_select.log 2>&1
  echo "[queue3] $(date +%H:%M:%S) servo done"
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select \
    sel_learned_s0=selector:results/r7/skillsel/learned_s0.npz sel_learned_s1=selector:results/r7/skillsel/learned_s1.npz \
    sel_shuffled_s0=selector:results/r7/skillsel/shuffled_s0.npz sel_shuffled_s1=selector:results/r7/skillsel/shuffled_s1.npz \
    sel_freq=selector:freq:results/r7/skillsel/freq.json sel_program=selector:program bot=bot > logs/r7_stage3_select.log 2>&1
  echo "[queue3] $(date +%H:%M:%S) stage 3 selection rooms done"
  if [ "$STAGE2" = 1 ]; then ./r7_lane.sh 2a_s1 2a_sh_s1 > logs/r7_lane_2a_lane2.log 2>&1; fi
  echo "[queue3] $(date +%H:%M:%S) lane 2 done"
) &
wait
echo "[queue3] $(date +%H:%M:%S) all done"
