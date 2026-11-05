#!/bin/bash
# Round 7 queue 4 (replaces queue 3; the pre-registered order): after the 1B-v1 lanes: select both seeds; then the two
# 1C runs on both lanes; then the 1C probes decide 2A's flags (r7_gate.py 1c); then, if 1B passed the reach gate, the
# four 2A runs (two lanes) and their selections; then the 1D servo and the Stage 3 arms. Never more than two jobs.
cd "$(dirname "$0")"
step() { ../.venv/bin/python -c "import json;print(json.load(open('$1/latest.json'))['step'])" 2>/dev/null || echo 0; }
until [ "$(step runs/r7_1bv1_s0)" -ge 16000 ] && [ "$(step runs/r7_1bv1_s1)" -ge 16000 ] && ! pgrep -f "r7_lane.sh 1bv1" > /dev/null; do sleep 20; done
echo "[queue4] $(date +%H:%M:%S) 1B-v1 trained; selecting both seeds"
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select --run runs/r7_1bv1_s0 > logs/r7_1bv1_s0_select.log 2>&1 &
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select --run runs/r7_1bv1_s1 > logs/r7_1bv1_s1_select.log 2>&1 &
wait
echo "[queue4] $(date +%H:%M:%S) 1B-v1 selections done: $(tail -1 logs/r7_1bv1_s0_select.log) | $(tail -1 logs/r7_1bv1_s1_select.log)"
if ../.venv/bin/python r7_gate.py 1b results/r7/select_r7_1bv1_s0.json results/r7/select_r7_1bv1_s1.json > logs/r7_gate_1b.log; then
  echo "[queue4] $(date +%H:%M:%S) 1B reach gate PASSED: Stage 2A is funded"; STAGE2=1
else
  echo "[queue4] $(date +%H:%M:%S) 1B reach gate FAILED on both seeds: Stage 2A is cut (kill rule)"; STAGE2=0
fi
./r7_lane.sh 1c_bh_s1 > logs/r7_lane_1c_bh.log 2>&1 &
./r7_lane.sh 1c_ow_s1 > logs/r7_lane_1c_ow.log 2>&1 &
wait
echo "[queue4] $(date +%H:%M:%S) 1C runs done; probes"
OMP_NUM_THREADS=4 ../.venv/bin/python r7_probes.py onset 1c_bh=runs/r7_1c_bh_s1/ckpt_016000/weights.safetensors 1c_ow=runs/r7_1c_ow_s1/ckpt_016000/weights.safetensors > logs/r7_onset_1c.log 2>&1
OMP_NUM_THREADS=4 ../.venv/bin/python r7_probes.py bearing 1c_bh=runs/r7_1c_bh_s1/ckpt_016000/weights.safetensors > logs/r7_bearing_1c.log 2>&1
FLAGS=$(../.venv/bin/python r7_gate.py 1c results/r7/onset_1c_bh.json results/r7/bearing_1c_bh.json results/r7/onset_1c_ow.json $STAGE2)
echo "[queue4] $(date +%H:%M:%S) 1C decision: 2A extra flags = '$FLAGS'"
if [ "$STAGE2" = 1 ]; then
  export R7_2A_EXTRA="$FLAGS"
  ./r7_lane.sh 2a_s0 2a_sh_s0 > logs/r7_lane_2a_lane1.log 2>&1 &
  ./r7_lane.sh 2a_s1 2a_sh_s1 > logs/r7_lane_2a_lane2.log 2>&1 &
  wait
  echo "[queue4] $(date +%H:%M:%S) 2A runs done; selecting"
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select --run runs/r7_2a_s0 > logs/r7_2a_s0_select.log 2>&1 &
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select --run runs/r7_2a_s1 > logs/r7_2a_s1_select.log 2>&1 &
  wait
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select --run runs/r7_2a_sh_s0 > logs/r7_2a_sh_s0_select.log 2>&1 &
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select --run runs/r7_2a_sh_s1 > logs/r7_2a_sh_s1_select.log 2>&1 &
  wait
  echo "[queue4] $(date +%H:%M:%S) 2A selections done"
fi
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_servo_baseline.py select > logs/r7_servo_select.log 2>&1 &
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select \
  sel_learned_s0=selector:results/r7/skillsel/learned_s0.npz sel_learned_s1=selector:results/r7/skillsel/learned_s1.npz \
  sel_shuffled_s0=selector:results/r7/skillsel/shuffled_s0.npz sel_shuffled_s1=selector:results/r7/skillsel/shuffled_s1.npz \
  sel_freq=selector:freq:results/r7/skillsel/freq.json sel_program=selector:program bot=bot > logs/r7_stage3_select.log 2>&1 &
wait
echo "[queue4] $(date +%H:%M:%S) servo and stage 3 selection rooms done; all done"
