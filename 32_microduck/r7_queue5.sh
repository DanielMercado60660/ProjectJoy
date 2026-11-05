#!/bin/bash
# Round 7 queue 5: after queue4's "all done": 1B-v1 confirmations on the 80 rooms (one look each, frozen picks); the
# 4A chicken-head pre-gate recording; the 4A test-time replays (step-5 pixel policies with the reflex on, 40 rooms);
# the 3A gate; 3B (decision-level DAgger round + retrain both seeds + selection) only if 3A passes, else a one-seed
# documentation run. Never more than two of my jobs at once. The fresh slice is NOT started here (method freeze first).
cd "$(dirname "$0")"
until grep -q "all done" logs/r7_queue4.log 2>/dev/null; do sleep 30; done
echo "[queue5] $(date +%H:%M:%S) queue4 done; 1B-v1 confirmations (frozen picks: s0 ckpt_012000, s1 ckpt_016000)"
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py confirm r7_1bv1_s0=runs/r7_1bv1_s0/ckpt_012000/weights.safetensors > logs/r7_confirm_1bv1_s0.log 2>&1 &
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py confirm r7_1bv1_s1=runs/r7_1bv1_s1/ckpt_016000/weights.safetensors > logs/r7_confirm_1bv1_s1.log 2>&1 &
wait
echo "[queue5] $(date +%H:%M:%S) 1B confirmations done; 4A pre-gate + replays"
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_gaze_pregate.py > logs/r7_gaze_pregate.log 2>&1 &
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select hr_s1_gaze=gaze:runs/p2_hr_s1/ckpt_016000/weights.safetensors r3_s1_gaze=gaze:runs/p2_r3_s1/ckpt_016000/weights.safetensors > logs/r7_gaze_replay.log 2>&1 &
wait
echo "[queue5] $(date +%H:%M:%S) 4A done; 3A gate"
if ../.venv/bin/python r7_gate.py 3a > logs/r7_gate_3a.log 2>&1; then
  echo "[queue5] $(date +%H:%M:%S) 3A PASSED: 3B dagger round (learned_s0 drives, program labels) then retrain both seeds"
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_skillsel.py dagger learned_s0 --n 300 > logs/r7_3b_dagger.log 2>&1
  for s in 0 1; do OMP_NUM_THREADS=2 ../.venv/bin/python p2_skillsel.py train learned --seed $s --extra results/r7/skillsel/dagger_learned_s0.json >> logs/r7_3b_train.log 2>&1; done
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select sel_learned_s0_dagger=selector:results/r7/skillsel/learned_s0_dagger.npz > logs/r7_3b_select_s0.log 2>&1 &
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select sel_learned_s1_dagger=selector:results/r7/skillsel/learned_s1_dagger.npz > logs/r7_3b_select_s1.log 2>&1 &
  wait
else
  echo "[queue5] $(date +%H:%M:%S) 3A not passed ($(tail -1 logs/r7_gate_3a.log | cut -c1-160)): 3B = one-seed documentation run"
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_skillsel.py dagger learned_s0 --n 300 > logs/r7_3b_dagger.log 2>&1
  OMP_NUM_THREADS=2 ../.venv/bin/python p2_skillsel.py train learned --seed 0 --extra results/r7/skillsel/dagger_learned_s0.json >> logs/r7_3b_train.log 2>&1
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select sel_learned_s0_dagger=selector:results/r7/skillsel/learned_s0_dagger.npz > logs/r7_3b_select_s0.log 2>&1
fi
echo "[queue5] $(date +%H:%M:%S) all done (3B included); freeze the method set, then r7_queue_fresh.sh"
