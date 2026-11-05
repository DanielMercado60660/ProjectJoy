#!/bin/bash
# R8 Gate C, part 2 orchestration after training starts (never more than two of my jobs at once):
#  1. when the first arm finishes, its turn goes to the random-encoder control lane (G3 control);
#  2. when both arms are done, evaluate them (one r8_job, sequential);
#  3. when the control is done, evaluate it; then the gate table and the curves.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
until [ -d runs/r8_lewm_ckpt/final ] || [ -d runs/r8_lewm_scratch/final ]; do sleep 30; done
echo "first arm done at $(date '+%T')"
WAIT_FOR=data/r8_rep/train_data_ready.json r8_rep/lane.sh lewm-rand runs/r8_lewm_rand --init random_encoder --steps 12000 --batch 32 > logs/r8_rep_lane_rand.log 2>&1 &
RAND=$!
until [ -d runs/r8_lewm_ckpt/final ] && [ -d runs/r8_lewm_scratch/final ]; do sleep 30; done
echo "both arms done at $(date '+%T')"
./r8_job.sh lewm-eval bash -c "$PY r8_rep/rep_eval.py eval runs/r8_lewm_ckpt > logs/r8_rep_eval_ckpt.log 2>&1; $PY r8_rep/rep_eval.py eval runs/r8_lewm_scratch > logs/r8_rep_eval_scratch.log 2>&1"
echo "arm evals done at $(date '+%T')"
wait $RAND
./r8_job.sh lewm-eval-rand $PY r8_rep/rep_eval.py eval runs/r8_lewm_rand > logs/r8_rep_eval_rand.log 2>&1
$PY r8_rep/rep_eval.py gates runs/r8_lewm_ckpt runs/r8_lewm_scratch runs/r8_lewm_rand > logs/r8_rep_gates.log 2>&1
$PY r8_rep/rep_curves.py runs/r8_lewm_ckpt runs/r8_lewm_scratch runs/r8_lewm_rand > logs/r8_rep_curves.log 2>&1
echo "all done at $(date '+%T')"
