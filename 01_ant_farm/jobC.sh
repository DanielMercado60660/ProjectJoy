#!/bin/bash
# GPU-slot job (<45 min). jobA.sh and jobC.sh are identical "tickets": whichever gets a slot
# first trains + evaluates the main PPO colony and renders the demo; the other trains a second
# PPO seed for a variance check.
set -x
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=../.venv/bin/python
if mkdir results/.claim_main 2>/dev/null; then
  $PY train.py --run ppo1 --minutes 28 --resume
  $PY evaluate.py --ppo checkpoints/ppo1 --out results/eval.json
  $PY demo.py --ppo checkpoints/ppo1 --es checkpoints/es1
else
  $PY train.py --run ppo1_seed1 --seed 1 --minutes 30 --resume
  $PY evaluate.py --baselines --ppo checkpoints/ppo1_seed1 --out results/eval_C.json
fi
