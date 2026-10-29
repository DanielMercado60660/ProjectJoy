#!/bin/bash
# GPU-slot job (<45 min): ppo2 = ppo1 continued (weights + Adam state + counters) with a 3x
# starvation penalty, because ppo1 starves next to a full nest once the food runs out.
set -x
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=../.venv/bin/python
if [ ! -d checkpoints/ppo2 ]; then
  mkdir -p checkpoints/ppo2
  cp checkpoints/ppo1/model.safetensors checkpoints/ppo1/state.pkl checkpoints/ppo1/config.json checkpoints/ppo2/
fi
$PY train.py --run ppo2 --death 3 --minutes 58 --resume
$PY evaluate.py --baselines --ppo checkpoints/ppo2 --out results/eval_D.json
