#!/bin/bash
# One GPU-slot job (<45 min): no-compass PPO ablation, the ES baseline, and their evaluation.
set -x
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=../.venv/bin/python
$PY train.py --run ppo_nocompass --no_compass --minutes 14 --resume
$PY es.py --run es1 --minutes 15 --resume
$PY train.py --run ppo_nocompass --no_compass --minutes 11 --resume
$PY evaluate.py --baselines --ppo checkpoints/ppo_nocompass --es checkpoints/es1 --out results/eval_B.json
