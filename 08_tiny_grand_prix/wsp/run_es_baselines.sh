#!/bin/zsh
# Matched-budget ES fine-tunes (1x = 25 wrapper calls x 64 episodes = 1600 episodes) + cost curve.
# usage: wsp/run_es_baselines.sh hack checkpoints/es_soft_best.npz
task=$1; start=$2
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for mult in 1 10 100; do
    ../.venv/bin/python wsp/es_finetune.py --task $task --start $start --episodes_budget $((1600*mult)) --seed $seed --tag ${task}_x${mult}_seed$seed
  done
done
../.venv/bin/python wsp/es_finetune.py --task $task --start $start --episodes_budget 1600000 --seed 0 --tag ${task}_x1000_seed0
