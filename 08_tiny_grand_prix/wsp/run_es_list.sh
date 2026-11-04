#!/bin/zsh
# usage: wsp/run_es_list.sh <task> <start.npz> "<mult:seed> <mult:seed> ..."   (1x = 1600 episodes)
task=$1; start=$2; list=$3
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
cd "$(dirname "$0")/.."
for ms in ${=list}; do
  mult=${ms%%:*}; seed=${ms##*:}
  ../.venv/bin/python wsp/es_finetune.py --task $task --start $start --episodes_budget $((1600*mult)) --seed $seed --tag ${task}_x${mult}_seed$seed
done
