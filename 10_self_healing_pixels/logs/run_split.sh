#!/bin/bash
# Usage: logs/run_split.sh <run_name> <mode> <target> <seed>
# 8000-step run; to respect the 45-min wall limit on a shared GPU, the first segment is stopped after the
# step-6000 checkpoint and the run is resumed from it (checkpoint has weights, Adam state, pool, RNG, step).
cd "$(dirname "$0")/.."
name=$1; mode=$2; tgt=$3; seed=$4
LOG=logs/train_$name.log
CMD="train.py --mode $mode --target $tgt --seed $seed --steps 8000 --ckpt_every 1000 --out checkpoints/$name"
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python $CMD --stop_after 6000 > $LOG 2>&1
echo "=== segment 1 ended $(date), resuming ===" >> $LOG
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python $CMD --resume >> $LOG 2>&1
