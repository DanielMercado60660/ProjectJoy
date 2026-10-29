#!/bin/bash
# Usage: logs/run_gallery.sh <target> <steps> <split_step>: regen model for the gallery, run in two segments (< 45 min each).
cd "$(dirname "$0")/.."
t=$1; n=${t}_regen; LOG=logs/train_$n.log
CMD="train.py --mode regen --target $t --steps $2 --ckpt_every 500 --out checkpoints/$n"
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python $CMD --stop_after $3 > $LOG 2>&1
echo "=== segment 1 ended $(date), resuming ===" >> $LOG
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python $CMD --resume >> $LOG 2>&1
