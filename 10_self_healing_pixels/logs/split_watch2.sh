#!/bin/bash
# Usage: logs/split_watch2.sh <run_name> <mode> <seed> <step>. Segment 2 slowed to ~0.7 s/it under GPU contention,
# so stop it after the checkpoint at <step> and resume once more (keeps every segment < 45 min).
cd "$(dirname "$0")/.."
n=$1; mode=$2; seed=$3; s=$4
until python3 -c "import json,sys;sys.exit(0 if json.load(open('checkpoints/$n/ckpt/meta.json'))['step']>=$s else 1)" 2>/dev/null; do sleep 3; done
pkill -f "out checkpoints/$n --resume"; sleep 3
echo "=== split_watch2: stopped after step-$s checkpoint $(date), resuming ===" >> logs/train_$n.log
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python train.py --mode $mode --target lizard --seed $seed --steps 8000 --ckpt_every 1000 --out checkpoints/$n --resume >> logs/train_$n.log 2>&1
