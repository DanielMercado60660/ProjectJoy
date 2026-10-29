#!/bin/bash
# Keep each run segment under the 45-min wall limit: after the step-7000 checkpoint, stop and resume.
cd "$(dirname "$0")/.."
m=$1
until python3 -c "import json,sys;sys.exit(0 if json.load(open('checkpoints/lizard_$m/ckpt/meta.json'))['step']>=7000 else 1)" 2>/dev/null; do sleep 3; done
pkill -f "train.py --mode $m --steps 8000" ; sleep 3
echo "=== split at step 7000 (45-min rule), resuming $(date) ===" >> logs/train_lizard_$m.log
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python train.py --mode $m --steps 8000 --ckpt_every 1000 --out checkpoints/lizard_$m --resume >> logs/train_lizard_$m.log 2>&1
