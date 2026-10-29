#!/bin/bash
# Usage: logs/split_watch.sh <run_name> <step>. GPU contention made seg-1 (6000 steps) too slow for the 45-min
# limit, so stop segment 1 right after the checkpoint at <step>; run_split.sh then resumes from it.
cd "$(dirname "$0")/.."
n=$1; s=$2
until python3 -c "import json,sys;sys.exit(0 if json.load(open('checkpoints/$n/ckpt/meta.json'))['step']>=$s else 1)" 2>/dev/null; do sleep 3; done
pkill -f "out checkpoints/$n --stop_after"
echo "=== split_watch: stopped segment 1 after step-$s checkpoint $(date) ===" >> logs/train_$n.log
