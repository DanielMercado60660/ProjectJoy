#!/bin/bash
# Bench + pilot pretraining in ONE gpu_run slot (the slot queue is contended, so we hold one slot
# for the benchmark and then size and run the pilot from the measured throughput).
#   ../bin/gpu_run bash run_pilot.sh > logs/pilot.log 2>&1
# Env overrides: DEPTH (8), BUDGET_MIN (training minutes the schedule is sized for, 90),
#                HARD_MIN (wall-clock stop, 105), SKIP_BENCH=1
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
DEPTH=${DEPTH:-8}
BUDGET_MIN=${BUDGET_MIN:-90}
HARD_MIN=${HARD_MIN:-105}
TOTAL_BATCH=131072
echo "[pilot] $(date) start"
if [ -z "$SKIP_BENCH" ]; then
  $PY bench.py --out media/bench.json
fi
# the GPU slot queue was long: cap the pilot so it ends by PILOT_END (local time), leaving
# time for the post stage + write-up before the 12:30 deadline
PILOT_END=${PILOT_END:-11:00}
LEFT=$(( ( $(date -j -f "%H:%M" "$PILOT_END" +%s) - $(date +%s) ) / 60 ))
[ "$LEFT" -lt 20 ] && LEFT=20
[ "$LEFT" -lt 40 ] && DEPTH=6
if [ "$LEFT" -lt "$HARD_MIN" ]; then HARD_MIN=$LEFT; BUDGET_MIN=$(( LEFT * 85 / 100 )); fi
echo "[pilot] time left until $PILOT_END: ${LEFT} min -> BUDGET_MIN=$BUDGET_MIN HARD_MIN=$HARD_MIN"
# wait for tokenization to finish (12 train shards)
while [ "$(ls data/tokens/train_*.npy 2>/dev/null | grep -vc tmp)" -lt 12 ]; do echo "[pilot] waiting for tokens"; sleep 30; done
# size the run from the measured pilot-config throughput (d$DEPTH, bf16, compiled, B=32)
# pick the largest depth (8, else 6) that reaches >= 8 tokens per scaling param in the budget
read DEPTH NITER <<< "$($PY - <<EOF
import json, glob, numpy as np
from gpt import GPT, config_for_depth
b = json.load(open("media/bench.json"))
avail = sum(np.load(f, mmap_mode="r").shape[0] for f in glob.glob("data/tokens/train_*.npy") if "tmp" not in f)
cands = [d for d in (8, 6) if d <= $DEPTH] or [6]
for d in cands:
    rows = [r for r in b["rows"] if r["depth"] == d and r["dtype"] == "bf16" and r["compile"]]
    tps = 0.85 * max(r["tok_per_s"] for r in rows)  # margin: shared GPU, eval/checkpoint overhead
    n = int(tps * $BUDGET_MIN * 60 / $TOTAL_BATCH)
    n = min(n, int(0.95 * avail / $TOTAL_BATCH))   # stay under one epoch
    sp = GPT(config_for_depth(d)).num_scaling_params()
    if n * $TOTAL_BATCH >= 8 * sp or d == cands[-1]:
        print(d, n)
        break
EOF
)"
echo "[pilot] $(date) depth=$DEPTH num_iterations=$NITER"
$PY train.py --depth $DEPTH --seq-len 1024 --device-batch 32 --total-batch $TOTAL_BATCH \
  --num-iterations $NITER --time-budget-min $HARD_MIN --eval-every 100 --eval-batches 16 \
  --sample-every $((NITER / 4)) --out checkpoints/pilot --resume
echo "[pilot] $(date) end"
# post-pilot stage (SFT + evals) in the same slot, if present; written while the pilot runs
if [ -f run_post.sh ]; then bash run_post.sh; fi
