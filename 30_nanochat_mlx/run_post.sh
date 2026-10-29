#!/bin/bash
# Post-pilot stage: inference benchmark, SFT (pretrained vs scratch), sandbox evals, plots, demo.
# Called at the end of run_pilot.sh (same gpu_run slot), or on its own:
#   ../bin/gpu_run bash run_post.sh > logs/post.log 2>&1
# Essentials first, so a late finish still leaves a demo. Past LATE (local time) the
# from-scratch comparison is skipped and eval episodes are cut to 100.
cd "$(dirname "$0")"
PY=../.venv/bin/python
LATE=${LATE:-11:30}
late() { [ "$(date +%s)" -gt "$(date -j -f "%H:%M" "$LATE" +%s)" ]; }
EP=200
echo "[post] $(date) start"
$PY engine.py --ckpt checkpoints/pilot/last --out media/inference_bench.json
$PY sft.py --base checkpoints/pilot/last --out checkpoints/sft --episodes 8000 --epochs 3
late && EP=100
$PY eval_sandbox.py --ckpt checkpoints/sft/last --policies lm,lm_constrained --episodes $EP --tag pretrained
$PY plot.py
$PY make_demo.py --ckpt checkpoints/sft/last
$PY eval_sandbox.py --policies expert,random,mlp --episodes $EP
if ! late; then
  $PY sft.py --scratch --base checkpoints/pilot/last --out checkpoints/sft_scratch --episodes 8000 --epochs 3
  $PY eval_sandbox.py --ckpt checkpoints/sft_scratch/last --policies lm,lm_constrained --episodes $EP --tag scratch
  $PY plot.py
  $PY make_demo.py --ckpt checkpoints/sft/last
else
  echo "[post] skipped scratch SFT (late)"
fi
echo "[post] $(date) end"
