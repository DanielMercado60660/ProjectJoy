#!/bin/bash
# Smoke test for the flow-head pipeline at tiny size (~1-2 min): train 5 steps on 64 LJ clips, resume to 10 ->
# generate 2 val sentences -> Mimi decode -> Whisper. Needs data/lj_lat.npy (mimi_io.py encode-latents), ../.venv-moshi,
# ../.venv-judge and the Gemma/Mimi/Whisper weights in the HF cache. The untrained model's CER is expected to be high.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
rm -rf runs/smoke_flow
A="--out runs/smoke_flow --mix lj --limit 64 --warmup 2 --val_every 5 --save_every 5 --val_n 16 --budget 3000 --K 2 --log_every 5"
$PY train_flow.py $A --steps 5 | tail -1                       # stops at step 5 with a checkpoint
$PY train_flow.py $A --steps 10 --resume | grep -E "resum|done"   # picks up at step 5, ends at 10
$PY evaluate_flow.py --run runs/smoke_flow --ckpt last --split val --n 2 --temp 0.4 --cfg 2.0 --tag smoke --mos 0 | tail -1
echo "SMOKE_FLOW OK"
