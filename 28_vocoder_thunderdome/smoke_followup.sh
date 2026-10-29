#!/bin/sh
# Follow-up (round 3) pipeline at tiny size (< 1 min): val split -> train_val (val selection) -> eval -> per-res MR-STFT.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
T=$(mktemp -d)
export OMP_NUM_THREADS=4
$PY prepare_data.py --n_train 8 --n_test 4 --out "$T/cache"
$PY prepare_val.py --n_val 3 --out "$T/val"
$PY train_val.py --data "$T/cache/train.npz" --val "$T/val/val.npz" --steps 80 --stop_step 60 --val_every 20 \
    --batch 4 --dim 64 --blocks 2 --out "$T/ckpt" --log "$T/log.jsonl"
test -s "$T/ckpt/best.safetensors" && test -s "$T/ckpt/last.safetensors"
$PY eval.py --data "$T/cache/test.npz" --split "$T/cache/split.json" --ckpt "$T/ckpt/best.safetensors" --gl_iters 4 --out "$T/res"
$PY mrstft_per_res.py --data "$T/cache/test.npz" --out "$T/pr.json" "gl4=$T/res/wavs/{id}_gl4.wav" "nn=$T/res/wavs/{id}_neural.wav"
$PY -c "import json; j=json.load(open('$T/ckpt/last.json')); assert len(j['val_hist'])==3, j['val_hist']; print('smoke_followup OK best step', j['best_step'])"
rm -rf "$T"
