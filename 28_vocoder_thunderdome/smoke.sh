#!/bin/sh
# End-to-end pipeline at tiny size (< 2 min): data prep -> train -> eval (GL + neural) -> demo -> bench.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
T=$(mktemp -d)
export OMP_NUM_THREADS=4
$PY prepare_data.py --n_train 8 --n_test 4 --out "$T/cache"
$PY train.py --data "$T/cache/train.npz" --steps 60 --batch 4 --dim 64 --blocks 2 --out "$T/ckpt/vocoder" --log "$T/log.jsonl"
$PY eval.py --data "$T/cache/test.npz" --split "$T/cache/split.json" --ckpt "$T/ckpt/vocoder.safetensors" --gl_iters 4,64 --out "$T/results"
$PY make_demo.py --results "$T/results" --cache "$T/cache" --media "$T/media" --ablation "$T/none"
test -s "$T/media/demo.html"
$PY -c "import json,sys; s=json.load(open('$T/results/metrics.json'))['summary']; assert 'neural' in s and 'gl64' in s; print('smoke OK', {k: round(v['mel_l1'],3) for k,v in s.items()})"
rm -rf "$T"
