#!/bin/bash
# End-to-end smoke check at tiny size (< 2 min): generate data, train 40 steps, evaluate, render demo.
# Writes only to a temp directory; does not touch data/, checkpoints/, results/ or media/.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
T=$(mktemp -d)
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
$PY make_data.py --out "$T/data" --n_train 256 --n_eval 40 --workers 2
$PY train.py --data "$T/data" --steps 40 --eval_every 20 --bs 16 --width 8 --out "$T/ck/unet"
$PY evaluate.py --data "$T/data" --ckpt "$T/ck/unet" --out "$T/res" > "$T/eval.txt"
grep -q "test_ood" "$T/res/eval.md"
$PY demo.py --data "$T/data" --ckpt "$T/ck/unet" --out "$T/media" --k 4
test -s "$T/media/ood_hardest.png" && test -s "$T/media/ood_hardest.gif"
$PY -c "import model as M, numpy as np, mlx.core as mx; m=M.load('$T/ck/unet'); print('reload ok', m(mx.zeros((1,96,96,3))).shape)"
rm -rf "$T"; echo "SMOKE OK"
