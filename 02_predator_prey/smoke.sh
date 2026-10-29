#!/bin/sh
# End-to-end smoke at tiny size (< 2 min): unit tests, 3 training iterations, eval, 60-tick render.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
OUT=${TMPDIR:-/tmp}/pp_smoke
mkdir -p "$OUT"
$PY -m pytest -q tests
$PY train.py --max-iters 3 --group 16 --eval-every 2 --snap-every 1 --device cpu --out "$OUT/ck" --log "$OUT/res"
$PY evaluate.py --ckpt "$OUT/ck" --n 16 --xn 8 --xsnaps 3 --results "$OUT/res" --media "$OUT/media"
$PY render.py --ckpt "$OUT/ck" --ticks 60 --media "$OUT/media" --name smoke
echo "smoke OK -> $OUT"
