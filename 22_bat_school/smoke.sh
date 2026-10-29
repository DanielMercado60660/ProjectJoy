#!/bin/bash
# End-to-end smoke check at tiny size (< 2 min): tests, tiny train, tiny eval, tiny demo.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
$PY -m pytest -q tests
$PY train.py --smoke | tail -2
$PY evaluate.py --ckpt checkpoints/smoke --n 8 --quick --out results/smoke_eval.json | tail -3
$PY demo.py --ckpt checkpoints/smoke --agent learned --maze 0 --max_steps 25 --out media/smoke_demo.gif
echo SMOKE OK
