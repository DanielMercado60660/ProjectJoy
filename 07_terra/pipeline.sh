#!/bin/bash
# Dataset generation + heuristic training in one GPU-slot reservation.
# Usage: ../bin/gpu_run bash pipeline.sh
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
PY=../.venv/bin/python
[ -f data/train.npz ] || $PY make_dataset.py --n_train 3000 --n_val 150 --goals 4
$PY train.py --steps 8000 --bs 16 --resume
