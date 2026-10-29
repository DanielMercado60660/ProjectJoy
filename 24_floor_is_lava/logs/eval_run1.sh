#!/bin/bash
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
PY=../.venv/bin/python
$PY evaluate.py --policy checkpoints/main_run1_end.npz --n 500 --out results/eval_run1_full.json
$PY evaluate.py --policy checkpoints/main_run1_end.npz --n 500 --gap_max 0.125 --out results/eval_run1_gap0125.json
$PY evaluate.py --policy checkpoints/main_run1_end.npz --n 200 --flat --out results/eval_run1_flat_push.json
