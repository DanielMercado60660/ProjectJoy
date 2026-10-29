#!/bin/bash
# Control for the retraining round: identical fine-tune, but on the SAME painter_train scenes with the
# random skin draw (draw 0) instead of the painter's pick. Then the same fresh re-attack on held-out families.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
$PY painter_finetune.py --steps 1500 --key random0 --out checkpoints/painter_round1/unet_ft_control
$PY painter.py --ckpt0 checkpoints/painter_round1/unet_ft_control --n_per_delta 100 --k 64 --out results/painter_round1_control
