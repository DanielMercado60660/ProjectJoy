#!/bin/bash
# Retraining round (run under ../bin/gpu_run): painted TRAINING-family data -> fine-tune U-Net0 -> re-attack held-out.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
$PY painter.py --families train --no_baselines --n_per_delta 250 --k 32 --seed_base 8000000 --out results/painter_train
$PY painter_finetune.py --steps 1500
$PY painter.py --ckpt0 checkpoints/painter_round1/unet_ft --n_per_delta 100 --k 64 --out results/painter_round1
