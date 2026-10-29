#!/bin/bash
# Job 4 (follow-up): the same train + merge pipeline on a COMPATIBLE skill pair:
# fwd (+v - 0.1|a|^2) vs cruise (+v - 1.0|a|^2, low-energy forward running).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 SKILL_PAIR=fwd_cruise
../.venv/bin/python pipeline.py --stage train --seeds 0 1 2 3 --root checkpoints/fwd_cruise --results results/fwd_cruise && \
../.venv/bin/python pipeline.py --stage merge --seeds 0 1 2 3 --root checkpoints/fwd_cruise --results results/fwd_cruise
