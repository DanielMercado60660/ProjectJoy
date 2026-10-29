#!/bin/bash
# Job 5 (follow-up): adaptation-cost stage for the compatible pair fwd vs cruise.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 SKILL_PAIR=fwd_cruise
../.venv/bin/python pipeline.py --stage adapt --seeds 0 1 2 3 --root checkpoints/fwd_cruise --results results/fwd_cruise
