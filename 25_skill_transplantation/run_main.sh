#!/bin/bash
# Job 1: train + merge for 4 seeds (one gpu_run slot, pool of 4 single-thread workers).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
../.venv/bin/python pipeline.py --stage train --seeds 0 1 2 3 && \
../.venv/bin/python pipeline.py --stage merge --seeds 0 1 2 3
