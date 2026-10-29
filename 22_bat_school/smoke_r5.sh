#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMBA_NUM_THREADS=4
# Run the whole smoke through ../bin/gpu_run; do not take nested locks.
exec ../.venv/bin/python smoke_r5.py --max-minutes 1.9
