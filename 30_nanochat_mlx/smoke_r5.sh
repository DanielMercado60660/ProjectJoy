#!/bin/sh
# Claude: run as ../bin/gpu_run sh smoke_r5.sh (one slot, no nested locks).
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
../.venv/bin/python -m pytest -q tests/test_r5_numpy.py
../.venv/bin/python r5_smoke.py
