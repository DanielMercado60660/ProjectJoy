#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
TAG="smoke_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "logs/r5/$TAG"
../.venv/bin/python -m pytest -q tests/test_r5_numpy.py tests/test_r5_t2_numpy.py --basetemp="logs/r5/$TAG/pytest"
../.venv/bin/python smoke_r5.py --tag "$TAG"
