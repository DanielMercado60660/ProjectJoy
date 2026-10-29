#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
exec ../.venv/bin/python r5_smoke.py
