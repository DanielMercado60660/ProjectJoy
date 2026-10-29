#!/bin/sh
# End-to-end smoke check at tiny size (< 2 minutes). Run from anywhere.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
../.venv/bin/python -m pytest -q tests
