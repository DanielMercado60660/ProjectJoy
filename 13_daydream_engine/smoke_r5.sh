#!/bin/bash
# Claude only. Turn 2 adds soft contacts, geometry/VJP checks and scheduled weighted training.
# Turn 3 adds five validated classical members, matched beta tuning and paired/resume evaluation.
# Acquire one shared slot for the entire smoke, not nested slots:
# ../bin/gpu_run bash smoke_r5.sh
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMBA_NUM_THREADS=4
export MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/results/r5/matplotlib_cache"
exec ../.venv/bin/python smoke_driver_r5.py --max-minutes 1.9
