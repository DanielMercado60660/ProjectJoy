#!/bin/sh
# Claude only: invokes MLX. Run this entire script through ../bin/gpu_run.
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export MLX_ENABLE_TF32=0 PYTHONDONTWRITEBYTECODE=1
PY=../.venv/bin/python
mkdir -p logs/r5 results/r5
$PY -m pytest -q -p no:cacheprovider tests/test_r5_numpy.py --basetemp=logs/r5/pytest_tmp
$PY r5_run.py calibrate --out results/r5/pilot.json --max-minutes 1.25
$PY r5_resume_probe.py
echo 'SMOKE R5 OK: pure tests, all engine paths attempted, recurrent parity, queue/latest, concurrent load, kill/resume, offline report'
