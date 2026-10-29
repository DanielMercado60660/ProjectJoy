#!/bin/bash
# Runner only: ../bin/gpu_run ./smoke_r5.sh. Keeps all receipts; never deletes old runs.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=../.venv/bin/python
"$PY" -m pytest -q tests/test_r5_numpy.py
"$PY" tests/check_r5_simulator.py
"$PY" - <<'PY'
import os
import signal
import subprocess
import sys

# Bound even Metal startup; this process group contains only this smoke's children.
job = subprocess.Popen([sys.executable, 'r5_gpu_checks.py'], start_new_session=True)
try:
    raise SystemExit(job.wait(timeout=115))
except subprocess.TimeoutExpired:
    os.killpg(job.pid, signal.SIGKILL)
    job.wait()
    raise SystemExit('R5_SMOKE_TIMEOUT: receipts retained; inspect logs/r5/smoke_*')
PY
