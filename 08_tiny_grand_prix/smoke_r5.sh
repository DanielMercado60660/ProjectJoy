#!/usr/bin/env bash
# Claude only: includes MLX. Run under ../bin/gpu_run bash ./smoke_r5.sh
set -euo pipefail
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
PY=../.venv/bin/python
# A hard overall guard keeps a pathological first Metal compile under two minutes.
if [[ "${1:-}" != "--inner" ]]; then
  exec "$PY" - "$0" <<'PYGUARD'
import os, signal, subprocess, sys
p = subprocess.Popen(['bash', sys.argv[1], '--inner'], start_new_session=True)
try:
    raise SystemExit(p.wait(timeout=115))
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    raise SystemExit('SMOKE TIMEOUT (115 s); retain logs and diagnose before allocating main budget')
PYGUARD
fi
TAG="r5_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/r5 results/r5
"$PY" -m pytest -q tests/test_r5_numpy.py --basetemp=results/r5/pytest_tmp
"$PY" bench_r5.py --max-minutes .3 --out "results/r5/${TAG}_rates.json"
"$PY" check_r5_mlx.py
# End-to-end and resume use small shapes, all six validation groups, both arms.
ARGS=(--envs 8 --horizon 8 --unroll 4 --minibatch-seqs 8 --epochs 1 --n-train 2 --n-val 1 --episode-steps 80 --eval-every 1 --save-every 1 --max-minutes .8)
"$PY" train_r5.py --tag "${TAG}_whole" "${ARGS[@]}" --max-updates 2
"$PY" train_r5.py --tag "$TAG" "${ARGS[@]}" --max-updates 1
"$PY" train_r5.py --tag "$TAG" "${ARGS[@]}" --max-updates 2 --resume "checkpoints/${TAG}_state.npz"
"$PY" - "$TAG" <<'PY'
import sys, numpy as np
name = sys.argv[1]
with np.load(f'checkpoints/{name}_state.npz') as a, np.load(f'checkpoints/{name}_whole_state.npz') as b:
    assert int(a['update']) == int(b['update']) == 2
    for k in a.files:
        if k.endswith('active_seconds'):
            continue
        assert np.array_equal(a[k], b[k]), f'exact resume mismatch: {k}'
print('weights, optimizer, hidden state, simulator, both RNG streams: EXACT RESUME PASS')
PY
"$PY" check_r5_kill_resume.py --tag "${TAG}_kill" --max-minutes .7
"$PY" tune_r5.py --tag "$TAG" --population 4 --episodes 6 --n-train 2 --n-val 1 --max-steps 80 --eval-every 1 --max-generations 1 --max-minutes .4
"$PY" tune_r5.py --tag "$TAG" --population 4 --episodes 6 --n-train 2 --n-val 1 --max-steps 80 --eval-every 1 --max-generations 2 --max-minutes .4 --resume "checkpoints/${TAG}_baseline_state.npz"
"$PY" r5_eval.py --tag "$TAG" --n 1 --max-steps 80 --max-minutes .4
"$PY" r5_eval.py --tag "$TAG" --split legacy --n 1 --max-steps 80 --max-minutes .3
"$PY" demo_r5.py --tag "$TAG" --max-steps 80 --max-minutes .3
printf 'SMOKE PASS; disposable tag: %s (keep receipts; do not commit smoke checkpoints)\n' "$TAG"
