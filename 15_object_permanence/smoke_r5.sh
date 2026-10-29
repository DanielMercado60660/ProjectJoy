#!/usr/bin/env bash
# RUNNER ONLY: imports MLX. Use ../bin/gpu_run bash ./smoke_r5.sh.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMBA_NUM_THREADS=4
PY=../.venv/bin/python
TAG=""
CHECK=("$PY" r5_mlx_check.py)
RESUME=("$PY" r5_resume_check.py)
TUNE=("$PY" r5_tune.py)
TESTS=(test_r5_numpy.py)
if [[ "${1:-}" == "--turn2" ]]; then
  TAG="_t2"
  CHECK+=(--turn2)
  RESUME+=(--turn2)
  TUNE+=(--turn2)
  TESTS+=(test_r5_turn2_numpy.py)
elif [[ -n "${1:-}" ]]; then
  echo 'Usage: smoke_r5.sh [--turn2]' >&2
  exit 2
fi
RUN_ID="smoke${TAG}_$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="results/r5/$RUN_ID"
CKPT_DIR="checkpoints/r5_$RUN_ID"
mkdir -p "$RUN_DIR" logs/r5
exec > >(tee "logs/r5/$RUN_ID.log") 2>&1
SECONDS=0
"$PY" -m pytest -q "${TESTS[@]}"
"${CHECK[@]}"
"${RESUME[@]}" --out "$CKPT_DIR"
"${TUNE[@]}" --out "$RUN_DIR/tuning.json" --tiny --screen-n 1 --confirm-n 1 --particles 16 --max-minutes .5
if [[ -n "$TAG" ]]; then
  # Exercise val-selected best.pkl as well as the independently tested resume path.
  "$PY" r5_turn2_train.py --out "$CKPT_DIR/selection" --tiny --steps 2 --batch 1 \
    --save-every 2 --val-count 1 --max-minutes .5
  "$PY" r5_cue_probe.py --checkpoint "$CKPT_DIR/selection/best.pkl" \
    --out "$RUN_DIR/cue_probe.json" --n 1 --max-minutes .5 --require-equivariant
fi
"$PY" r5_evaluate.py --checkpoint "$CKPT_DIR/resumed/last.pkl" --tuning "$RUN_DIR/tuning.json" \
  --thresholds "$RUN_DIR/thresholds.json" --calibrate --n 1 --max-minutes .5
"$PY" r5_evaluate.py --checkpoint "$CKPT_DIR/resumed/last.pkl" --tuning "$RUN_DIR/tuning.json" \
  --thresholds "$RUN_DIR/thresholds.json" --split smoke_fresh --n 1 --out "$RUN_DIR/fresh.json" --max-minutes .5
"$PY" r5_evaluate.py --checkpoint "$CKPT_DIR/resumed/last.pkl" --tuning "$RUN_DIR/tuning.json" \
  --thresholds "$RUN_DIR/thresholds.json" --split hard_test --n 1 --out "$RUN_DIR/legacy.json" --max-minutes .5
"$PY" r5_demo.py --evaluation "$RUN_DIR/fresh.json" --out "media/r5_$RUN_ID" --max-minutes .25
"$PY" - "$RUN_DIR" <<'PY'
import json, pathlib, sys
for name in ('fresh', 'legacy'):
    d = json.loads((pathlib.Path(sys.argv[1])/(name+'.json')).read_text())
    assert d['complete'] and d['completed_n'] == 1
    assert set(d['summary']) == {'gaussian', 'mixture', 'kalman', 'particle'}
    for rows in d['per_sequence'].values():
        assert set(rows) == {'never', 'random', 'uncertainty'}
        for policy in ('random', 'uncertainty'):
            assert rows[policy][0]['peek']['cost_views'] == 1
print('SMOKE_ARTIFACTS_OK all four trackers / three policies / both splits')
PY
if (( SECONDS >= 120 )); then
  echo "SMOKE_SLOW elapsed=${SECONDS}s (pipeline passed but under-2-minute contract failed)"
  exit 1
fi
echo "SMOKE_R5_OK elapsed=${SECONDS}s artifacts=$RUN_DIR checkpoint=$CKPT_DIR/resumed/last.pkl"
