#!/bin/bash
# One GPU-slot job: train + evaluate several (agent, seed) runs back to back.
# usage: ../bin/gpu_run ./sweep.sh <env: base|tv> <steps> <agent:seed> [agent:seed ...]
set -uo pipefail
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
PY=../.venv/bin/python
ENV=$1; STEPS=$2; shift 2
TVFLAG=""; [ "$ENV" = "tv" ] && TVFLAG="--noisy_tv"
for spec in "$@"; do
  AG=${spec%%:*}; SEED=${spec##*:}; REAL_AG=${AG%long}
  NAME=${AG}_${ENV}_s${SEED}
  echo "=== $(date +%H:%M:%S) $NAME"
  if [ -f runs/$NAME/ckpt.pkl ] && [ ! -f runs/$NAME/DONE ]; then RES="--resume runs/$NAME/ckpt.pkl"; else RES=""; fi
  if [ ! -f runs/$NAME/DONE ]; then
    $PY train.py --agent $REAL_AG --seed $SEED --n_envs 128 --total_steps $STEPS $TVFLAG --out runs/$NAME $RES && touch runs/$NAME/DONE
  fi
  $PY evaluate.py --ckpt runs/$NAME/ckpt.pkl $TVFLAG --seed $((10000 + SEED)) --out runs/eval_${NAME}.json
done
echo "=== $(date +%H:%M:%S) sweep done"
