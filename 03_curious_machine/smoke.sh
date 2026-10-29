#!/bin/bash
# End-to-end smoke check at tiny size (< 2 minutes): tests, train, resume, evaluate, render.
set -euo pipefail
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=../.venv/bin/python
OUT=$(mktemp -d)
$PY -m pytest -q
$PY train.py --agent rnd --n_envs 16 --rollout 32 --total_steps 4096 --hidden 64 --ckpt_every 4 --out $OUT/rnd --stop_after_updates 4
$PY train.py --agent rnd --n_envs 16 --rollout 32 --total_steps 4096 --hidden 64 --ckpt_every 4 --out $OUT/rnd --resume $OUT/rnd/ckpt.pkl
$PY evaluate.py --ckpt $OUT/rnd/ckpt.pkl --n_episodes 4 --noisy_tv --out $OUT/eval.json
$PY evaluate.py --random --n_episodes 4 --out $OUT/eval_random.json
$PY render.py --ckpt $OUT/rnd/ckpt.pkl --out $OUT/demo.gif
ls -la $OUT
echo "SMOKE OK"
