#!/bin/bash
# End-to-end smoke check at tiny size (< 2 min): sim invariants, tiny PPO train + reload,
# quick baseline tuning + evaluation, demo page build. Writes only to a temp dir.
set -euo pipefail
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=../.venv/bin/python
TMP=$(mktemp -d)
$PY -m pytest -q tests
$PY train.py --max-iters 2 --envs 2 --val-every 1 --out "$TMP/ppo" --log "$TMP/log.jsonl" > /dev/null
$PY evaluate.py --quick --episodes 2 --ckpt "$TMP/ppo_best" --out "$TMP" > /dev/null
$PY make_demo.py --runs "$TMP" --out "$TMP/demo.html" --png "$TMP/summary.png"
echo "smoke OK ($TMP)"
