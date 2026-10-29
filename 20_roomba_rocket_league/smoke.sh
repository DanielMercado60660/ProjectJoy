#!/bin/bash
# End-to-end smoke test at tiny size (~20 s): train 3 PPO updates, evaluate, render one clip.
# Writes only to a temp dir; does not touch checkpoints/ or media/.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
PY=../.venv/bin/python
TMP=$(mktemp -d)
$PY ppo.py --max-updates 3 --envs 32 --horizon 64 --minibatch 512 --out "$TMP/ppo.npz" --log "$TMP/train.jsonl"
$PY -c "import json; r=[json.loads(l) for l in open('$TMP/train.jsonl')]; import math; assert len(r)==3 and all(math.isfinite(x['vloss']) for x in r); print('train log ok')"
$PY eval.py --ckpt "$TMP/ppo.npz" --episodes 20 --out "$TMP/eval.json"
$PY demo.py --ckpt "$TMP/ppo.npz" --outdir "$TMP/media" --quick
test -s "$TMP/media/ppo_vs_goalie.gif"
echo "SMOKE OK ($TMP)"
