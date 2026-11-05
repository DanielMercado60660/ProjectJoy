#!/bin/bash
# End-to-end smoke test at tiny size (~40 s): train 3 PPO updates, evaluate, render one clip (round 1-4 pipeline),
# then the round-6 pipeline (population PPO with learned defender, dev eval, demo, pytest).
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
# round 6
$PY r6_train.py --learn-def 1 --total-steps 1e6 --max-updates 3 --envs 64 --horizon 32 --minibatch 512 \
    --snap-every 4096 --out "$TMP/r6/r6" --log "$TMP/r6/r6.jsonl"
$PY r6_eval.py --mode dev --att "r6=$TMP/r6/r6.pkl" scripted --opps goalie,keeper_r --sets heldout --n 20 --out "$TMP/r6_dev.json"
$PY r6_demo.py --att "r6=$TMP/r6/r6.pkl" --seed-base 999 --outdir "$TMP/media" --quick
test -s "$TMP/media/r6_r6_vs_interceptor.gif"
$PY -m pytest -q tests/test_r6.py
echo "SMOKE OK ($TMP)"
