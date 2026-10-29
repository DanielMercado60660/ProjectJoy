#!/bin/bash
# End-to-end smoke test at tiny size (well under 2 minutes): sim -> train -> eval -> plan -> gif.
# Writes only into smoke_out/ so it never touches the real data, checkpoint or media.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLBACKEND=Agg
O=smoke_out
rm -rf $O && mkdir -p $O
$PY sim.py --n-train 500 --n-test 300 --frames 60 --out $O/data
$PY train.py --data $O/data --steps 150 --max-minutes 0.5 --hidden 64 --depth 2 --unroll 5 --log-every 50 --out $O/ckpt
$PY eval.py --data $O/data --ckpt $O/ckpt --horizon 25 --n 100 --out $O/eval.json --plot $O/horizon.png
$PY plan.py --data $O/data --ckpt $O/ckpt --tasks 10 --cands 64 --T 20 --out $O/plan.json --plot $O/plan.png
$PY make_gif.py --data $O/data --ckpt $O/ckpt --frames 10 --out $O/demo.gif --px 80
$PY -c "import json; e=json.load(open('$O/eval.json')); assert e['by_horizon']['model']['25'] < 5; print('SMOKE OK')"
rm -rf $O
