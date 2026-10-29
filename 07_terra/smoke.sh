#!/bin/bash
# End-to-end smoke check at tiny size (< 2 min). Does not touch real checkpoints/results/media.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
TMP=$(mktemp -d)
trap 'rm -rf "$TMP" checkpoints/smoke_* ' EXIT
export OMP_NUM_THREADS=2
echo "== unit tests (planners, D* Lite vs Dijkstra, JS parity, scenarios)"
$PY -m pytest -q
echo "== tiny dataset"
$PY make_dataset.py --n_train 12 --n_val 4 --goals 2 --out "$TMP/data"
echo "== tiny heuristic training (+ resume)"
$PY train.py --data "$TMP/data" --steps 20 --ckpt_every 10 --tag smoke_h | tail -2
$PY train.py --data "$TMP/data" --steps 30 --ckpt_every 10 --tag smoke_h --resume | grep -E "resumed|done"
echo "== tiny prior training"
$PY train_prior.py --data "$TMP/data" --steps 20 --tag smoke_p | tail -1
echo "== tiny evals"
$PY eval_heuristic.py --maps 2 --per_map 1 --tag smoke_h --out "$TMP/h.json" --no_figs | head -4
$PY run_fog.py --maps 2 --per_map 1 --sweep_maps 0 --tag smoke_p --out "$TMP/fog.json" --no_figs > /dev/null && echo "fog ok"
$PY -c "import sys; sys.path.insert(0,'.'); from terra import scenarios; r=scenarios.run_bridge_benchmark(range(900000,900003), per_map=1, reps=1); print('bridge scenario ok,', len(r), 'queries, consistent:', all(x['consistent'] for x in r))"
echo "== demo build"
$PY build_demo.py
echo "SMOKE OK"
