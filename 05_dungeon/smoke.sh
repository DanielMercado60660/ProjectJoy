#!/bin/bash
# End-to-end smoke test at tiny size (< 2 min): levels -> PPO -> designer -> evaluation.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
rm -rf checkpoints/smoke_solver logs/smoke_solver.jsonl
$PY -m pytest -q tests
$PY train_solver.py --name smoke_solver --envs 64 --minibatches 2 --steps 8192 --eval-every 1000 --eval-n 10 --log-every 1
$PY evolve_designer.py --solver checkpoints/smoke_solver --out smoke --gens 3 --pop 16 --K 2 --fresh-K 2
$PY evaluate.py --quick --solvers smoke_solver --designer-runs smoke
echo "SMOKE OK"
