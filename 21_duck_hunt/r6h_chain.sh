#!/bin/sh
# R6 hunter: stage 1 (learned vs scripted), then population self-play (resumable), then eval + demo.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
PY="$(cd .. && pwd)"/.venv/bin/python
GR="$(cd .. && pwd)"/bin/gpu_run
$GR $PY train_hunter.py stage1 --steps 12000000 > results/r6_logs/hunter_stage1.txt 2>&1
$GR $PY train_hunter.py selfplay --gens 8 --steps 5000000 > results/r6_logs/hunter_selfplay.txt 2>&1
for i in 1 2; do
  $GR $PY train_hunter.py selfplay --gens 8 --steps 5000000 --resume >> results/r6_logs/hunter_selfplay.txt 2>&1
done
$GR $PY eval_hunter.py > results/r6_logs/hunter_eval.txt 2>&1
