#!/bin/sh
# R6 flight final chain B: pure PPO (no classical prior), budget 2.3e8 env steps per seed, resumable.
# Usage: sh r6_final_chainB.sh <seed> [extra args]
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
PY="$(cd .. && pwd)"/.venv/bin/python
GR="$(cd .. && pwd)"/bin/gpu_run
s=$1; shift
B=230000000
$GR $PY train_flight_r6.py --run f_pure_s$s --seed $s --steps $B --total $B --max_wall 2580 \
   --snap_every 10000000 --eval_every 4000000 "$@" > results/r6_logs/f_pure_s$s.txt 2>&1
for i in 1 2 3; do
  $GR $PY train_flight_r6.py --run f_pure_s$s --steps $B --resume --max_wall 2580 >> results/r6_logs/f_pure_s$s.txt 2>&1
done
