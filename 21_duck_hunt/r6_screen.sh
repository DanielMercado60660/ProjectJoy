#!/bin/sh
# R6 flight screening runs (training courses only). Usage: sh r6_screen.sh <chain>
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
PY="$(cd .. && pwd)"/.venv/bin/python
GR="$(cd .. && pwd)"/bin/gpu_run
S=30000000
run() { name=$1; shift; $GR $PY train_flight_r6.py --run $name --steps $S "$@" > results/r6_logs/$name.txt 2>&1; }
if [ "$1" = a ]; then
  run s_time --reward time
  run s_res --reward time --residual 1 --init_log_std -1.6
  run s_time_g995 --reward time --gamma 0.995
fi
if [ "$1" = b ]; then
  run s_orig256 --reward orig
  run s_time_lin --reward time --lr_sched linear --lr 5e-4
  run s_time_h256 --reward time --hid 256
fi
