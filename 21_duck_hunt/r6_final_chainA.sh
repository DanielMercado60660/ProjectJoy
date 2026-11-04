#!/bin/sh
# R6 flight final chain A: PD+ fair retunes, then the residual recipe for 3 seeds.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
PY="$(cd .. && pwd)"/.venv/bin/python
GR="$(cd .. && pwd)"/bin/gpu_run
$GR $PY pd_plus_r6.py wide > results/r6_logs/pd_plus_wide.txt 2>&1
$GR $PY pd_plus_r6.py cont > results/r6_logs/pd_plus_cont.txt 2>&1
# residual base = better of (reviewer PD+, wide PD+) on the training courses
BASE=$($PY -c "
import json
a=json.load(open('results/pd_plus_tuning.json'))['train_score']; b=json.load(open('results/r6_pd_plus_wide_tuning.json'))['train_score']
print('checkpoints/pd_plus_params.json' if a>=b else 'checkpoints/pd_plus_r6_wide_params.json')")
echo "residual base: $BASE" > results/r6_logs/residual_base.txt
for s in 0 1 2; do
  $GR $PY train_flight_r6.py --run f_res_s$s --seed $s --reward time --residual 1 --init_log_std -1.6 \
     --pd_plus_params $BASE --steps 60000000 --lr_sched linear --lr 3e-4 --max_wall 2580 > results/r6_logs/f_res_s$s.txt 2>&1
  # finishes the run if max_wall stopped it (no-op otherwise)
  $GR $PY train_flight_r6.py --run f_res_s$s --steps 60000000 --resume --max_wall 2580 >> results/r6_logs/f_res_s$s.txt 2>&1
done
