#!/bin/bash
# Phase 2, amendment 6: the per-pick checks for one or more recovery-round runs, one after another.
# Each name's pick comes from results/p2/select_p2_<name>.json (written by p2_lane.sh).
#   ./p2_checks.sh c_s0 c_s1 > logs/p2_checks1.log 2>&1
cd "$(dirname "$0")"
PY=../.venv/bin/python
for name in "$@"; do
  sel=results/p2/select_p2_$name.json
  if [ ! -f $sel ]; then echo "[checks] $(date +%H:%M:%S) SKIP $name (no $sel)"; continue; fi
  w=$($PY -c "import json,os;print(os.path.relpath(json.load(open('$sel'))['weights']))")
  n=p2_$name
  echo "[checks] $(date +%H:%M:%S) $n: $w"
  OMP_NUM_THREADS=4 ../bin/gpu_run $PY p2_diag.py $w > logs/${n}_diag.log 2>&1
  OMP_NUM_THREADS=4 ../bin/gpu_run $PY p2_aux_split.py $w > logs/${n}_aux_split.log 2>&1
  OMP_NUM_THREADS=4 ../bin/gpu_run $PY p2_wz_check.py $n=$w > logs/${n}_wz_check.log 2>&1
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run $PY p2_vx_closed.py $w > logs/${n}_vx_closed.log 2>&1
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run $PY p2_shadow_dev.py $n=$w > logs/${n}_shadow_dev.log 2>&1
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run $PY p2_other.py $n=$w > logs/${n}_other.log 2>&1
  for f in diag aux_split wz_check vx_closed shadow_dev other; do
    grep -q Traceback logs/${n}_$f.log && echo "[checks] $(date +%H:%M:%S) $n: $f FAILED (logs/${n}_$f.log)"
  done
  echo "[checks] $(date +%H:%M:%S) done $n"
done
