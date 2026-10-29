#!/bin/bash
# One GPU-slot evaluation session (<= $1 minutes, default 40): baseline round robin, then
# keep evaluating every 10th generation of the main run against the suite as new ones appear.
#   EVAL_TAG=evalA ../bin/gpu_run bash eval_slot.sh 40
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 EVAL_TAG=${EVAL_TAG:-evalA}
PY=../.venv/bin/python
MIN=${1:-40}
T0=$(date +%s)
left() { python3 -c "print(round(max(0.0, $MIN - ($(date +%s) - $T0)/60), 2))"; }
$PY evaluate.py baselines --max-minutes $(left)
while [ "$(python3 -c "print(int($(left) > 1.5))")" = 1 ]; do
  $PY evaluate.py suite --run ${RUN:-main} --gens ${GENS:-every10} --max-minutes $(left)
  sleep 30
done
$PY evaluate.py elo > /dev/null
echo "eval slot done after $(( ($(date +%s) - T0) / 60 )) min"
