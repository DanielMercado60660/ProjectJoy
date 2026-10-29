#!/bin/bash
# One GPU-slot session (<= 45 min wall): train for $1 minutes, then spend up to $2 minutes
# on evaluation (baseline round robin first, then the suite on every 5th generation).
#   ../bin/gpu_run bash slot.sh 36 8
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=../.venv/bin/python
RUN=${RUN:-main}
T0=$(date +%s)
$PY train.py --run $RUN --minutes $1 ${TRAIN_ARGS}
LEFT=$(python3 -c "print(max(0.5, $1 + $2 - ($(date +%s) - $T0)/60))")
$PY evaluate.py baselines --max-minutes $LEFT  # no-op if evalA already did it
LEFT=$(python3 -c "print(max(0.5, $1 + $2 - ($(date +%s) - $T0)/60))")
$PY evaluate.py suite --run $RUN --gens every10 --max-minutes $LEFT
$PY evaluate.py elo > /dev/null
echo "slot done after $(( ($(date +%s) - T0) / 60 )) min"
