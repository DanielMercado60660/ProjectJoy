#!/bin/bash
# Final evaluation session for generation $1 (<= 45 min):
#   suite on every 10th gen, generation round robin, latency, matched-budget comparison, Elo, plots.
#   ../bin/gpu_run bash final_eval.sh 172
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 EVAL_TAG=${EVAL_TAG:-final}
PY=../.venv/bin/python
G=$1
T0=$(date +%s)
left() { python3 -c "print(round(max(0.5, 44 - ($(date +%s) - $T0)/60), 2))"; }
$PY evaluate.py suite --gens $G --max-minutes $(left)
$PY evaluate.py latency --gen $G
$PY evaluate.py budget --gen $G --max-minutes $(left)
RRG=$(python3 -c "print(','.join(str(x) for x in sorted(set([0,2,5,10,20,40,70,100,136,$G]))))")
$PY evaluate.py rr --gens $RRG --max-minutes $(left)
$PY evaluate.py suite --gens every10 --max-minutes $(left)
$PY evaluate.py elo > /dev/null
$PY evaluate.py plots
$PY evaluate.py report > /dev/null
echo "final eval done after $(( ($(date +%s) - T0) / 60 )) min"
