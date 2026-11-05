#!/bin/sh
# R6 LeWM S1 training lane: one run, resumed after every 44-min guard stop (exit 3), each invocation via gpu_run.
#   sh lewm/lane.sh runs/lewm_s1 20000            # LeWM
#   sh lewm/lane.sh runs/lewm_rand 20000 --random-encoder
# Do not edit while a lane runs (sh re-reads the file).
cd "$(dirname "$0")/.."
RUN=$1; STEPS=$2; shift 2
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
if [ -f "$RUN/ckpt/state.json" ]; then
  ../bin/gpu_run ../.venv/bin/python lewm/train.py --run "$RUN" --resume
else
  ../bin/gpu_run ../.venv/bin/python lewm/train.py --run "$RUN" --steps "$STEPS" "$@"
fi
rc=$?
while [ $rc -eq 3 ]; do
  ../bin/gpu_run ../.venv/bin/python lewm/train.py --run "$RUN" --resume
  rc=$?
done
echo "lane $RUN finished rc=$rc"
