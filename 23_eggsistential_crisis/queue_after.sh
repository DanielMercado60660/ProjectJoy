#!/bin/bash
# usage: queue_after.sh <pid-to-wait-for> <label> <python script + args...>
# Waits for my previous job, checks AC power, then runs the script through the GPU slot lock.
cd "$(dirname "$0")"
W=$1; LABEL=$2; shift 2
while kill -0 "$W" 2>/dev/null; do sleep 15; done
if pmset -g batt | grep -q "Battery Power"; then
  echo "$(date '+%Y-%m-%d %H:%M') on battery: NOT launching $LABEL (waiting for power)" >> STATUS.md; exit 0
fi
echo "$(date '+%Y-%m-%d %H:%M') launching $LABEL via gpu_run: $*" >> STATUS.md
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
exec ../bin/gpu_run ../.venv/bin/python -W ignore "$@"
