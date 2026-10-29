#!/bin/bash
# One GPU-slot job that drains a task queue, running up to WIDTH tasks in parallel (each with
# 1 numba/OMP thread, so the job stays within the 4-thread cap). Tasks are lines of
#   "<env> <steps> <agent>:<seed>"  in $QUEUE (default runs/queue.txt), consumed from the top,
# so tasks can be appended while the slot is held. Stops starting new batches after MAX_MIN min.
# usage: ../bin/gpu_run ./worker.sh [MAX_MIN] [WIDTH]
cd "$(dirname "$0")"
MAX_MIN=${1:-36}
WIDTH=${2:-3}
Q=${QUEUE:-runs/queue.txt}
START=$(date +%s)
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
while true; do
  [ $(( ($(date +%s) - START) / 60 )) -ge $MAX_MIN ] && { echo "worker: time budget used"; break; }
  [ -s "$Q" ] || { echo "worker: queue empty"; break; }
  i=0
  while [ $i -lt $WIDTH ] && [ -s "$Q" ]; do
    LINE=$(head -n 1 "$Q"); tail -n +2 "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
    [ -z "$LINE" ] && continue
    TAG=$(echo $LINE | tr ' :' '__')
    echo "worker: $(date +%H:%M:%S) start: $LINE"
    ./sweep.sh $LINE > runs/task_$TAG.log 2>&1 &
    i=$((i + 1))
  done
  wait
  echo "worker: $(date +%H:%M:%S) batch done"
done
