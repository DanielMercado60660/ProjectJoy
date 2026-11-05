#!/bin/bash
# R8 Gate C: run one LeWM arm to completion as a chain of <= 44-min invocations through r8_job.sh (resume
# between them). Logs: logs/r8_rep_<tag>_<n>.log.
#   [WAIT_FOR=data/r8_rep/train_data_ready.json] r8_rep/lane.sh lewm-ckpt runs/r8_lewm_ckpt --init ckpt:tworooms --steps 12000 --batch 32
set -u
cd "$(dirname "$0")/.."
tag="$1"; run="$2"; shift 2
if [ -n "${WAIT_FOR:-}" ]; then until [ -s "$WAIT_FOR" ]; do sleep 10; done; fi   # e.g. the training-data marker
i=0
if [ ! -d "$run/ckpt" ]; then
  i=1
  ./r8_job.sh "$tag" ../.venv/bin/python r8_rep/rep_train.py --run "$run" "$@" > "logs/r8_rep_${tag}_$i.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] && [ $rc -ne 3 ]; then echo "[$tag] invocation $i failed rc=$rc"; exit 1; fi
fi
while [ ! -d "$run/final" ]; do
  i=$((i + 1))
  if [ $i -gt 10 ]; then echo "[$tag] too many invocations"; exit 2; fi
  ./r8_job.sh "$tag" ../.venv/bin/python r8_rep/rep_train.py --run "$run" --resume > "logs/r8_rep_${tag}_$i.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] && [ $rc -ne 3 ]; then echo "[$tag] invocation $i failed rc=$rc"; exit 1; fi
done
echo "[$tag] done after $i invocations at $(date '+%T')"
