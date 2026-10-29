#!/bin/bash
# Keep re-queuing gpu_run sessions until every task in session.py is done (or the 12:15 cutoff).
cd "$(dirname "$0")"
while true; do
  [ "$(date +%H%M)" -ge 1215 ] && break
  ../.venv/bin/python -c "import session as S, sys; sys.exit(0 if all(S.done(t[0]) for t in S.TASKS) else 1)" && break
  ../bin/gpu_run ../.venv/bin/python session.py --minutes 42 "$@"
  sleep 30
done
