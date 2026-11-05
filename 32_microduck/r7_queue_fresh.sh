#!/bin/bash
# Round 7 Stage 5: every frozen arm once on the 96-room fresh reserved slice, two jobs at a time.
# Arms come from results/r7/fresh_arms.txt (one NAME=WEIGHTS per line; weights may be a path, 'bot',
# 'servo:lamp', 'selector:<spec>' or 'gaze:<path>'). Written only after the method set froze (STATUS.md).
cd "$(dirname "$0")"
run_one() { OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py fresh "$1" > "logs/r7_fresh_${1%%=*}.log" 2>&1; echo "[fresh] $(date +%H:%M:%S) done ${1%%=*}: $(tail -1 logs/r7_fresh_${1%%=*}.log | cut -c1-200)"; }
n=0
while read -r arm; do
  [ -z "$arm" ] && continue
  case "$arm" in \#*) continue ;; esac
  run_one "$arm" &
  n=$((n + 1))
  if [ $((n % 2)) -eq 0 ]; then wait; fi
done < "${1:-results/r7/fresh_arms.txt}"
wait
echo "[fresh] $(date +%H:%M:%S) all arms done"
