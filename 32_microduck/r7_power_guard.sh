#!/bin/bash
# Round 7 power guard (the brief's battery rule). Every 20 s: on AC, SIGCONT every job of mine; on battery at >= 50%,
# keep only the oldest of my jobs running and SIGSTOP the others (with their pool workers); below 50%, SIGSTOP all.
# Jobs are matched by exact command lines (p2_train on runs/r7_*, r7_eval, the servo, the 3B dagger round, the gaze
# pre-gate); PIDs come from ps, never from a pgrep of this script's own text. Transitions go to logs/r7_power.log.
cd "$(dirname "$0")"
PAT='MacOS/Python (p2_train\.py runs/r7_|r7_eval\.py|p2_servo_baseline\.py|p2_skillsel\.py dagger|r7_gaze_pregate\.py)'
mine() { ps -axo pid,command | grep -E "$PAT" | grep -v grep | awk '{print $1}' | sort -n; }
kids() { pgrep -P "$1" 2>/dev/null; }
sig() { local s=$1 p=$2; for c in $(kids "$p"); do kill -"$s" "$c" 2>/dev/null; done; kill -"$s" "$p" 2>/dev/null; }
last=""
while true; do
  src=$(pmset -g batt | head -1)
  pct=$(pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d %)
  jobs=$(mine)
  if echo "$src" | grep -q "AC Power"; then
    state="ac"
    for p in $jobs; do sig CONT "$p"; done
  elif [ -n "$pct" ] && [ "$pct" -lt 50 ]; then
    state="battery_low_${pct}"
    for p in $jobs; do sig STOP "$p"; done
  else
    state="battery_${pct}"
    first=1
    for p in $jobs; do
      if [ $first = 1 ]; then sig CONT "$p"; first=0; else sig STOP "$p"; fi
    done
  fi
  if [ "$state" != "$last" ]; then
    echo "[power] $(date +%Y-%m-%dT%H:%M:%S) $state; jobs: $(echo $jobs | tr '\n' ' ')" >> logs/r7_power.log
    last=$state
  fi
  sleep 20
done
