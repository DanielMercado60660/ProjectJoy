#!/bin/bash
# Round 8 power guard (the owner's battery rule). Every 20 s, for every job holding an r8_job.sh slot:
# on AC, SIGCONT all; on battery at >= 50%, keep only the oldest job running and SIGSTOP the rest (whole process
# trees: r8_job.sh -> gpu_run -> python -> pool workers); below 50%, SIGSTOP all. Transitions: logs/r8_power.log.
cd "$(dirname "$0")"
desc() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do echo "$c"; desc "$c"; done; }
sigtree() { local s=$1 p=$2 c; for c in $(desc "$p"); do kill -"$s" "$c" 2>/dev/null; done; kill -"$s" "$p" 2>/dev/null; }
jobs_oldest_first() {  # "<start time> <pid>" per held slot, oldest first
  local d pid t
  for d in logs/r8_slots/slot*; do
    [ -d "$d" ] || continue
    pid=$(cat "$d/pid" 2>/dev/null) || continue
    t=$(cut -d' ' -f2- "$d/info" 2>/dev/null)
    kill -0 "$pid" 2>/dev/null && echo "$t|$pid"
  done | sort | cut -d'|' -f2
}
last=""
while true; do
  src=$(pmset -g batt | head -1)
  pct=$(pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d %)
  jobs=$(jobs_oldest_first)
  if echo "$src" | grep -q "AC Power"; then
    state="ac"
    for p in $jobs; do sigtree CONT "$p"; done
  elif [ -n "$pct" ] && [ "$pct" -lt 50 ]; then
    state="battery_low_${pct}"
    for p in $jobs; do sigtree STOP "$p"; done
  else
    state="battery_${pct}"
    first=1
    for p in $jobs; do
      if [ $first = 1 ]; then sigtree CONT "$p"; first=0; else sigtree STOP "$p"; fi
    done
  fi
  if [ "$state" != "$last" ]; then
    echo "[power] $(date +%Y-%m-%dT%H:%M:%S) $state; jobs: $(echo $jobs | tr '\n' ' ')" >> logs/r8_power.log
    last=$state
  fi
  sleep 20
done
