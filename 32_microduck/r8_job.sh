#!/bin/bash
# Round 8 job throttle (all R8 builders share it): at most R8_MAX (default 3, the owner's standing cap) of our
# long jobs run at once; on battery at most 1, and none below 50%. Each job also goes through ../bin/gpu_run.
#
#   ./r8_job.sh <tag> <command...>        e.g. ./r8_job.sh bam-gate ../.venv/bin/python m0_gate.py ...
#
# Slots are directories under logs/r8_slots/ (mkdir is atomic); a slot whose owner PID is gone is reclaimed.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SLOTS="$HERE/logs/r8_slots"
mkdir -p "$SLOTS"
MAX="${R8_MAX:-3}"
tag="${1:?usage: r8_job.sh <tag> <command...>}"
shift

allowed() {
  local b pct
  b="$(pmset -g batt 2>/dev/null)"
  if echo "$b" | grep -q "AC Power"; then echo "$MAX"; return; fi
  pct="$(echo "$b" | grep -o '[0-9]*%' | head -1 | tr -d '%')"
  if [ -n "$pct" ] && [ "$pct" -ge 50 ]; then echo 1; else echo 0; fi
}

held=""
cleanup() { [ -n "$held" ] && rm -rf "$held"; }
trap cleanup EXIT
trap 'exit 130' INT TERM

waited=0
while [ -z "$held" ]; do
  n="$(allowed)"
  for i in $(seq 1 "$n"); do
    d="$SLOTS/slot$i"
    if [ -d "$d" ]; then
      pid="$(cat "$d/pid" 2>/dev/null || true)"
      if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then rm -rf "$d"; fi
    fi
    if mkdir "$d" 2>/dev/null; then
      echo $$ > "$d/pid"
      echo "$tag $(date '+%F %T')" > "$d/info"
      held="$d"
      break
    fi
  done
  if [ -z "$held" ]; then
    if [ $((waited % 300)) -eq 0 ]; then
      echo "[r8_job] $tag waiting for a slot ($n allowed; busy: $(cat "$SLOTS"/*/info 2>/dev/null | tr '\n' ';'))" >&2
    fi
    sleep 5
    waited=$((waited + 5))
  fi
done
echo "[r8_job] $tag holds $(basename "$held") at $(date '+%T')" >&2
"$HERE/../bin/gpu_run" "$@"
