#!/bin/bash
# Is the GPU free of other projects' work? Prints other gpu_run jobs (not 30_nanochat_mlx) + load.
others=""
while read -r pid rest; do
  [ -z "$pid" ] && continue
  cwd=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')
  if [[ "$cwd" != *30_nanochat_mlx* ]]; then others+="$pid $cwd :: ${rest:0:120}"$'\n'; fi
done < <(ps -axo pid,command | grep "bin/gpu_run" | grep -v grep)
echo "$(date '+%H:%M:%S') load: $(sysctl -n vm.loadavg)"
if [ -n "$others" ]; then echo "BUSY: other gpu_run jobs:"; printf "%s" "$others"; exit 1; else echo "IDLE: no other gpu_run jobs"; fi
