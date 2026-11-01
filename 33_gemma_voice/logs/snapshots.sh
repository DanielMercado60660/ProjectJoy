# usage: snapshots.sh <run dir> <interval> <log file to watch for a finish marker>
cd /Users/danielmercado/ProjectJoy/33_gemma_voice
run=$1; every=$2; log=$3
until grep -qE "^done step|Traceback" $log 2>/dev/null; do
  s=$(python3 -c "import json;print(json.load(open('$run/last/meta.json'))['step'])" 2>/dev/null)
  if [ -n "$s" ] && [ $((s % every)) -eq 0 ] && [ ! -d $run/snap_step$s ]; then
    mkdir -p $run/snap_step$s.tmp && cp $run/last/model.safetensors $run/last/meta.json $run/snap_step$s.tmp/ && mv $run/snap_step$s.tmp $run/snap_step$s && echo "snap $s"
  fi
  sleep 10
done
s=$(python3 -c "import json;print(json.load(open('$run/last/meta.json'))['step'])"); mkdir -p $run/snap_step$s && cp $run/last/model.safetensors $run/last/meta.json $run/snap_step$s/ && echo "final snap $s"
