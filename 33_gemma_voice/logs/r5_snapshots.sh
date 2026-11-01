cd /Users/danielmercado/ProjectJoy/33_gemma_voice
until grep -q "R5_EXIT" logs/r5.log; do
  s=$(python3 -c "import json;print(json.load(open('runs/r5/last/meta.json'))['step'])" 2>/dev/null)
  if [ -n "$s" ] && [ $((s % 2000)) -eq 0 ] && [ ! -d runs/r5/snap_step$s ]; then
    mkdir -p runs/r5/snap_step$s.tmp && cp runs/r5/last/model.safetensors runs/r5/last/meta.json runs/r5/snap_step$s.tmp/ && mv runs/r5/snap_step$s.tmp runs/r5/snap_step$s && echo "snap $s"
  fi
  sleep 30
done
echo SNAPS_DONE
