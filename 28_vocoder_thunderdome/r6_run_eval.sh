#!/bin/sh
# Round 6: vocode + score one or more systems on one or more sets, then UTMOS them (one model load).
#   ./r6_run_eval.sh "val100" "gt gl64 old"                       # baselines
#   ./r6_run_eval.sh "val100" "gan_s4000=checkpoints/r6_runs/gan/snaps/G_4000.safetensors"
# A system is a name (gt, glNN, old = checkpoints/followup_val3k/best, the generator 27 uses) or name=checkpoint.
# Run through ../bin/gpu_run (it is >1 min); MLX/numpy in ../.venv, UTMOS in ../.venv-sac, 4 threads each.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
SETS="$1"; SYSTEMS="$2"
JOBS=$(mktemp)
echo "[" > "$JOBS"; first=1
for s in $SETS; do
  for spec in $SYSTEMS; do
    name=${spec%%=*}; ck=""
    case "$spec" in *=*) ck=${spec#*=};; esac
    [ "$name" = old ] && ck=checkpoints/followup_val3k/best.safetensors
    if [ -n "$ck" ]; then ../.venv/bin/python r6_eval.py --set "$s" --system "$name" --ckpt "$ck"
    else ../.venv/bin/python r6_eval.py --set "$s" --system "$name"; fi
    [ $first = 1 ] || echo "," >> "$JOBS"; first=0
    echo "{\"wavdir\": \"results_r6/wavs/$s/$name\", \"out\": \"results_r6/mos/$s/$name.json\"}" >> "$JOBS"
  done
done
echo "]" >> "$JOBS"
../.venv-sac/bin/python r6_utmos.py --jobs "$JOBS" --threads 4
rm -f "$JOBS"
