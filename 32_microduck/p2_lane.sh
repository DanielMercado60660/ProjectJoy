#!/bin/bash
# Phase 2, amendments 6-7: one lane of recovery-round / history-pilot runs. Each run trains to 16,000 steps (continuing
# with --resume after each 45-min guard stop), then gets its set-lamp dev selection. Three lanes at once.
#   ./p2_lane.sh c_s0 r3_s0 r2_s1 > logs/p2_lane1.log 2>&1
cd "$(dirname "$0")"
MAIN="--index results/m5/bc_index.jsonl --index episodes/dagger_filtered_index.jsonl --index episodes/dagger2_filtered_index.jsonl"
COMMON="--img 64 --aux-weight 1 --act leaky --steps 16000 --ckpt-every 4000 --max-minutes 45"
REC="--recovery-index episodes/recovery_index.jsonl --recovery-frac 0.5"
HEADS="--vx-head binary --turn-head discrete"
for name in "$@"; do
  arm=${name%_s*}; seed=${name##*_s}
  case $arm in
    c)  extra="" ;;
    r1) extra="$REC" ;;
    r2) extra="$REC $HEADS" ;;
    r3) extra="$HEADS" ;;
    e)  extra="$HEADS --p-end 0.2" ;;          # amendment 7
    h)  extra="$HEADS --stream" ;;
    hr) extra="$HEADS --stream --stream-reset" ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  run=runs/p2_$name
  echo "[lane] $(date +%H:%M:%S) start $run"
  if [ ! -f $run/latest.json ]; then
    OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_train.py $run $MAIN $COMMON $extra --seed $seed >> logs/p2_${name}_train.log 2>&1
  fi
  if [ ! -f $run/latest.json ]; then echo "[lane] $(date +%H:%M:%S) FAILED $run (no checkpoint; see logs/p2_${name}_train.log)"; continue; fi
  for i in 1 2 3; do
    step=$(../.venv/bin/python -c "import json;print(json.load(open('$run/latest.json'))['step'])")
    [ "$step" -ge 16000 ] && break
    echo "[lane] $(date +%H:%M:%S) resume $run at $step"
    OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_train.py $run --resume --steps 16000 --max-minutes 45 >> logs/p2_${name}_train.log 2>&1
  done
  echo "[lane] $(date +%H:%M:%S) trained $run; selecting"
  OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_select.py $run > logs/p2_${name}_select.log 2>&1
  echo "[lane] $(date +%H:%M:%S) done $run: $(tail -1 logs/p2_${name}_select.log)"
done
