#!/bin/bash
# Round 7 (design/R7_CAMPAIGN.md): one lane of training runs. Each run trains to 16,000 steps on the Hr data
# (bc_index + filtered DAgger 1-2), continuing with --resume after each 45-min guard stop. Selection and
# confirmation are run separately (r7_eval.py). Two lanes at most (AGENTS.md / the owner's 2-job cap).
#   ./r7_lane.sh 1b_s0 1b_s1 > logs/r7_lane1.log 2>&1
# Arms: 1b (Hr + belief vector, no pixels), 1c_bh (R3 + belief heads), 1c_ow (R3 + onset weighting x20),
#       2a (Hr + belief vector + pixels), 2a_sh (2a with the shuffled-vector control).
cd "$(dirname "$0")"
MAIN="--index results/m5/bc_index.jsonl --index episodes/dagger_filtered_index.jsonl --index episodes/dagger2_filtered_index.jsonl"
COMMON="--img 64 --aux-weight 1 --act leaky --steps 16000 --ckpt-every 4000 --max-minutes 44"
R3="--vx-head binary --turn-head discrete"
HR="$R3 --stream --stream-reset"
for name in "$@"; do
  arm=${name%_s*}; seed=${name##*_s}
  case $arm in
    1b|1bv1) extra="$HR --mem-vector --no-pixels" ;;
    1c_bh) extra="$R3 --belief-heads" ;;
    1c_ow) extra="$R3 --onset-weight 20" ;;
    2a)    extra="$HR --mem-vector $R7_2A_EXTRA" ;;                       # R7_2A_EXTRA: the 1C decision (r7_gate.py 1c)
    2a_sh) extra="$HR --mem-vector --shuffle-vector $R7_2A_EXTRA" ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  run=runs/r7_$name
  echo "[lane] $(date +%H:%M:%S) start $run"
  if [ ! -f $run/latest.json ]; then
    OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_train.py $run $MAIN $COMMON $extra --seed $seed >> logs/r7_${name}_train.log 2>&1
  fi
  if [ ! -f $run/latest.json ]; then echo "[lane] $(date +%H:%M:%S) FAILED $run (no checkpoint; see logs/r7_${name}_train.log)"; continue; fi
  for i in 1 2 3 4; do
    step=$(../.venv/bin/python -c "import json;print(json.load(open('$run/latest.json'))['step'])")
    [ "$step" -ge 16000 ] && break
    echo "[lane] $(date +%H:%M:%S) resume $run at $step"
    OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_train.py $run --resume --steps 16000 --max-minutes 44 >> logs/r7_${name}_train.log 2>&1
  done
  echo "[lane] $(date +%H:%M:%S) trained $run: $(tail -1 logs/r7_${name}_train.log)"
done
