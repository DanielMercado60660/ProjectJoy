#!/bin/sh
# Chain resumable <=43-min chunks of one r6_train.py run until it reaches <until_step>, scoring every new 4k-step
# generator snapshot on val100 (r6_run_eval.sh: metrics + UTMOS) between chunks, so at most 2 jobs run at once.
#   ./r6_chain.sh <out_dir> <wait_pid|0> <until_step> [first-launch args...]
# Each chunk / scoring step is its own ../bin/gpu_run job; logs go to results_r6/train_<name>_c<k>.log, chain_<name>.log.
cd "$(dirname "$0")"
OUT=$1; WAIT=$2; UNTIL=$3; shift 3
NAME=$(basename "$OUT")
score() {
  for f in "$OUT"/snaps/G_*.safetensors; do
    [ -e "$f" ] || continue
    st=${f##*G_}; st=${st%.safetensors}
    [ $((st % 4000)) = 0 ] || [ "$st" = "$UNTIL" ] || continue
    [ -e "results_r6/mos/val100/${NAME}_s$st.json" ] && continue
    OMP_NUM_THREADS=4 ../bin/gpu_run ./r6_run_eval.sh val100 "${NAME}_s$st=$f" 2>&1 | grep -E '^val100|"n"' | cut -c1-160
  done
}
[ "$WAIT" != 0 ] && while kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
k=1
while :; do
  score
  step=$(../.venv/bin/python -c "import json,sys; print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null || echo 0)
  [ "$step" -ge "$UNTIL" ] && break
  [ -e "$OUT/STOP" ] && { echo "STOP file, chain ends at step $step"; exit 0; }
  while [ -e "results_r6/train_${NAME}_c$k.log" ]; do k=$((k+1)); done
  OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python r6_train.py --out "$OUT" --max_minutes 43 --stop_at "$UNTIL" "$@" > "results_r6/train_${NAME}_c$k.log" 2>&1
  grep -q "saved at step" "results_r6/train_${NAME}_c$k.log" || { echo "chunk $k failed"; exit 1; }
done
echo "chain $NAME done at step $step"
