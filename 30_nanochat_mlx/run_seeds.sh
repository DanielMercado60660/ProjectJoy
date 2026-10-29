#!/bin/bash
# Extra SFT seeds (SEED=1,2,...) for the data-efficiency comparison (single-seed results were noisy).
#   ../bin/gpu_run bash run_seeds.sh > logs/seeds.log 2>&1
cd "$(dirname "$0")"
PY=../.venv/bin/python
S=${SEED:-1}
echo "[seeds] $(date) start"
for N in 300 1000; do
  EPOCHS=$(( N == 300 ? 20 : 6 ))
  $PY sft.py --base checkpoints/pilot/last --out checkpoints/sft_n${N}_s$S --episodes $N --epochs $EPOCHS --seed $S
  $PY sft.py --scratch --base checkpoints/pilot/last --out checkpoints/sft_n${N}_scratch_s$S --episodes $N --epochs $EPOCHS --seed $S
  $PY eval_sandbox.py --ckpt checkpoints/sft_n${N}_s$S/last --policies lm --episodes 100 --tag "pretrained,n=$N,seed=$S" --out media/sandbox_eval_seeds_s$S.json
  $PY eval_sandbox.py --ckpt checkpoints/sft_n${N}_scratch_s$S/last --policies lm --episodes 100 --tag "scratch,n=$N,seed=$S" --out media/sandbox_eval_seeds_s$S.json
done
echo "[seeds] $(date) end"
