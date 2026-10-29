#!/bin/bash
# Extra experiments after the main post stage (written once the full-data SFT hit 100%):
#  - KV-cache inference benchmark (first-token latency, decode tok/s)
#  - SFT data-efficiency: 300 / 1000 training episodes, pretrained vs from-scratch
#  - 3-delivery tasks (longer than anything in training) for all LM policies
#   ../bin/gpu_run bash run_extra.sh > logs/extra.log 2>&1
cd "$(dirname "$0")"
PY=../.venv/bin/python
echo "[extra] $(date) start"
$PY engine.py --ckpt checkpoints/pilot/last --out media/inference_bench.json
for N in 300 1000; do
  EPOCHS=$(( N == 300 ? 20 : 6 ))
  $PY sft.py --base checkpoints/pilot/last --out checkpoints/sft_n$N --episodes $N --epochs $EPOCHS
  $PY sft.py --scratch --base checkpoints/pilot/last --out checkpoints/sft_n${N}_scratch --episodes $N --epochs $EPOCHS
  $PY eval_sandbox.py --ckpt checkpoints/sft_n$N/last --policies lm,lm_constrained --episodes 100 --tag "pretrained,n=$N" --out media/sandbox_eval_extra.json
  $PY eval_sandbox.py --ckpt checkpoints/sft_n${N}_scratch/last --policies lm,lm_constrained --episodes 100 --tag "scratch,n=$N" --out media/sandbox_eval_extra.json
done
# wait for the main post stage (it writes the scratch SFT model) before the 3-delivery evals
while ! grep -q "\[post\] .* end" logs/pilot.log; do sleep 20; done
$PY eval_sandbox.py --ckpt checkpoints/sft/last --policies lm,lm_constrained --episodes 100 --splits test3 --tag pretrained --out media/sandbox_eval_extra.json
$PY eval_sandbox.py --ckpt checkpoints/sft_scratch/last --policies lm,lm_constrained --episodes 100 --splits test3 --tag scratch --out media/sandbox_eval_extra.json
$PY eval_sandbox.py --policies expert,random,mlp --episodes 100 --splits test3 --out media/sandbox_eval_extra.json
$PY plot.py
$PY make_demo.py --ckpt checkpoints/sft/last
echo "[extra] $(date) end"
