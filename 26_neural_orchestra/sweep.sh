#!/bin/bash
# All training for the results plus a quick check-eval to results/eval_sweep.json
# (the reported eval is the separate command in README.md) (one GPU slot, ~6 min when a slot is free).
# Order: MLP beta sweep + unconditional ablation -> eval -> AR (GRU decoder) VAE -> eval again including AR.
# No `set -e`: an AR failure must not lose the MLP evaluation.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
PY=../.venv/bin/python
for b in 1.0 0.5 0.25; do
  $PY train.py --beta $b --out checkpoints/vae_cond_b$b
done
$PY train.py --no_cond --beta 0.5 --out checkpoints/vae_uncond_b0.5
MLP="checkpoints/vae_cond_b1.0/best checkpoints/vae_cond_b0.5/best checkpoints/vae_cond_b0.25/best checkpoints/vae_uncond_b0.5/best"
$PY eval.py --vaes $MLP --out results/eval_sweep.json > logs/eval_sweep.log 2>&1 && echo "MLP eval done"
$PY train.py --arch ar --beta 1.0 --free_bits 10 --epochs 150 --warmup 25 --out checkpoints/arvae_cond_fb10
if [ -f checkpoints/arvae_cond_fb10/best/vae.safetensors ]; then
  $PY eval.py --vaes $MLP checkpoints/arvae_cond_fb10/best --out results/eval_sweep.json > logs/eval_sweep.log 2>&1 && echo "full eval done"
fi
echo "sweep finished"
