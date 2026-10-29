#!/bin/bash
# Extra held-out evaluations after the blind ablation run (one gpu_run job, 4 workers).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
PY=../.venv/bin/python
$PY evaluate.py --policy checkpoints/blind_latest.npz --baselines 0 --n 500 --out results/eval_blind_full.json
$PY evaluate.py --policy checkpoints/policy.npz --baselines 0 --mask_stones --push 0 --n 500 --out results/eval_mask_stones.json
$PY evaluate.py --policy checkpoints/policy.npz --n 500 --push 80 --out results/eval_push80_full.json
$PY evaluate.py --policy checkpoints/policy.npz --n 200 --flat --push 80 --out results/eval_flat_push80.json
$PY evaluate.py --policy checkpoints/policy.npz --n 200 --flat --push 120 --out results/eval_flat_push120.json
$PY evaluate.py --policy checkpoints/policy.npz --n 300 --gap_max 0.25 --push 0 --out results/eval_ood_gap025.json
