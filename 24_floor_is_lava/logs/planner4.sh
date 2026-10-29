#!/bin/bash
# Planner baseline re-tune round 4 (warm start from round 3), then held-out eval.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
PY=../.venv/bin/python
$PY tune_cpg.py --planner --warm checkpoints/planner3_params.json --seed 3 --n_random 60 --n_layouts 40 \
    --cem_gens 16 --cem_pop 48 --final_layouts 192 --out checkpoints/planner4_params.json
$PY evaluate.py --policy checkpoints/policy.npz --planner checkpoints/planner4_params.json --n 500 --out results/eval_planner4_full.json
$PY evaluate.py --policy checkpoints/policy.npz --planner checkpoints/planner4_params.json --n 500 --gap_max 0.125 --push 0 --out results/eval_planner4_gap0125.json
$PY evaluate.py --policy checkpoints/policy.npz --planner checkpoints/planner4_params.json --n 200 --flat --push 80 --out results/eval_planner4_flat_push80.json
$PY evaluate.py --policy checkpoints/policy.npz --planner checkpoints/planner4_params.json --n 200 --flat --push 40 --out results/eval_planner4_flat_push40.json
$PY evaluate.py --policy checkpoints/policy.npz --planner checkpoints/planner4_params.json --n 200 --flat --push 120 --out results/eval_planner4_flat_push120.json
$PY evaluate.py --policy checkpoints/policy.npz --planner checkpoints/planner4_params.json --n 300 --gap_max 0.25 --push 0 --out results/eval_planner4_ood_gap025.json
