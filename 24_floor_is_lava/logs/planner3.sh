#!/bin/bash
# Planner baseline re-tune round 3 (warm start from round 2), then held-out eval.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
PY=../.venv/bin/python
$PY tune_cpg.py --planner --warm checkpoints/planner2_params.json --seed 2 --n_random 60 --n_layouts 40 \
    --cem_gens 20 --cem_pop 48 --final_layouts 192 --out checkpoints/planner3_params.json
$PY evaluate.py --policy checkpoints/policy.npz --planner checkpoints/planner3_params.json --n 500 --out results/eval_planner3_full.json
$PY evaluate.py --policy checkpoints/policy.npz --planner checkpoints/planner3_params.json --n 500 --gap_max 0.125 --push 0 --out results/eval_planner3_gap0125.json
$PY evaluate.py --policy checkpoints/policy.npz --planner checkpoints/planner3_params.json --n 200 --flat --push 80 --out results/eval_planner3_flat_push80.json
