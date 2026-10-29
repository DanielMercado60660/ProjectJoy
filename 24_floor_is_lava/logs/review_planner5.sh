#!/bin/bash
# Reviewer: planner round 5 with a widened freq/stride range (warm start from round 4), then held-out evals.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
PY=../.venv/bin/python
$PY review_tune_wide.py --planner --warm checkpoints/planner4_params.json --seed 5 --n_random 120 --n_layouts 40 \
    --cem_gens 10 --cem_pop 48 --final_layouts 192 --out checkpoints/review_planner5_wide_params.json
$PY evaluate.py --policy checkpoints/policy.npz --planner checkpoints/review_planner5_wide_params.json --n 500 --out results/review_eval_planner5_full.json
