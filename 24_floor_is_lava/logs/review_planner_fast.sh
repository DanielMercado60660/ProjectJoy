#!/bin/bash
# Reviewer: CPG + planner tuned ONLY over fast gaits (freq 3.2-5 Hz, stride 0.22-0.45 m), warm start from the
# fast open-loop gait found on flat walkways (review_fast_corner.py), then held-out eval on the same 500 layouts.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
PY=../.venv/bin/python
$PY review_tune_wide.py --fast_corner --planner --warm checkpoints/review_fast_corner_params.json --seed 7 --n_random 120 \
    --n_layouts 40 --cem_gens 10 --cem_pop 48 --final_layouts 192 --out checkpoints/review_planner_fast_params.json
$PY evaluate.py --policy checkpoints/policy.npz --planner checkpoints/review_planner_fast_params.json --n 500 --out results/review_eval_planner_fast_full.json
