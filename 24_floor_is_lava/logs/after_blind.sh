#!/bin/bash
# One gpu_run job: extra evals, then planner baseline re-tune (round 2), then its held-out eval.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
PY=../.venv/bin/python
bash logs/eval_extra.sh
$PY tune_cpg.py --planner --warm checkpoints/planner_params.json --seed 1 --n_random 100 --n_layouts 32 \
    --cem_gens 24 --cem_pop 48 --final_layouts 128 --out checkpoints/planner2_params.json
$PY evaluate.py --policy checkpoints/policy.npz --planner checkpoints/planner2_params.json --n 500 --out results/eval_planner2_full.json
$PY evaluate.py --policy checkpoints/policy.npz --planner checkpoints/planner2_params.json --n 500 --gap_max 0.125 --push 0 --out results/eval_planner2_gap0125.json
