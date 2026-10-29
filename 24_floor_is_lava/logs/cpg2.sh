#!/bin/bash
# Open-loop CPG re-tune round 2 (warm start from round 1) + held-out evals of it alone.
# Waits for this project's planner-4 job (pid 51219) so only one job runs at a time.
cd "$(dirname "$0")/.."
while kill -0 51219 2>/dev/null; do sleep 10; done
export OMP_NUM_THREADS=1
PY=../.venv/bin/python
../bin/gpu_run bash -c '
PY=../.venv/bin/python
$PY tune_cpg.py --warm checkpoints/cpg_params.json --seed 1 --n_random 60 --n_layouts 40 --cem_gens 20 --cem_pop 48 --final_layouts 192 --out checkpoints/cpg2_params.json
$PY evaluate.py --policy "" --planner "" --cpg checkpoints/cpg2_params.json --n 500 --out results/eval_cpg2_full.json
$PY evaluate.py --policy "" --planner "" --cpg checkpoints/cpg2_params.json --n 500 --gap_max 0.125 --push 0 --out results/eval_cpg2_gap0125.json
$PY evaluate.py --policy "" --planner "" --cpg checkpoints/cpg2_params.json --n 200 --flat --push 80 --out results/eval_cpg2_flat_push80.json
'
