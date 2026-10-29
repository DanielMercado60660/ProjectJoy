#!/bin/bash
# All remaining experiments in ONE GPU-slot reservation, in priority order (~40 min).
# Usage: ../bin/gpu_run bash pipeline2.sh
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
PY=../.venv/bin/python
echo "[pipeline2] $(date +%H:%M) eval_heuristic"; $PY eval_heuristic.py --maps 100 --per_map 3 > logs/eval_heuristic.log 2>&1
echo "[pipeline2] $(date +%H:%M) run_bridge";     $PY run_bridge.py --maps 150 --per_map 2 > logs/bridge.log 2>&1
echo "[pipeline2] $(date +%H:%M) train_prior";    $PY train_prior.py --steps 3000 > logs/train_prior.log 2>&1
echo "[pipeline2] $(date +%H:%M) run_fog";        $PY run_fog.py --maps 100 --per_map 2 > logs/fog.log 2>&1
echo "[pipeline2] $(date +%H:%M) train hnet16";   $PY train.py --steps 6000 --bs 16 --base 16 --tag hnet16 > logs/train_hnet16.log 2>&1
echo "[pipeline2] $(date +%H:%M) eval hnet16";    $PY eval_heuristic.py --maps 100 --per_map 3 --tag hnet16 --out results/heuristic_hnet16.json --no_figs > logs/eval_hnet16.log 2>&1
echo "[pipeline2] $(date +%H:%M) build_demo";     $PY build_demo.py > logs/build_demo.log 2>&1
echo "[pipeline2] $(date +%H:%M) done"
