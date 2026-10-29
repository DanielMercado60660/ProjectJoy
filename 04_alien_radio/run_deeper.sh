#!/bin/bash
# Follow-up: does a harder game (K=20 random candidates instead of 5) push the agents towards
# messages that identify objects exactly?   ../bin/gpu_run bash run_deeper.sh > runs/run_deeper.log 2>&1
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
PY=../.venv/bin/python
$PY train.py --stage full --method gumbel reinforce --P 5 --K 20 --cluster_frac 0 --tag _K20
$PY analyze.py
$PY make_demo.py
