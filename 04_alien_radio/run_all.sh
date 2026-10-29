#!/bin/bash
# Pipeline behind the reported numbers. Stage 2 goes through the GPU lock:
#   ../bin/gpu_run bash run_all.sh > runs/run_all.log 2>&1
# Stage 1 runs are ~15 s each and were run directly:
#   ../.venv/bin/python train.py --stage color --method gumbel reinforce nocomm --P 8 --fresh
#   bash sweep_color.sh > runs/sweep_color.log
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
PY=../.venv/bin/python
# Stage 2 (main): colour x shape x location, 5 independent seeds per method, 15k steps each
$PY train.py --stage full --method gumbel reinforce nocomm compositional holistic --P 5
# Variants (same steps, 5 seeds): REINFORCE entropy schedule, and channel size
$PY train.py --stage full --method reinforce --P 5 --entropy_final 0.01 --tag _decay
$PY train.py --stage full --method reinforce --P 5 --entropy_coef 0.3 --tag _ent0.3
$PY train.py --stage full --method gumbel --P 5 --V 16 --tag _V16
$PY train.py --stage full --method gumbel --P 5 --L 2 --V 16 --tag _L2V16
$PY train.py --stage full --method gumbel --P 5 --L 4 --V 8 --tag _L4
$PY analyze.py
$PY make_demo.py
