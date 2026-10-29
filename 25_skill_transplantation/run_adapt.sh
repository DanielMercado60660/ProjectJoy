#!/bin/bash
# Job 2: adaptation-cost stage for 4 seeds (resumable: finished runs are skipped, PPO runs resume
# from their checkpoints), then the scripted CPG baseline tuning.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
../.venv/bin/python pipeline.py --stage adapt --seeds 0 1 2 3
../.venv/bin/python cpg.py --iters 40 --pop 48
