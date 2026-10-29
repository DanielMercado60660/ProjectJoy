#!/bin/bash
# Follow-up (round 3): extra PPO training seeds, run as ONE gpu_run job.
# Single-threaded processes in parallel (at most 4 threads total).
# Iteration caps match the original runs (main beta=0.5: 307 iters; beta=0 ablation: 342 iters); wall cap MIN minutes.
# Usage:  MIN=11 SEEDS="1 2 3" B0SEEDS="1" ../bin/gpu_run ./train_seeds.sh      (job 1)
#         MIN=11 SEEDS="4 5" B0SEEDS="2 3" ../bin/gpu_run ./train_seeds.sh      (job 2)
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=../.venv/bin/python
MIN=${MIN:-12}
SEEDS=${SEEDS:-"1 2 3"}
B0SEEDS=${B0SEEDS:-"1"}
for s in $SEEDS; do
  $PY train.py --seed $s --minutes $MIN --max-iters 307 --out checkpoints/seeds/ppo_s$s \
      --log runs/seeds/train_s$s.jsonl > runs/seeds/train_s${s}_stdout.txt 2>&1 &
done
for s in $B0SEEDS; do
  $PY train.py --seed $s --beta 0 --minutes $MIN --max-iters 342 --out checkpoints/seeds/beta0_s$s \
      --log runs/seeds/train_beta0_s$s.jsonl > runs/seeds/train_beta0_s${s}_stdout.txt 2>&1 &
done
wait
echo "seeds done: $SEEDS / beta0: $B0SEEDS"
