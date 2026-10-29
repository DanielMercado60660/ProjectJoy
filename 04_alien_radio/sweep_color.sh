#!/bin/bash
# Stage-1 sensitivity sweep: 8 seeds per setting, 3000 steps; each run takes ~15 s.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
PY=../.venv/bin/python
D=checkpoints/sweep
run(){ $PY train.py --stage color --P 8 --ckpt_dir $D --fresh "$@" | grep done; }
for tau in 1 2; do for ge in 0.0 0.1 0.3 1.0; do
  run --method gumbel --tau $tau --gumbel_entropy_coef $ge --tag _tau${tau}_ent${ge}
done; done
for e in 0.03 0.1 0.3 1.0; do
  run --method reinforce --entropy_coef $e --tag _ent$e
done
run --method reinforce --entropy_coef 0.3 --lr 1e-3 --tag _ent0.3_lr1e-3
run --method reinforce --entropy_coef 0.3 --steps 9000 --tag _ent0.3_9k
