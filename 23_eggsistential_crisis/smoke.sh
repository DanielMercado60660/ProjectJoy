#!/bin/bash
# End-to-end pipeline check at tiny size (< 2 min). Writes only under smoke_out/.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 PYTHONWARNINGS=ignore
rm -rf smoke_out; mkdir -p smoke_out
$PY -m pytest -q test_parity.py
$PY train.py --run smoke --max_iters 2 --n_envs 16 --horizon 16 --ckpt_every 1 --log_dir smoke_out --ckpt_dir smoke_out
$PY baselines.py --out smoke_out/baselines.json --n_eps 4 --pop 2 --iters 1 --log smoke_out/baselines.log
$PY evaluate.py --ckpt smoke_out/smoke/latest.npz --baselines smoke_out/baselines.json --quick --baselines_fa smoke_out/none.json --out smoke_out/eval.json > smoke_out/eval.txt
$PY render_demo.py --ckpt smoke_out/smoke/latest.npz --n 4 --media smoke_out --baselines smoke_out/baselines.json
ls smoke_out
echo "SMOKE OK"
