#!/bin/sh
# Reviewer side probe: NumPy (and MLX CPU) with the shelf's 4-thread BLAS cap vs uncapped, on four
# throughput batches. Run from 31_silicon_decathlon/ via ../bin/gpu_run on an idle machine.
# Output: results/probes/numpy_threads/{capped,uncapped}/<event>_provisional.json
set -e
cd "$(dirname "$0")/../../.."
PY=../.venv/bin/python
ARGS="--only numpy_cpu mlx_cpu --no-task --repeats 3 --budget 0.5 --true-fp32"
for ev in "e07_terra 16" "e09_alphazero 256" "e11_physics_detective 1000" "e29_image_gen 64"; do
  set -- $ev
  echo "== $1 b=$2 capped (VECLIB_MAXIMUM_THREADS=4)"
  VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 MLX_ENABLE_TF32=0 $PY run_event.py $1 --batches $2 $ARGS --out results/probes/numpy_threads/capped | grep -E "numpy|mlx_cpu|wrote"
  echo "== $1 b=$2 uncapped"
  env -u VECLIB_MAXIMUM_THREADS -u OMP_NUM_THREADS MLX_ENABLE_TF32=0 $PY run_event.py $1 --batches $2 $ARGS --out results/probes/numpy_threads/uncapped | grep -E "numpy|mlx_cpu|wrote"
done
