cd "$(dirname "$0")/../.."
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=../.venv/bin/python
../bin/gpu_run $PY seeds_eval.py h2h --pairs seed1/136:main/136,seed2/136:main/136,seed1/136:seed2/136
until [ -f checkpoints/seed1/gen_170.json ] && [ -f checkpoints/seed2/gen_170.json ]; do sleep 10; done
../bin/gpu_run $PY seeds_eval.py suite --runs seed1,seed2 --gens 170
../bin/gpu_run $PY seeds_eval.py h2h --pairs seed1/170:seed1/136,seed2/170:seed2/136
echo "queue2 done $(date +%H:%M)"
