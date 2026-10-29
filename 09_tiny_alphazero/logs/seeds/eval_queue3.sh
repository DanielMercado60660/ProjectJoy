cd "$(dirname "$0")/../.."
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=../.venv/bin/python
../bin/gpu_run $PY seeds_eval.py h2h --pairs main/170:main/136
until [ -f checkpoints/seed1/gen_191.json ] && [ -f checkpoints/seed2/gen_191.json ]; do sleep 10; done
../bin/gpu_run $PY seeds_eval.py suite --runs seed1,seed2 --gens 191
../bin/gpu_run $PY seeds_eval.py h2h --pairs seed1/191:seed1/136,seed2/191:seed2/136,seed1/191:main/191,seed2/191:main/191,seed1/191:seed2/191
echo "queue3 done $(date +%H:%M)"
