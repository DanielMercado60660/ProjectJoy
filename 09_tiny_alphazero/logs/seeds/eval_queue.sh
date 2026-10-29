cd "$(dirname "$0")/../.."
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=../.venv/bin/python
until grep -q '^done' logs/seeds/noflip_seed2.log; do sleep 10; done
../bin/gpu_run $PY seeds_eval.py suite --runs seed1,seed2 --gens 40,70
until [ -f checkpoints/seed1/gen_100.json ] && [ -f checkpoints/seed2/gen_100.json ]; do sleep 10; done
../bin/gpu_run $PY seeds_eval.py suite --runs seed1,seed2 --gens 100
until [ -f checkpoints/seed1/gen_136.json ] && [ -f checkpoints/seed2/gen_136.json ]; do sleep 10; done
../bin/gpu_run $PY seeds_eval.py suite --runs seed1,seed2 --gens 136
../bin/gpu_run $PY seeds_eval.py h2h --pairs seed1/136:seed1/100,seed2/136:seed2/100,seed1/100:main/100,seed2/100:main/100,seed1/100:seed2/100
echo "queue done $(date +%H:%M)"
