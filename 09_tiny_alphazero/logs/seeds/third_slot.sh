cd "$(dirname "$0")/../.."
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
until grep -q '^done' logs/seeds/noflip_seed1.log; do sleep 10; done
../bin/gpu_run ../.venv/bin/python train.py --run noflip_seed2 --seed 2 --no-flip --gens 40 --minutes 30 > logs/seeds/noflip_seed2.log 2>&1
echo "noflip2 done $(date +%H:%M)"
