set -e
E="../bin/gpu_run ../.venv/bin/python evaluate.py --split val --n 100"
$E --run runs/r3 --ckpt best --temp 0.8 --tag val100_t0.8
$E --run runs/r3 --ckpt best --temp 0.6 --tag val100_t0.6
$E --run runs/scratch --ckpt best --temp 0.8 --tag val100_t0.8
$E --run runs/scratch --ckpt best --temp 0.6 --tag val100_t0.6
$E --run runs/r2 --ckpt best --temp 0.8 --tag val100_t0.8
$E --run runs/r1 --ckpt best --temp 0.8 --tag val100_t0.8
$E --run runs/main --ckpt best --temp 0.8 --tag val100_t0.8
../bin/gpu_run ../.venv/bin/python score.py ceiling --nq 8 16 32
echo BATCH_DONE
