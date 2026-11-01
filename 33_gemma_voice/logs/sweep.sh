cd /Users/danielmercado/ProjectJoy/33_gemma_voice
for cfg in "0.2 1" "0.6 1" "0.4 4" "0.6 4"; do
  set -- $cfg
  ../bin/gpu_run ../.venv/bin/python evaluate.py --run runs/r5 --ckpt snap_step16000 --split val --n 100 --temp $1 --best_of $2 --tag val100_t$1_bo$2 2>&1 | grep '^{"run"'
done
echo SWEEP_DONE
