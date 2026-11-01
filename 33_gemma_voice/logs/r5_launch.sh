cd /Users/danielmercado/ProjectJoy/33_gemma_voice
until [ -f data/hifitts_codes32.json ]; do sleep 30; done
echo "hifi merged: $(cat data/hifitts_codes32.json | head -c 300)"
../bin/gpu_run ../.venv/bin/python train.py --out runs/r5 --libritts data/libritts_codes32.npz,data/hifitts_codes32.npz \
  --lj_repeat 4 --nq 4 --text chars --compute bfloat16 --steps 20000 --warmup 500 --max_minutes 180 \
  --lr 3e-5 --lr_new 3e-4 --cb_weight 4,2,1,1 --corrupt 0.15 --select mean --val_every 500 --save_every 500
echo R5_EXIT $?
