#!/bin/bash
# f1: first flow-matching probe. Waits for the Hi-Fi latents (file-existence trigger), then 60 min from r5's backbone on
# r5's mix (LJ x4 + LibriTTS-R clean-100 + Hi-Fi TTS), 10% text dropout (for CFG), snapshots every 1500 steps, then
# judges each snapshot on val-100 (t0.8) and last at t0.8, t1.0 and t0.8 + CFG 1.5.
cd /Users/danielmercado/ProjectJoy/33_gemma_voice
while [ ! -f data/hifitts_lat.npy.json ]; do sleep 30; done
../bin/gpu_run ../.venv/bin/python train_flow.py --out runs/f1 --init_from runs/r5/snap_step16000 \
  --mix lj:4,libritts,hifitts --K 2 --lr 5e-5 --lr_new 3e-4 --warmup 300 --steps 6000 --max_minutes 60 \
  --text_drop 0.1 --val_every 500 --save_every 500 --snap_every 1500 2>&1 | grep -v Warn
for s in runs/f1/snap_step*; do
  n=$(basename $s)
  ../bin/gpu_run ../.venv/bin/python evaluate_flow.py --run runs/f1 --ckpt $n --tag val_t0.8_$n 2>&1 | grep -v Warn | tail -1
done
for args in "--temp 0.8 --tag val_t0.8_last" "--temp 1.0 --tag val_t1.0_last" "--temp 0.8 --cfg 1.5 --tag val_t0.8_cfg1.5_last"; do
  ../bin/gpu_run ../.venv/bin/python evaluate_flow.py --run runs/f1 --ckpt last $args 2>&1 | grep -v Warn | tail -1
done
echo F1_DONE
