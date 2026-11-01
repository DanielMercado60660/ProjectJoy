#!/bin/bash
# f2: 3 h continuing from f1 on the same sets, LJ x2 (was x4), full cosine. Then val evals per snapshot + test on last.
cd /Users/danielmercado/ProjectJoy/33_gemma_voice
../bin/gpu_run ../.venv/bin/python train_flow.py --out runs/f2 --init_from runs/f1/last \
  --mix lj:2,libritts,hifitts --K 2 --lr 5e-5 --lr_new 3e-4 --warmup 200 --steps 11000 --max_minutes 185 \
  --text_drop 0.1 --val_every 1000 --save_every 500 --snap_every 2500 2>&1 | grep --line-buffered -v Warn
for s in runs/f2/snap_step*; do
  n=$(basename $s)
  ../bin/gpu_run ../.venv/bin/python evaluate_flow.py --run runs/f2 --ckpt $n --temp 0.4 --cfg 2.0 --tag val_t0.4_cfg2.0_$n 2>&1 | grep -v Warn | tail -1
done
for args in "--temp 0.4 --cfg 2.0 --tag val_t0.4_cfg2.0_last" "--temp 0.3 --cfg 2.0 --tag val_t0.3_cfg2.0_last" \
            "--temp 0.6 --cfg 2.5 --tag val_t0.6_cfg2.5_last" "--split test --temp 0.4 --cfg 2.0 --tag test_t0.4_cfg2.0_last"; do
  ../bin/gpu_run ../.venv/bin/python evaluate_flow.py --run runs/f2 --ckpt last $args 2>&1 | grep -v Warn | tail -1
done
echo F2_DONE
