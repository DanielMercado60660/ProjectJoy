#!/bin/bash
# Overnight: after f2 (+ its evals) and the other-500 download, encode LibriTTS-R clean-360 + other-500 latents,
# then f3 = 6 h from f2 on everything (LJ x2, clean-100, Hi-Fi, 360+500), then evals.
cd /Users/danielmercado/ProjectJoy/33_gemma_voice
X=../data/libritts_r/extracted/LibriTTS_R
until grep -q F2_DONE logs/f2.log 2>/dev/null; do sleep 60; done
until grep -qE "OTHER500_DONE|MD5_MISMATCH" logs/other500_finish.log 2>/dev/null || ! pgrep -f other500_finish.sh >/dev/null; do sleep 60; done
ROOTS="$X/train-clean-360"
if grep -q OTHER500_DONE logs/other500_finish.log; then ROOTS="$ROOTS $X/train-other-500"; else echo "other-500 missing: encoding clean-360 only"; fi
echo "encode $ROOTS $(date)"
../bin/gpu_run ../.venv-moshi/bin/python mimi_io.py encode-latents-dirs --roots $ROOTS \
  --out data/librittsx_lat.npy --meta data/librittsx_meta.npz 2>&1 | grep --line-buffered -vi warn
[ -f data/librittsx_lat.npy.json ] || { echo "ENCODE_FAILED"; exit 1; }
INIT=runs/f2/last; [ -f $INIT/meta.json ] || INIT=runs/f1/last
echo "f3 start $(date), init $INIT"
../bin/gpu_run ../.venv/bin/python train_flow.py --out runs/f3 --init_from $INIT \
  --mix lj:2,libritts,hifitts,librittsx --K 2 --lr 5e-5 --lr_new 3e-4 --warmup 300 --steps 20500 --max_minutes 360 \
  --text_drop 0.1 --val_every 2000 --save_every 500 --snap_every 5000 2>&1 | grep --line-buffered -v Warn
for s in runs/f3/snap_step*; do
  n=$(basename $s)
  ../bin/gpu_run ../.venv/bin/python evaluate_flow.py --run runs/f3 --ckpt $n --temp 0.4 --cfg 2.0 --tag val_t0.4_cfg2.0_$n 2>&1 | grep -v Warn | tail -1
done
for args in "--temp 0.4 --cfg 2.0 --tag val_t0.4_cfg2.0_last" "--temp 0.3 --cfg 2.0 --tag val_t0.3_cfg2.0_last" \
            "--split test --temp 0.4 --cfg 2.0 --tag test_t0.4_cfg2.0_last"; do
  ../bin/gpu_run ../.venv/bin/python evaluate_flow.py --run runs/f3 --ckpt last $args 2>&1 | grep -v Warn | tail -1
done
echo F3_DONE $(date)
