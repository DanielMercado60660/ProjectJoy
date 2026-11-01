#!/bin/bash
cd /Users/danielmercado/ProjectJoy/33_gemma_voice
run() { tag=$1; shift; ../bin/gpu_run ../.venv/bin/python evaluate_flow.py --run runs/f1 --ckpt last --split val --tag $tag "$@" 2>&1 | grep -v Warn | tail -1; }
run val_t0.6_cfg1.5_last --temp 0.6 --cfg 1.5
run val_t0.8_cfg2.0_last --temp 0.8 --cfg 2.0
run val_t0.6_cfg2.5_last --temp 0.6 --cfg 2.5
run val_t0.8_cfg2.0_n32_last --temp 0.8 --cfg 2.0 --n_steps 32
echo SWEEP_DONE
