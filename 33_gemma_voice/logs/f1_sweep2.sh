#!/bin/bash
cd /Users/danielmercado/ProjectJoy/33_gemma_voice
run() { tag=$1; shift; ../bin/gpu_run ../.venv/bin/python evaluate_flow.py --run runs/f1 --ckpt last --split val --tag $tag "$@" 2>&1 | grep -v Warn | tail -1; }
run val_t0.4_cfg2.0_last --temp 0.4 --cfg 2.0
run val_t0.5_cfg2.5_last --temp 0.5 --cfg 2.5
echo SWEEP_DONE
