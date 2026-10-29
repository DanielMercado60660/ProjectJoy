#!/bin/zsh
# End-to-end pipeline at tiny size (~1-2 min): data -> acoustic train/resume -> export -> baseline -> CTC -> eval -> demo.
# Everything goes to runs/smoke_e2e (gitignored); the real checkpoints are untouched.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
S=runs/smoke_e2e; rm -rf $S; mkdir -p $S
$PY text.py
$PY prepare_data.py --limit 24 --out $S/cache
$PY train.py --cache $S/cache --limit 24 --steps 20 --val_every 10 --n_val 8 --warm 10 --prior_steps 10 --out $S/acoustic --log $S/log.jsonl
$PY train.py --cache $S/cache --limit 24 --steps 30 --val_every 10 --n_val 8 --warm 10 --prior_steps 10 --out $S/acoustic --log $S/log.jsonl --resume
$PY finalize.py --src $S/acoustic --out $S/tts --cache $S/cache --n_val 4
$PY ctc.py --cache $S/cache --limit 24 --steps 10 --out $S/ctc --log $S/ctc_log.jsonl
$PY baseline.py --cache $S/cache --n_train 24 --n_val 4 --iters 1 --Ks 1 --out $S/baseline.npz --tune_out $S/bt.json --ctc $S/ctc.safetensors | tail -2
$PY eval.py --cache $S/cache --n 3 --no_gl --tts $S/tts.safetensors --baseline $S/baseline.npz --ctc $S/ctc.safetensors --out $S/results | tail -8
$PY make_demo.py --cache $S/cache --n_held 1 --tts $S/tts.safetensors --baseline $S/baseline.npz --media $S/media
$PY synth.py "Doctor Smith paid \$15 in 1963." --tts_ckpt $S/tts.safetensors --vocoder gl --out $S/synth.wav
echo "SMOKE OK"
