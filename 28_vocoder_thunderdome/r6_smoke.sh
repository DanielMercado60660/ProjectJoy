#!/bin/sh
# Round 6 end-to-end at tiny size (< 2 min): data prep (20 train clips, 3 clips per eval set) -> GAN train 12 steps with a
# kill/resume in the middle -> recon train -> eval (GL + GAN snapshot) -> CLI vocode -> tests. UTMOS is not run here
# (it lives in ../.venv-sac and is exercised by the real scoring runs).
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
T=$(mktemp -d)
export OMP_NUM_THREADS=4
$PY r6_prepare.py --out "$T/c" --n_train 20 --n_eval 3 --sets val100,test27
$PY r6_train.py --mode gan --mpd small --mrd_ch 8 --batch 2 --frames 33 --cache "$T/c" --out "$T/gan" \
    --total_steps 12 --snap_every 6 --val_every 6 --n_val 2 --stop_at 6
$PY r6_train.py --mode gan --out "$T/gan" --stop_at 0          # resumes from state.json at step 6, runs to 12
$PY -c "import json; s=json.load(open('$T/gan/state.json')); assert s['step']==12, s['step']; print('resume OK, step', s['step'])"
$PY r6_train.py --mode recon --batch 2 --frames 33 --cache "$T/c" --out "$T/rec" --total_steps 6 --snap_every 6 --val_every 6 --n_val 2
$PY r6_eval.py --set test27 --system gl64 --cache "$T/c" --out "$T/res"
$PY r6_eval.py --set test27 --system gan --ckpt "$T/gan/snaps/G_12.safetensors" --cache "$T/c" --out "$T/res"
$PY r6_vocoder.py --ckpt "$T/gan/snaps/G_12.safetensors" --wav ../data/ljspeech/LJSpeech-1.1/wavs/LJ045-0004.wav --out "$T/cli.wav"
$PY -m pytest -q test_r6.py
rm -rf "$T"
echo "r6 smoke OK"
