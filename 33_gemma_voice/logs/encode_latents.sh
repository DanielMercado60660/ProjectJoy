#!/bin/bash
# Mimi projected latents (space 'c', fp16 .npy) aligned with the existing code files: LJ, LibriTTS-R clean-100, Hi-Fi TTS.
cd /Users/danielmercado/ProjectJoy/33_gemma_voice
../bin/gpu_run ../.venv-moshi/bin/python mimi_io.py encode-latents --codes data/codes32.npz --set lj --out data/lj_lat.npy 2>&1 | grep -v -i warn
../bin/gpu_run ../.venv-moshi/bin/python mimi_io.py encode-latents --codes data/libritts_codes32.npz --set libritts --root ../data/libritts_r/extracted/LibriTTS_R/train-clean-100 --out data/libritts_lat.npy 2>&1 | grep -v -i warn
../bin/gpu_run ../.venv-moshi/bin/python mimi_io.py encode-latents --codes data/hifitts_codes32.npz --set hifitts --root ../data/hifitts/extracted --out data/hifitts_lat.npy 2>&1 | grep -v -i warn
echo LATENTS_DONE
