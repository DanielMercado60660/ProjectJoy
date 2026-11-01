# 33 · Gemma Voice

Can a small *text* model learn to talk? This project takes Google's **Gemma 3 270M** (pretrained on text only) and
teaches it to speak. It reads text one character at a time and writes **Mimi**'s continuous audio latent, one frame every 80
ms, through a small flow-matching head. Kyutai's Mimi decoder turns that latent into 24 kHz audio. Training took about 12 h
on a laptop GPU, with about 700 h of public speech.

**Result (held-out test, project 27's 100 LJSpeech clips; Whisper large-v3-turbo and UTMOS):** CER 1.05%, WER 2.66%,
UTMOS 3.20. The shelf's dedicated TTS (project 27 v2) scores 1.32% / 3.67% / 2.96. The naturalness gain is outside the noise
(+0.24 [+0.17, +0.32]); intelligibility is level (CER within noise, WER just ahead). It still falls well short of its own
ceiling: Mimi's latent round trip scores 4.28 on val and, on an informal listen, sounds close to the real recording (4.40 on
test). Listen: [media/listen_f3.html](media/listen_f3.html). Numbers,
history and caveats: [RESULTS.md](RESULTS.md).

## Pipeline
```
text --one Gemma token per character--> [BOS text] [AUDIO_START] + speaker embedding
     --Gemma 3 270M backbone (fine-tuned)--> hidden state per frame
     --flow head (rectified flow, 28M, 16 Euler steps, text guidance 2.0)--> 512-d Mimi latent (12.5 frames/s)
     --stop head--> end of utterance
latents --Mimi quantizer output projections + decoder--> 24 kHz waveform
```
- **Backbone** (`gemma.py`): our own MLX port of Gemma 3 270M, checked against mlx-lm (`parity.py`: max logit diff 2e-4
  in true fp32, 100% top-1 agreement). Its 262K-row text embedding stays frozen and its text head is never computed.
- **Flow model** (`tts_flow.py`): each frame's latent is projected back in as the next input. A 6-block adaLN MLP
  predicts the flow velocity from the backbone state, and a linear stop head ends the utterance. Training drops the text on
  10% of clips, so sampling can use classifier-free guidance. 130M trainable parameters.
- **Codec** (`mimi_io.py`): Kyutai's Mimi through `moshi_mlx`, in a side venv (`../.venv-moshi`, MLX 0.26.5). The target is
  the latent at the quantizer's input ("space c"); its round trip scores Whisper CER 0.38%, UTMOS 4.28 on val.
- **Judges:** Whisper large-v3-turbo through mlx-whisper (`judge.py`, `../.venv-judge`) for CER/WER, and UTMOS22 through
  project 27's `mos.py` (`../.venv-sac`, CPU) for naturalness.
- **Earlier rounds** predicted 4 delayed Mimi codebooks (`tts.py`, `train.py`, `evaluate.py`). That route capped
  naturalness at 3.11 and reached 2.30. f1 starts from its round-2 backbone (r5).

## Run it (from this folder)
```sh
# one-time side venvs (the codec and judges pin their own MLX / torch, so they stay out of the shared venv)
uv venv --python 3.12 ../.venv-moshi && uv pip install --python ../.venv-moshi/bin/python moshi_mlx mlx-lm soundfile scipy
# ../.venv-judge: mlx-whisper, num2words;  ../.venv-sac: 27's UTMOS environment (see ../27_pocket_voice_foundry)

# weights: checkpoints/f3/last (the model) -- not in git and not on the Hugging Face mirror yet; train it with the commands below
# data: LJSpeech in ../data/ljspeech; LibriTTS-R and Hi-Fi TTS from openslr (141, 109) in ../data/libritts_r, ../data/hifitts

# listen to the shipped model on val (generate, decode, Whisper + UTMOS)
../bin/gpu_run ../.venv/bin/python evaluate_flow.py --run checkpoints/f3 --ckpt last --split val --temp 0.4 --cfg 2.0 --tag val
./smoke_flow.sh                                                                  # flow pipeline end to end, tiny, ~15 s

# rebuild from scratch: Mimi codes (round 2), then latents
../bin/gpu_run ../.venv-moshi/bin/python mimi_io.py encode --out data/codes32.npz
../bin/gpu_run ../.venv-moshi/bin/python mimi_io.py encode-libritts --root ../data/libritts_r/extracted/LibriTTS_R/train-clean-100 --out data/libritts_codes32.npz
../bin/gpu_run ../.venv-moshi/bin/python mimi_io.py encode-hifitts --root ../data/hifitts/extracted --out data/hifitts_codes32.npz
../bin/gpu_run ../.venv-moshi/bin/python mimi_io.py encode-latents --codes data/codes32.npz --set lj --out data/lj_lat.npy
../bin/gpu_run ../.venv-moshi/bin/python mimi_io.py encode-latents --codes data/libritts_codes32.npz --set libritts \
  --root ../data/libritts_r/extracted/LibriTTS_R/train-clean-100 --out data/libritts_lat.npy
../bin/gpu_run ../.venv-moshi/bin/python mimi_io.py encode-latents --codes data/hifitts_codes32.npz --set hifitts \
  --root ../data/hifitts/extracted --out data/hifitts_lat.npy
X=../data/libritts_r/extracted/LibriTTS_R
../bin/gpu_run ../.venv-moshi/bin/python mimi_io.py encode-latents-dirs --roots $X/train-clean-360 $X/train-other-500 \
  --out data/librittsx_lat.npy --meta data/librittsx_meta.npz

# r5: Gemma -> 4 Mimi codebooks on LJ x4 + LibriTTS-R clean-100 + Hi-Fi TTS (3 h); its step-16000 snapshot seeds f1
../bin/gpu_run ../.venv/bin/python train.py --out runs/r5 --libritts data/libritts_codes32.npz,data/hifitts_codes32.npz \
  --lj_repeat 4 --nq 4 --text chars --compute bfloat16 --steps 20000 --warmup 500 --max_minutes 180 \
  --lr 3e-5 --lr_new 3e-4 --cb_weight 4,2,1,1 --corrupt 0.15 --select mean --val_every 500 --save_every 500
# flow head: f1 (1 h) -> f2 (2.9 h) -> f3 (5.1 h, adds clean-360 + other-500)
F="--K 2 --lr 5e-5 --lr_new 3e-4 --text_drop 0.1 --save_every 500"
../bin/gpu_run ../.venv/bin/python train_flow.py --out runs/f1 --init_from runs/r5/snap_step16000 $F \
  --mix lj:4,libritts,hifitts --warmup 300 --steps 6000 --max_minutes 60 --val_every 500 --snap_every 1500
../bin/gpu_run ../.venv/bin/python train_flow.py --out runs/f2 --init_from runs/f1/last $F \
  --mix lj:2,libritts,hifitts --warmup 200 --steps 11000 --max_minutes 185 --val_every 1000 --snap_every 2500
../bin/gpu_run ../.venv/bin/python train_flow.py --out runs/f3 --init_from runs/f2/last $F \
  --mix lj:2,libritts,hifitts,librittsx --warmup 300 --steps 20500 --max_minutes 360 --val_every 2000 --snap_every 5000
../bin/gpu_run ../.venv/bin/python evaluate_flow.py --run runs/f3 --ckpt last --split test --temp 0.4 --cfg 2.0 --tag test_t0.4_cfg2.0_last
../.venv/bin/python compare_27.py                                 # paired bootstrap vs project 27 -> results/test_vs_27.json
python3 make_listen_f1.py --run f3 --f1_tag val_t0.4_cfg2.0_last --label "..." --prev f2 --prev_label "..."   # listening page
```
In zsh, write `${=F}` instead of `$F` (zsh doesn't split unquoted variables). Round 1's commands (15-minute LJ-only
codebook runs, random-init ablation) are on the first line of `logs/r3.log` and `logs/scratch.log`; RESULTS.md appendix
A has the results.

## Files
| file | what |
|---|---|
| `gemma.py` | Gemma 3 text decoder in MLX (config-driven), weight loading, tokenizer, progress-RoPE option |
| `parity.py` | port vs mlx-lm, cached vs full forward, greedy continuation |
| `tts_flow.py` | flow model: latent in/out, rectified-flow head, stop head, loss, cached sampler with guidance |
| `train_flow.py` | flow training: memmapped latent sets with repeats, speaker table, init from a codebook or flow run, snapshots, resume |
| `evaluate_flow.py` | generate val/test clips, decode, Whisper CER/WER, UTMOS, length ratio, RTF, first-frame latency |
| `mimi_io.py` | Mimi encode/decode, codebook and latent round trips, latent stores (runs in `../.venv-moshi`) |
| `judge.py` | Whisper large-v3-turbo judge with a shared text normalizer (`../.venv-judge`) |
| `compare_27.py`, `paired_bootstrap.py` | paired bootstraps: f3 vs project 27; round 1's Gemma vs random init |
| `make_listen_f1.py`, `make_demo.py` | listening pages: flow runs (`media/listen_f3.html`), round 1 (`media/demo.html`) |
| `tts.py`, `train.py`, `evaluate.py` | rounds 1–2: Gemma → delayed Mimi codebooks |
| `tts_mel.py`, `train_mel.py`, `evaluate_mel.py`, `vocoder_model.py` | parked: Gemma → 27's mel + 28's vocoder |
| `qwen_io.py`, `sac_io.py`, `xy_io.py`, `make_lists_codecs.py` | codec round trips for the comparison in RESULTS §2 |
| `score.py`, `ctc.py`, `ctc_blocks.py`, `mel.py` | round 1's judge: project 27's CTC recognizer (copied with provenance) |
| `diag_text.py`, `diag_tf.py`, `phase0_prep.py` | diagnostics (text use, teacher-forced resynthesis) and the judge switch |
| `smoke_flow.sh`, `smoke.sh` | end-to-end smoke tests: flow path, codebook path |
| `research/` | background reports (codecs, small LLM TTS, alignment, data), the T5Gemma-TTS notes and the dataset survey |
| `results/` | every eval summary and per-clip judge/UTMOS file behind RESULTS.md |
| `checkpoints/f3/last` | the shipped model: trainable weights + `meta.json` (497 MB; gitignored) |

Weights come from the Hugging Face cache: `google/gemma-3-270m` (Gemma terms of use),
`kyutai/moshiko-mlx-q8/tokenizer-e351c8d8-checkpoint125.safetensors` (Mimi, CC-BY-4.0) and
`mlx-community/whisper-large-v3-turbo` (MIT). Data: LJSpeech (public domain), LibriTTS-R (CC BY 4.0),
Hi-Fi TTS (CC BY 4.0).
