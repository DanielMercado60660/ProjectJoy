# 28 · Vocoder Thunderdome

Hold the acoustic representation fixed, then race waveform reconstructors on it. First satisfying
version: a classical **Griffin–Lim** baseline vs one small **neural vocoder trained in MLX**, scored on
held-out LJSpeech clips, with level-matched blind listening.

**The fixed mel** (`mel.py`, used by every script, the metrics and the spectrograms):
16 kHz (LJSpeech 22.05 kHz resampled with `resample_poly(320, 441)`), n_fft 1024, Hann window 1024,
hop 256 (16 ms), 80 Slaney mel bands 0–8 kHz, feature = `ln(max(mel·|STFT|, 1e-5))`, centred frames.

**Data split** (`prepare_data.py`, seed 0): 400 training clips (42.5 min) from LJSpeech chapters
LJ001–LJ040, 24 held-out clips (2.0 min, 2–7 s each) from chapters LJ045–LJ050. LJSpeech has a single
speaker, so "held-out" means held-out text/chapters, not a held-out speaker. The 8 held-out clips with the
highest share of word-initial plosives (p t k b d g) are tagged as the transient subset.

**Contenders**
- `griffin_lim.py`: mel → linear magnitude by non-negative least squares through the mel filterbank (pinv
  initialisation + 60 multiplicative updates), then fast Griffin–Lim (momentum 0.99), numpy on CPU.
  The main setting is 64 iterations; the eval also sweeps 8/16/32/64/128.
- `model.py`: Vocos-style vocoder. Conv → 8 ConvNeXt blocks (dim 384) at frame rate → per-frame
  log-magnitude + phase for 513 bins → inverse STFT (same n_fft/hop) → waveform. 7.74 M params.
  Loss: multi-resolution STFT (spectral convergence + log-magnitude L1 at 512/1024/2048) + 2 × mel L1.
  No adversarial loss. Running at frame rate rather than sample rate is what makes it trainable in under
  20 minutes.
- Ablation (run after the write-up): the same model trained on 3000 clips (`checkpoints/vocoder_3k`). It ran all
  25,000 steps on a quieter machine, while the main run was cut by its wall-clock cap at 17,195 steps, so it differs
  in data *and* LR schedule; it is not a pure data-size ablation. The demo includes it as a fourth blind option.

**Result in one line** (details and metric caveats in `RESULTS.md`): Griffin–Lim wins
held-out pitch (GPE 3.6% vs 7.7%) and mel L1 (0.100 vs 0.195, which is GL's own objective); the neural model wins
transients (+1.7 vs +2.5 dB pre-onset smear) and speed (~8× on CPU, ~100× on GPU, under load); it overfits its
400 clips.

## Run it (from this folder)
```sh
../.venv/bin/python prepare_data.py                      # -> cache/{train,test}.npz, cache/split.json (~3 s)
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python train.py \
    --steps 25000 --batch 32 --dim 384 --blocks 8 --max_minutes 17   # -> checkpoints/vocoder.{safetensors,json}, train_log.jsonl
OMP_NUM_THREADS=4 ../.venv/bin/python eval.py            # -> results/metrics.json, results/wavs/ (all held-out outputs)
OMP_NUM_THREADS=4 ../.venv/bin/python bench_rtf.py --out results/bench_rtf_idle.json  # RTF, latency, memory, uptime; won't overwrite
../.venv/bin/python make_demo.py                         # -> media/*.wav, media/spec_*.png, media/demo.html
./smoke.sh                                               # prep/train/eval/demo at tiny size, ~10 s
../.venv/bin/python plots.py                             # -> media/train_curve.png, media/quality_vs_budget.png

# overfit check: same eval on 24 *training* clips
OMP_NUM_THREADS=4 ../.venv/bin/python eval.py --data cache/train.npz --gl_iters 64 --limit 24 --out results_trainsplit
# 3000-clip ablation (same model/seed/--steps; ran all 25k steps, main run was capped at 17.2k; same held-out set)
../.venv/bin/python prepare_data.py --n_train 3000 --out cache_big
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python train.py --data cache_big/train.npz --steps 25000 --batch 32 \
    --dim 384 --blocks 8 --max_minutes 17 --seed 0 --out checkpoints/vocoder_3k --log train_log_3k.jsonl
OMP_NUM_THREADS=4 ../.venv/bin/python eval.py --ckpt checkpoints/vocoder_3k.safetensors --gl_iters 64 --out results_3k
```
### Follow-up (round 3): data size alone + validation split (results in `RESULTS.md`, section "Follow-up (round 3)")
```sh
../.venv/bin/python prepare_data.py --n_train 3000 --out cache_big    # 3000 train clips (first 400 = main set), ~9 s, 1.6 GB
../.venv/bin/python prepare_val.py --n_val 32 --out cache_val         # 32 val clips, chapters LJ041-LJ044 (disjoint from train and test)
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python train_val.py --val_every 500   # 3000 clips, main-run schedule (--steps 25000), stop at 17195;
                                                                      # -> checkpoints/followup_val3k/{best,last}.*, results/followup_val3k/train_log.jsonl (~9.4 min)
OMP_NUM_THREADS=4 ../.venv/bin/python eval.py --ckpt checkpoints/followup_val3k/best.safetensors --gl_iters 64 --out results/followup_val3k/best
OMP_NUM_THREADS=4 ../.venv/bin/python eval.py --ckpt checkpoints/followup_val3k/last.safetensors --gl_iters 64 --out results/followup_val3k/last
OMP_NUM_THREADS=4 ../.venv/bin/python eval.py --data cache_val/val.npz --split cache/split.json --ckpt checkpoints/vocoder.safetensors \
    --gl_iters 64 --out results/followup_val3k/valset_main400                       # context: GL 64 + main model on the val split
# (same with --ckpt checkpoints/vocoder_3k.safetensors --out .../valset_ablation3k and --ckpt checkpoints/followup_val3k/last.safetensors --out .../valset_last)
../.venv/bin/python mrstft_per_res.py --out results/followup_val3k/mrstft_per_res.json \
    gl64=results/wavs/{id}_gl64.wav neural400=results/wavs/{id}_neural.wav neural3k_ablation=results_3k/wavs/{id}_neural.wav \
    val3k_best=results/followup_val3k/best/wavs/{id}_neural.wav val3k_last=results/followup_val3k/last/wavs/{id}_neural.wav
../.venv/bin/python make_demo_followup.py                             # -> media/followup_val3k/demo.html (+ wavs, spectrograms)
../.venv/bin/python bootstrap_followup.py                             # paired clip bootstrap CIs -> results/followup_val3k/bootstrap_gpe.json
./smoke_followup.sh                                                   # tiny end-to-end check of the follow-up scripts, ~25 s
```
`cache_big/` was deleted after the run (rebuild with the first command); `cache_val/` and `results/followup_val3k/*/wavs/`
are gitignored and regenerable.
Follow-up result in one line: with only the data changed, held-out GPE 7.7% → 4.6%, still worse than GL 64's 3.6%;
the transient lead over GL varies 0.38–0.73 dB between nearby checkpoints, so treat the transient "win" above as fragile.

`cache/`, `cache_big/` and `results/wavs/` are derived and regenerable (folder `.gitignore`); rerun
`prepare_data.py` and `eval.py` to rebuild them.
`bench_rtf.py` is standalone and safe to rerun on an idle machine; it records `uptime` before/after and refuses to
overwrite an existing `--out` file (`--overwrite` to force).
Open `media/demo.html` in any browser (fully offline: audio and images are embedded).

## Files
| path | what |
|---|---|
| `mel.py` | the one mel definition, numpy STFT/iSTFT, LJSpeech loader + resampler |
| `prepare_data.py` | clip selection, split, plosive tagging, cache |
| `griffin_lim.py` | NNLS mel inversion + fast Griffin–Lim |
| `model.py` | MLX vocoder, differentiable MLX STFT, training loss, checkpoint loader |
| `train.py` | training loop with wall-clock cap; logs `train_log.jsonl`, log text `train.log` |
| `vocoders.py` | uniform `mel -> wav` functions for every method |
| `metrics.py` | MR-STFT distance, mel L1, YIN pitch/voicing/periodicity, transient metrics |
| `eval.py` | held-out evaluation of all methods |
| `bench_rtf.py` | speed / latency / memory bench |
| `make_demo.py`, `demo_template.html` | loudness matching, spectrograms, blind A/B/C page |
| `checkpoints/vocoder.safetensors` (+ `.json`) | main run: trained weights + config + step (used by the demo) |
| `checkpoints/vocoder_3k.safetensors` (+ `.json`) | 3000-clip ablation run (full 25k-step schedule) |
| `results_trainsplit/`, `results_3k/` | overfit check and ablation metrics |
| `plots.py` | training curve and GL budget sweep plots |
| `media/` | demo WAVs (gt / gl64 / neural / neural3k per clip, −23 LUFS), spectrogram PNGs, `demo.html` |
| `results/` | `metrics.json`, `bench_rtf.json` (load-4.0 run), `wavs/` (raw float outputs for all 24 held-out clips) |
| `prepare_val.py`, `train_val.py` | follow-up: validation split (LJ041–044); training with val-loss checkpoint selection |
| `mrstft_per_res.py` | follow-up: MR-STFT per resolution from saved eval WAVs |
| `bootstrap_followup.py` | follow-up (added in review): paired bootstrap over the 24 test clips for the quoted CIs |
| `make_demo_followup.py`, `demo_followup_template.html` | follow-up blind page `media/followup_val3k/demo.html` |
| `checkpoints/followup_val3k/{best,last}.*` | follow-up: 3000 clips at the main run's 17,195 steps; best = val-selected (step 16,500) |
| `results/followup_val3k/` | follow-up metrics (test: `best/`, `last/`; val: `valset_*/`), per-res MR-STFT, train log |
