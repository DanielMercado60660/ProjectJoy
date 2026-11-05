# 27 · Pocket Voice Foundry

A tiny single-speaker text-to-speech system trained from scratch in MLX on LJSpeech 1.1 (public domain).
First satisfying version: **train the text-to-acoustics stage and borrow an existing compatible vocoder**, so the
question "does the acoustic model work?" can be answered on its own. Every result is reported three ways:
the predicted mel itself, the mel through project 28's trained neural vocoder, and the mel through Griffin–Lim, with the
**real recording's mel through the same vocoder** as the upper bound.

**Result (held-out test, 100 clips; RESULTS.md):** the learned model (23.9 M params, 85 min of training) is at least as legible to a
CTC recognizer proxy as the real recording. Its CER is 5.8 %, below the real recording's mel at 7.0 %, which shows the proxy's bias towards
"typical" synthetic speech. (Reviewer: that margin is significant only for the checkpoint chosen by val CER on this same
recognizer; the training loop's own pick scores 6.6 %. Fed the real recording's timing, the TTS scores 8.0 %, worse than
real. RESULTS §7.) It is ~11 CER points better than a template-concatenation baseline tuned over 412 configurations
(16.8 %). Spectrally it is still clearly not the recording: MCD-like 4.4 dB, against 0.6 dB for the vocoder alone. Numbers
and abbreviations are handled by the normalizer; rare and irregular words come out as spelled. Under load, TTS + neural
vocoder renders a 7 s sentence in ~6 ms. Listening preference: not measured; the blind page is `media/demo.html`.

## Pipeline
```
text --normalize--> characters --encoder--> h, mu --duration predictor--> durations --length regulator + decoder--> 80-band log-mel
                                               (training only: MAS on log N(mel_t; mu_n, I) gives the alignment)
log-mel --> 28's Vocos-style neural vocoder (copied checkpoint)  or  Griffin-Lim 64 --> 16 kHz waveform
```
- **Text** (`text.py`): no phonemizer or G2P is installed, so the input is normalized characters: lowercase, accents stripped,
  common abbreviations (mr., mrs., dr., st., lt., ...) and simple numbers (cardinals, years, ordinals, $ amounts, %, clock
  times, 1,200-style numbers) expanded; punctuation `, . ? ! ; : - ' "` kept as pause cues. 37 symbols + pad. Training text is
  LJSpeech's own normalized column (numbers already spelled out there; it still contains "Mr." etc.).
- **Alignment** (`mas.py`, `model.py`): no forced aligner is installed, so the model aligns itself, Grad-TTS style. The encoder
  outputs a per-character mean `mu` in normalized-mel space; monotonic alignment search (numba, CPU) finds the best monotonic
  path of real mel frames to characters under a unit-variance Gaussian, and that path gives the durations and the frame→character
  gather for the decoder. A straight-line diagonal prior is mixed in for the first 2,000 steps and then fades out.
- **Acoustic model** (`model.py`): FastSpeech 2-style, 23.9 M parameters: char embedding + conv prenet + 4 FFT blocks (d 256,
  2 heads, conv FFN 1024, kernel 9) → `mu` head and duration predictor (2 convs on stop-gradient encoder output, log-duration MSE)
  → length regulator → 4 FFT blocks → linear to 80 mel bands. Losses: L1 mel + prior MSE + log-duration MSE. No pitch or energy
  predictors (not built; see RESULTS "next steps").
- **Vocoders**: copied from `../28_vocoder_thunderdome/` (provenance headers in each file): `mel.py` (the mel definition:
  16 kHz, n_fft 1024, hop 256, 80 Slaney bands, natural-log magnitude mel), `griffin_lim.py` (NNLS mel inversion + fast
  Griffin–Lim, 64 iterations), `vocoder_model.py` (Vocos-style ConvNeXt vocoder) and its checkpoint
  `checkpoints/vocoder28/vocoder.{safetensors,json}` = 28's `checkpoints/followup_val3k/best` (3,000 clips from chapters LJ001–040,
  val-selected at step 16,500; 7.7 M params, 31 MB). Nothing in 28 was imported or modified.
- **Baseline** (`baseline.py`): template concatenation with no neural network and no help from the learned model. Each character
  is K left-to-right states with a mean mel; alignment by classical flat-start Viterbi training (uniform segmentation, then 5
  rounds of re-estimate means / re-align with the same monotonic search); synthesis repeats each state's mean for its average
  duration, looked up by the longest context seen often enough (7, 5 or 3 characters centred on it, else the character alone),
  then optionally Gaussian-smooths in time. K, context widths and count thresholds, smoothing width and duration scale were tuned
  on the validation split over 3 rounds (each time the best setting sat on the grid edge, the grid was widened), selected by the
  recognizer's val CER (the main Measure; ties by val DTW log-mel L1). Final: K=2, 5-char contexts seen ≥2 times (55,933 of them)
  backing off to 3-char (≥2) and single characters, dur_scale 0.985, no smoothing. In effect a small unit-concatenation system
  built from state means.
- **Intelligibility proxy** (`ctc.py`): no ASR is installed, so a small CTC character recognizer (conv + 4 transformer blocks,
  ~8 M params) is trained on *real* training-split mels only, and scored by character error rate on held-out real mels
  (the floor) vs synthesized mels.

## Data split (same chapter convention as 28)
`prepare_data.py`: train = chapters LJ001–LJ040 (10,555 clips of 1–10 s, 19.1 h), validation = LJ041–LJ044 (865 clips;
baseline tuning, duration pace, checkpoint choice), test = LJ045–LJ050 (1,536 clips; never trained or tuned on). LJSpeech has one
speaker, so "held out" means held-out sentences and chapters, not a held-out voice. Evaluation uses a fixed random 100-clip
subset of test (seed 0).

## Run it (from this folder)
```sh
../.venv/bin/python prepare_data.py                         # -> cache/{train,val,test}.npz, cache/info.json (~1 min, 4 procs)
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python train.py --out runs/train_ckpt/acoustic --max_minutes 40   # run 1
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python ctc.py --max_minutes 14              # recognizer (val-selected save) -> checkpoints/ctc/
../.venv/bin/python finalize.py --src runs/train_ckpt/acoustic_best --out runs/run1_tts_export/tts      # fp16 export + duration pace fitted on val
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python train.py --out runs/train_ckpt/acoustic --max_minutes 45 --resume --anneal_to 0.1   # run 2
../.venv/bin/python finalize.py --src runs/train_ckpt/acoustic_best --out runs/run2_best_export/tts     # run 2's best by val loss (step 7,000)
../.venv/bin/python finalize.py --src runs/train_ckpt/acoustic --out runs/run2_final_export/tts         # end of run 2 (step 12,911)
../.venv/bin/python val_compare.py runs/run1_tts_export/tts.safetensors runs/run2_best_export/tts.safetensors runs/run2_final_export/tts.safetensors
mkdir -p checkpoints/acoustic && cp runs/run2_final_export/tts.* checkpoints/acoustic/   # val-CER winner = shipped model
# baseline: 3 tuning rounds on val (each overwrites checkpoints/baseline/baseline.npz; rounds 2 and 3 select the same config)
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python baseline.py --ctc checkpoints/ctc/ctc.safetensors --Ks 1,2,3 --contexts mono,tri_min50,tri_min10   # round 1 (as run it swept 2 dur scales; the code now sweeps 4)
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python baseline.py --ctc checkpoints/ctc/ctc.safetensors --Ks 2,3 --sigmas 0,0.5,1 --contexts mono,tri_min50,tri_min10,tri_min5,tri_min2,quin_min10_tri5,quin_min5_tri5,quin_min2_tri2
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python baseline.py --ctc checkpoints/ctc/ctc.safetensors --Ks 2 --sigmas 0,0.5 --contexts quin_min2_tri2,quin_min1_tri1,sept_min5_quin2_tri2,sept_min2_quin2_tri2,sept_min1_quin1_tri1
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python eval.py --n 100   # -> results/eval_summary.json, eval_clips.jsonl, hard_sentences_eval.json (results/run1/ = same with run 1 shipped)
../.venv/bin/python bench.py                                 # -> results/bench.json (latency, RTF, uptime)
../.venv/bin/python make_demo.py                             # -> media/*.wav, media/mels.png, media/demo.html
../.venv/bin/python plots.py                                 # -> media/train_curve.png, media/alignment.png
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python eval.py --n 100 --tts runs/run1_tts_export/tts.safetensors --out results/run1   # run-1 reference
# reviewer additions (RESULTS §7):
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python timing_split.py   # -> results/timing_split.json (~15 s)
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python eval.py --n 100 --no_gl --tts runs/run2_best_export/tts.safetensors --out results/review_step7000 && rm -rf results/review_step7000/wavs   # loss-rule checkpoint
../.venv/bin/python report.py                                # -> results/tables.md (tables used in RESULTS.md)
../.venv/bin/python synth.py "Dr. Smith paid \$15 on March 3rd, 1963." --system tts --vocoder neural --out media/synth.wav
./smoke.sh                                                   # whole pipeline at tiny size into runs/smoke_e2e (~1 min)
```

## Files
| path | what |
|---|---|
| `text.py` | normalizer, symbol set, number/abbreviation expansion |
| `prepare_data.py` | chapter split, 16 kHz resample, log-mel cache |
| `mas.py` | numba monotonic alignment search, DTW (L1 and Euclidean) |
| `model.py` | MLX acoustic model, losses, MAS-driven alignment, inference |
| `train.py` | training loop, checkpoint/resume (weights + optimizer + step + data position + seeds + config), val selection |
| `finalize.py` | export fp16 inference checkpoint; choose duration pace/rounding on val |
| `val_compare.py` | compare exported checkpoints on val (recognizer CER, DTW L1, length) → `results/val_compare.json` |
| `baseline.py` | template-concatenation baseline, Viterbi training, validation tuning grid |
| `ctc.py` | CTC recognizer (intelligibility proxy), CER/WER |
| `eval.py` | held-out evaluation of every (acoustic, vocoder) pair + the hard-sentence set |
| `timing_split.py` | reviewer diagnostic: baseline/TTS re-synthesized with borrowed (TTS-predicted or oracle) durations → `results/timing_split.json` |
| `bench.py` | first-audio latency and real-time factor (under load, with `uptime`) |
| `synth.py` | text → wav for any system/vocoder pair, reloading the checkpoints |
| `make_demo.py`, `demo_template.html` | level-matched WAVs and the offline blind listening page |
| `plots.py` | training curves, alignment figure |
| `report.py` | prints the RESULTS tables from the result JSONs → `results/tables.md` |
| `hard_sentences.txt` | hand-written pronunciation set (numbers, abbreviations, irregular spellings, rare names) |
| `mel.py`, `griffin_lim.py`, `vocoder_model.py` | copied from 28 (provenance header in each) |
| `checkpoints/acoustic/tts.*` | shipped acoustic model, step 12,911 (end of run 2), used by the demo (fp16 weights + config + pace; 48 MB) |
| `checkpoints/baseline/baseline.npz` | tuned baseline tables (selected by val CER). `baseline.py` also writes `baseline_dtw.npz` here (the config val DTW-L1 alone would pick); it was moved to `runs/` to keep checkpoints small |
| `checkpoints/ctc/ctc.*` | recognizer |
| `checkpoints/vocoder28/` | 28's vocoder checkpoint (copied) |
| `results/` | `eval_summary.json`, `eval_clips.jsonl`, `hard_sentences_eval.json` (shipped model), `run1/` (same files for the run-1 checkpoint), `review_step7000/` (reviewer: the training loop's val-loss pick, step 7,000, on the same test clips, no GL), `timing_split.json`, `bench.json`, `baseline_tuning.json` (all 3 rounds), `tables.md`, training/CTC logs (`train_log.jsonl`, `ctc_log.jsonl`, `*.log`) |
| `media/` | `demo.html`, WAVs, `mels.png`, `alignment.png`, `train_curve.png` |

Training checkpoints with optimizer state (~290 MB each) live in `runs/train_ckpt/` (gitignored); `cache/` is regenerable.
The run-1 and step-7,000 exports used by `val_compare.py`, `results/run1/` and `results/review_step7000/` are also in the
gitignored `runs/` (2.6 GB on this disk). The committed checkpoints (~124 MB: acoustic 46, ctc 30, vocoder28 30, baseline
18) are all needed: acoustic + vocoder for the demo, baseline + ctc for the eval. `results/wavs/` (137 MB of eval WAVs) is
gitignored.

## Follow-up (2026-10-31): flow-matching TTS, multi-speaker pretraining, vocoder check

**Result (RESULTS.md, "Follow-up"):** a 44 M-parameter flow-matching (Matcha-style OT-CFM) acoustic model on v1's own MAS
alignment + duration predictor, with pitch/energy predictors, raises UTMOS on the same 100 test clips from 1.95 (v1) to
**2.96** through the same vocoder, with Whisper intelligibility level with v1 (1.32 % vs 1.14 % CER, CI spans 0). That beats
Gemma Voice (project 33: 2.45 % CER, UTMOS 2.35). Pretraining on 337 h of LibriTTS-R + Hi-Fi TTS before an LJ fine-tune
ties the LJ-only model on test. 28's vocoder is the naturalness ceiling (real mel through it: 2.59; real recording: 4.40). A
short GAN fine-tune of it failed its gate, and halves the new model's margin over v1 to +0.5. Listener preference: not
measured; blind page `media/followup.html`.

*Reviewer (REVIEW_FOLLOWUP.md):* the 2.96 comes after choosing the sampler temperature by val UTMOS. At the pre-registered
default (temp 0.667) B scores 2.87, still **+0.93 [+0.86, +1.00]** over v1. The run's unselected last step gives +0.94.
WER is not level with v1: +0.83 points [+0.00, +1.65]. The GAN vocoder is better mainly on v1's mels (+0.48 there, +0.11 on
real mels), so the "+0.5 through a better vocoder" is not a measurement with a good vocoder. Speed under load was about 2×
slower than idle (B 57 ms vs 30 ms per clip). The loose `media/followup_*.wav` files carry system names in their
filenames, so listen through the page for blind picks. The page's final tally lists v1, B and C twice each (once per
label wording: held-out groups vs hard groups), so add the two rows by hand. The Gemma row reads 33's WAVs from 33's
gitignored `runs/`, and `hard_fm.py` imports 33's `judge.normalize`, so those two pieces are not standalone.

*Builder, after the review:* three of these are fixed. The page's tally now uses one label per system (page regenerated; every WAV
byte-identical). `hard_fm.py` now carries a verbatim, provenance-marked copy of the judge normalizer (same `hard_summary.json`). And
`eval_fm.py`'s sampler flags now default to the checkpoint's stored sampler (shipped checkpoints: Euler 10, temp 0.5). The Gemma
row still reads 33's WAVs.

```
text --normalize--> chars --encoder(+speaker)--> h, mu --dur/pitch/energy predictors--> durations, per-char pitch & energy
     --length regulator--> [mu, h + pitch/energy emb] per frame --OT-CFM decoder, Euler 10 steps, temp 0.5--> 80-band log-mel
     --> 28's vocoder --> 16 kHz          (training only: MAS on log N(mel; mu, I), exactly as v1)
```

Run it (from this folder; every MLX process caps itself at 16 GB / 2 GB cache):
```sh
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python prep_multi.py      # -> cache2/{lj,libritts,hifi} (12 GB, ~12 min)
../.venv/bin/python prep_wav16.py                                                         # -> cache2/lj/wav16.* (vocoder GAN only)
# B: LJ only, 2 x 45 min
../bin/gpu_run ../.venv/bin/python train_fm.py --sources lj --out runs/fm/lj/ckpt --max_minutes 45
../bin/gpu_run ../.venv/bin/python train_fm.py --sources lj --out runs/fm/lj/ckpt --max_minutes 45 --resume --anneal_to 0.1
# C: multi-speaker pretraining 2 x 45 min, then LJ fine-tune 45 min
../bin/gpu_run ../.venv/bin/python train_fm.py --sources lj,libritts,hifi --out runs/fm/multi/ckpt --max_minutes 45
../bin/gpu_run ../.venv/bin/python train_fm.py --sources lj,libritts,hifi --out runs/fm/multi/ckpt --max_minutes 45 --resume
../bin/gpu_run ../.venv/bin/python train_fm.py --sources lj --init runs/fm/multi/ckpt --out runs/fm/ft/ckpt --lr 2e-4 --warm 500 --max_minutes 45
# export a snapshot (pace fitted on val), generate + proxy-score, then Whisper judge + UTMOS
../.venv/bin/python finalize_fm.py --src runs/fm/lj/ckpt_snap14000 --out runs/fm/export/lj_s14000
../bin/gpu_run ../.venv/bin/python eval_fm.py --ckpt runs/fm/export/lj_s14000 --name fm_lj --split test --temp 0.5   # also --system v1|real, --vocoder, --split val|hard
./score_fm.sh test_fm_lj                                       # ../.venv-judge (33's judge.py, as-is) + ../.venv-sac (mos.py, UTMOS22)
runs/fm/grid.sh                                                # the val sampler grid used for selection
../bin/gpu_run ../.venv/bin/python voc_check.py && ./score_fm.sh vc_lj_real vc_lj_voc vc_libritts_real vc_libritts_voc vc_hifi_real vc_hifi_voc
../bin/gpu_run ../.venv/bin/python train_voc_gan.py --out runs/voc/gan --max_minutes 45   # GAN fine-tune of 28's vocoder (failed its gate)
../.venv-judge/bin/python hard_fm.py hard_p27v1 hard_fm_lj hard_fm_ft ; python3 endings_fm.py
../.venv/bin/python report_fm.py                               # -> results/fm/tables.md (every follow-up table)
../.venv/bin/python make_demo_fm.py                            # -> media/followup.html + media/followup_*.wav
../.venv/bin/python synth_fm.py "Dr. Smith paid \$15 on March 3rd, 1963." --out media/fm_synth.wav
./smoke_fm.sh                                                  # follow-up pipeline end to end at tiny size (~40 s)
../.venv/bin/python review_fm.py                               # reviewer: selection effect on test -> results/fm/review_tables.md (inputs: see its docstring)
```
(*Reviewer:* `smoke_fm.sh` covers prep, training, resume, fine-tune, export and synthesis (20 s here). It does not run
`eval_fm.py`, the judge, UTMOS, `report_fm.py` or `make_demo_fm.py`. Also, `eval_fm.py`'s CLI defaults (`--temp 0.667`,
`--n_steps 10`) override a checkpoint's stored sampler, so pass `--temp 0.5` to reproduce the shipped setting, as the
command above does. `synth_fm.py` and `make_demo_fm.py` do use the stored sampler.)
The exact sequence that ran is in `runs/fm/chain1.sh`, `runs/fm/chain2.sh` and STATUS.md (gitignored `runs/` holds the logs).

| path | what |
|---|---|
| `pitch.py` | numpy F0 tracker (normalized autocorrelation), frame-aligned with `mel.py` |
| `prep_multi.py`, `prep_wav16.py` | multi-source mel + F0 cache (`cache2/`, memory-mapped fp16); LJ 16 kHz waveforms for the vocoder GAN |
| `model_fm.py` | flow-matching acoustic model, losses, MAS alignment, ODE sampling, `cap_memory()` |
| `train_fm.py` | training (single or multi-source, `--init` fine-tune, `--resume`), snapshots |
| `finalize_fm.py`, `eval_fm.py`, `synth_fm.py` | export with val pace; generation + proxy metrics + judge lists; text → wav |
| `score_fm.sh`, `mos.py`, `report_fm.py`, `hard_fm.py`, `endings_fm.py` | Whisper + UTMOS scoring, tables, hard set, sentence-ending check |
| `voc_check.py`, `train_voc_gan.py` | 28's vocoder on other speakers; GAN fine-tune of it |
| `make_demo_fm.py`, `followup_template.html` | blind page (per-viewer shuffle, labels hidden until reveal, per-system tally) |
| `smoke_fm.sh` | tiny end-to-end check |
| `review_fm.py` | reviewer diagnostic: test at the default sampler and the final step → `results/fm/review_tables.md` |
| `checkpoints/fm/tts_fm_lj.*` | **shipped**: B, LJ only, step 14,000, fp16, pace + val-chosen sampler in the JSON (88 MB) |
| `checkpoints/fm/tts_fm_ft.*` | C, pretrain + LJ fine-tune, step 4,000 (88 MB) |
| `results/fm/` | `lists/` (judge lists), `judge/`, `mos/`, `proxy/`, `tables.md`, `systems.json`, `hard_summary.json`, `endings.json` |

The new checkpoints add 176 MB to `checkpoints/`. `cache2/` (12 GB) and `runs/` (training checkpoints, generated WAVs, the
GAN vocoder snapshots) are regenerable and gitignored.
