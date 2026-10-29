# 28 · Vocoder Thunderdome — results

**Headline (revised in cross-review):** after one 17-minute training run, the MLX neural vocoder
**loses to Griffin–Lim on held-out pitch** (gross pitch error 7.7% vs 3.6%, f0 RMSE 41.7 vs 34.6 cents) and on
mel L1 (0.195 vs 0.100). Mel L1 is GL's own objective, though: NNLS fits the given mel, GL then iterates
consistency with that magnitude on the same 1024/256 STFT, and mel L1 re-measures the mel of the result. GL's small
MR-STFT edge (0.936 vs 0.945) is likewise entirely the 1024/256 term (0.808 vs 0.938); at 512/128 and 2048/512 the
neural model is better (0.903 vs 0.996, 0.993 vs 1.006), but those are its training loss. Pitch, voicing and the
transient measures are the metrics neither method optimises. On transients the neural model wins: energy ahead of
plosive onsets is +1.7 dB over ground truth vs +2.5 dB for GL, and a level-confound check below rules out "it's just
quieter" (the win is modest: the 0.75 dB gap is about what a 2 ms misalignment shifts this measure). Rise time is a tie (13.2 vs 13.2 ms). It is also faster: about 8× faster than 64-iteration GL on CPU
(MLX CPU RTF 0.0035 vs numpy 0.0282) and about 100× faster on the GPU (RTF 0.00029), measured at load 4.0
(`results/bench_rtf.json`) and confirmed on a near-idle machine (load 2.1; `results/bench_rtf_idle.json`: CPU 0.0035 vs 0.0271, 7.8×; GPU 0.00030, about 90×). The main reason it loses is **memorisation**. On 24 of
its own 400 training clips it ties GL on mel L1 (0.100 vs 0.100) and beats it on the rest: MR-STFT 0.71 vs 0.94, gross pitch
error 3.6% vs 4.0%. On held-out chapters it falls back to 0.195 / 0.945 / 7.7%. 42 minutes of audio is too
little for a 7.7 M-parameter model trained for about 220 epochs. **Ablation:** the same model trained on
3000 clips (322 min) *with a full 25k-step LR schedule* beats GL on held-out MR-STFT on all 24 clips (0.812 vs 0.936; it also
edges GL at GL's own 1024/256 resolution, 0.801 vs 0.808), on voicing (3.7% vs 4.1%) and on transients. It closes two thirds
of the mel-L1 gap (0.132 vs GL 0.100) and narrows the pitch gap (GPE 4.7% vs 3.6%), but it still loses mel L1 on every
clip and still loses pitch. Treat the ablation as more data *plus* a full LR schedule (see the caveat in the ablation section).

All numbers: 24 held-out LJSpeech clips (chapters LJ045–LJ050, 2.0 min total), ground-truth mels from
the one fixed mel definition (16 kHz, n_fft 1024, hop 256, 80 Slaney bands 0–8 kHz, natural-log magnitude).
LJSpeech has one speaker, so "held-out" means held-out text/chapters only.

## Setup actually run
| | Griffin–Lim (baseline) | Neural (learned) |
|---|---|---|
| method | NNLS mel→linear (pinv + 60 multiplicative updates), fast GL momentum 0.99, random phase seed 0 | Vocos-style: conv + 8 ConvNeXt blocks (dim 384) at frame rate → log-mag + phase → iSTFT |
| params / training | none | 7.74 M params; 17,195 steps × batch 32 × 64 frames (~1 s segments), AdamW 1e-3 cosine, 17.0 min wall (wall-clock cap hit; LR was at ~24% of peak, not fully annealed) |
| loss | — | MR-STFT (SC + log-mag L1 at 512/1024/2048) + 2 × mel L1, no adversarial term |
| input | identical ground-truth log-mels | identical ground-truth log-mels |
| hardware | numpy, CPU, `OMP_NUM_THREADS=4` | MLX GPU (also timed on MLX CPU) |

Training curve: `media/train_curve.png` (from `train_log.jsonl`). Loss 7.31 → 0.93; train-batch mel L1 1.96 → 0.11.

## Quality on held-out clips (`results/metrics.json`, written by `eval.py`)
Lower is better everywhere. Transient columns cover the 8 plosive-heavy held-out clips (249 detected
ground-truth onsets); ground-truth 10–90% rise time is 7.6 ms.

| method | MR-STFT | mel L1 | f0 RMSE (cents) | gross pitch err % | voicing err % | periodicity RMSE | pre-onset excess dB | rise time ms |
|---|---|---|---|---|---|---|---|---|
| GL 8 it | 0.965 | 0.137 | 39.1 | 5.18 | 5.16 | 0.066 | +3.56 | 14.2 |
| GL 16 it | 0.941 | 0.118 | 37.0 | 4.36 | 4.53 | 0.064 | +3.13 | 14.3 |
| GL 32 it | **0.935** | 0.106 | 35.7 | 3.92 | 4.16 | 0.063 | +2.72 | 13.7 |
| **GL 64 it (baseline)** | 0.936 | 0.100 | 34.6 | **3.56** | **4.08** | 0.061 | +2.49 | **13.2** |
| GL 128 it | 0.939 | **0.097** | **34.2** | 3.79 | 4.14 | 0.062 | +2.43 | 13.4 |
| **neural** | 0.945 | 0.195 | 41.7 | 7.72 | 4.51 | **0.058** | **+1.73** | 13.2 |

Per clip (GL 64 vs neural): neural has lower mel L1 on 0 of 24 clips, lower MR-STFT on 8 of 24, and less
pre-onset smearing on 7 of the 8 plosive clips.

**Train vs held-out check** (`results_trainsplit/metrics.json`: `eval.py --data cache/train.npz --limit 24`,
the first 24 training clips, same scripts):

| | mel L1 train | mel L1 held-out | MR-STFT train | MR-STFT held-out | GPE % train | GPE % held-out |
|---|---|---|---|---|---|---|
| GL 64 | 0.100 | 0.100 | 0.939 | 0.936 | 4.03 | 3.56 |
| neural | 0.100 | 0.195 | 0.711 | 0.945 | 3.57 | 7.72 |

Griffin–Lim is split-independent, as it should be. The neural vocoder's held-out mel error is about double
its training error, so it overfit. At step ~3k the train-batch and held-out mel L1 were both about 0.23,
so the gap opened later in training. (Cross-review: the train-batch value is in `train.log`, ~0.236 at step
2.3k; the held-out value at step ~3k is **not in any file**, since `train.py` never evaluates held-out data. Treat it as
unverified.) The checkpoint is overwritten every 1000 steps, so there is no
early-stopped checkpoint to compare.

**Is the transient win just less high-band energy?** No. Over the same 8 clips, whole-clip energy above
2 kHz relative to ground truth is −1.44 dB for neural and −1.13 dB for GL 64. So neural is 0.3 dB quieter
up there overall, but 0.75 dB lower before onsets, and *higher* than GL in the 5–30 ms after onsets
(−0.55 vs −0.98 dB). Pre-onset minus post-onset (a level-free smear contrast) is 2.3 dB for neural and
3.5 dB for GL 64.

### Metric definitions (`metrics.py`)
- **MR-STFT** = mean spectral convergence + mean |Δ log magnitude| over (512/128, 1024/256, 2048/512).
  This is also the neural training loss, so it is biased toward the neural model. Griffin–Lim directly
  optimises consistency with the 1024/256 magnitude, so it is biased toward GL too. Per-resolution means over the
  24 held-out clips (cross-review, recomputed from `results/wavs`, `results_3k/wavs`), 512/1024/2048: GL 64
  0.996 / **0.808** / 1.006; neural 0.903 / 0.938 / 0.993; 3000-clip 0.778 / 0.801 / 0.857. GL's MR-STFT edge
  over the main neural model is all in the 1024/256 term.
- **mel L1** is measured on the same 1024/256 STFT that GL's NNLS + phase iterations fit at inference time, so
  it is structurally favourable to GL (it is the one metric still improving at 128 iterations). The neural model
  also trains on it (2 × mel L1), but amortised over the training set rather than solved per test clip.
- **Neutral metrics:** pitch (YIN), voicing, periodicity and the transient measures are optimised by neither method.
- **Level and alignment** (cross-review check): outputs are scored raw, not loudness-matched. Mean RMS level vs
  ground truth is −0.46 dB (GL 64), −0.81 dB (neural), −0.32 dB (3000-clip). Removing the median log-mel offset
  moves neural mel L1 only 0.195 → 0.188 (GL unchanged at 0.100), so level does not explain the gap. All methods
  are frame-synchronous with the ground truth (same centred STFT); broadband-envelope cross-correlation puts every
  output within −1..+4 ms, with the same spread for GL and neural. The pre-onset measure is shift-sensitive:
  moving the output envelope by −2 / 0 / +2 ms gives GL +3.03 / +2.46 / +2.10 dB and neural +2.79 / +1.73 / +0.92 dB.
  The ordering holds at any matched shift, but the 0.75 dB gap is the same size as a 2 ms misalignment.
- **mel L1** = mean |Δ log-mel| with the project mel (see the bias note above).
- **Pitch**: YIN (60–400 Hz, 10 ms hop, threshold 0.15) on output vs ground truth. f0 RMSE in cents on
  frames voiced in both (excluding >20% errors), gross pitch error = share of both-voiced frames >20% off,
  voicing error = disagreement on the voiced/unvoiced decision, periodicity = 1 − min CMND (clipped to 0..1),
  RMSE on ground-truth voiced frames.
- **Transients**: energy envelope above 2 kHz (4th-order Butterworth high-pass, 2 ms RMS, 1 ms hop). Onsets are
  ground-truth points with a >15 dB rise within 10 ms. *Pre-onset excess* = output energy in the 30–5 ms
  before each onset relative to ground truth (dB; 0 is perfect, positive means energy smeared ahead of the
  burst). *Post-onset excess* is the same for 5–30 ms after the onset, and *HF level* is the whole-clip energy
  above 2 kHz relative to ground truth (both are level-confound checks). *Rise time* = 10→90% of the envelope rise.

## Speed and memory (`results/bench_rtf.json` under load; `results/bench_rtf_idle.json` near idle)

**Idle rerun (orchestrator, 19:49, load 2.09 / 2.22 / 4.08, nothing else training; `results/bench_rtf_idle.json`, stdout in `results/bench_rtf_idle_stdout.txt`):** GL 32 RTF 0.0167 (first output 19.2 ms), GL 64 RTF 0.0271 (30.9 ms), neural MLX GPU RTF 0.00030 (1.0 ms), neural MLX CPU RTF 0.0035 (4.1 ms). These agree with the load-4.0 column below to within about 6%, so the speed conclusions stand: 7.8× on matched CPU, about 90× GPU vs CPU GL. The table below keeps the under-load runs.
`bench_rtf.py` is standalone and meant to be rerun on an idle machine. RTF is whole-utterance synthesis
time ÷ audio duration, median of 3 per clip over all 24 clips. First-output latency is the time to synthesise
the first 64 mel frames (≈1.0 s of audio), median of 5. It was run twice. The file holds the second run (19:24,
load averages 4.0 / 8.0 / 10.2). The first run (19:04, load 13.7 / 13.9 / 10.8, ten builders training) was
overwritten, and its numbers were copied from the console. **They are not traceable to any file** (kept in the
table for context only; use the load-4.0 column). `bench_rtf.py` now refuses to overwrite an existing `--out`.
**Both are under-load numbers.**

| method | RTF (load 13.7, console only, no file) | RTF (load 4.0, in file) | × real time (load 4.0) | first-output latency (load 13.7 console / 4.0 file) | peak memory (longest 6.9 s clip) |
|---|---|---|---|---|---|
| GL 32 it (numpy CPU, 4 threads) | 0.0249 | 0.0178 | 56× | 23.6 / 20.4 ms | 22.1 MB (tracemalloc) |
| GL 64 it (numpy CPU, 4 threads) | 0.0374 | 0.0282 | 35× | 38.5 / 31.6 ms | 22.1 MB (tracemalloc) |
| neural, MLX GPU | 0.00034 | 0.00029 | 3,440× | 1.1 / 1.0 ms | 78.5 MB (MLX peak, incl. 31 MB fp32 weights) |
| neural, MLX CPU | 0.0051 | 0.0035 | 284× | 5.2 / 4.2 ms | 53.4 MB (MLX peak) |

Matched hardware (CPU vs CPU): neural is 7–8× faster than GL 64. Neural on GPU vs GL on CPU: about 100×.
The 3000-clip ablation model has the same architecture, so it has the same speed (not benched separately).
Process max RSS for the whole bench process (Python + MLX + numpy + all clips loaded): 863 MB (`max_rss_mb`). The per-clip timings in
`metrics.json` (`rtf_under_load`, one run each) are noisier: GL 128 iterations came out faster than GL 64 there,
so use `bench_rtf.json` for speed. The neural vocoder is non-streaming. Its receptive field is ±27 frames,
so a streaming version would need about 0.43 s of look-ahead. Streaming was **not measured**.

"At matched quality targets": no neural operating point reaches Griffin–Lim's held-out mel L1, so a
matched-quality speed comparison is **not possible yet**. The GL budget sweep (`media/quality_vs_budget.png`)
shows GL's quality saturating at around 32–64 iterations.

## Ablation: 3000 training clips (`results_3k/metrics.json`, `results_3k_trainsplit/metrics.json`)
Run after the write-up was complete. Same model, seed, batch, `--steps 25000` and 17-min cap, and the
same 24 held-out clips. The training set is a superset: 3000 clips (322 min) from LJ001–LJ040, including
the original 400. Checkpoint: `checkpoints/vocoder_3k.safetensors`; log: `train_log_3k.jsonl`.
**Caveat:** the machine was quieter during this run, so it finished all 25,000 steps in 14.2 min and annealed its
LR to the end. The main run hit the cap at step 17,195 with the LR at ~24% of peak. The difference below is
therefore *more data + a full schedule*, not data alone.

| held-out (24 clips) | MR-STFT | mel L1 | f0 RMSE ¢ | GPE % | VDE % | periodicity | pre-onset dB | post-onset dB | HF level dB | rise ms |
|---|---|---|---|---|---|---|---|---|---|---|
| GL 64 | 0.936 | **0.100** | **34.6** | **3.56** | 4.08 | 0.061 | +2.49 | −0.98 | −1.13 | 13.2 |
| neural, 400 clips (main) | 0.945 | 0.195 | 41.7 | 7.72 | 4.51 | **0.058** | **+1.73** | −0.55 | −1.44 | 13.2 |
| neural, 3000 clips | **0.812** | 0.132 | 36.3 | 4.66 | **3.72** | 0.060 | +1.76 | **−0.46** | **−1.00** | **12.5** |

Per clip, the 3000-clip model vs GL 64: lower MR-STFT on 24/24, lower mel L1 on 0/24, less pre-onset smearing
on 7/8 plosive clips, lower GPE on 3/24. Train vs held-out for the 3000-clip model (same first 24 training
clips as above): mel L1 0.113 train vs 0.132 held-out, MR-STFT 0.775 vs 0.812. The memorisation gap
(0.100 → 0.195 for the main run) is mostly gone.

## Listening demo
`media/demo.html` (6.3 MB, fully offline: audio and spectrograms embedded). Six held-out clips, four of them
plosive-heavy. For each clip, ground truth, GL 64, the main neural model and the 3000-clip ablation are shuffled
behind A/B/C/D (fixed seed 28; since cross-review the ground-truth slot is spread across A–D, because the original
per-clip permutation put ground truth in slot A on 5 of 6 clips), and a reveal button shows the labels, per-clip metrics and spectrograms. All WAVs are integrated-loudness matched to
−23 LUFS with a BS.1770 K-weighted, gated meter implemented in `make_demo.py` (`pyloudnorm` isn't installed),
with one shared peak cap per clip so all versions stay matched. Files: `media/<id>_{gt,gl64,neural,neural3k}.wav`,
`media/spec_<id>.png` (full spectrogram plus a ±80 ms zoom on the strongest onset).
No listening test with people was run (**not measured**).

What the spectrograms show: Griffin–Lim keeps the harmonic comb but blurs it into horizontal streaks and
smears energy across onsets, which sounds phasey or "underwater". The neural output has crisper onsets in the
zooms, but above about 3 kHz its harmonics turn to noise, which sounds hoarse or breathy and is consistent
with its worse gross pitch error. The 3000-clip model restores much more of the harmonic structure up to
about 5 kHz. Its fricative zooms show a faint periodic dotting pattern. The cause was not investigated;
frame-rate phase discontinuities in the iSTFT head are a guess.

## What worked
- Vocos-style frame-rate vocoder plus a differentiable MLX STFT: about 25 ms per step unloaded and 60 ms
  under load. Loss fell steadily for the whole run with no instability and no adversarial term.
- One shared mel module kept every number on the same features. The STFT/iSTFT round trip is exact
  (max error 9e-8).
- Transients are the one quality axis where the learned model already wins on held-out data, and the win survives the level-confound check.
- More data plus a full LR schedule: 3000 clips with all 25k steps cut held-out mel L1 from 0.195 to 0.132 and turned
  MR-STFT into a clear win over GL. The two factors are not separated.

## What didn't
- Generalisation: held-out mel L1 is twice the training value, the clearest result of the run.
- Without an adversarial or phase loss, high-frequency harmonics come out as noise, which shows up as pitch errors.
- The wall-clock cap (17 min) cut the cosine schedule at step 17.2k of 25k, so the learning rate never annealed.

## Surprises
- Even under load, Griffin–Lim at 64 iterations runs 27–35× real time on 4 CPU threads. It is a much
  tougher baseline than its reputation suggests once the NNLS mel inversion is done properly.
- More GL iterations stop helping MR-STFT after 32 (0.935 → 0.939 at 128). The phase solution gets more
  self-consistent at 1024/256 but no better at the other resolutions.
- The neural model's memorisation is very visible: on training clips it beats GL clearly.

## Next steps
1. Separate data from schedule: rerun 400 clips with a full schedule, or 3000 clips at 17k steps. Train on the full
   LJ001–LJ040 set (~10k clips) and early-stop on a separate validation split (not the 24 test clips).
2. Add a multi-period/multi-scale discriminator for the high-band harmonics.
3. (Idle rerun done, see Speed.) Add a streaming (chunked, look-ahead) mode and measure true first-packet latency.
4. "Go deeper" items: imperfect mels from Pocket Voice Foundry, and diffusion/flow vocoders at several step budgets.

## Gaps
- `pyloudnorm` and `librosa` aren't installed, so the mel filterbank, YIN and BS.1770 loudness are
  re-implemented here (filterbank formulas follow librosa's Slaney defaults).
- Not measured: human listening scores, streaming latency, idle-machine speed.

## Follow-up (round 3): data size alone, with a validation split (`results/followup_val3k/`)
**Why:** the review found that the neural model loses the neutral pitch metric (GPE 7.7% vs GL 3.6%), overfits its
400 clips, and that the 3000-clip ablation changed data *and* LR schedule together, with no validation split.

**What was run (one run, 9.4 min, `train_val.py` via `gpu_run`, launched at load 4.2 (`results/followup_val3k/launch.txt`); val-set evals ran alongside it and logged load 8.9–14.3, so no speed numbers from it):**
the main run's exact recipe (same model, batch 32 × 64 frames, AdamW, seed 0, the same cosine schedule object over
`--steps 25000`, stopped at **step 17,195**, the step where the main run's wall-clock cap hit, LR 2.35e-4 ≈ 24% of
peak) on the **3000-clip** set (`prepare_data.py --n_train 3000`; its first 400 clips are the main run's 400: same seed-0 shuffle, greedy in-order
filter; verified in review by recomputing the 3000-clip ID list and comparing its first 400 with `cache/split.json`, in order). It did not hit its own 15-min cap. Training-batch loss tracks the old 3000-clip ablation log within ~1–2%
at matched steps (EMA 1.920 vs 1.921 at step 1k, 1.349 vs 1.351 at 5k), so the three runs form a clean triangle:
main (400 clips, 17,195 steps) → **3000 @ 17,195 (`last`)** isolates data; `last` → old ablation (3000 @ 25,000,
fully annealed) isolates schedule.
**Validation split** (`prepare_val.py`): 32 clips (2.7 min, 2–7 s) from chapters **LJ041–LJ044**, disjoint from training
(LJ001–040) and test (LJ045–050). Every 500 steps `train_val.py` computes the training objective on whole val clips
(**val_loss, the selection criterion, fixed before any test eval**) and YIN GPE (logged only). The test set is never
read during training. Best val_loss 1.1238 at step 16,500 (`checkpoints/followup_val3k/best.*`); last 1.1304 at
17,195 (`checkpoints/followup_val3k/last.*`). Val loss had flattened, not still falling: 1.1384 / 1.1307 / 1.1238 / 1.1274 / 1.1304 at steps 15.5k / 16k / 16.5k /
17k / 17,195, so selection had almost nothing to choose: `best` ≈ `last`. Curves: `media/followup_val3k/val_curve.png`, log `results/followup_val3k/train_log.jsonl`.

**Held-out test, same 24 clips and same `eval.py`** (`results/followup_val3k/{best,last}/metrics.json`; GL 64 rows there
reproduce `results/metrics.json` exactly). Neutral metrics first:

| 24 test clips | GPE % | f0 RMSE ¢ | VDE % | periodicity | pre-onset dB | post-onset dB | rise ms | mel L1* | MR-STFT* |
|---|---|---|---|---|---|---|---|---|---|
| GL 64 | **3.56** | **34.6** | 4.08 | 0.061 | +2.49 | −0.98 | 13.2 | **0.100** | 0.936 |
| neural, 400 clips @17,195 (main) | 7.72 | 41.7 | 4.51 | **0.058** | **+1.73** | −0.55 | 13.2 | 0.195 | 0.945 |
| **neural, 3000 @17,195, `last`** (data-only change) | 4.59 | 37.2 | 4.06 | 0.062 | +1.76 | −0.61 | 12.9 | 0.139 | 0.820 |
| neural, 3000 @16,500, `best` (val-selected) | 4.52 | 38.5 | 4.06 | 0.064 | +2.11 | −0.31 | **12.4** | 0.142 | 0.825 |
| neural, 3000 @25,000 (old ablation) | 4.66 | 36.3 | **3.72** | 0.060 | +1.76 | −0.46 | 12.5 | 0.132 | **0.812** |

\* mel L1 and the 1024/256 MR-STFT term are Griffin–Lim's own objective; all MR-STFT terms are in the neural training loss.

MR-STFT per resolution (`results/followup_val3k/mrstft_per_res.json`, written by `mrstft_per_res.py`; it reproduces the
reviewer's hand-computed table to 3 decimals): GL 64 0.996 / 0.808 / 1.006 (512 / 1024 / 2048); main 0.903 / 0.938 /
0.993; 3000 `last` 0.787 / 0.809 / 0.864; `best` 0.791 / 0.813 / 0.869; old ablation 0.778 / 0.801 / 0.857.
At GL's own 1024/256 resolution `last` is level with GL (0.809 vs 0.808); `best` is slightly behind (0.813).

**Per clip** (`best` vs GL 64 / vs main): lower GPE on 5/24 / 21/24, lower MR-STFT on 24/24 / 24/24, lower mel L1 on
0/24 / 24/24, less pre-onset smear than GL on 6/8 plosive clips. Paired bootstrap over the 24 clips (5000 resamples, seed 0;
`bootstrap_followup.py` → `results/followup_val3k/bootstrap_gpe.json`, added in review because the builder's CIs were
computed inline and matched only to ~±0.03): GPE `best` − GL = +0.95 points, 95% CI [+0.44, +1.47]; `best` − main = −3.20,
CI [−4.11, −2.33] (`last`: +1.02 [+0.48, +1.56] and −3.13 [−4.10, −2.17]; `last` − old ablation −0.08 [−0.51, +0.34]).
Clips, not speakers, are the resampling unit (one speaker). These CIs cover clip sampling only, not training-seed variance.

**Validation set, for context** (`results/followup_val3k/valset_*/metrics.json`, `eval.py` on the 32 val clips): GPE GL 64
4.30%, main 8.91%, old 3000 ablation 5.65%, new run `last` 6.02% (`valset_last/`; matches the in-training value in
`train_log.jsonl`). As on the test set, GL is best, the main model worst, and both 3000-clip models sit in between; the two 3000-clip models swap order between val and test (6.02 vs 5.65 on val, 4.59 vs 4.66 on test), i.e. they are within noise.

**What this says:**
- **The data-size gain is large, and on pitch and the spectral losses it is most of the old ablation's gain** (one seed
  per run, so this is one draw of each, not a measured decomposition). With the schedule held fixed, 400 → 3000
  clips takes held-out GPE from 7.7% to 4.6% (CI excludes zero), f0 RMSE 41.7 → 37.2 ¢, voicing error 4.5 → 4.1%, mel L1
  0.195 → 0.139, MR-STFT 0.945 → 0.820. Finishing the LR schedule (17k → 25k steps) adds small gains on everything except GPE: voicing
  error 4.06 → 3.72%, f0 RMSE 37.2 → 36.3 ¢, periodicity 0.062 → 0.060, mel L1 0.139 → 0.132, MR-STFT 0.820 → 0.812;
  GPE doesn't move (4.59 → 4.66%, CI [−0.34, +0.51]). Voicing is the exception to "almost all data": data takes VDE
  4.51 → 4.06 and the schedule 4.06 → 3.72, so the schedule is ~40% of that gain (clip-paired CI −0.62 to −0.07). With one
  seed per run and `best` vs `last` (700 steps apart) differing by ~40–60% as much as the schedule on mel L1 and MR-STFT
  (`bootstrap_gpe.json`, `other_metrics`), the small schedule effects are not separable from run-to-run jitter. The
  review's "data + schedule, not separated" is now separated for this seed: data dominates GPE, f0, mel L1 and MR-STFT;
  voicing gets a real share from the schedule. More data also made periodicity slightly worse (0.058 → 0.062).
- **The neural vocoder still loses pitch to Griffin–Lim** (4.5% vs 3.6% GPE, CI excludes zero; it wins only 5 of 24 clips
  on GPE). It ties GL on voicing (4.06 vs 4.08%), and periodicity is a wash (0.064/0.062 vs 0.061).
- **Early stopping didn't matter at this budget.** Val loss had flattened over the last ~1,200 steps; `best` and `last`
  are 700 steps apart, level on GPE and VDE, and differ by small amounts on f0 RMSE (+1.4 ¢), periodicity, mel L1 and
  MR-STFT (the last three outside clip-level CIs, but tiny) and on transients (next bullet).
- The transient measure moves by 0.35 dB between two checkpoints 700 steps apart (+2.11 vs +1.76 dB pre-onset excess),
  about half the main model's 0.75 dB win over GL; the 3000-clip models' pre-onset margin over GL is 0.72 (`last`),
  0.73 (old ablation) or 0.38 dB (`best`) depending on checkpoint. The earlier headline "on transients the neural model
  wins" should therefore be read as: ahead of GL on this one measure for every checkpoint tested, by a margin that is
  the size of a 2 ms envelope shift (review) or a 700-step checkpoint change (here), so not a robust audible claim.

**Not run / not measured:** the pitch-aware loss term (a low-frequency-weighted or 4096-point STFT term) was **skipped**:
the follow-up rules allow one main run, and the term is only interpretable alongside a plain run, so it wasn't run. Also
not measured: seed variance of the neural runs (one seed each), GL phase-seed variance, human listening scores, and any
speed numbers for this checkpoint (same architecture as the main model, so the speed numbers above apply unchanged).

**Demo:** `media/followup_val3k/demo.html` (6.3 MB, offline, 0 external references), a separate blind page. Same 6 held-out
clips; ground truth, GL 64, the main model and the val-selected 3000-clip model (`best`) are shuffled behind A–D (seed
2803, ground-truth slot balanced: B, D, A, C, B, C). Every embedded WAV measures −23.00 LUFS. The original `media/demo.html`
is unchanged.
