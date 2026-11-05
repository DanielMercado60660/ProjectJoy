# 27 · Pocket Voice Foundry: results

All headline numbers are on **held-out test sentences** (chapters LJ045–LJ050, never trained or tuned on), a fixed seed-0
subset of 100 clips (≈6.6 s each, 11 min of speech). Validation (LJ041–044) was used only for checkpoint choice, duration
pace, baseline tuning and the recognizer's own checkpoint save. Every table is printed by `report.py` from the JSON files named in each heading
(`results/tables.md` is the raw printout). LJSpeech has one speaker, so "held out" means unseen sentences and chapters,
not an unseen voice.

## Headline

- **Intelligibility proxy.** The shipped acoustic model (step 12,911, `checkpoints/acoustic/`) is at least as legible to the
  CTC recognizer proxy as the real recording. Its corpus CER is 5.8 %, **lower than the real recording's mel at 7.0 %** (paired
  −1.0 points, 95 % CI [−1.8, −0.2]). That is a bias of the proxy, not super-human speech (§1). *Reviewer:* the CI excludes 0
  only for this checkpoint, which was picked by val CER on the same recognizer. The training loop's own pick (step 7,000) is
  −0.3 [−1.1, +0.5]. Fed the real recording's timing, the TTS decoder is *worse* than real (+1.2 [+0.3, +2.2]), so the "win"
  comes from the TTS's averaged timing (§7).
- **Against the baseline.** The model beats a template-concatenation baseline tuned over 412 configurations, which scores
  16.8 % CER. Paired difference −11.2 points, 95 % CI [−12.2, −10.1].
- **Vocoder.** Project 28's vocoder, borrowed to isolate the acoustic stage, adds no measurable recognizer damage: +0.15
  points on real mels, CI [−0.12, +0.42].
- **Remaining gap.** The gap to real speech is spectral, not lexical. Mel distance (MCD-like) from the real recording is 4.4 dB
  for the TTS, against 0.6 dB for the vocoder alone.
- **Pronunciation.** Numbers and abbreviations come out right. Rare and irregular words come out as spelled (§3).
- **Speed (under load).** TTS + neural vocoder renders a 7-second sentence in about 6 ms, RTF 0.0009.
- **Listening preference: not measured.** The blind page `media/demo.html` exists, but nobody has taken it.

## 1. Intelligibility proxy and spectral distance (`results/eval_summary.json`, `results/eval_clips.jsonl`)

Rows: "mel, no vocoder" feeds the predicted log-mel straight to the recognizer (the acoustic stage alone). The vocoder rows
synthesize a waveform, re-analyse it to log-mel with the same `mel.py`, and score that.

| mel source | output | CER % (CTC proxy) [95% CI of the clip-mean CER] | WER % | DTW log-mel L1 | MCD-like dB |
|---|---|---|---|---|---|
| real mel (upper bound) | mel, no vocoder | 7.0 [6.3, 8.4] | 30.5 | 0 (reference) | 0 (reference) |
| real mel (upper bound) | 28's neural vocoder | 7.1 [6.5, 8.5] | 31.2 | 0.143 | 0.63 |
| real mel (upper bound) | Griffin-Lim 64 | 6.8 [6.2, 8.2] | 30.8 | 0.101 | 0.75 |
| learned TTS | mel, no vocoder | 5.8 [5.3, 7.3] | 26.2 | 0.740 | 4.39 |
| learned TTS | 28's neural vocoder | 6.2 [5.7, 8.0] | 27.5 | 0.836 | 4.60 |
| learned TTS | Griffin-Lim 64 | 6.0 [5.5, 7.5] | 26.4 | 0.746 | 4.43 |
| baseline (template concat) | mel, no vocoder | 16.8 [16.1, 19.0] | 51.5 | 0.839 | 5.05 |
| baseline (template concat) | 28's neural vocoder | 18.4 [17.2, 20.2] | 51.5 | 0.983 | 5.10 |
| baseline (template concat) | Griffin-Lim 64 | 17.1 [16.2, 19.1] | 51.3 | 0.837 | 5.03 |

The bracket is the bootstrap CI of the *clip-mean* CER, which runs about 1 point above the corpus CER (total edits / total
characters) because short clips have more errors per character.

Paired per-clip differences (bootstrap 95 % CI; `paired_cer_diff_*` in `eval_summary.json`):

| comparison | mel, no vocoder | 28's neural vocoder | Griffin–Lim 64 |
|---|---|---|---|
| TTS − real mel (points) | −1.03 [−1.82, −0.20] | −0.52 [−1.44, +0.51] | −0.87 [−1.64, −0.03] |
| TTS − baseline (points) | −11.2 [−12.2, −10.1] | −11.9 [−13.0, −10.7] | −11.1 [−12.2, −10.0] |
| vocoder on real mel − real mel (points) | n/a | +0.15 [−0.12, +0.42] | −0.09 [−0.38, +0.18] |

Both vocoders keep what the recognizer needs. In spectral terms, though, the acoustic stage is still about 5× further from the
real recording (DTW log-mel L1 0.74) than the vocoder's own damage (0.10–0.14).

WER is high everywhere (30 % even on real speech) because the recognizer is a character model with no lexicon or language
model, so one wrong letter makes the whole word wrong. CER is the main number.

**What the proxy is, and its limits.** `ctc.py` is a conv + 4 transformer-block network, 7.8 M parameters, with CTC over 28
symbols. It was trained for 14 min (4,315 steps, light SpecAugment) on real *training-split* mels only. The checkpoint was
chosen by val CER (8.2 % on 64 val clips at step 4,000). Its floor on held-out real speech is the 7.0 % CER above. Its limits,
measured where possible:
- **It is biased towards this TTS.** Both models learned the same speaker from the same 19 h of audio, so both learned the same
  "typical" mapping from text to spectrum. The TTS emits close to that average rendering, which is exactly what the recognizer
  expects. The shipped model scores *significantly better* than real speech (above), so the proxy can't rank the TTS
  against real speech. The *direction* is not an artefact of choosing the shipped checkpoint by val CER: the run-1 checkpoint (6.3 % vs 7.0 %,
  paired −0.4 [−1.3, +0.4]; `results/run1/eval_summary.json`) and the training loop's own val-loss pick, step 7,000 (6.6 %,
  −0.3 [−1.1, +0.5]; `results/review_step7000/`), point the same way. *Reviewer correction:* the *significance* is partly an
  artefact of that choice, since only the val-CER-selected checkpoint has a CI that excludes 0. Most of the margin also comes from timing:
  the same decoder fed the real recording's MAS durations scores 8.0 %, worse than real (§7). It can say "fully legible to a recognizer of this speaker", and it can rank TTS against the baseline,
  where the gap is 11 points.
- **It does not simply reward blur.** Smoothing the *real* mel in time makes it much worse: CER 8.5 % at σ = 1 frame, 24.5 % at
  σ = 2, 47.6 % at σ = 3 (`recognizer_probe_cer`). The TTS is only mildly oversmoothed: its per-band temporal standard
  deviation is 0.94 of the real recording's (baseline: 0.74; `gv_ratio`).
- **A listener would hear what it ignores.** `media/mels.png` shows the flat upper bands and blurred harmonics typical of an
  L1-trained mel model. MCD-like catches some of this; listening preference would catch more, and it is **not measured**.
- **MCD-like is not standard MCD.** It is the cepstrum (13 DCT coefficients, c0 dropped) of the natural-log mel, DTW-aligned,
  not the standard MCD on vocoder-analysis mel-cepstra. Compare it only within this table.
- **Val reuse and leakage (reviewer).** Every val-side choice uses the head of the val split, not a random sample: the
  recognizer save uses the first 64 val clips, while TTS checkpoint choice, duration pace and baseline tuning use the first 100. So the
  same val clips select the scorer, the TTS checkpoint and the baseline. Test is untouched by all of them, so the test numbers
  are clean held-out numbers, but both systems are tuned to *this* recognizer. Text leakage: 0 of the 100 eval clips' normalized
  text appears in the training text (LJSpeech itself has 5/1,536 test and 4/865 val exact duplicates of training sentences,
  none in the eval subset), and 2 eval clips share one 8-word phrase with training text. 28's vocoder was trained on LJ001–040
  only (per 28's README).

## 2. Two acoustic runs and the checkpoint choice (`results/train_log.jsonl`, `results/val_compare.json`, `results/run1/`)

- **Run 1 (first builder).** 40 min, steps 0–6,690, LR warm-up then inverse-sqrt from 1e-3. Val mel L1 (64 val clips,
  teacher-forced on MAS durations) went 0.399 (1k) → 0.375 (2k) → 0.361 (3k) → **0.359 (4k)** → 0.360 (5k) → 0.362 (6k). Train
  mel L1 ended at 0.322.
- **Run 2 (resume).** 45 min, steps 6,690–12,911, LR annealed linearly from 5.5e-4 to 5.5e-5. Train mel L1 0.322 → 0.283, but
  val mel L1 *rose*: 0.3636 (7k) → 0.3633 (8k) → 0.365 (9k) → 0.368 (10k) → 0.369 (11k) → 0.370 (12k). By the training
  loop's own criterion (val mel + val duration) the best step was 7,000, which wins on duration loss while being worse on mel.
- **Overfitting.** 12.9k steps are about 36 epochs over 10,555 clips. In L1 terms the model overfits after step ~4k.
- **Choice on validation.** Finalized candidates were compared on 100 val clips with the recognizer and DTW L1
  (`val_compare.py`), using the same rule as the baseline (min val CER, ties by DTW). This rule was adopted when choosing
  among the three exports, to match the baseline's selection. It was not pre-registered in the training loop, whose own rule
  (val mel + duration loss) is shown above. The choice used only the val lines under
  the table; the test columns are reported after the fact. (`runs/` is gitignored; the chosen export is the shipped
  `checkpoints/acoustic/tts.*`.)

| output | run-1 CER % | shipped CER % | paired diff, points [95% CI] | run-1 DTW L1 | shipped DTW L1 | run-1 MCD-like | shipped MCD-like |
|---|---|---|---|---|---|---|---|
| mel, no vocoder | 6.3 | 5.8 | -0.60 [-1.04, -0.16] | 0.723 | 0.740 | 4.36 | 4.39 |
| 28's neural vocoder | 6.8 | 6.2 | -0.72 [-1.41, -0.12] | 0.827 | 0.836 | 4.57 | 4.60 |
| Griffin-Lim 64 | 6.4 | 6.0 | -0.59 [-1.05, -0.13] | 0.729 | 0.746 | 4.39 | 4.43 |

Global-variance ratio (TTS mel / real): run 1 0.905, shipped 0.939. Duration: run-1 length ratio 0.974, shipped 0.979.
- val (100 clips) `runs/run1_tts_export/tts.safetensors` step 6000: CER 7.36 %, DTW L1 0.7136, length ratio 1.001
- val (100 clips) `runs/run2_best_export/tts.safetensors` step 7000: CER 7.51 %, DTW L1 0.7162, length ratio 1.005
- val (100 clips) `runs/run2_final_export/tts.safetensors` step 12911: CER 6.56 %, DTW L1 0.7260, length ratio 1.006

**Surprise: L1 and intelligibility disagree.** Annealing made the model *worse* in mel L1 but *more* intelligible, on both
val (7.4 → 6.6 %) and test (−0.6 points, CI [−1.0, −0.2]). It also made the output less smooth (GV ratio 0.905 → 0.939).
The likeliest reading is that L1 punishes the sharper, more committed spectra that a longer-trained model produces, while the
recognizer rewards them. L1 alone would have picked step 6,000 or 7,000. The run-1 model's full test results are kept in
`results/run1/`. No run 3: val L1 was rising, and the CER gain alone didn't justify another 45 min.

Alignment (`media/alignment.png`): monotonic alignment search (MAS) learned a clean monotonic alignment from characters with no
external aligner. The diagonal prior was only needed for the first 2,000 steps. Checkpoint/resume was verified before the long
run: kill at step 80, resume, val mel continued 0.4835 → 0.4833 (step 100) → 0.463 (step 200). Run 2
itself was a resume from step 6,690, and train loss continued smoothly (0.322 → 0.322 at step 6,700).

## 3. Durations (`results/eval_summary.json`: `tts_duration`, `baseline_duration`)

| system | total length ratio (geo-mean) | mean abs length error % | per-char duration MAE ms | per-char duration corr |
|---|---|---|---|---|
| learned TTS | 0.979 | 5.7 | 27.3 | 0.52 |
| baseline (template concat) | 0.948 | 6.7 | 43.0 | 0.06 |

The total-length columns are neutral. The per-character columns compare against MAS durations of the real mel under the
*trained TTS model* (model-derived pseudo-truth; no forced aligner is installed), so they favour the TTS's own segmentation
convention. The baseline also puts at least 1 frame in each of its 2 states (≥ 2 frames per character), while MAS gives
many 1-frame characters. Read its 0.06 correlation as "different convention", not "random durations". Duration pace was
fitted on val (shipped model: pace 0.959, ceil rounding; val length ratio 1.042 → 1.006). On test it comes out 2.1 % short.

## 4. Pronunciation: the hard-sentence set (`results/hard_sentences_eval.json`)

Twelve hand-written sentences (`hard_sentences.txt`) aimed at what a character-input TTS with a rule normalizer gets wrong:
numbers, currency, ordinals, years, clock times, abbreviations, irregular spellings and rare names. There are no recordings,
so scoring is recognizer-only. A "hard word" fails when its best substring match in the transcript has character error
above 1/3.

| system | hard words failed (char err > 1/3) | sentence CER mean % |
|---|---|---|
| learned TTS / mel, no vocoder | 16 / 53 | 17.0 |
| learned TTS / 28's neural vocoder | 18 / 53 | 17.0 |
| baseline (template concat) / mel, no vocoder | 20 / 53 | 26.6 |
| baseline (template concat) / 28's neural vocoder | 20 / 53 | 26.7 |

Per sentence (28's neural vocoder; the number after each failed word is how often it occurs in the training text):

| # | sentence | TTS CER % | TTS failed words | baseline CER % | baseline failed words |
|---|---|---|---|---|---|
| 1 | Dr. Smith paid $15.50 on March 3rd, 1963. | 6 | none | 15 | fifty cents (1) |
| 2 | The 21st Amendment passed in 1933 with 36 states. | 0 | none | 7 | none |
| 3 | Approximately 45% of 1,200 respondents disagreed. | 6 | none | 20 | approximately (72) |
| 4 | Mr. and Mrs. Featherstonehaugh visited Worcester and Leicester. | 25 | featherstonehaugh (0), worcester (0) | 24 | featherstonehaugh (0), worcester (0) |
| 5 | The colonel read the epitome of the yacht's itinerary. | 15 | yacht's (0) | 28 | epitome (0), yacht's (0) |
| 6 | Lieutenant Kowalczyk drove through Albuquerque, Schenectady, and Tucson. | 41 | lieutenant (10), kowalczyk (0), albuquerque (0), schenectady (0) | 48 | lieutenant (10), kowalczyk (0), albuquerque (0), schenectady (0) |
| 7 | Photosynthesis, mitochondria, and chlorophyll are biology words. | 30 | photosynthesis (1), chlorophyll (16) | 30 | photosynthesis (1), biology (5) |
| 8 | She sells seashells by the seashore, quite thoroughly. | 25 | sells (0), seashells (0), seashore (0), thoroughly (10) | 33 | seashore (0), thoroughly (10) |
| 9 | The quick brown fox jumps over the lazy dog. | 19 | quick (7), lazy (0) | 40 | brown (46), jumps (1) |
| 10 | Is the meeting at 10:30 or at 11:45? | 4 | none | 10 | none |
| 11 | The psychologist's rhythm was wholly unique, though not quixotic. | 27 | psychologist's (0), rhythm (0), unique (2) | 48 | psychologist's (0), rhythm (0), unique (2) |
| 12 | In 1066, William crossed the Channel with 7,000 knights. | 7 | none | 19 | knights (0) |

What this shows:
- **The normalizer gets 12/12 sentences right.** This is a deterministic check of `text.normalize`: "$15.50" → "fifteen
  dollars and fifty cents", "3rd" → "third", "10:30" → "ten thirty", "1,200" → "one thousand two hundred", "1066" → "ten
  sixty-six", and "Mr./Mrs./Dr." are expanded. The TTS read all five number sentences with 0–7 % CER and no failed number
  word. Numbers are a normalizer problem here, and the normalizer handles these cases.
- **TTS failures are rare and irregular words.** 12 of the 18 failed words never occur in the training text. The other 6
  occur 1–16 times: chlorophyll 16, lieutenant 10, thoroughly 10, quick 7, unique 2, photosynthesis 1. The transcripts show
  spelling-driven pronunciations: "feether stown ho" for Featherstonehaugh, "le tent colel sint" for Lieutenant Kowalczyk,
  "olpboquir senacty and toson" for Albuquerque, Schenectady and Tucson.
- **The hard set is small and noisy.** Run 1's model failed 17 words, the shipped model 18 (16 when its mel is fed to the
  recognizer directly). Individual words flip between checkpoints: colonel failed in run 1 and passes now, while
  "sells/seashells" now fail. With 53 words, a difference of 1–2 failures means nothing. TTS vs baseline is clearer on
  sentence CER (17 % vs 27 %) than on word failures (18 vs 20).
- **The measure scores spelling consistency, not correct pronunciation.** The recognizer outputs letters. A TTS that correctly
  said "kernel" for *colonel* or "wuster" for *Worcester* would be marked as a failure, and a spelling pronunciation would pass.
  For irregular words the pass/fail labels are unreliable in both directions.
- **Not measured:** human judgement of whether each hard word was pronounced acceptably.

## 5. Speed, UNDER LOAD (`results/bench.json`)

Other builders were training on the same Mac (`uptime` load averages below). The bench uses 10 test sentences (mean 6.8 s of
audio) and reports the median per sentence. Neural pipelines are best of 3 warm repeats; Griffin–Lim is a single run.
"First audio" = text → whole mel (the acoustic model is non-autoregressive) → vocoder on the first 71 frames (≈1.1 s of
audio).

`uptime` before: `3:16  up 4 days,  8:40, 2 users, load averages: 9.36 12.50 14.51`  
`uptime` after: `3:16  up 4 days,  8:40, 2 users, load averages: 9.54 12.43 14.46`

| pipeline | acoustic ms (median) | first audio ms (median) | full utterance ms (median) | RTF median | RTF max |
|---|---|---|---|---|---|
| tts/neural | 4.4 | 5.8 | 6.4 | 0.0009 | 0.0015 |
| tts/gl | 12.1 | 52.3 | 236.7 | 0.0340 | 0.0357 |
| baseline/neural | 0.9 | 2.8 | 3.5 | 0.0006 | 0.0007 |
| baseline/gl | 0.8 | 42.9 | 213.3 | 0.0323 | 0.0333 |

TTS + neural vocoder on the M5 Max GPU, warm: about 6 ms for a 7-second sentence, roughly 1,000× faster than real time.
Griffin–Lim on CPU dominates its pipelines. An earlier bench of the run-1 model under heavier load (load averages
12.8/15.9/22.3) gave 9.3 ms and RTF 0.0013 for the same pipeline. Its file was overwritten, so treat that figure as
anecdotal; `results/bench.json` holds the run above. Cold start (first call, kernel compilation) was excluded by a warm-up and is **not measured**.
Streaming / chunked synthesis was not built.

## 6. Baseline: what it is and how hard it was tuned (`results/baseline_tuning.json`)

A cheap classical system with no neural network and no help from the learned model. Training is flat-start Viterbi on
K-state-per-character mean-mel templates (6 passes over all 10,555 training clips, the same data as the TTS). Synthesis
concatenates the state means of the longest context seen often enough, with average durations. It was tuned on 100 val clips by
the same recognizer's CER (the main Measure; ties by DTW log-mel L1). **412 configurations over 3 rounds**; each round widened
the grid because the best setting sat on its edge:

| round | swept | best val CER |
|---|---|---|
| 1 | K 1–3; contexts: char only / 3-char ≥50 / 3-char ≥10; smoothing σ 0–5; 2 duration scales | 22.6 % (K 2, 3-char ≥10, σ 0) |
| 2 | K 2–3; 3-char ≥5 / ≥2; 5-char contexts ≥10 / ≥5 / ≥2; σ 0–1; 4 duration scales | **15.9 %** (K 2, 5-char ≥2, σ 0) |
| 3 | K 2; 5-char ≥1; 7-char ≥5 / ≥2 / ≥1; σ 0–0.5 | 15.9 % again (7-char variants: 16.1–16.6 %) |

- selected: K=2, context=quin_min2_tri2, dur_scale=0.985, sigma=0.0: val CER 15.9 %, val DTW L1 0.823, val length ratio 0.991
- selected_by_dtw_l1: K=2, context=sept_min5_quin2_tri2, dur_scale=1.085, sigma=0.5: val CER 17.0 %, val DTW L1 0.806, val length ratio 1.076

The selected baseline uses 55,933 five-character contexts, backing off to 3-char contexts and then single characters. Every
parameter ended at an interior optimum except smoothing σ = 0, which is the bottom of its range. Tuning took the baseline from
55.3 % (char only) to 15.9 % val CER; the TTS's val CER is 6.6 %.

Both systems were selected the same way: the TTS checkpoint (from 3 candidates) and the baseline configuration (from 412) were
each chosen on val CER with the same recognizer. Compute: baseline tuning ≈ 17 min (1,035 s, mostly CPU); TTS training 85 min
of GPU time over 2 runs. Data: identical (all training chapters).

## 7. Reviewer additions: timing vs acoustics, and the selection effect (`results/timing_split.json`, `results/review_step7000/`)

Added by the cross-reviewer (`timing_split.py`, a new file; nothing above was re-run). Same 100 test clips (ids read from
`eval_summary.json`), same recognizer and vocoder; the shelf baseline and shipped TTS rows reproduce eval.py's per-clip CERs
exactly. Question: how much of the TTS-vs-baseline gap, and of TTS-beats-real, is *timing* rather than *acoustics*? Each
system's mel is re-synthesized with borrowed per-character durations: the TTS duration predictor's (with its val-fitted pace), or
"oracle" MAS durations of the real test mel under the trained model (the pseudo-truth §3 uses). The baseline's K = 2 state
means get the borrowed frames per character (largest-remainder split in proportion to its own state durations; a 1-frame
character keeps only its longer state), or per character with ≥ 1 frame per state, or per *word* (word + following
space/punctuation gets the borrowed total, spread over its states in the baseline's own proportions). Only the TTS-duration rows
use no oracle information. The oracle rows are diagnostics. Nothing was selected on test, and the val column is a sanity check, not tuning.

| system | CER % | CER % with 28's vocoder | minus real mel, points [95% CI] | minus shelf baseline, points [95% CI] | length ratio | val CER % (first 100 val clips) |
|---|---|---|---|---|---|---|
| baseline, own durations (shelf baseline) | 16.8 | 18.4 | +10.1 [+9.1, +11.3] | n/a | 0.948 | 15.9 |
| baseline + TTS-predicted durations, per char | 25.9 | 27.9 | +19.3 [+17.9, +20.8] | +9.2 [+8.2, +10.3] | 0.979 | 24.8 |
| baseline + TTS durations, per char, >=1 frame/state | 25.9 | 27.5 | +19.2 [+18.0, +20.5] | +9.1 [+8.1, +10.0] | 0.991 | 24.1 |
| baseline + TTS durations, per word | 20.1 | 21.5 | +13.0 [+11.8, +14.2] | +2.8 [+2.0, +3.6] | 0.979 | 19.1 |
| baseline + oracle (real-mel MAS) durations, per char | 32.7 | 35.2 | +26.1 [+24.8, +27.6] | +16.0 [+14.9, +17.1] | 1.000 | 33.0 |
| baseline + oracle durations, per char, >=1 frame/state | 29.9 | 31.1 | +23.2 [+21.8, +24.8] | +13.1 [+12.0, +14.2] | 1.061 | 28.9 |
| baseline + oracle durations, per word | 22.3 | 24.0 | +15.3 [+13.9, +16.7] | +5.2 [+4.3, +6.0] | 1.000 | 22.2 |
| TTS, own durations (shipped) | 5.8 | 6.2 | -1.0 [-1.8, -0.2] | n/a | 0.979 | 6.6 |
| TTS decoder + oracle durations | 8.0 | 8.1 | +1.2 [+0.3, +2.2] | n/a | 1.000 | 9.1 |

What this says:
- **The baseline's gap is acoustic, not timing.** No borrowed timing helps it. The best variant, the TTS's word-level
  timing, is 2.8 points *worse* than its own (CI [+2.0, +3.6]), and the same holds on val. Its own durations were tuned
  jointly with its templates (state means estimated under its own Viterbi segmentation). MAS's character boundaries follow a
  different convention (many 1-frame characters), so per-character transfer hurts most. **The headline TTS − baseline gap
  (−11.2 points) stands.** A stronger cheap baseline would need better acoustics (e.g. real unit selection over stored
  frames, not state means), not better durations.
- **"TTS beats real" is mostly timing.** The same TTS decoder fed the real recording's durations scores 8.0 %, 1.2 points
  *worse* than the real mel (CI [+0.3, +2.2]) and 2.2 points worse than with its own predicted durations. The recognizer
  prefers the duration predictor's averaged, regular timing to the speaker's real timing. That is the proxy-bias reading of §1,
  now localized. Caveat: the decoder was trained on MAS durations, so oracle durations are in-distribution for it.
  The oracle durations are pseudo-truth from the trained model, not hand labels.
- **Selection effect, measured against the pre-registered rule.** The training loop's own rule (val mel + duration loss)
  applied to all 85 min of training picks step 7,000 (`runs/run2_best_export`). On the same test clips (no GL;
  `results/review_step7000/eval_summary.json`) it scores 6.6 % mel / 6.7 % neural CER, vs the shipped 5.8 / 6.2 %. Paired,
  shipped − step 7,000 = −0.74 [−1.22, −0.27] (mel) and −0.40 [−0.93, +0.08] (neural). Its TTS − real is −0.30 [−1.05, +0.53],
  and its TTS − baseline is −10.4 [−11.4, −9.5] (mel). So choosing the checkpoint by val CER on the scoring recognizer is worth
  about 0.4–0.7 CER points on test. That is enough to make "beats real" significant, and small next to the 10–11-point gap to the baseline.
  Hard set for step 7,000: 15/53 failed words (neural), vs 18/53 shipped.
- **Blind demo fixed.** `media/demo.html` showed the *system name* of the clip you marked "prefer" in its footer tally before
  reveal, and every label sat in the DOM (a hidden span) and in the page's JSON as plain text. The template now tallies letters only,
  stores labels base64-encoded, and writes them into the DOM only on "Reveal". `make_demo.py` regenerated the page, and all
  WAVs came out byte-identical. The shuffle is still one fixed permutation (seed 7) for every viewer, and the loose WAVs in
  `media/` carry system names in their filenames. Only the page itself is blind.

## What worked
- **MAS from scratch on character input.** No aligner and no phonemizer, yet held-out text came out fully legible to the
  recognizer proxy.
- **Borrowing 28's vocoder isolated the acoustic stage as planned.** Real mel → vocoder costs nothing measurable in CER, so the
  remaining gap (MCD-like 4.4 dB) belongs to the acoustic model.
- **Tuning the baseline hard mattered.** Val CER fell from 22.6 % to 15.9 %. With only round 1, the TTS would have looked about
  7 points better than it is.
- **Choosing the checkpoint on the Measure rather than the loss.** The step L1 preferred is 0.6 CER points worse on test.
  (*Reviewer:* the training loop's own val-loss pick, step 7,000, is 0.7 points worse, CI [−1.2, −0.3]. This gain is
  measured on the same recognizer that did the selecting, so it is partly "fit to the proxy". It also did not carry over to
  the hard set: step 7,000 fails 15/53 hard words through the neural vocoder, the shipped model 18/53.)

## What didn't
- **The intelligibility proxy saturates and is biased.** Synthetic speech beats real speech on it (significantly only for the val-CER-selected checkpoint; §7), so it can't rank quality at
  the top.
- **Rare and irregular words come out as spelled** (no G2P is installed).
- **Mel L1 overfits after ~4k steps** with this 23.9 M-parameter model on 19 h of audio. Annealing didn't fix the L1.
- **No pitch or energy predictor.** FastSpeech 2's variance adaptor beyond duration was not built; prosody is whatever the
  decoder infers from the characters.

## Surprises
- **The synthetic mel is more legible to the recognizer than the real recording** (5.8 % vs 7.0 % CER, CI excludes 0). Both
  models learned the same average speaker from the same data, and the most "typical" rendering wins. That doesn't make it
  better speech. (*Reviewer:* the CI excludes 0 only for the val-CER-selected checkpoint. With the real recording's timing
  forced in, the TTS loses to real, 8.0 % vs 7.0 %, so "typical" here is mostly typical *timing*; §7.)
- **Overfitting in L1 made the model more intelligible.** Run 2 got worse in val mel L1 but better in val and test CER, and
  less oversmoothed.
- **Griffin–Lim scores as well as the neural vocoder on the proxy** and has the lower DTW L1 (0.101 vs 0.143). GL iterates to
  match the target magnitude, so re-analysing its waveform recovers that mel almost exactly. Mel-domain metrics flatter it,
  because GL's typical phase artefacts barely change the magnitude mel. Whether listeners hear it as worse is not measured.
- **Classic concatenation got much stronger with 5-character contexts** (22.6 → 15.9 % val CER), and nothing longer helped.

## Not measured
- Listening preference: no listeners. `media/demo.html` is ready for a blind test.
- Human judgement of pronunciation on the hard set.
- Cold-start latency, streaming latency, and speed on an idle machine (all speed numbers are under load).
- Intelligibility with a real ASR system (none is installed).
- A self-trained vocoder and expressive controls ("Go deeper").

## Next steps
1. Pitch and energy predictors (needs an F0 extractor; a simple autocorrelation tracker in numpy would do), then expressive
   controls.
2. A less biased intelligibility proxy: train the recognizer on chapters the TTS never saw, so it stops rewarding "typical".
3. Fight the oversmoothing: a postnet, or adversarial/flow losses on the mel. MCD-like 4.4 dB is the number to move. Also
   regularize, or stop on CER rather than L1, since L1 overfits early.
4. A small exception lexicon for irregular words, which character input can't learn from 19 h of audio.

---

# Follow-up (2026-10-31): flow matching, pitch/energy, 337 h of multi-speaker pretraining, and a vocoder check

Goal: the best tiny single-speaker (LJSpeech) non-streaming TTS we can make on this Mac, keeping v1's MAS alignment,
duration predictor, splits and its exact 100 test clips. All tables below are printed by `report_fm.py` from
`results/fm/{judge,mos,proxy}/*.json` (raw printout: `results/fm/tables.md`). Nothing above this line was re-run or edited.

## Follow-up headline (test, 27's 100 clips; every system scored once after selection on val)

| system | Whisper CER % [95% CI] | Whisper WER % | UTMOS [95% CI] |
|---|---|---|---|
| real recording | 0.54 [0.23, 0.92] | 1.60 | 4.40 [4.38, 4.41] |
| real mel → 28's vocoder (vocoder ceiling) | 0.43 [0.18, 0.72] | 1.54 | 2.59 [2.53, 2.65] |
| 27 v1 (FastSpeech-style, L1) → 28's vocoder | 1.14 [0.69, 1.70] | 2.84 | 1.95 [1.88, 2.01] |
| **B: flow matching, LJ only → 28's vocoder (shipped)** | 1.32 [0.88, 1.78] | 3.67 | **2.96 [2.90, 3.01]** |
| C: flow matching, 337 h pretrain + LJ fine-tune → 28's vocoder | 1.49 [0.98, 2.06] | 3.67 | 2.95 [2.89, 3.01] |
| Gemma Voice (project 33), best-of-4, Mimi 24 kHz | 2.45 [1.76, 3.20] | 5.62 | 2.35 [2.29, 2.40] |

Paired against 27 v1 (same clips): B is **+1.01 UTMOS [+0.95, +1.08]** at CER +0.17 points [−0.33, +0.65]; C is +1.00
[+0.94, +1.06] at CER +0.34 [+0.00, +0.71]. B vs C: CER −0.17 [−0.60, +0.25], UTMOS +0.01 [−0.04, +0.06], a tie. Both beat
Gemma Voice on both measures (CER −1.1 to −1.0 points, UTMOS +0.6). Whisper intelligibility sits at the ceiling for every
27-family system: the gaps to the real recording (0.8–1.0 points) are a handful of words per 100 clips.

*Reviewer (REVIEW_FOLLOWUP.md):* three qualifiers. (1) **+1.01 is after tuning the sampler to UTMOS.** At the pre-registered
default sampler (Euler 10, temp 0.667) the same B checkpoint is **+0.93 [+0.86, +1.00]** over v1 on test, and the run's
unselected last step (17,719) is +0.94 [+0.88, +1.00]. Choosing temp 0.5 by val UTMOS bought +0.08 [+0.07, +0.10] on test
(`review_fm.py` → `results/fm/review_tables.md`). (2) **WER is not level.** Paired WER: B − v1 +0.83 [+0.00, +1.65], C − v1
+0.83 [+0.12, +1.51] (62 vs 48 word errors in 1,691 words). Against the real recording it is 62 vs 27 word errors, i.e. 35
more per 100 clips, which is more than "a handful". (3) Paired against Gemma: B CER −1.13 [−1.85, −0.46], WER −1.95
[−3.33, −0.64], UTMOS +0.61 [+0.54, +0.68]; C CER −0.96 [−1.65, −0.26]. So "beats Gemma on both" holds.

What the follow-up says, in short:
- **The modern decoder is what moved naturalness.** A Matcha-style flow-matching decoder on 27's own MAS/duration
  front end lifts UTMOS by about 1.0 over v1, through the same vocoder, with intelligibility unchanged within noise.
  (*Reviewer:* B differs from v1 in four ways at once: the decoder, the pitch/energy predictors, size (44 M vs 24 M) and
  training (~70 vs 85 min, a different LR schedule). No ablation separates them, so "the decoder did it" is the likely
  reading but it is not measured. "Unchanged within noise" is true for CER only; WER is +0.83 points, see the note above.)
- **Pretraining on 337 h of other speakers did not measurably help this single-speaker model.** It helped the pretrained
  model's LJ val loss and gave a small, consistent val CER edge (9/9 matched sampler settings), but that edge did not
  survive on test, and UTMOS is tied.
- **28's vocoder is the naturalness ceiling, and part of the new model's win is that it is easier on that vocoder.**
  Real mel → 28's vocoder scores 2.59, far below the recording (4.40), and B (2.96) scores *above* it. A short GAN fine-tune of
  the vocoder (approved add-on, failed its pre-agreed gate) lifts real mel and v1 but not the flow-matching models, which
  halves their margin over v1 to +0.52 (§F5).
- **UTMOS is a proxy with visible biases here** (it rates a TTS above the real mel through the same vocoder), so the
  naturalness ranking needs listeners: `media/followup.html` is the blind page. Listener preference: **not measured**.

## F1. What was built

| piece | file | notes |
|---|---|---|
| data | `prep_multi.py`, `pitch.py`, `prep_wav16.py` | 27's exact mel (`mel.py`) + a numpy normalized-autocorrelation F0 tracker aligned to mel frames, for LJ (27's split and keep rule; mels bit-identical to `cache/`), LibriTTS-R train-clean-100 (1–12 s; 6 held-out speakers as in 33), Hi-Fi TTS (train manifests, 1–12 s). Streamed to memory-mapped fp16 in `cache2/` (12 GB, 11.7 min on 4 processes). |
| model | `model_fm.py` | 44.0 M params. Char encoder (6 FFT blocks, d 384, 4 heads, conv-FFN k 5) + speaker embedding → prior μ; **27's MAS** on log N(x; μ, I) with v1's diagonal prior for the first 2k steps; **27's duration predictor** (log-duration MSE); FastPitch-style per-character pitch (speaker-normalized log-F0 over voiced frames) and energy (utterance-centred log mel energy) predictors, teacher-forced in training; **OT-CFM decoder** (Matcha-TTS loss, σ_min 1e-4): 6-layer frame transformer, d 384, adaLN-zero conditioning on flow time + speaker, input = [x_t, μ aligned, encoder output aligned]. Euler / midpoint ODE sampling. |
| training | `train_fm.py` | weights + optimizer + step + epoch + batch position + seeds in each checkpoint; fp16 snapshots every 2k steps; MLX capped at 16 GB / cache 2 GB in every process (`model_fm.cap_memory`). Resume verified before the long runs (STATUS 00:20). |
| export / eval | `finalize_fm.py`, `eval_fm.py`, `score_fm.sh`, `mos.py`, `report_fm.py`, `hard_fm.py`, `endings_fm.py` | pace fitted on the first 100 val clips (v1's rule); generation seeded per clip; Whisper judge = `../33_gemma_voice/judge.py` run as-is in `../.venv-judge`; UTMOS22 in `../.venv-sac` |
| vocoder | `train_voc_gan.py`, `voc_check.py` | GAN fine-tune of 28's generator (MPD + MRD, LSGAN + feature matching + 45 × mel L1); multi-speaker vocoder check |
| demo | `make_demo_fm.py`, `followup_template.html`, `synth_fm.py` | `media/followup.html`, `media/followup_*.wav` |

Data actually used: LJ train 19.1 h (10,555 clips), LibriTTS-R 37.4 h train (241 speakers; 0.75 h / 6 speakers held out),
Hi-Fi TTS 280.8 h (310,850 clips, 10 speakers): 337.2 h, 252 speakers in pretraining. This is the same data behind project
33's scaled run; nothing else is on disk (the rest of LibriTTS-R would need a download).

Compute (one GPU job at a time through `../bin/gpu_run`; AC power): B = 2 × 45 min on LJ (B2 resumed with LR annealed to 10%,
as v1's run 2); C = 2 × 45 min multi-speaker pretraining (19,866 steps, ≈2.6 passes over 337 h) + 45 min LJ fine-tune
(LR 2e-4, fresh optimizer). Steps/s 2.3–3.7 (44 M params, 12k mel frames per batch, peak 6.5 GB). v1 had 85 min.
(*Reviewer:* the multi-speaker log peaks at 6.9 GB. Compute is not matched: C's selected model had 90 + 18 min against B's
~70 min to step 14,000, which favours C, so the "no gain from pretraining" null is if anything conservative.)

## F2. Measurement: what was added and why

- **Whisper judge** (33's `judge.py`, large-v3-turbo, temperature 0): reproduces the brief's numbers exactly on the reference
  lists (real 0.54/1.60, real mel → voc28 0.43/1.54, v1 1.14/2.84, Gemma bo4 2.45/5.62), and v1 regenerated from its
  checkpoint here reproduces 1.14/2.84 and v1's CTC-proxy numbers (5.8/6.2 %, MCD-like 4.60 dB).
- **UTMOS22-strong** (VoiceMOS 2022 winner, 16 kHz native; torch.hub `tarepan/SpeechMOS:v1.2.0`, approved by the user, run on
  CPU in the existing `../.venv-sac`, nothing installed). Per-clip scores, bootstrap CIs; paired differences resample the same
  clips. It is a *proxy* for MOS trained on other systems' listening tests; it has not been checked against listeners here.
- **Selection (pre-registered in STATUS.md at 00:25, before any FM number):** candidates = 2k-step snapshots of each
  system's last run + its end; scored on val-100 (33's seed-0 sample of LJ041–044) with Euler 10 / temp 0.667; pick max val
  UTMOS among candidates within 0.3 Whisper-CER points of the best; then the sampler grid {Euler 10, Euler 32, midpoint 10} ×
  temp {0.5, 0.667, 0.8} on that checkpoint, same rule. Test was scored once per system afterwards. The same rule picked
  B over C for shipping (val CER 0.98 vs 0.89, both within 0.3 points; UTMOS 3.015 vs 3.014). That is a coin flip, and C is shipped too. *Builder, after the review:* applying the rule *across* systems to pick the shipped one was decided after the test numbers
  were already known (B 1.32 vs C 1.49). It was not part of the pre-registered per-system rule, so read "B is shipped" as a
  convenience, not a result.
- **Caveat: double use of the predictor.** Checkpoints and sampler settings were chosen by val UTMOS, and UTMOS is the
  headline naturalness number. Test is held out, but the system is tuned *to UTMOS*. That is the same kind of selection effect the
  v1 reviewer measured for the CTC proxy (0.4–0.7 points). The selection rule mostly moved temperature (below).
- *Reviewer:* the STATUS.md line holding the rule is timestamped **00:22**, not 00:25, and the resume check is the 00:18 line,
  not 00:20. Both still come before the first FM number (01:08). I re-derived every pick from `tables.md`, and the rule
  reproduces them. Two caveats. (a) The 0.3-point CER tolerance hardly binds. It excludes only C s2000 (1.32 > 1.15), which
  also has the lowest UTMOS. Every other checkpoint candidate and every sampler setting is inside it. So in practice the
  rule was "maximum val UTMOS". UTMOS rises monotonically as
  temperature falls, so the chosen **temp 0.5 is the floor of the grid** (the same "optimum at the grid edge" caveat
  REVIEW.md raised for σ = 0). (b) The STATUS rule covers choices *within* a system. Using it to pick B over C for shipping
  is an extension that is not written in STATUS before test. It is harmless here because UTMOS differs by 0.001. The
  measured test-side cost of selection: B at the default sampler is −0.084 [−0.099, −0.069] UTMOS below the shipped
  setting. CER is +0.45 [−0.01, +1.27], almost all from one clip where Whisper skipped the first 3 s of near-identical
  audio (LJ046-0005, 35 of 174 edits). Selection is worth about 0.08 of the 1.0 UTMOS gain.

## F3. Selection on val (`results/fm/tables.md`, val sections)

Default sampler (Euler 10, temp 0.667), val-100, Whisper CER % / UTMOS:

| | B (LJ only) | | C (pretrain + fine-tune) |
|---|---|---|---|
| B1 end, step 8,369 (45 min) | 0.83 / 2.83 | pretrained only (P2 end, LJ speaker id, no fine-tune) | 1.27 / 2.53 |
| step 10,000 | 0.86 / 2.92 | fine-tune step 2,000 | 1.32 / 2.82 |
| step 12,000 | 0.86 / 2.90 | step 4,000 ← chosen | 0.85 / 2.93 |
| step 14,000 ← chosen | 0.95 / 2.93 | step 6,000 | 0.99 / 2.85 |
| step 16,000 | 0.98 / 2.92 | step 8,000 | 0.87 / 2.80 |
| step 17,719 (end) | 0.93 / 2.93 | step 9,819 (end) | 1.10 / 2.86 |
| 27 v1 (reference) | 0.96 / 1.92 | real mel → 28's vocoder | 0.27 / 2.58 |

- **The decoder learns fast.** After 13 min (step 2,000) B already scored UTMOS 1.84 on 3 clips; after 45 min it beat v1 by
  0.9 on val. Val UTMOS plateaus at ~2.92 from step 10k on; later snapshots trade CER noise.
- **The fine-tune is necessary.** The pretrained model speaking as speaker "lj" is 0.4 UTMOS worse than after 18 min of LJ
  fine-tuning. Its LJ val CFM loss (0.365 after 90 min) was above B1's (0.346 after 45 min); fine-tuning reached 0.334, below
  B2's 0.339.
- **Sampler:** lower temperature raises UTMOS and costs a little CER, for both systems (B: t0.5 3.02 / t0.667 2.93 / t0.8
  2.84; CER 0.98 / 0.95 / 0.92). 32 Euler steps or midpoint (20 NFE) are no better than 10 Euler steps. The rule chose Euler
  10, temp 0.5 for both. C has lower val CER than B at **all 9** matched settings (by 0.09–0.24 points), which is the only
  signal for pretraining; on test it flips (B 1.32 vs C 1.49, CI spans 0).
  *Reviewer:* the 9 settings share the same 100 clips and the same two checkpoints. They are one comparison seen 9 times,
  not 9 independent signals. On test at the default sampler, C is again below B (1.53 vs 1.76, −0.23 [−1.12, +0.45]), so the
  test "flip" depends on the sampler too. Read the whole B-vs-C CER question as noise.

## F4. Test proxies and spectra (`results/fm/proxy/test_*.json`)

| system | 27's CTC proxy CER %: mel / vocoded | DTW log-mel L1 | MCD-like dB | GV ratio | length ratio | gen time per clip, median |
|---|---|---|---|---|---|---|
| real mel → 28's vocoder | 7.0 / 7.1 | 0.143 | 0.63 | 1.000 | 1.000 | 2.5 ms (vocoder only) |
| 27 v1 | 5.8 / 6.2 | 0.836 | 4.60 | 0.939 | 0.979 | 8 ms |
| B | 6.4 / 6.2 | 0.838 | 4.66 | 0.936 | 0.983 | 57 ms (RTF 0.009) |
| C | 7.2 / 7.3 | 0.834 | 4.55 | 0.918 | 0.987 | 51 ms (RTF 0.008) |

- **Mel distances did not move.** The new models are as far from the recording as v1 in DTW L1 and MCD-like dB, yet ~1.0
  UTMOS better. A flow-matching decoder samples a plausible spectrum rather than the conditional mean, so it adds natural
  texture without getting closer to *this* recording. v1's §1 already warned that MCD-like is not the number to optimize.
- **27's CTC proxy prefers v1** (5.8 % vs 6.4 %); it was trained on the same speaker and rewards v1's averaged rendering, as
  v1's reviewer found. Whisper says they are level.
- **Speed:** 10 ODE steps + vocoder take ~55 ms for a 6.6 s clip (≈110× real time; measured while the GAN vocoder was
  training, so under load). v1 is ~7× faster; both are far faster than real time.
  *Reviewer:* "under load" was worth about 2×. The same B checkpoint and settings, re-run on a quieter machine (load ≈ 3,
  `results/fm/proxy/test_review_fm_lj_t0667.json`), take **30 ms median (RTF 0.0045, ≈ 220× real time)**. The builder's
  own GAN-vocoder test runs, made after the GAN job ended, agree: B 30 ms, v1 5.5 ms, so v1 is about 5× faster.

## F5. The vocoder is the ceiling (`voc_check.py`, `train_voc_gan.py`)

28's vocoder was trained for ~15 min without any adversarial loss. UTMOS shows what that costs: real recording 4.40 → its
mel through 28's vocoder 2.59 → through Griffin–Lim 64 2.03 (test). Intelligibility is untouched (0.43 % CER).

**Multi-speaker check (20 real clips per source, real mel → 28's vocoder):** LJ val 4.41 → 2.54, LibriTTS-R held-out speakers
4.32 → 2.36, Hi-Fi TTS 3.87 → 1.94; Whisper CER unchanged (0.0 → 0.0, 0.68 → 0.91, 0.08 → 0.08). The vocoder costs ~1.9
UTMOS on *every* speaker, LJ included, so its problem is quality, not speaker generalization; the shipped model only speaks
LJ, so only LJ matters for the final system.

**GAN fine-tune (user-approved add-on, pre-agreed gate: real mel → new vocoder ≥ 3.0 val UTMOS after 45 min, else stop).**
28's 7.7 M generator + new 3.0 M discriminators (5 periods + 3 STFT resolutions, narrower than HiFi-GAN's), LJ train, 1 s
segments, batch 16, 4,754 steps in 45 min. Val real mel: 2.58 → 2.57 (step 2k) → **2.73 [2.68, 2.78] (end): gate failed,
vocoder training stopped.** Scored anyway as a partial result (test, paired, same mels):

| mels | 28's vocoder | GAN fine-tune | difference |
|---|---|---|---|
| real mel | 2.59 | 2.69 | +0.11 [+0.06, +0.15] |
| 27 v1 | 1.95 | 2.43 | **+0.48 [+0.44, +0.52]** |
| B | 2.96 | 2.95 | −0.01 [−0.06, +0.04] |
| C | 2.95 | 2.95 | +0.01 [−0.04, +0.06] |

Whisper CER is unchanged within noise for every row (v1 +0.17 [+0.00, +0.44]).
- **The vocoder fix and the acoustic fix overlap.** Whatever the GAN repaired (most likely the buzzy rendering of v1's
  oversmoothed harmonics) is something the flow-matching mels never triggered. Through the better vocoder the new models' margin over v1 is
  **+0.52 [+0.46, +0.58]**, half the +1.0 seen through 28's vocoder. Some of the headline gain is "the flow-matching model is easier on a weak vocoder";
  half of it survives a better vocoder.
  *Reviewer:* "a better vocoder" overstates this. The GAN snapshot is only +0.11 better on *real* mels (and failed its gate).
  Its gain is concentrated on v1's mels (+0.48), so it is mostly a vocoder that repairs v1's particular artifact, not a
  generally better one. What the margin would be through a good vocoder (real mel → ≥ 3.5) is **not measured**. It could
  be anywhere from about +0.5 to +1.0. Likewise "ceiling" is loose, since B scores above real mel → 28's vocoder. Read it as
  "the main bottleneck": nothing through 28's vocoder gets above ~3.0 while the recording is 4.40.
- **UTMOS ranks B (2.96) above the real mel through the same vocoder (2.59).** That is not better-than-real speech; it says
  this vocoder renders a real spectrum's fine detail worse than a model's sampled one. Read UTMOS differences between systems
  with *different* acoustic sources and the same weak vocoder with care.
- On val, the GAN snapshot was slightly worse than 28's vocoder for B and C (2.95 vs 3.02), so the shipped system keeps 28's
  vocoder; the GAN snapshot stays in `runs/voc/` (gitignored) as a starting point, not a deliverable.

## F6. Pronunciation and a learned failure (`results/fm/hard_summary.json`, `results/fm/endings.json`)

- **Hard sentences, Whisper-scored (12 sentences, 53 hard words, same rule as §4):** v1 9 failed / 12.8 % CER, B 8 / 12.1 %,
  C 8 / 10.3 %. Four failures are shared by every system and are scorer noise: Whisper writes "$15.50", "3rd", "1,200" as
  digits and the judge expands them differently from 27's normalizer. Whisper scores real spellings ("colonel" for a /kernel/
  sound), so unlike the CTC proxy it rewards correct pronunciation. On 53 words, 8 vs 9 means nothing. Character input still says
  "featherstonehaugh", "kowalczyk", "quixotic" as spelled.
  *Reviewer correction (`hard_summary.json`):* **five** failures are shared by every system, not four. "thirty" also fails
  everywhere because Whisper writes "10:30" as "10.30", which the judge expands to "ten point three". That leaves v1 4, B 3
  and C 3 real failures out of 48 scorable words. The spelled-out readings are B's: B fails featherstonehaugh, kowalczyk and
  quixotic. v1 fails kowalczyk, albuquerque, schenectady and quixotic. C gets all three of B's words right and fails colonel,
  yacht's and tucson instead.
- **Sentence endings (good kind of failure).** LJSpeech's reader says "end quote" after quotations. On val, 12 clips end with it:
  real 0/12 dropped by Whisper, v1 2/12, B 4/12, C 5/12 (chosen settings), *pretrained-only 1/12*. On test (5 such clips):
  real 1, v1 2, B 1, C 3. The flow-matching models learned to swallow this formulaic, low-energy coda from LJ itself (the
  multi-speaker model, which saw little LJ, keeps it), and higher temperature drops it less often (C with Euler 10: 5 → 4 → 3 of 12 at t
  0.5/0.667/0.8). It accounts for a large share of the CER gap to real speech. Small counts; a listener check would settle it.
  *Reviewer:* the "large share" holds on **val only**. Counting Whisper character edits in end-quote clips: on val, B has 77
  more edits than the recording in total and 42 of them are in end-quote clips; for C it is 55 of 68. On **test**, B drops
  "end quote" in 1 of 5 clips, the same as the real recording, and only 9 of its 77 extra edits are in those clips (C: 25
  of 94). Excluding end-quote clips, test CER is real 0.41, v1 0.93, B 1.14, C 1.15, so the test gap to real is mostly
  elsewhere.

## F7. What worked, what didn't, next steps

Worked:
- Swapping v1's L1 decoder for an OT-CFM decoder on the same MAS + duration front end: +1.0 UTMOS through the same vocoder
  (+0.5 through a better one), same intelligibility, trained in 45–90 min.
- The data pipeline: 337 h of three corpora in 27's exact mel space in 12 minutes, memory-mapped so training stays at 6.5 GB.
- Measuring the vocoder separately (it caps everything at ~2.6 UTMOS) before attributing gains to the acoustic model.

Didn't:
- **Multi-speaker pretraining** (90 min on 337 h) gave no test gain over 90 min of LJ only. At this model size and
  budget, 19 h of one speaker is enough; the pretraining budget was ~2.6 passes, so a longer run, a bigger model, or a
  phoneme front end might change this. Not tested.
- **GAN vocoder fine-tune** failed its gate (2.73 < 3.0). The discriminators have about a tenth of HiFi-GAN's parameters, the run was
  4.8k steps, and the generator kept its non-adversarial init. A full-size discriminator and several hours are the next
  thing to try; 28 is the natural home for that.
- Pitch/energy predictors were always on (no ablation was run: no GPU budget left for a third arm), so their contribution is
  **not measured**.

Not measured: listener preference (blind page ready), pitch/energy ablation, speed on an idle machine, Whisper on
streaming/chunked output (non-streaming only).

Next: (1) take the blind page `media/followup.html`, since UTMOS alone can't rank these; (2) a real GAN vocoder at full
discriminator size, then re-rank; (3) the pitch/energy ablation; (4) the "end quote" fade: per-character duration check
on codas, or condition the sampler's temperature on position.
