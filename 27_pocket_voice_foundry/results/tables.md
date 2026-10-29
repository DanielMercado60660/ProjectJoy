### Held-out test, 100 clips (TTS checkpoint step 12911)

| mel source | output | CER % (CTC proxy) [95% CI, clip mean] | WER % | DTW log-mel L1 | MCD-like dB |
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

### Durations on the same test clips (reference = MAS of the real mel under the trained model: model-derived pseudo-truth)

| system | total length ratio (geo-mean) | mean abs length error % | per-char duration MAE ms | per-char duration corr |
|---|---|---|---|---|
| learned TTS | 0.979 | 5.7 | 27.3 | 0.52 |
| baseline (template concat) | 0.948 | 6.7 | 43.0 | 0.06 |

### Hard-sentence set (hard_sentences.txt, recognizer-only, no recordings)

| system | hard words failed (char err > 1/3) | sentence CER mean % |
|---|---|---|
| learned TTS / mel, no vocoder | 16 / 53 | 17.0 |
| learned TTS / 28's neural vocoder | 18 / 53 | 17.0 |
| baseline (template concat) / mel, no vocoder | 20 / 53 | 26.6 |
| baseline (template concat) / 28's neural vocoder | 20 / 53 | 26.7 |

### Run 1 (step 6000) vs shipped (step 12911), same 100 test clips

| output | run-1 CER % | shipped CER % | paired diff, points [95% CI] | run-1 DTW L1 | shipped DTW L1 | run-1 MCD-like | shipped MCD-like |
|---|---|---|---|---|---|---|---|
| mel, no vocoder | 6.3 | 5.8 | -0.60 [-1.04, -0.16] | 0.723 | 0.740 | 4.36 | 4.39 |
| 28's neural vocoder | 6.8 | 6.2 | -0.72 [-1.41, -0.12] | 0.827 | 0.836 | 4.57 | 4.60 |
| Griffin-Lim 64 | 6.4 | 6.0 | -0.59 [-1.05, -0.13] | 0.729 | 0.746 | 4.39 | 4.43 |

Global-variance ratio (TTS mel / real): run 1 0.905, shipped 0.939. Duration: run-1 length ratio 0.974, shipped 0.979.
- val (100 clips) `runs/run1_tts_export/tts.safetensors` step 6000: CER 7.36 %, DTW L1 0.7136, length ratio 1.001
- val (100 clips) `runs/run2_best_export/tts.safetensors` step 7000: CER 7.51 %, DTW L1 0.7162, length ratio 1.005
- val (100 clips) `runs/run2_final_export/tts.safetensors` step 12911: CER 6.56 %, DTW L1 0.7260, length ratio 1.006

### Hard sentences one by one (28's neural vocoder; failed hard words with their count in the training text)

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

### Speed, UNDER LOAD (other builders training concurrently)

`uptime` before: `3:16  up 4 days,  8:40, 2 users, load averages: 9.36 12.50 14.51`  
`uptime` after: `3:16  up 4 days,  8:40, 2 users, load averages: 9.54 12.43 14.46`

| pipeline | acoustic ms (median) | first audio ms (median) | full utterance ms (median) | RTF median | RTF max |
|---|---|---|---|---|---|
| tts/neural | 4.4 | 5.8 | 6.4 | 0.0009 | 0.0015 |
| tts/gl | 12.1 | 52.3 | 236.7 | 0.0340 | 0.0357 |
| baseline/neural | 0.9 | 2.8 | 3.5 | 0.0006 | 0.0007 |
| baseline/gl | 0.8 | 42.9 | 213.3 | 0.0323 | 0.0333 |

### Baseline tuning on val (100 val clips, 412 configs over 3 rounds; min val CER (CTC recognizer on the baseline mel), tie -> val DTW log-mel L1)

- selected: K=2, context=quin_min2_tri2, dur_scale=0.985, sigma=0.0: val CER 15.9 %, val DTW L1 0.823, val length ratio 0.991
- selected_by_dtw_l1: K=2, context=sept_min5_quin2_tri2, dur_scale=1.085, sigma=0.5: val CER 17.0 %, val DTW L1 0.806, val length ratio 1.076

### Timing vs acoustics (reviewer diagnostic, timing_split.py; same 100 test clips; paired differences on mel, no vocoder)

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
