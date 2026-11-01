# Results: Gemma Voice

**Status: finished 2026-11-01.** Built over three rounds (2026-10-30 to 11-01) on the M5 Max. Every number below comes from a
file in `results/`, a log in `logs/` or an entry in `STATUS.md` (kept on disk, not published). One training seed per
configuration.

## Headline

Held-out test: project 27's 100 clips (LJ045–LJ050). Whisper large-v3-turbo transcribes; UTMOS22 predicts naturalness
(27's `mos.py`). Each model was scored on test once per decoding mode (r5 twice: single take and best-of-4, both planned),
at sampling settings chosen on val.

| system | Whisper CER % | WER % | UTMOS [95% CI] |
|---|---|---|---|
| real recording | 0.54 | 1.60 | 4.40 |
| Mimi latent round trip, the target this model predicts (**val-100**, not test) | 0.38 | 1.23 | 4.28 |
| **f3: Gemma 3 270M + flow head on Mimi latents** | **1.05** | **2.66** | **3.20 [3.14, 3.26]** |
| f2: same, before the extra 364 h of data | 1.10 | 2.78 | 3.13 |
| project 27 v2, shipped (23.9M flow-matching mel model + project 28's vocoder) | 1.32 | 3.67 | 2.96 [2.90, 3.01] |
| project 27 v1 (FastSpeech + 28's vocoder) | 1.14 | 2.84 | 1.95 |
| r5: Gemma predicting 4 Mimi codebooks (round 2's best) | 3.63 | 7.21 | 2.30 (val) |
| r3: round 1 (LJ only, 15 min) | 13.03 | 23.83 | not measured |

Paired on the same 100 clips (`compare_27.py`, `results/test_vs_27.json`, bootstrap 95% CI; positive = f3 better):

| f3 vs | UTMOS | clip-mean CER (points) | clip-mean WER (points) | f3 higher UTMOS on |
|---|---|---|---|---|
| 27 v2 shipped | **+0.24 [+0.17, +0.32]** | +0.22 [−0.26, +0.69] | +1.04 [+0.05, +2.12] | 75 / 100 clips |
| 27 v1 | +1.25 [+1.17, +1.34] | +0.34 [−0.38, +1.25] | +0.66 [−0.55, +2.33] | 100 / 100 |
| f2 | +0.07 [+0.02, +0.13] | −0.06 [−0.45, +0.27] | −0.16 [−1.04, +0.47] | 54 / 100 |
| real recording | −1.20 [−1.26, −1.13] | −0.70 [−1.26, −0.21] | −1.52 [−2.60, −0.59] | 0 / 100 |

- **A text-only 270M language model learned to talk, and by UTMOS it is the most natural voice on the shelf** (no other
  project's test result scores above 3.20; only 27 has UTMOS numbers). It beats project 27 on naturalness by a margin
  outside the noise. On intelligibility it is level with 27: the CER difference is within noise, and WER is just ahead
  (the interval barely excludes zero).
- **The move that mattered: a flow-matching head on Mimi's continuous latent instead of discrete codebooks** (§3).
  Predicting 4 Mimi codebooks caps naturalness at that format's own round trip (UTMOS 3.11), and r5 reached 2.30. Swapping
  the codebook heads for a 28M flow head that writes the latent the codebooks quantize raised the ceiling to 4.28. Starting
  from r5's backbone, it reached the old cap after about 3.9 h of flow training (f2: val 3.12 vs the cap's 3.11, within
  noise) and passed it with f3 (test 3.20; there is no test UTMOS for the 4-codebook round trip).
- **More data and training helped a little, at most.** f2 → f3 added 364 h of LibriTTS-R (2,057 new speakers) and 5.1 h
  of training: +0.07 UTMOS on test, no change on val (3.12 both), and no change in intelligibility. Data and steps were
  not separated (§4).
- **Not done: closing the gap to the ceiling.** f3 is at 3.20 against the latent round trip's 4.28. Listening agrees: the
  round trip sounds close to the real recording, and f3 does not (an informal listen, not a blind test). The codec isn't the
  limit; the model's predictions are. Extrapolating (not measured): at f2 → f3's rate, more data alone is unlikely to
  close a 1-point gap on this machine (§8).
- **Speed** (batch 1, KV cache, bf16): generation RTF 0.19 with guidance, which needs two backbone passes per frame (0.11
  without, measured on f1; same architecture). The first frame is ready 24 ms after the call. Mimi decoding (RTF 0.006–0.007) runs after generation in a separate
  process, so end-to-end streaming latency was not measured.
- **Pace:** with guidance, speech comes out about 7% faster than the recording (median length ratio 0.93).
- **Model size:** 130M trained parameters (100.3M backbone, 28.0M flow head, 1.5M speaker table, 0.3M input projection).
  Gemma's 168M text embedding stays frozen. Training ran about 12 h in total: r5 3 h, f1 1 h, f2 2.9 h, f3 5.1 h.
  Data: about 700 distinct hours from 2,309 training speakers (LJSpeech, LibriTTS-R clean-100/360/other-500, Hi-Fi TTS;
  the speaker table has 2,315 rows, 6 of them for held-out clean-100 val speakers).

## 1. The path (val-100, Whisper CER / WER %, UTMOS)

| run | what changed | train time | CER / WER | UTMOS |
|---|---|---|---|---|
| r3 | Gemma 3 270M → 4 delayed Mimi codebooks, LJ only (round 1, appendix) | 15 min | 12.70 / 22.83 | – |
| r4 | + LibriTTS-R clean-100 (38 h, 247 speakers), speaker table | 23 min | 6.52 / 12.37 | – |
| r5 | + Hi-Fi TTS (283 h, 10 speakers), LJ ×4, step 16,000 of 18,916 | 3 h | 2.00 / 4.14 | 2.30 |
| f1 | flow head on Mimi latents, initialized from r5, LJ ×4 | 1 h | 0.66 / 1.79 | 2.80 |
| f2 | continue f1, LJ ×2, 11k-step cosine | 2.9 h | 0.57 / 1.40 | 3.12 |
| f3 | continue f2, + LibriTTS-R clean-360 + other-500 (364 h, 2,057 speakers) | 5.1 h | 0.61 / 1.62 | 3.12 |

f1–f3 are sampled at temperature 0.4 with guidance 2.0; r3–r5 at their best temperatures. No flow run produced a runaway
clip (frame cap) or an early stop on val or test. On val, f3's snapshots score 3.11 / 3.17 / 3.19 / 3.18 UTMOS at steps
5k / 10k / 15k / 20k and 3.12 at the end. Differences of about 0.05 are within 100-clip noise, so `last` was kept as
planned.

## 2. The judge and the ceilings decided the design

Round 1 scored with project 27's character CTC recognizer, trained on one speaker's real mels. It made Mimi look lossy
(4 codebooks: 18.6% CER) and put a "codec ceiling" above 27. Whisper large-v3-turbo (`judge.py`, `results/phase0_summary.md`)
showed that ceiling was a judge artifact: Mimi-4 round trips score 1.09% CER on test, and the whole gap to 27 was the
model. Round trip ceilings on val-100 (`results/judge/`, `results/utmos/`):

| round trip | Whisper CER / WER | UTMOS |
|---|---|---|
| real recording | 0.23 / 1.06 | 4.40 |
| Qwen3-TTS-Tokenizer-12Hz, 16 codebooks | 0.32 / 1.18 | 4.44 |
| SAC (16 kHz, 37.5 Hz) | 0.76 / 1.57 | 4.44 |
| XY-Tokenizer, 8 codebooks | 0.41 / 1.23 | 4.26 |
| Mimi, 16 codebooks | 0.40 / 1.29 | 4.23 |
| **Mimi continuous latent** (quantizer input projections, codebook lookup skipped) | **0.38 / 1.23** | **4.28** |
| Mimi, 8 codebooks | 0.55 / 1.51 | 3.92 |
| Mimi, 4 codebooks (r3–r5's target) | 0.70 / 2.35 | 3.11 |
| real mel → project 28's vocoder (27's audio path) | 0.27 / 1.06 | 2.58 |
| Mimi raw encoder output → decoder (wrong space) | 159 / – | 1.49 |

Every codec is close to lossless for intelligibility, so representations were chosen by what a small model can predict.
Naturalness is where they differ: 4 codebooks cap it at 3.11, and the r1–r3 diagnostics (appendix §A.2) showed the finer
codebooks are close to unpredictable for this model. The continuous latent keeps Mimi's 12.5 Hz frame rate (one step per
frame, so a 10 s clip is 125 steps) and has the ceiling of 16 codebooks without predicting 16 discrete tokens.

## 3. The flow head (`tts_flow.py`, `train_flow.py`, `evaluate_flow.py`)

```
text (one char per Gemma token) -> Gemma 3 270M backbone -> hidden h_t
   h_t -> flow head: rectified flow, adaLN MLP (768 wide, 6 blocks, 28M) -> 512-d Mimi latent for frame t
   h_t -> stop head (BCE)
   latent_t -> in_proj -> next backbone input (with noise U(0, 0.2) in training)
latents -> Mimi quantizer output projections -> Mimi decoder -> 24 kHz audio
```
- **Target:** Mimi's latent at the quantizer input ("space c": `rvq_first.input_proj(z)` and `rvq_rest.input_proj(z)`
  concatenated, 512-d), normalized per dimension with the training set's mean and std. Stored as fp16 (20-clip check:
  0.29 CER, 4.25 UTMOS). Frames = ceil(samples at 24 kHz / 1920), asserted per clip.
- **Training:** flow loss over K = 2 noise draws per frame, stop BCE (positive weight 5), 10% of clips with the text
  dropped (for classifier-free guidance). Backbone LR 5e-5, new parts 3e-4, cosine. f1 started from r5's backbone and
  speaker table.
- **Sampling:** 16 Euler steps per frame; temperature = std of the starting noise; guidance on the text.

f1 sampling sweep (val-100, CER / WER, UTMOS, length ratio):

| temp | guidance | CER / WER | UTMOS | length ratio |
|---|---|---|---|---|
| 1.0 | off | 6.20 / 11.14 | 1.73 | – |
| 0.8 | off | 2.62 / 5.88 | 2.25 | ~0.97 |
| 0.8 | 1.5 | 1.80 / 3.36 | 2.43 | 0.94 |
| 0.8 | 2.0 | 1.29 / 2.24 | 2.52 | 0.91 |
| 0.8 | 2.0, 32 Euler steps | 1.20 / 2.13 | 2.47 | – |
| 0.6 | 1.5 | 1.13 / 2.41 | 2.69 | 0.93 |
| 0.6 | 2.5 | 0.92 / 1.96 | 2.69 | 0.90 |
| 0.5 | 2.5 | 0.82 / 2.01 | 2.75 | 0.89 |
| **0.4** | **2.0** | **0.66 / 1.79** | **2.80** | 0.91 |

Lower temperature and guidance both help, and twice the Euler steps don't. On f2 and f3, temperature 0.3 gives slightly
clearer words with no gain in UTMOS (f2 0.45 / 1.29, 3.06; f3 0.59 / 1.29, 3.14). On f2, 0.6 with guidance 2.5 is worse
on both (0.73 / 1.68, 3.06).
Guidance shortens speech, which is where the 7% fast pace comes from.

## 4. Data, epochs and scaling
- **Data beat epochs everywhere.** r3's val loss flattened by epoch 4–5 on LJ alone. Adding LibriTTS-R clean-100 halved the
  error (r4), and adding Hi-Fi TTS halved it again (r5). r5 was best at about 4.4 passes and flat after about 3. An LJ-only
  fine-tune from r5 (r6, lower LR) made it worse within 250 steps (2.63 vs 2.00 CER).
- **Val loss is a poor checkpoint selector for the codebook model.** r5's LJ val loss rose from step 9k to 16k while
  Whisper CER fell from 2.92 to 2.00, so codebook checkpoints were picked by Whisper on val. For the flow model, val loss
  and val UTMOS moved together through f2 (0.392 → 0.339, UTMOS 2.80 → 3.12) but not in f3: val loss fell to 0.323 while
  val UTMOS stayed at 3.12.
- **f2 → f3:** the extra 364 h and 5.1 h of training lowered val loss (0.339 → 0.323). Test UTMOS rose +0.07
  ([+0.02, +0.13], higher on 54/100 clips); val UTMOS didn't move (3.12 vs 3.12), and val words were slightly worse
  (0.61 / 1.62 vs 0.57 / 1.40, within noise). Data and extra steps were not separated. other-500 is the noisier half of
  LibriTTS-R and was not filtered.

## 5. What didn't work
- **Gemma's subword tokens as text input:** too sparse for 10.5k sentences; spelling the text one character per token
  fixed it (appendix §A.2).
- **16 delayed codebooks:** the acoustic heads barely beat chance (appendix §A.2). 4 worked better, but capped naturalness.
- **Gemma → project 27's mel + 28's vocoder (mel1):** 78% Whisper CER on val after 39 min; blurry mels made Whisper
  hallucinate. Parked.
- **LJ-only fine-tuning after multi-speaker training (r6):** overfit within 250 steps.
- **Best-of-4 reranking** (r5, picked by 27's recognizer): test CER 3.63 → 2.45 at four times the cost. Superseded by the
  flow head, which needs no reranking.
- **More Euler steps** (32 vs 16): no gain.

## 6. Correctness checks
- **Gemma port** vs mlx-lm in true fp32: max logit diff 2.2e-4, 100% top-1 agreement; cached vs full forward 1.8e-4
  (appendix §A.4).
- **Flow model:** the KV-cached generation path matches a full forward pass to fp32 kernel noise (the same size as
  padded vs unpadded full forward).
- **Latent store:** frame counts asserted per clip at encode time; 0 unreadable files among 275,696 clean-360/other-500
  clips.
- **Continuing runs:** f2's step-0 val loss (0.3924) equals f1's final, and f3's step-0 (0.3392) equals f2's final, so
  the weights and latent normalization carry over. When the speaker table grew for f3, the old rows were kept unchanged
  (3-step smoke check).
- **Shipped checkpoint:** `checkpoints/f3/last` (the trainable weights and `meta.json`, 497 MB) reloads and reproduces
  f3's val-100 run exactly: same transcript on 100 of 100 clips, CER 0.61, WER 1.62, UTMOS 3.121
  (`results/eval_f3_ckpt_check.json`).
- **Smoke tests:** `./smoke_flow.sh` trains 5 steps on 64 clips, resumes to 10, then generates, decodes and judges 2
  sentences (about 15 s on an idle GPU, timed once). `./smoke.sh` covers the round-1 codebook path.

## 7. Caveats
- **UTMOS is a predictor, not listeners.** It ranks 27 v2 (2.96) above real mels through the same vocoder (2.59), so treat
  it as a ranking among systems, not absolute quality. A blind listening test was not run.
- **100 clips per evaluation, one seed.** Val differences under about 0.05 UTMOS or 0.2 CER points are within noise.
- **Settings were chosen on val:** temperature 0.4 and guidance 2.0 came from f1's val sweep and were reused unchanged for f2,
  f3 and test. f2 and f3 were each scored on test once. f3 was planned as the final model before either test score
  existed.
- **The comparison with 27 is cross-pipeline.** 27 makes 22.05 kHz mels and vocodes them; 33 makes 24 kHz Mimi audio.
  Both are judged on the same clips with the same Whisper and normalizer. 33 trained on about 700 h, 27 v2 on LJ only
  (19 h), so this compares systems, not architectures at matched data.
- **Pace:** about 7% faster than the recording, a side effect of guidance. Not corrected.
- **Multi-speaker:** speaker IDs were trained for 2,309 voices, but only LJ (speaker 0) was evaluated.
- **Not tried:** T5Gemma 2 (encoder-decoder) as the backbone, a larger flow head, a depth transformer for codebooks,
  Qwen3-12Hz codes, streaming Mimi decode, the reverse (ASR) direction, voice prompts.

## 8. If this is picked up again
The gap from f3 (3.20) to its own ceiling (4.28) belongs to the model. The cheapest test of where it sits is a larger flow head
on f3 for 2–3 h: if UTMOS barely moves, the backbone is the limit, and the next step is a bigger or encoder-decoder
backbone (T5Gemma 2, or Gemma 3 1B) rather than more data. Data that's ready if needed: Jenny (single speaker, already
downloaded, not encoded), HiFiTTS-2, and a filtered Emilia slice.

## Appendix A: Round 1 (2026-10-30), Gemma → 4 Mimi codebooks on LJSpeech, 15-minute runs

Kept as written at the time, with section numbers prefixed by A. Two of its conclusions were later overturned by the Whisper judge (§2): the "codec ceiling" above project 27 was an artifact of 27's recognizer, and "unlikely to beat 27" no longer holds. The measurements, the random-init ablation and the diagnostics stand.

### A.0 Round 1 headline (held-out test, 27's 100 clips, scored with 27's CTC recognizer)

| system | CER % (27's CTC proxy, corpus) | clip-mean CER [95% CI] | runaway clips (hit frame cap) | median length ratio |
|---|---|---|---|---|
| real recording | 6.94 | | | |
| project 27 FastSpeech + 28's vocoder (audio path) | 6.2 (27's RESULTS) | | | |
| Mimi round trip, 4 codebooks (the ceiling for this model) | 18.55 | | | |
| **Gemma TTS** (Gemma 3 270M init, 4 codebooks, 15 min, temp 0.2) | **37.96** | 38.4 [36.5, 40.4] | 1 | 1.015 |
| same architecture, **random init**, same recipe and budget (temp 0.4) | 66.94 | 67.0 [66.0, 67.9] | 7 | 1.164 |

- **Borrowing Gemma matters, at this budget.** With identical recipe, data and 15-minute budget, Gemma init beats random init on
  97 of 100 test clips; paired difference in clip-mean CER 28.6 points [26.6, 30.5] (`results/test_paired_bootstrap.json`;
  corpus-CER difference 29.0). The recipe (backbone LR 3e-5, warmup, loss weights) was tuned over five val runs *for the
  Gemma init* and reused unchanged for random init, which was still improving when its 15 minutes ran out (§A.3, §A.5). So this
  measures the head start Gemma gives under a fine-tuning recipe; a from-scratch model with its own tuned LR or more steps
  would likely narrow the gap. Not tested.
- **It talks, badly.** Words are recognizable but slurred: "return to new orleans and joining the marine corps" →
  heard as "bueture thonew olens and toy m mwrriecorp" (val, temp 0.8). Listen: `media/demo.html`.
- **It does not beat project 27, and is unlikely to on this proxy.** Mimi itself costs 18.6% CER at 4 codebooks and still 8.5%
  at 32 (§A.1), above 27's 6.2%. (The round trip is a practical bar, not a hard bound: 27's own TTS scored below the real
  recordings on this recognizer.) This project's bar is the Mimi ceiling; it covers about 60% of the gap from random init (67%)
  to it (19%).
- **Speed** (batch 1, KV cache, bf16): generation runs at RTF 0.035 (~29× faster than real time); the first frame's codes are ready
  13 ms after the call (prefill + 4 steps). Mimi decoding runs afterwards in a separate process (RTF 0.005); there is no
  streaming decode, so end-to-end first-audio latency was not measured. Training (padded positions/s): 18–24k tok/s fresh,
  ~11–15k sustained (`logs/*.log`).
- **Listening preference: not measured.** Nobody has taken the demo page blind.


### A.1 The codec ceiling (`results/ceiling_nq8_16_32.json`, `results/ceiling_nq1_2_4.json`)
Real LJSpeech clips through Mimi encode → decode, scored with the same recognizer path. Our pipeline reproduces 27's real-
recording number (6.94 vs 27's 7.0), so the scoring matches.

| | real | 1 cb | 2 cb | 4 cb | 8 cb | 16 cb | 32 cb |
|---|---|---|---|---|---|---|---|
| val-100 | 7.31 | 96.89 | 31.22 | 20.05 | 13.23 | 10.21 | 8.88 |
| test-100 | 6.94 | 97.02 | 29.37 | 18.55 | 12.27 | 9.38 | 8.51 |

Codebook 0 (Mimi's "semantic" codebook) alone is unintelligible to this recognizer (97%). Intelligibility comes from the
first few acoustic codebooks. That shaped everything below.

### A.2 How we got there (val-100, temperature 0.8 unless noted; the protocol fixed test until the end)

| run | text input | codebooks | loss weights | LR backbone / new | val CER % | val cb0 loss | notes |
|---|---|---|---|---|---|---|---|
| `main` step 500 | Gemma subwords | 16 | equal | 1e-4 / 1e-3 | 73.78 | 4.163 | babbles; memorized and diverged by step 770 |
| `r1` step 700 | subwords | 16 | 20,4,1… | 3e-5 / 3e-4 | 67.71 | 3.338 | + 15% input-code corruption, zero-init heads; step 700 is from a 4.5-min continuation (peak 2e-5 / 2e-4, `logs/r1b.log`) |
| `r2` step 803 | **characters** | 16 | 20,4,1… | 3e-5 / 3e-4 | 64.59 | 2.756 | cb0 learns much faster |
| `r3` step 800 | characters | **4** | 4,2,1,1 | 3e-5 / 3e-4 | 45.81 (t0.8) · 42.00 (t0.6) · 39.27 (t0.4) · **38.41 (t0.2)** | 2.663 | shipped |
| `scratch` step 882 | characters | 4 | 4,2,1,1 | 3e-5 / 3e-4 | 69.32 (t0.8) · 68.35 (t0.6) · **66.93 (t0.4)** · 74.16 (t0.2, 67 runaways) | 3.709 | random init ablation |

What each step taught us:
1. **First run: it learned the voice before the words.** Step 500 produced LJ-sounding audio of about the right length with
   the wrong words (the "good failure", `media/failure_step500/`). Val loss then rose on every codebook while train loss fell
   (memorization), and jumped 6.06 → 6.66 in 20 steps at peak LR (instability).
2. **Is the text used at all?** (`diag_text.py`: teacher-forced val loss on codebook 0 with the right text, another clip's
   text, or no text.)

   | checkpoint | correct text | swapped text | BOS only | gen vs real length corr. |
   |---|---|---|---|---|
   | `main` step 500 | 4.163 | 4.477 | 5.364 | 0.76 |
   | `r1` step 690 | 3.344 | 7.051 | 7.356 | 0.88 |

   Barely at first; strongly after the recipe change (lower LR, content-weighted loss, corrupted fed-back codes, zero-init
   heads: the original random head init started the loss at 10.2 instead of ln 2049 = 7.6).
3. **Subword tokens are too sparse for 10.5k sentences.** 15,687 distinct Gemma tokens in train; 4.1% of val token occurrences
   never appear in train and 11.1% appear fewer than 5 times. Spelling the text out (every character is one Gemma token)
   lowered val cb0 loss at matched steps (step 700: 2.876 vs 3.338).
4. **The bottleneck was the acoustic codebooks, not alignment** (`diag_tf.py`: predict each step from the real history,
   decode, score; val-20):

   | checkpoint | real codes | model cb0 + real rest | real cb0 + model rest | all model (argmax) | all model (sampled) | top-1 acc cb0/1/2/3 |
   |---|---|---|---|---|---|---|
   | `r2` (16 cb) | 9.05 | 8.96 | 62.38 | 62.61 | 64.87 | .27 / .10 / .05 / .05 |
   | `r3` (4 cb) | 18.56 | 18.42 | 41.42 | 41.65 | 47.17 | .34 / .13 / .06 / .07 |

   Caveat: "model cb0 + real rest" ≈ real does not prove the model's cb0 is right, since cb0 contributes little to this
   recognizer (§A.1). The text-conditioning evidence is the swapped-text gap above. The lesson is the right-hand side:
   with 16 delayed codebooks, each input step is a sum of 16 embeddings from 16 different frames, and the acoustic heads
   barely beat chance. Cutting to 4 codebooks (MusicGen's regime) let them learn, and free-running CER fell from 64.6 to 45.8.
5. **Lower temperature helps the Gemma model a lot (45.8 → 38.4), the random-init one little (69.3 → 66.9 at 0.4) and it loops
   at 0.2** (67 of 100 clips hit the frame cap). Each
   model's temperature was chosen on val; test was scored once at that setting.
6. **Data-limited.** In every Gemma-init run val loss flattens by epoch 4–5 (the random-init run was still improving at its
   cutoff, §A.3) while train loss keeps falling (r3: weighted val ~3.86 vs
   train 3.40 at step 900). Continuing `r1` for another 240 steps made val cb0 worse (3.338 → 3.384).

### A.3 Where Gemma helps (`logs/r3.log` vs `logs/scratch.log`, val loss per codebook at step 800)

| init | cb0 | cb1 | cb2 | cb3 |
|---|---|---|---|---|
| Gemma 3 270M | 2.663 | 4.256 | 5.599 | 6.116 |
| random | 3.844 | 4.243 | 5.608 | 6.052 |

All of the advantage is in codebook 0, the one that has to follow the text; the acoustic codebooks come out the same. At this
budget, the pretrained text model's gift is reading, not voicing. The random-init run had not converged: its best checkpoint is
its last step (882), and its val cb0 was still falling (4.039 → 3.844 → 3.709 at steps 700 / 800 / 882) while Gemma's had
flattened (val mean 4.659 at 800, 4.667 at 900). Its final cb0 (3.71) is where Gemma init was at roughly step 370, so random
init needed about 2.4× the steps to reach the same cb0 loss here, under a backbone LR chosen for fine-tuning.

### A.4 Correctness checks
- **Gemma port** (`parity.py`, `results/parity.json`), true fp32 (`MLX_ENABLE_TF32=0`) vs mlx-lm 0.26.4: max |logit diff| 2.2e-4
  (logit scale 52), 100% top-1 agreement on 206 tokens, NLL 0.96 nats/token; cached decoding vs one full forward over 700 tokens
  (crosses the 512 window): 1.8e-4. Greedy: "The capital of France is Paris. It is the most visited city in the world."
  With MLX's default TF32 the diffs are ~0.08 (finding 2 of the Silicon Decathlon).
- **Generation path = training path:** feeding a real clip through the KV-cached sampler gives the training logits (cb0 argmax
  agreement 97%, max diff 0.31 in bf16).
- **bf16 activations with fp32 master weights** match fp32 training step for step (step 100 loss 5.549 vs 5.555) at 1.6× speed.
- **Resume:** `--stop_at 10` + `--resume` tracks an uninterrupted run (step 11: 0.40 vs 0.44, step 20: 0.0021 vs 0.0019); two
  uninterrupted runs differ by as much (GPU scatter-add in the embedding backward is not bitwise deterministic).
- **Mimi RVQ prefix:** encoding with k codebooks equals the first k of 32 (asserted in `mimi_io.py roundtrip`). Batched encoding
  is *not* equal to single-clip encoding (6–75% code agreement on 12 clips; an ad-hoc check whose output was not saved), so
  every clip is encoded alone.
- `./smoke.sh`: encode 9 clips → train → resume → generate → decode → score, ~20 s.

### A.5 Caveats
- **The judge is narrow.** 27's recognizer is a character CTC model trained on one speaker's real 16 kHz mels. It penalizes
  codec artifacts and has no language model. A human might find the samples more (or less) intelligible. No general ASR
  (e.g. Whisper) is installed; that would be a second, independent judge.
- **One seed per configuration**, and the recipe was tuned on val over five runs. No model output was scored on test until the
  final two scores. (The codec ceiling, which needs no model, was measured on test-100 alongside val-100 before training.)
- **The ablation is matched, not tuned.** Same command except `--init random` (same seed, data order, LR, 15 min; 904 vs 882
  steps). The backbone LR (3e-5) and the rest of the recipe were tuned for fine-tuning Gemma; random init got no tuning except
  its sampling temperature, chosen on val. It was still improving at the cutoff (§A.3). The 28.6-point gap is the head start at
  this budget, not the gap to the best from-scratch model.
- Temperature was chosen per model on val (0.2 for Gemma, 0.4 for random init). The gap doesn't depend on that choice: at
  matched temperatures on val it is 39.3 vs 66.9 (0.4) and 45.8 vs 69.3 (0.8).
- **Reference text.** The model reads LJSpeech's normalized column, and scoring uses 27's normalized references. These differ
  on 2 of 100 test clips ("Mrs."/"missus", "Lt."/"lieutenant"), the same for both models.
- **The ablation's frozen text table is random** (regenerated deterministically from seed 0; `build_model(init="random")`).
  Training it as well would add 168M trainable params; not tried.
- **Protocol change.** The protocol fixed 16 codebooks before training; it was changed to 4 on val evidence (§A.2.4), and the
  loss weights changed with it. Recorded in `STATUS.md` before any test scoring.
- **Thermal throttling.** Training throughput fell from ~21k to ~12–14k tok/s within minutes (laptop), so runs were capped by
  wall time and step counts differ slightly (r3 904, scratch 882 steps in 15 min).
- Splits are 27's by chapter, but the 1–10 s keep rule computed on 22.05 kHz audio keeps 10,515 / 861 / 1,534 clips vs 27's
  10,555 / 865 / 1,536. All 100 val and 100 test eval clips are present.
