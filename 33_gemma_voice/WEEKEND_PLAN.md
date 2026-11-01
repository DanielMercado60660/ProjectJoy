# Gemma Voice weekend plan: make Gemma 270M one heck of a TTS

Goal: the best LM-generated TTS we can get from **Gemma 3 270M** (T5Gemma 2 270M-270M as a time-boxed alternative),
judged fairly against project 27 (24M FastSpeech + LJ vocoder: 6.2% CER on 27's recognizer).

Sources: four research reports in `research/` (written by Codex agents with live web search, 2026-10-30):
`small_llm_tts.md` (case studies), `codec.md` (representation), `alignment_architecture.md` (design),
`data_training_eval.md` (data, recipe, judges). This plan is the synthesis; the reports carry the citations.

## What the research agrees on
1. **Judge first.** 27's recognizer was trained on LJ mels; it scores 27 better than real audio and punishes codec
   artifacts. Every report puts an independent ASR judge (Whisper-class, MLX) before any architecture change.
2. **Two separate bottlenecks:** alignment (text progress: slurring, skipped letters) and representation (fine Mimi
   codebooks are nearly unpredictable at our scale). Fix them one at a time, each against a matched control.
3. **Published small TTS LMs lean on data and structure, not size.** Spark 102k h, Llasa 250k h, Kani-TTS 370M
   ~80k h. The closest small-data counterpoint is NVIDIA T5-TTS (~1.8k h, 220M) with **explicit alignment**
   (attention prior + CTC). Nobody shows 19 h being enough; more clean, speaker-labelled data is expected to help.
4. **Representations that small LMs handle:** one token stream per frame (BiCodec, XCodec2, NeuCodec), or semantic
   tokens + a separate renderer (CosyVoice, Chatterbox), or Mimi with a trained depth transformer (CSM, Kyutai).
   Our 4-16 summed delayed Mimi heads match none of the proven recipes.
5. **Keep the decoder-only Gemma as the primary bet** (it works end to end and beats random init by 29 points).
   T5Gemma 2's decoder merges self- and cross-attention, so the T5-TTS prior/CTC recipe doesn't drop in; porting is
   >1 day of risk. Time-box it only if the main path stalls.
6. **Defer:** CFG, DPO/GRPO with ASR rewards, synthetic distillation, until supervised speech is aligned.

## The plan (each phase gated; every run resumable, checkpoint every 250 steps)

**Phase 0: fair judge (~2 h, mostly CPU/GPU eval).**
Install an MLX ASR judge (Whisper large-v3-turbo via `mlx-whisper`, or Parakeet MLX) in a side venv. Rescore on
val-100 and test-100 with CER and WER: real audio, 28's vocoder on real mels, Mimi 4/8/16/32 round trips, project 27,
our r3 and scratch. Output: the real gap to 27 and each representation's ceiling under a fair judge.
*Gate:* the judge transcribes real LJ audio at a low error rate (expect ~2-4% WER).

**Phase 1: representation bake-off (~6 GPU-h, LJ only, same recipe, 45 min each).**
Measure round-trip ceilings first (cheap), and train only candidates whose ceiling clearly beats Mimi-4.
- A. **Mimi 4 codebooks** (control, = r3).
- B. **Gemma predicts project 27's log-mel** in 4-frame chunks (~15.6 steps/s), L1+L2 loss plus a stop head, rendered
  by 28's LJ vocoder. No codec ceiling beyond the vocoder (real mel through it: 7.1% on 27's judge), and the same output
  path as 27, so it's the most direct way to beat 27. Risk: blurry or drifting autoregressive mel.
- C. **SAC** (ACL 2026, Soul-AILab/SAC): two single-codebook streams (semantic 12.5 Hz + acoustic 25 Hz, 16,384 entries
  each, 37.5 tokens/s, 16 kHz). LibriSpeech reconstruction WER 2.53%, UTMOS 4.27; with Qwen3-0.6B (100k h) TTS WER 1.06%.
  No deep RVQ stack to predict. PyTorch, so it runs in a side venv for encode/decode only. User's shortlist; top pick.
- D. **Qwen3-TTS-Tokenizer-12Hz** (Apache-2.0, MLX via mlx-audio, the tokenizer behind Mei's Qwen voice): cheap to test;
  multi-codebook like Mimi.
- E. **XY-Tokenizer** (ACL 2026): 12.5 Hz, 8 x 1,024 RVQ, ~1 kbps, text-aligned semantics; beat Mimi-8 in a matched AR
  TTS test (WER 1.81% vs 4.30%). Still 8 residual codebooks.
- (Research also suggested NanoCodec and BiCodec; lower priority now.)
Per candidate (user's test list): reconstruction on LJ val-100, Whisper CER/WER, speaker similarity (mama-mei's
`eres2net.onnx` speaker embedding vs the real clip), a tiny text-to-token probe (15-min Gemma run, identical recipe),
generated-speech CER under the judge, codec decode RTF.
*Gate:* pick the representation with the best val score under the fair judge (ties go to the simpler one).

**Phase 2: more data (~8-10 GPU-h).**
Winner + LJ / LibriTTS-R clean-100 (54 h, encoding now) / Hi-Fi TTS (292 h, downloading), with a learned **speaker ID**
embedding and balanced sampling (~50% LJ). Compare against LJ-only at **matched tokens** from the same init. Keep LJ val
and a held-out multi-speaker val.
*Gate:* mixture beats LJ-only on LJ val.

**Phase 3: alignment (~6 GPU-h).**
Give the decoder-only model explicit text progress: a small head predicts which character/word each audio step is on
(targets from 27's monotonic alignment search, or ASR word timestamps), and the predicted position is fed back as
input; never ground truth at inference. Compare with the Phase 2 winner.
*Gate:* fewer deletions and lower CER on val, without breaking prosody (listen).

**Phase 4: main run + decoding (~3-4 GPU-h).**
2-3 h run with everything that passed. On val only: sweep temperature / top-k, try best-of-N (2-4) with ASR
reranking (report single-sample and reranked). Then test once, demo page (real / 27 / ours), reviewer.

**Stretch (only with spare time):** T5Gemma 2 port time-boxed to 3 h for parity; depth transformer for 8+ Mimi books;
CFG; ASR-preference DPO.

## Needs your OK before starting
- Phase 0 judge: a side venv (`.venv-judge`) with `mlx-whisper` (or `parakeet-mlx`) plus the model weights
  (Whisper large-v3-turbo ~1.6 GB). Exact sizes confirmed before downloading.
- Phase 1 C/D: `mlx-audio` in a side venv plus NanoCodec / BiCodec weights (sizes confirmed first). B needs nothing new.
