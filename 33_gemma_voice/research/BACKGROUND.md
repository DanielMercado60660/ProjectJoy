# Background for the research agents (Gemma Voice, round 2)

## Goal (weekend project, 2026-10-31 to 11-01)
Turn **Gemma 3 270M** (decoder-only) or **T5Gemma 2 270M-270M** (Google's encoder-decoder built from Gemma 3 270M) into
the best text-to-speech model we can. It must be a language-model TTS: the Gemma model generates the speech
representation. Curiosity project, not a product; licenses are not the main constraint. Baseline to beat: project 27,
a 24M FastSpeech-style model + LJSpeech vocoder trained in 85 min (6.2% CER on our proxy, close to real recordings).

## Hardware and software constraints
- One Apple M5 Max laptop, 128 GB unified memory. Measured bf16 matmul peak ~60 TFLOP/s; sustained training throttles
  ~20% within minutes. Our Gemma 3 270M fine-tune runs ~18-21k tokens/s fresh, ~11-15k sustained (bf16 activations,
  fp32 master weights).
- **Training and inference in MLX (Python)**, our own code. No PyTorch training. A codec or judge may run in a separate
  venv (we already run Kyutai Mimi through moshi_mlx in its own venv; mlx-audio 0.6.x exists and runs many TTS/codec
  models; torch 2.14 is installed but only for conversion).
- Weekend budget: roughly 30-48 hours of GPU time total, in resumable runs of 15 min (recipe tests) to 2-3 h (main).

## Base models
- Gemma 3 270M: 18 layers, d_model 640, 4 query heads / 1 KV head, head_dim 256, MLP 2048, 5 sliding-window (512)
  layers per 1 global layer, RoPE 10k local / 1M global, 262,144-token vocab (~170M of the 270M params are the
  embedding). Our MLX port matches mlx-lm to 2e-4 (fp32).
- T5Gemma 2 270M-270M: encoder-decoder adapted from Gemma 3 270M; tied embeddings; decoder merges self- and
  cross-attention; also ships a SigLIP vision encoder we would not use. Not ported yet.

## Data (local or downloading)
- LJSpeech 1.1: one female speaker, ~24 h (we train on ~19 h, chapters LJ001-040; val LJ041-044; test LJ045-050,
  the 100 test clips are project 27's).
- LibriTTS-R train-clean-100: ~54 h, ~250 speakers, restored audio, normalized text (encoding now).
- Hi-Fi TTS: 292 h, 10 speakers (>=17 h each), 44.1 kHz (downloading).
- Could add: LibriTTS-R train-clean-360 (~190 h), VCTK, Emilia-YODAS slices, Libriheavy small.

## What we built and measured so far (round 1, 2026-10-30)
- Gemma 3 270M + Kyutai Mimi codec (24 kHz, 12.5 frames/s, RVQ 2048-entry codebooks), MusicGen-style delay pattern,
  sum of codebook embeddings in, one linear head per codebook out, frozen text embedding, text spelled one character
  per token (subword tokens were too sparse for 10.5k sentences), cb loss weights, 15% random corruption of fed-back
  codes, zero-init heads, lr 3e-5 backbone / 3e-4 new params, bf16.
- Test CER (project 27's LJ-only CTC recognizer as judge): Gemma TTS 38.0% (4 codebooks, temp 0.2), same model random
  init 66.9%, real recordings 6.9%, Mimi round-trip ceiling 18.6% (4 cb), 12.3% (8), 9.4% (16), 8.5% (32), codebook 0
  alone 97%. Generation RTF 0.035.
- Diagnostics: the model does condition on text (cb0 loss 3.3 with correct text vs 7.1 with swapped text) but
  alignment is soft (slurring, skipped letters). With 16 delayed codebooks, the acoustic heads barely learn
  (teacher-forced top-1 accuracy 3-10% for cb1-15); at 4 codebooks 13% / 6% / 7%. Every run is data-limited: val loss
  flattens after 4-5 epochs on 19 h while train loss keeps falling. Lower temperature helps a lot.
- The judge is biased: 27's recognizer was trained on LJ mels and scores 27 better than real audio. We plan to add an
  independent ASR judge (Whisper-class, MLX).

## Your output
Write ONE markdown report at the path given in your task. Cite sources (URLs) for factual claims and say when
something is your inference. End with a section "Recommendations for us", ranked, each with expected impact,
effort, and risk on THIS setup (270M, MLX, one laptop, a weekend). Be concrete: numbers, hyperparameters, file
formats, MLX availability. Do not install packages, download models or data, run git, or edit anything outside
your report file.
