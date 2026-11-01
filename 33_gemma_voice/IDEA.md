# 33. Gemma Voice — Teach a small borrowed backbone to speak, then to listen
Status: **finished 2026-11-01** (first version 2026-10-30, then rounds 2–3: more data, Whisper judge, flow head on Mimi latents). See README.md and RESULTS.md; this file is the original idea note
Scope: Expandable

**Build:** Take Gemma 3 270M as a ready-made backbone (skip pretraining) and give it a voice: text in,
neural-codec audio tokens out, decoded to a waveform. Then the reverse direction, speech in → text out.
The point is to see if we can make a voice model ourselves, not to ship one.

**Tinker with:** codec choice (Kyutai Mimi at 12.5 frames/s vs a mel + project 28's vocoder), how to lay
out several codebooks (flatten, delay pattern, small depth transformer), a separate audio output head vs
growing the 262K text vocab, full fine-tune vs LoRA, single speaker (LJSpeech) vs multi-speaker
(LibriTTS-R), speaker prompts.

**First satisfying version:** Gemma 3 270M + Mimi tokens, trained on LJSpeech, reads held-out sentences
aloud. Compare against project 27's FastSpeech model on the same held-out clips with the same CTC
recognizer proxy (27: 5.8% CER, 23.9M params, 85 min of training).

**Go deeper:**
- Multi-speaker TTS on LibriTTS-R with a short voice prompt.
- ASR: an audio encoder (frozen or small trained conformer) → adapter → Gemma 3 270M decoder, on
  LibriSpeech. Alternative shape: T5Gemma 2 270M-270M (encoder-decoder, the Whisper shape).
- Streaming: first-audio latency and real-time factor while generating.

**Measure:** CER via a recognizer proxy, first-audio latency, real-time factor, tok/s while training,
listening preference (blind demo page), held-out sentences and hard pronunciations.

**Starting numbers (estimates, not measured):**
- Gemma 3 270M is about 170M embedding params + about 100M transformer params, with a 262K vocab.
  A fine-tune should run around 16k tok/s on the M5 Max, scaled from nanochat d12's 27.7 TFLOP/s
  sustained (`../30_nanochat_mlx/efficiency/REPORT.md`).
- Mimi with 8 codebooks ≈ 100 audio tokens per second of speech. LJSpeech (24 h) ≈ 8.6M tokens, about
  9 min per epoch. LibriTTS-R (585 h) ≈ 210M tokens, about 3.5 h per epoch.
- Memory gotcha: the logits for 16k tokens over a 262K vocab are ~17 GB in fp32. Use a small audio head,
  or chunk the cross-entropy.

**Shelf notes:**
- LJSpeech is at `../data/ljspeech/LJSpeech-1.1`. Reuse 27's text normalizer and recognizer proxy and
  28's vocoder by copying them (with provenance notes), not by importing.
- Needs approval before building: download the Gemma 3 270M weights (Gemma terms; this repo is public
  under MIT, so check what can be published) and the Mimi codec (CC-BY-4.0). It also needs an MLX port
  of Mimi, or Kyutai's `moshi_mlx` (an install; check it runs on MLX 0.32.4).
- LibriTTS-R / LibriSpeech are not on disk yet.

**Links:** https://huggingface.co/google/gemma-3-270m · https://huggingface.co/google/t5gemma-2-270m-270m ·
https://huggingface.co/docs/transformers/model_doc/mimi · https://github.com/hexgrad/kokoro (82M non-LLM TTS, a
reference point)
