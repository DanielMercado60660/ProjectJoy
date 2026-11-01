# T5Gemma-TTS (Aratako) — what it does and what we can borrow

Sources: https://github.com/Aratako/T5Gemma-TTS (code read at `../vendor/t5gemma_tts_src/T5Gemma-TTS`, cloned
2026-10-31), tech report https://arxiv.org/abs/2604.01760, model https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b.

## What it is
- **Backbone:** original T5Gemma **2B-2B** (`google/t5gemma-2b-2b-ul2`), 4B total, encoder-decoder. Text goes to the
  encoder as **subword tokens** (no phonemes); the decoder generates audio tokens and reads text through
  cross-attention in every layer. Architecture and base code from VoiceStar (arXiv 2505.19462).
- **Codec:** XCodec2, **one codebook** at 50 tokens/s, 65,536-entry vocabulary (`--audio_vocab_size 65536`,
  `--encodec_sr 50`, folder `xcodec2_1cb`). Anime-XCodec2-44.1kHz variant for Japanese; original 16 kHz XCodec2 for EN/ZH.
- **Data:** ~170k hours EN/ZH/JA (English via Emilia-YODAS; LibriHeavy too, per the report's caveat about LibriSpeech).
- **Recipe** (`examples/training/t5gemma_2b-2b.sh`): from scratch on 8 GPUs, 143,391 steps, max 30,000 tokens per
  batch x 4 gradient accumulation, **ScaledAdam lr 0.035**, warmup 2%, bf16, gradient checkpointing,
  `--no_loss_on_prefix 1`, `--neighbor_prompt_prob 0.5` (half the time a same-speaker "neighbor" clip is the voice
  prompt: zero-shot cloning), clips 0.2-40 s, text up to 500 tokens, `--prune_text_modules 2`.
- **Inference:** temperature 0.8, top-k 30, top-p 0.9; target duration given or estimated from phoneme count and
  per-language pacing (`duration_estimator.py`).

## The key idea: PM-RoPE (progress-monitoring RoPE), `models/t5gemma.py`
- Positions are **normalized progress**, not token indices: encoder token i -> `i / (L_text - 1) * 2000`; decoder step t
  -> `t / (L_audio - 1) * 2000` (`progress_scale 2000`). Training uses the true audio length; inference uses the
  estimated total length (and clamps at 2000).
- These positions drive RoPE in the self-attentions **and** in the cross-attention (separate rotary embeddings for
  decoder queries and encoder keys), so the query-key rotation encodes the *difference in progress* between an audio
  step and a text token. Attention is pulled toward "the part of the text at the same fraction".
- Report ablation: turning PM-RoPE off at inference (same checkpoint) raises CER 0.129 -> 0.982 and drops duration
  accuracy 79% -> 46%. It is the load-bearing part.

## What transfers to Gemma Voice (33)
1. **Progress positions in our decoder-only Gemma 3.** Our audio queries already attend to the text prefix through
   self-attention, so the same trick applies: text positions `i/(Lt-1)*P`, audio-step positions `s/(S-1)*P`, with the
   total step count S known in training and estimated at inference (frames/char, or a small length predictor). Direct
   attack on our main error class (skipped or slurred letters in rare names and acronyms).
2. **Bidirectional text (prefix-LM mask).** T5Gemma's encoder reads the whole sentence in both directions. In our
   model each letter only sees earlier letters. Letting the text prefix attend fully within itself (as Google did to
   build T5Gemma from Gemma) is a mask change.
3. Together, 1 + 2 are structurally close to **T5Gemma 2's** decoder, which merges self- and cross-attention into one
   attention over [encoder states; decoder tokens]. A full T5Gemma 2 270M-270M port is the faithful version; the
   decoder-only variant is the cheap test of whether it would pay off.
4. Not borrowing for now: XCodec2 (4x more steps/s than our 4-codebook Mimi delay layout; 65k-way head),
   ScaledAdam, neighbor-prompt cloning (we have speaker IDs and one target voice).

## Scale caveat
170k hours and 4B params on 8 GPUs. We have ~400 h and 270M on one laptop. What transfers is the alignment mechanism,
not the numbers.
