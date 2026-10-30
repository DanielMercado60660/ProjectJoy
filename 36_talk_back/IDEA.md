# 36. Talk Back — A tiny speech-to-speech loop, then a toy full-duplex one
Status: idea, not built (written 2026-10-30)
Scope: Expandable

**Build:** A local voice conversation on the M5 Max. Start as a cascade: ears + brain from Gemma 4 E2B
(audio in, text out), mouth from project 33's own TTS. Then move towards one model that listens while it
speaks, in the spirit of Moshi or Realtime-Venus, at toy scale.

**Tinker with:** cascade vs end-to-end; turn-taking (voice-activity endpointing vs a learned "speak now"
token); barge-in (stop talking when interrupted); acting on the newest audio rather than queueing
(Silicon Decathlon round 5 found that dropping stale frames halved the error under overload); two-stream
token layouts (user audio + own audio + text); synthetic two-speaker conversations made with 33's TTS.

**First satisfying version:** a push-to-talk cascade. E2B transcribes and answers, 33's TTS speaks.
Measure end-to-end latency per stage (audio in → text → first audio out) and real-time factor.

**Go deeper:** streaming in both directions; interruption handling; a toy duplex model on Mimi tokens
trained on synthetic conversations; the same loop on the iPad.

**Measure:** time to first audio after the user stops, per-stage latency (p50/p95), barge-in reaction
time, intelligibility of replies (recognizer proxy), and missed or overlapping turns.

**Starting facts:**
- Gemma 4 E2B: 2.3B effective / 5.1B total params, ~300M audio encoder (30 s max), text out only,
  Apache 2.0. MLX audio support had a gibberish-output bug (mlx-vlm issue #903), fixed on main in April
  2026; test the exact checkpoint before building on it.
- Realtime-Venus is two 9B models. Moshi pretrained on ~7M hours of audio. True duplex at small scale is
  a research toy; the cascade is the realistic first step.
- Silicon Decathlon round 5: for small models the loop around the model (sensor rendering, result pickup)
  costs about as much as the model itself.

**Shelf notes:** Depends on 33 for the mouth. Needs approved downloads (E2B weights) and probably an
mlx-vlm install. Headless rule: render a demo to file (recorded conversation + latency timeline).

**Links:** https://huggingface.co/google/gemma-4-E2B · https://github.com/inclusionAI/Realtime-Venus ·
https://github.com/Blaizzy/mlx-vlm/issues/903
