# 34. Backbone Bake-off — Which small architecture learns the most per FLOP on a Mac?
Status: idea, not built (written 2026-10-30)
Scope: Expandable → Long-run

**Build:** Pretrain small language models from scratch in MLX with different backbones, at equal training
FLOPs, and compare. Grows out of `../30_nanochat_mlx` (pure-MLX nanochat port, fused softcap+CE kernel).

**Tinker with:**
1. Nanochat-style transformer (the baseline we already have).
2. LFM3-style tri-hybrid: gated short convs (kernel 7) + sliding-window attention (256) + global
   attention. LFM3's 24 layers are 13 conv / 5 SWA / 6 global.
3. MoE (Ling 2.0-style: sigmoid routing, shared expert, low activation ratio).
4. Mamba-3 hybrid: later. MLX has no fused scan kernel, so this would mostly measure our kernel-writing.

**First satisfying version:** 2–3 backbones at ~100M params, a few hours each on the same data and token
budget, one GPU job at a time. Report val bits/byte vs training FLOPs, tok/s and MFU on the sustained
(throttled) plateau, and decode tok/s.

**Go deeper:** the winner as a ~250M model trained for ~96 h, on the M5 Max or on the M4 Pro mini once
it has been measured. Then branch the checkpoint for SFT, reasoning RL, ASR/TTS heads.

**Measure:** val bpb per training FLOP, sustained tok/s and MFU vs the 60.3 TFLOP/s bf16 peak, peak memory,
decode tok/s and first-token latency, and a small downstream eval (e.g. CORE-style tasks).

**Starting numbers (measured on the M5 Max, `../30_nanochat_mlx/efficiency/REPORT.md`):**
- bf16 matmul peak 60.3 TFLOP/s cold. Throttles ~20% within ~4 min under sustained load.
- d6 (16.9M): ~203k tok/s sustained. d12: ~42k tok/s = 27.7 TFLOP/s sustained (46% MFU).
- Estimate for 250M non-embedding params: ~1.75 GFLOP/token → ~16–17k tok/s → 96 h ≈ 5.5–6B tokens
  (~22 tokens/param). Compare Gemma 3 270M at 6T tokens: this is a science project, not a better backbone.
- The M4 Pro mini has never been measured. Run `../30_nanochat_mlx/efficiency/bench_step.py` there
  (~4 min) before sizing anything for it. Don't link the two Macs for one run (a synchronous run goes at
  the slower machine's pace); use them for separate experiments.

**Shelf notes:** Data candidates: FineWeb-Edu, Cosmopedia, FineMath, a code set (downloads need approval).
Pre-tokenize before a long run. A 96 h laptop run means the laptop is unusable for 4 days.

**Links:** https://www.liquid.ai/blog/lfm3 · https://huggingface.co/blog/RichardBian/ling-mini-2-moe-release
