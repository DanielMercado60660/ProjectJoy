# 35. FP8 on Silicon — Does low-precision training survive on a chip with no FP8 math?
Status: idea, not built (written 2026-10-30). One probe run, see "Starting numbers".
Scope: Quick-start

**Build:** Ling 2.0 (Ant Group) trained in FP8 (blockwise scaling, FP8 optimizer states) and reports a
loss curve nearly identical to BF16 over 1T+ tokens, with a big throughput gain on FP8 tensor-core GPUs.
Ask the Mac version of the question: in MLX on the M5 Max, what does FP8 cost in quality, and does it buy
any speed or memory?

**Tinker with:** emulated FP8 training (fake-quantize weights and activations to e4m3 in the forward pass,
straight-through gradients, fp32 master weights) vs bf16 on nanochat d6/d12; blockwise vs per-tensor
scaling; FP8 optimizer states (`mx.to_fp8` / `mx.from_fp8` storage); MXFP8 / MXFP4 / NVFP4 frozen bases
with bf16 LoRA (QLoRA-style); FP8 weights for decode.

**First satisfying version:** nanochat d6, bf16 vs emulated-FP8 forward, matched tokens and seeds. Report
the val bpb gap, plus the speed and memory of each path.

**Go deeper:** FP8 optimizer states (memory saved vs loss change); a quantized-base LoRA fine-tune of a
borrowed backbone (ties into 33); decode tok/s with fp8/fp4 weights.

**Measure:** val bpb gap vs bf16, tok/s, peak memory, decode tok/s, and loss spikes/divergence.

**Starting numbers (probe, 2026-10-30, MLX 0.32.4, M5 Max, load ~3; `probes/quantized_matmul_probe.py`):**
- MLX has no FP8 dtype. `mx.to_fp8` stores e4m3 as uint8. `mx.quantize` supports `affine`, `mxfp4`,
  `mxfp8`, `nvfp4`.
- Quantized matmul vs bf16 (single quick runs, not interleaved):
  - 8192×1024 @ 1024×4096: bf16 2.23 ms; mxfp8 1.31×, mxfp4 1.36×, nvfp4 1.42× faster (the bf16 run
    here reached only 31 TFLOP/s).
  - 8192×2048 @ 2048×8192: bf16 4.51 ms (61 TFLOP/s); mxfp8 **0.86×** (slower), fp4 modes 0.92–0.93×.
  - Batch-1 decode (1×2048 @ 2048×8192): quantized 1.11–1.19× faster (bandwidth-bound).
- Gradients flow to the activations through `quantized_matmul`, but not to the quantized weights ("no
  gradient wrt scales in mxfp8"). So real FP8 *weight* training isn't available; use emulation or
  frozen quantized bases.
- Reading: the weights get dequantized and the math runs at bf16 rate. FP8 here saves memory and helps
  decode; it isn't a training speedup the way FP8 tensor cores are on Ling's GPUs.

**Links:** https://huggingface.co/blog/RichardBian/ling-mini-2-moe-release ·
https://ml-explore.github.io/mlx/build/html/python/data_types.html
