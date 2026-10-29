# Efficiency pass, 2026-10-22 afternoon

> **Stopped early at 14:07.** The coordinator stopped all GPU work because the user needed
> the machine. A last 10-minute interleaved d6 run (base vs kernel, to measure both thermal
> plateaus) was killed after 4 windows, and its partial reading was discarded. Not finished:
> (a) the d6 *baseline* plateau, so the d6 sustained gain is inferred from the 4.5-minute
> interleaved run, not measured at the plateau; (b) the optional fixed-wall-time loss-cost
> runs for training-changing ideas. Those were skipped on purpose, because neither showed a
> speed gain worth paying for. After the stop, only `pytest -q` was run (11 passed). The
> earlier `./smoke.sh` and `pytest` runs (14:05, kernel already the default) also passed.

Goal: honest idle-GPU throughput numbers for the real pretraining step on the M5 Max, plus
any pure-systems speedups that keep the training math unchanged. The hypotheses came from the
Codex audit. Each one was implemented behind a flag (default off),
parity-checked, and then benchmarked against the current kept set in the same process.

## Method

**Harness (`bench_step.py`).** It times exactly what `train.py` times per optimizer update:
`set_schedule` plus `train.make_update(...)`, which is the function `train.py main()` now calls.
That covers 4 accumulated micro-batches from the real `TokenStream` loader, compiled
loss+grad, MuonAdamW, bf16 compute and fp32 master weights. `mx.synchronize()` brackets every
timed window, and tok/s = accum·B·T·updates / window time. Each variant gets 8 warm-up
updates and then 5 timed windows. Variants in one process are **interleaved** (A B C, then
C B A, ...) so drift, thermals and background load hit every variant equally. The first
variant is always the current kept set, so every comparison carries its own re-baseline.
The bf16 8192² matmul peak is measured at the start and end of every process. MFU = model
FLOPs/token (6·N_matmul + 12·L·h·d·T, upstream's convention) × tok/s ÷ the mean of those two
peaks. That is model FLOPs vs measured dense-GEMM throughput, not a hardware counter.

Configs: **d6** at the pilot config (B=32, T=1024, accumulation 4, 131,072 tok/update);
**d12** at B=16, T=1024, accumulation 4 (65,536 tok/update, the audit's d12 reference).
The M4 Pro handoff command runs d12 with accumulation 8, which amortizes the optimizer
further; that config was measured separately at the end.

**Old `bench.py` vs this harness.** `bench.py` applies a full MuonAdamW update after every
16-row batch (accumulation 1), on synthetic tokens, with 3 warm-up and 10 timed steps. That
overweights the optimizer about 8× relative to the pilot, which does one update per 4×32 rows.

**Parity (`parity.py`, `parity_sdpa.py`; results in `parity.json` and `parity_log.txt`).**
The checks use d6, real batches of 4×1024, and two starting points: `init` (init weights with
the zero-init output projections perturbed) and `pilot` (the trained
`checkpoints/pilot/last` weights AND optimizer state, read-only). They measure the loss rel
error, the gradient rel error, and the rel error of the parameter *delta* after 2 updates,
plus a 30-update loss-trajectory comparison from init. `default_vs_orig` runs the edited
code with every flag off against the untouched pre-pass source (`efficiency/orig/`). It gives
identical losses and a delta rel of 8e-11 to 3e-10, the same as the GPU's own run-to-run
noise (`base` vs `base`).

The noise floors are measured, not assumed. Regrouping the same examples into bigger GEMMs
(`g2`, mathematically identical) moves the 2-update delta by 0.93% at `pilot`, and moves the
Muon matrices' delta by 3.2% at `init`. Muon orthogonalizes in bf16 and Adam's first steps
are roughly sign(g), so any change in floating-point order moves the deltas by about 1%. The
audit's 1e-2 delta gate therefore sits right at the noise floor. A candidate passes the delta
gate if it is within the audit tolerance OR within 1.5× the measured `g2` floor. Model-path
changes must also stay within 1.5× the gradient floor (full batch vs the mean of its two
halves), in bf16 and in fp32 compute.

## Headline: idle baseline vs final

All numbers come from an idle GPU (no other projects' `gpu_run` jobs, 1-minute load average
1.9–4.8, recorded per run in `bench_runs.jsonl`). The two MFU columns are model FLOPs/token ×
tok/s divided by (a) **60.3 TFLOP/s**, the cold bf16 8192² matmul peak, which is the same
reference as RESULTS.md §2, and (b) that run's own start/end matmul peak.

| config | measure | baseline (pilot code) | final (`ce_mode=kernel`) | gain |
|---|---|---|---|---|
| d6, 32×1024, accum 4 | interleaved, 5 windows (R2) | 204,830 tok/s | 217,494 tok/s | **+6.2%** (min > base max) |
| d6, 32×1024, accum 4 | interleaved, 10 windows over 4.5 min (F) | 211,167 tok/s · 38.8% MFU | 233,648 tok/s · **43.0%** | **+10.6%**, faster in all 10 window pairs |
| d6, 32×1024, accum 4 | sustained plateau, 10 min, final only | not measured | **~203,000 tok/s** (22.5 TF/s, **37%**) | n/a |
| d12, 16×1024, accum 4 | interleaved (R1/R2) | 41,043 / 41,443 tok/s · 45–46% | 41,673 tok/s · 46% | +0.6% (noise) |
| d12, 16×1024, accum 8 (handoff) | interleaved, 8 windows (F) | 42,552 tok/s · 47.0% | 43,346 tok/s · 47.5% | +1.9% (noise level) |
| d12, 16×1024, accum 8 (handoff) | sustained plateau, 30 windows / 7 min, final only | not measured | **~42,000 tok/s** (27.7 TF/s, **46%**) | n/a |

(MFU vs the throttled same-run peak is higher: d6 final 52.6%, d12 final 57.6%. See LOG.tsv.)

**The biggest effect is not a code change: the GPU throttles under sustained load.** This
M5 Max is a MacBook Pro (Mac17,6) on AC power. In every run tok/s falls steadily for about
4 minutes and then plateaus. d12 at the handoff config goes from 52.4k tok/s in the first
window to a 41.9k plateau (−20%). The 8192² matmul measured right after a run reads 38–51
TFLOP/s instead of 60. Consequences:
- The review note's single 10-step readings (d6 230k, d12 47k) were real, but they are
  fresh-GPU numbers. Short single-variant processes here reproduce them (kernel d6 247–256k,
  d12 48–50k). A long run lives on the plateau.
- Every comparison above interleaves base and candidate in one process, so both see the same
  thermal state. Sequential processes are not comparable: the same configuration drifted 6%
  between two back-to-back processes (R3).
- `pmset -g therm` recorded no thermal warning, so this is ordinary DVFS clock management,
  not an alarm state.

## Profile of one update (`profile_step.py`, `profile.jsonl`)

| piece | d6 (32×1024, accum 4) | d12 (16×1024, accum 4) |
|---|---|---|
| one micro-batch loss+grad (compiled) | 161 ms (fwd alone 42 ms) | 370 ms (fwd 99 ms) |
| MuonAdamW update (once per 4 micro-batches) | 11 ms = **1.7%** of an update | 62 ms = **4.0%** |
| softcap + CE fwd+bwd, reference | 17.9 ms/micro = **10.9%** | 9.1 ms/micro = 2.4% |
| softcap + CE fwd+bwd, fused kernel | **5.0 ms** | 2.7 ms |
| softcap + CE, bf16 capped logits | 13.0 ms | 6.7 ms |
| causal SDPA fwd+bwd, all layers | 41 ms/micro = **25%** | 86 ms/micro = 22% |
| numpy loader | 0.4 ms/update = 0.06% | 0.2 ms = 0.01% |

This profile explains almost every result below. The optimizer is a small fraction of a real
4-micro-batch update, so Muon tweaks cannot move the total. CE was the one large, fixable
piece at d6. The loader is free.

## What was tried (one row per attempt in LOG.tsv; parity in parity.json)

| # | change | parity | d6 | d12 | decision |
|---|---|---|---|---|---|
| 1 | fused softcap+CE Metal kernel (`fused_ce.py`, `ce_mode=kernel`) | pass: loss rel ≤2e-7; grad rel 4.6e-3 vs floor 4.9e-3; fp32 grad 3.3e-4 vs floor 3.5e-4; delta 1.14% (1.5× floor = 1.39%); SFT mask and all-zero mask OK; 30-update loss trajectory within 0.010 | **+6.2% / +10.6%**, −2.6 GB peak memory | +0.6–1.9% | **kept, on by default** |
| 2 | on-device accumulation + async_eval, one sync per update | pass, bit-level (delta 1e-10, same as the GPU's own noise) | −1.3% | +1.1% | discarded (no gain) |
| 3 | 2× device batch, same examples (64×2 / 32×2) | defines the reorder floor (0.93% delta at pilot) | −1.2% | −0.7% | discarded; 1.8× memory |
| 4 | SDPA backward tiles 512×512 (env) | pass, causal leakage 0 | +2.1, +4.1, +2.1, +2.4% (paired) | +0.6 to +3.4% | discarded: consistent but under the 3% gate, and it is an internal knob. Opt-in: `MLX_SDPA_VJP_BQ=512 MLX_SDPA_VJP_BK=512` |
| 4 | SDPA tiles 256×256, 256×512 | pass | −5 to −10% | not run | discarded |
| 5 | one QKV GEMM | pass | +1.2% | −0.8% | discarded |
| 6 | batched (stacked) Muon | pass | +0.8% | −1.6% | discarded |
| 7 | Polar Express via `mx.addmm` | pass (0.97% delta vs 0.93% floor) | −1.0% | +0.4% | discarded |
| 8 | prefetch thread for the loader | not built: the loader is 0.06% of an update | n/a | n/a | not attempted |
| 9 | bf16 capped logits [changes training] | n/a | −5.3% vs kernel | −1.3% | discarded (slower than the exact kernel) |
| 10 | Muon 4 or 3 Polar Express iterations [changes training] | n/a; worst per-tensor update change 34% / 49% | −0.8%, −1.7% | +0.3%, +0.5% | discarded; with no speed gain, the loss-cost run was skipped |
| – | another custom Metal kernel | no dominant uncovered op (SDPA is already fused; the remaining time is spread across many small compiled elementwise ops: rms_norm 1.4 ms, relu² epilogue 1.7 ms/layer, embedding 1.7 ms) | | | not attempted |

Noise: the 5-window spreads were 4–9% because of the thermal decay inside a run. Paired
window-by-window comparisons are tighter: kernel vs base won 10 of 10 pairs at d6.

## Audit predictions vs reality

| audit item | predicted | measured |
|---|---|---|
| fix the benchmark (accumulation) | `bench.py` overweights the optimizer | confirmed: the optimizer is 1.7% (d6) / 4% (d12) of a real update |
| #2 fast accumulation | d6 +1–6%, d12 0–3% | −1.3% / +1.1%: the per-micro-batch sync costs nothing measurable |
| #3 bigger device batch | d6 +3–10%, d12 +3–12% | −1.2% / −0.7%: B=32 (d6) and B=16 (d12) already saturate the GEMMs |
| #4 SDPA tiles | 0–8% / 0–6%, regression plausible | 512: +2.7% / +1.8%; 256: −5 to −10% (both directions as warned) |
| #5 QKV GEMM | d6 +1–5%, d12 0–3% | noise |
| #1 fused softcap+CE | d6 +5–15%, d12 +1–5% | **d6 +6 to +11%, d12 +1–2%**: inside the range |
| #6 vmap/batched Muon | 1–4% / 1–5% in real training | noise (the whole optimizer is 1.7–4%) |
| #7 addmm | 0–2% / 0–3% | noise; parity held (rejection risk did not materialize) |
| #8 prefetch | 0–2% / 0–1% | loader is 0.06%, so not built |
| #9 bf16cap | d6 +2–8% | ≈0%: 0.947× the kernel, which was 1.062× base (indirect estimate), so roughly base speed |
| #10 fewer Muon iterations | 0.5–2% / 1–4% | not measurable |
| ceiling | d6 50–60%, d12 60–70% MFU | fresh GPU (short single-variant runs): d6 ~51%, d12 ~61% vs that run's mean matmul peak. Sustained: d6 37–43%, d12 46–48% vs the cold 60.3 peak |

The audit's ranking was right about where the headroom was (CE first). It was optimistic on
the accumulation, batch and optimizer items, because a 4-micro-batch update is already
dominated by compiled GEMMs and the fused SDPA. The audit's API claims all checked out against
the installed MLX 0.32.4: `mx.fast.cross_entropy` (fp32 output; `nn.losses` casts it back to
the logits dtype), the `MLX_SDPA_VJP_BQ/BK` env knobs (read once per process, not per call),
`addmm(alpha, beta)`, and no `donate_argnums` in `mx.compile`. One snippet needed a fix: the
compiled accumulation functions need `outputs=model.state`, because `nn.value_and_grad`
writes traced parameters into the module during the call. Without it, the next eval fails
with "array without a primitive".

## Recommended flags

- **On (now the default): `--ce-mode kernel`.** It is pure systems and passed parity. It gives
  +6–11% at d6, about +2% at d12, and less memory.
- **Optional opt-in:** `MLX_SDPA_VJP_BQ=512 MLX_SDPA_VJP_BK=512` (environment), worth about
  +2–3% at d6 and +2% at d12. It passed parity, but it is below the gate and is an
  undocumented MLX knob.
- **Off:** everything else (`--accum-mode fast`, `--fused-qkv`, `--muon-batched`,
  `--muon-addmm`: no gain). `--ce-mode bf16cap` and `--ns-steps <5` change training and gave
  no gain. Every other new flag defaults to off. With them off and `--ce-mode ref`, the code
  reproduces the original path (`default_vs_orig`: identical losses, delta at GPU-noise level).

## What thermal throttling means for multi-day runs

Under sustained load the measured bf16 matmul peak drops from **~60 to ~39 TFLOP/s**, and
training tok/s falls about 20% within ~4 minutes, then holds a plateau (d12 handoff config:
52k → 42k tok/s over 7 minutes; 30 windows, flat for the last ~10). A multi-day run lives
entirely on the plateau. Size it from plateau numbers, and measure the target machine the same
way: a long enough run, reading the last windows. A Mac mini's cooling differs from this
laptop's, so its plateau/fresh ratio is unknown (not measured).

## Bottom line

On this M5 Max the port is near its practical ceiling for these model sizes. The only real win
is the fused softcap+CE kernel: +6–11% at d6 and about +2% at d12. d12 runs at 46–48% of the
cold matmul peak sustained, and about 61% of the same-time (throttled) peak. That gap is
thermal, not software. For sizing a long run, use the **sustained plateau**: d12 at the
handoff config ≈ **42k tok/s ≈ 27.7 model TFLOP/s** here. Do not use the fresh-GPU 50k.

## Files

- `bench_step.py`: corrected harness (real `make_update`, real loader, interleaved variants)
- `profile_step.py`: per-piece timing; `profile.jsonl`
- `parity.py`, `parity_sdpa.py`: parity gates; `parity.json`, `parity_log.txt`
- `bench_runs.jsonl`: every benchmark run (per-window tok/s, matmul peaks, load averages,
  peak memory); `summarize_runs.py` prints it as a table
- `LOG.tsv`: one row per attempt; `final_runs.log`, `final_long.log`: raw logs of the final runs
- `gpu_idle.sh`: lists other projects' `gpu_run` jobs and the load average
- `checkpoints/pilot/last/optim.safetensors` (needed by the `pilot` parity start) is gitignored, so it exists only on this machine
- `orig/`: the untouched pre-pass `gpt.py`, `optim.py`, `train.py`, `bench.py`, `data.py`,
  used by `default_vs_orig`
- Code changes outside `efficiency/`: `fused_ce.py` (new), `gpt.py` (`ce_mode`, `fused_qkv`,
  `raw_logits`), `optim.py` (`batched`, `ns_addmm`, `ns_steps` plumbing), `train.py`
  (`make_update` shared with the harness; flags `--ce-mode` (default kernel), `--accum-mode`,
  `--fused-qkv`, `--muon-batched`, `--muon-addmm`, `--ns-steps`), and
  `tests/test_efficiency.py`. `--ce-mode ref` reproduces the pilot's exact code path.
