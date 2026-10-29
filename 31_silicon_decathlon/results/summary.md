# Silicon Decathlon: results summary

Generated 2026-10-22 18:25 by `make_report.py` from `results/*.json`.

| event | source | model | params | batches | unit |
|---|---|---|---|---|---|
| 05_dungeon | FINAL (`results/e05_dungeon.json`) | PPO-GRU navigator (MLP enc 55-128-128, GRU 128, pi/v heads) | 1.56e+05 | 1, 512 | agent steps (one recurrent step, hidden state in/out) |
| 07_terra | FINAL (`results/e07_terra.json`) | HNet U-Net heuristic (base 32, GroupNorm, 4 downsamplings) | 1.97e+06 | 1, 16 | heuristic fields (one 96x96 map + goal) |
| 08_grand_prix | FINAL (`results/e08_grand_prix.json`) | ES car policy MLP 13-32-32-2 (tanh) | 1.57e+03 | 1, 256 | car steps (one observation -> one action) |
| 09_alphazero | FINAL (`results/e09_alphazero.json`) | PVNet 5x64 ResNet (policy + value) | 3.76e+05 | 1, 256 | positions |
| 10_nca | FINAL (`results/e10_nca.json`) | NCA lizard_regen (perceive 3x3 + MLP 48-128-16) | 8.32e+03 | 1, 32 | NCA steps of one 72x72 creature |
| 11_physics_detective | FINAL (`results/e11_physics_detective.json`) | Deep-Sets posterior net (enc 42-192-192-128, head 129-192-192-24, 8-comp MoG) | 1.37e+05 | 1, 1000 | datasets (posterior queries; each = up to 15 pushes x 20 frames) |
| 15_object_permanence | FINAL (`results/e15_object_permanence.json`) | CoordConv CNN (4 conv) + FC512 + GRU(256) + Gaussian head, 3 ball slots | 2.99e+06 | 1, 50 | tracker steps (one 64x64 frame for one stream, GRU state in/out) |
| 26_orchestra | FINAL (`results/e26_orchestra.json`) | drum VAE decoder (MLP 24-256-512-864, GELU) | 5.81e+05 | 1, 2000 | patterns |
| 29_image_gen | FINAL (`results/e29_image_gen.json`) | Flow-matching U-Net 64x[1,2,2], attention at 8x8 (EMA) | 6.47e+06 | 1, 64 | U-Net evaluations (one 32x32 image, one Euler step) |
| 30_nanochat | FINAL (`results/e30_nanochat.json`) | Nanochat d6 pilot GPT (6x384, 3 heads, vocab 8192) | 1.69e+07 | 1, 256 | tokens |

## Medal table

Per event and batch size, the three fastest qualifying athletes by median latency (= highest throughput at that batch). Core ML athletes whose compute plan placed no op on the requested unit are excluded (they ran on the CPU; see RESULTS.md, Core ML at batch 1). TF32-pass athletes (`mlx_gpu_tf32`, `mlx_gpu_compiled_tf32`) are exhibition only: they ran in a separate process, not interleaved with the others, and were judged by an error tolerance without the task check, so they get no medal (see the TF32 table below).

| athlete | gold | silver | bronze | total |
|---|---|---|---|---|
| coreml_ane | 5 | 4 | 1 | 10 |
| coreml_cpu | 4 | 7 | 0 | 11 |
| numpy_cpu | 4 | 2 | 1 | 7 |
| coreml_gpu | 3 | 2 | 2 | 7 |
| mlx_gpu_fp16 | 3 | 1 | 2 | 6 |
| mlx_gpu_compiled | 1 | 2 | 7 | 10 |
| mlx_gpu | 0 | 1 | 1 | 2 |
| mlx_gpu_bf16 | 0 | 1 | 0 | 1 |
| mlx_cpu | 0 | 0 | 6 | 6 |

Podium per event:

| event | batch | gold | silver | bronze |
|---|---|---|---|---|
| 05_dungeon | 1 | numpy_cpu (0.0188 ms) | coreml_cpu (0.0477 ms) | mlx_cpu (0.0924 ms) |
| 05_dungeon | 512 | coreml_ane (0.216 ms) | coreml_cpu (0.219 ms) | mlx_gpu_compiled (0.272 ms) |
| 07_terra | 1 | coreml_ane (0.549 ms) | coreml_gpu (1.05 ms) | mlx_gpu_compiled (1.5 ms) |
| 07_terra | 16 | coreml_gpu (5.46 ms) | mlx_gpu_fp16 (5.87 ms) | coreml_ane (6.39 ms) |
| 08_grand_prix | 1 | numpy_cpu (0.00283 ms) | coreml_cpu (0.0268 ms) | mlx_cpu (0.031 ms) |
| 08_grand_prix | 256 | numpy_cpu (0.0285 ms) | coreml_cpu (0.036 ms) | mlx_cpu (0.0468 ms) |
| 09_alphazero | 1 | coreml_cpu (0.0542 ms) | numpy_cpu (0.318 ms) | mlx_cpu (0.391 ms) |
| 09_alphazero | 256 | coreml_gpu (0.512 ms) | coreml_ane (0.795 ms) | mlx_gpu_compiled (1.16 ms) |
| 10_nca | 1 | coreml_ane (0.163 ms) | coreml_cpu (0.179 ms) | mlx_gpu_compiled (0.234 ms) |
| 10_nca | 32 | mlx_gpu_fp16 (1.37 ms) | coreml_ane (1.47 ms) | coreml_gpu (1.64 ms) |
| 11_physics_detective | 1 | coreml_cpu (0.0487 ms) | numpy_cpu (0.0583 ms) | mlx_cpu (0.0892 ms) |
| 11_physics_detective | 1000 | coreml_gpu (0.527 ms) | mlx_gpu_compiled (0.535 ms) | mlx_gpu (0.547 ms) |
| 15_object_permanence | 1 | coreml_cpu (0.0951 ms) | coreml_ane (0.165 ms) | numpy_cpu (0.25 ms) |
| 15_object_permanence | 50 | coreml_ane (0.399 ms) | coreml_gpu (0.515 ms) | mlx_gpu_compiled (0.731 ms) |
| 26_orchestra | 1 | numpy_cpu (0.0245 ms) | coreml_cpu (0.0485 ms) | mlx_cpu (0.0559 ms) |
| 26_orchestra | 2000 | mlx_gpu_compiled (0.513 ms) | mlx_gpu (0.526 ms) | mlx_gpu_fp16 (0.786 ms) |
| 29_image_gen | 1 | coreml_ane (0.497 ms) | coreml_cpu (1.78 ms) | mlx_gpu_fp16 (3.12 ms) |
| 29_image_gen | 64 | mlx_gpu_fp16 (15.1 ms) | mlx_gpu_compiled (21.1 ms) | coreml_gpu (21.2 ms) |
| 30_nanochat | 1 | coreml_cpu (0.383 ms) | coreml_ane (0.592 ms) | mlx_gpu_compiled (0.599 ms) |
| 30_nanochat | 256 | mlx_gpu_fp16 (1.18 ms) | mlx_gpu_bf16 (1.19 ms) | mlx_gpu_compiled (1.57 ms) |

## TF32 exhibition (speed cost of true fp32)

MLX GPU with its default TF32 matmuls, timed in a separate process (`run_event.py all --tf32-pass`) against the true-fp32 athletes of the scored pass. Cross-process differences of a few percent are noise (e.g. e15 b=50 TF32 is slower). Tolerance check only, no task check; no medals.

| event | batch | athlete | true fp32 median ms | TF32 median ms | fp32 / TF32 | TF32 within tolerance |
|---|---|---|---|---|---|---|
| 05_dungeon | 1 | mlx_gpu | 0.2223 | 0.2185 | 1.02 | yes |
| 05_dungeon | 1 | mlx_gpu_compiled | 0.1957 | 0.1892 | 1.03 | yes |
| 05_dungeon | 512 | mlx_gpu | 0.2995 | 0.2914 | 1.03 | yes |
| 05_dungeon | 512 | mlx_gpu_compiled | 0.2715 | 0.2619 | 1.04 | yes |
| 07_terra | 1 | mlx_gpu | 1.651 | 1.66 | 0.99 | yes |
| 07_terra | 1 | mlx_gpu_compiled | 1.5 | 1.504 | 1.00 | yes |
| 07_terra | 16 | mlx_gpu | 7.585 | 7.65 | 0.99 | yes |
| 07_terra | 16 | mlx_gpu_compiled | 7.006 | 7.036 | 1.00 | yes |
| 08_grand_prix | 1 | mlx_gpu | 0.1849 | 0.1805 | 1.02 | yes |
| 08_grand_prix | 1 | mlx_gpu_compiled | 0.1807 | 0.1804 | 1.00 | yes |
| 08_grand_prix | 256 | mlx_gpu | 0.1654 | 0.16 | 1.03 | yes |
| 08_grand_prix | 256 | mlx_gpu_compiled | 0.1641 | 0.1596 | 1.03 | yes |
| 09_alphazero | 1 | mlx_gpu | 0.6919 | 0.6991 | 0.99 | yes |
| 09_alphazero | 1 | mlx_gpu_compiled | 0.5075 | 0.507 | 1.00 | yes |
| 09_alphazero | 256 | mlx_gpu | 1.405 | 1.402 | 1.00 | yes |
| 09_alphazero | 256 | mlx_gpu_compiled | 1.158 | 1.15 | 1.01 | yes |
| 10_nca | 1 | mlx_gpu | 0.299 | 0.296 | 1.01 | yes |
| 10_nca | 1 | mlx_gpu_compiled | 0.2344 | 0.2322 | 1.01 | yes |
| 10_nca | 32 | mlx_gpu | 1.784 | 1.767 | 1.01 | yes |
| 10_nca | 32 | mlx_gpu_compiled | 1.689 | 1.669 | 1.01 | yes |
| 11_physics_detective | 1 | mlx_gpu | 0.2455 | 0.2614 | 0.94 | yes |
| 11_physics_detective | 1 | mlx_gpu_compiled | 0.2298 | 0.2464 | 0.93 | yes |
| 11_physics_detective | 1000 | mlx_gpu | 0.5475 | 0.4917 | 1.11 | yes |
| 11_physics_detective | 1000 | mlx_gpu_compiled | 0.5346 | 0.4772 | 1.12 | yes |
| 15_object_permanence | 1 | mlx_gpu | 0.3733 | 0.3739 | 1.00 | yes |
| 15_object_permanence | 1 | mlx_gpu_compiled | 0.3403 | 0.3365 | 1.01 | yes |
| 15_object_permanence | 50 | mlx_gpu | 0.7764 | 0.7933 | 0.98 | yes |
| 15_object_permanence | 50 | mlx_gpu_compiled | 0.7312 | 0.743 | 0.98 | yes |
| 26_orchestra | 1 | mlx_gpu | 0.1612 | 0.1608 | 1.00 | yes |
| 26_orchestra | 1 | mlx_gpu_compiled | 0.1591 | 0.1584 | 1.00 | yes |
| 26_orchestra | 2000 | mlx_gpu | 0.526 | 0.4028 | 1.31 | yes |
| 26_orchestra | 2000 | mlx_gpu_compiled | 0.5132 | 0.3918 | 1.31 | yes |
| 29_image_gen | 1 | mlx_gpu | 3.686 | 3.695 | 1.00 | yes |
| 29_image_gen | 1 | mlx_gpu_compiled | 3.314 | 3.3 | 1.00 | yes |
| 29_image_gen | 64 | mlx_gpu | 23.36 | 23.02 | 1.01 | yes |
| 29_image_gen | 64 | mlx_gpu_compiled | 21.13 | 20.84 | 1.01 | yes |
| 30_nanochat | 1 | mlx_gpu | 0.6717 | 0.6683 | 1.01 | yes |
| 30_nanochat | 1 | mlx_gpu_compiled | 0.5988 | 0.584 | 1.03 | yes |
| 30_nanochat | 256 | mlx_gpu | 1.673 | 1.298 | 1.29 | yes |
| 30_nanochat | 256 | mlx_gpu_compiled | 1.571 | 1.198 | 1.31 | yes |

## Crossover (best accelerator vs best CPU engine, qualifying athletes; TF32 exhibition rows excluded)

ratio = best accelerator median / best CPU-class median; < 1 means the GPU or Neural Engine wins.

| event | batch | params x batch | numpy ms | best CPU | best accelerator | ratio | winner |
|---|---|---|---|---|---|---|---|
| 08_grand_prix | 1 | 1.57e+03 | 0.00283 | numpy_cpu 0.00283 ms | mlx_gpu_compiled 0.181 ms | 63.79 | numpy_cpu |
| 10_nca | 1 | 8.32e+03 | 0.665 | coreml_cpu 0.179 ms | coreml_ane 0.163 ms | 0.91 | coreml_ane |
| 11_physics_detective | 1 | 1.37e+05 | 0.0583 | coreml_cpu 0.0487 ms | mlx_gpu_compiled 0.23 ms | 4.72 | coreml_cpu |
| 05_dungeon | 1 | 1.56e+05 | 0.0188 | numpy_cpu 0.0188 ms | mlx_gpu_compiled 0.196 ms | 10.42 | numpy_cpu |
| 10_nca | 32 | 2.66e+05 | 20.4 | coreml_cpu 5.38 ms | mlx_gpu_fp16 1.37 ms | 0.26 | mlx_gpu_fp16 |
| 09_alphazero | 1 | 3.76e+05 | 0.318 | coreml_cpu 0.0542 ms | mlx_gpu_compiled 0.507 ms | 9.36 | coreml_cpu |
| 08_grand_prix | 256 | 4.02e+05 | 0.0285 | numpy_cpu 0.0285 ms | mlx_gpu_compiled 0.164 ms | 5.75 | numpy_cpu |
| 26_orchestra | 1 | 5.81e+05 | 0.0245 | numpy_cpu 0.0245 ms | mlx_gpu_compiled 0.159 ms | 6.49 | numpy_cpu |
| 07_terra | 1 | 1.97e+06 | 23.7 | coreml_cpu 2.83 ms | coreml_ane 0.549 ms | 0.19 | coreml_ane |
| 15_object_permanence | 1 | 2.99e+06 | 0.25 | coreml_cpu 0.0951 ms | coreml_ane 0.165 ms | 1.73 | coreml_cpu |
| 29_image_gen | 1 | 6.47e+06 | 17.6 | coreml_cpu 1.78 ms | coreml_ane 0.497 ms | 0.28 | coreml_ane |
| 30_nanochat | 1 | 1.69e+07 | 0.748 | coreml_cpu 0.383 ms | coreml_ane 0.592 ms | 1.54 | coreml_cpu |
| 07_terra | 16 | 3.15e+07 | 419 | coreml_cpu 36.8 ms | coreml_gpu 5.46 ms | 0.15 | coreml_gpu |
| 05_dungeon | 512 | 8.01e+07 | 0.646 | coreml_cpu 0.219 ms | coreml_ane 0.216 ms | 0.99 | coreml_ane |
| 09_alphazero | 256 | 9.63e+07 | 16 | coreml_cpu 3.8 ms | coreml_gpu 0.512 ms | 0.13 | coreml_gpu |
| 11_physics_detective | 1000 | 1.37e+08 | 36.6 | coreml_cpu 1.02 ms | coreml_gpu 0.527 ms | 0.52 | coreml_gpu |
| 15_object_permanence | 50 | 1.49e+08 | 5.96 | coreml_cpu 2.04 ms | coreml_ane 0.399 ms | 0.20 | coreml_ane |
| 29_image_gen | 64 | 4.14e+08 | 1.21e+03 | coreml_cpu 73.5 ms | mlx_gpu_fp16 15.1 ms | 0.21 | mlx_gpu_fp16 |
| 26_orchestra | 2000 | 1.16e+09 | 17 | mlx_cpu 10.2 ms | mlx_gpu_compiled 0.513 ms | 0.05 | mlx_gpu_compiled |
| 30_nanochat | 256 | 4.33e+09 | 12.8 | coreml_cpu 4.08 ms | mlx_gpu_fp16 1.18 ms | 0.29 | mlx_gpu_fp16 |

## 05_dungeon  (FINAL)

PPO-GRU navigator (MLP enc 55-128-128, GRU 128, pi/v heads). Params 1.56e+05. One item = agent steps (one recurrent step, hidden state in/out). Checkpoint `../05_dungeon/checkpoints/solver_cur3/weights.safetensors`.

solver_cur3 = PPO-GRU after 3 curriculum rounds (first policy in 05's demo.html). Task check uses greedy actions (05's eval samples actions; greedy makes the check deterministic). q8/q4: enc1 (in 55) stays fp32. Core ML packages are fp16. At batch 512 CPU fp32 engines differ from the MLX GPU reference (logits ~0.03) because MLX GPU runs fp32 matmuls in TF32 by default here (MLX_ENABLE_TF32='' turns it off). *(Reviewer: this note predates the true-fp32 reference; this run used true fp32, and the CPU fp32 engines agree with it to ~1e-6, see the tables. The event module's note is fixed.)*

MLX GPU fp32 reference: true fp32 (matmul rel err 5e-07). `*_tf32` athletes come from `results/e05_dungeon_tf32.json` (TF32 on, timed in a separate process, not interleaved with the others; agreement vs the saved true-fp32 reference).

Run 2026-10-22 18:02:31; load avg before 1.5/1.5/2.0, after 2.0/1.7/2.0; Now drawing from 'AC Power'; noise probe before/after CPU 0.17 ms, GPU 1.67 ms / CPU 0.17 ms, GPU 1.56 ms.

### batch 1

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 0.01879 | 0.01921 | 53,217 | 1 | logits_max_err 9.54e-07, value_max_err 0, h_max_err 1.19e-07, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 20/20 (ref 20) |  |  | gold |
| mlx_cpu | 0.09238 | 0.103 | 10,825 | 0.203 | logits_max_err 3.58e-07, value_max_err 0, h_max_err 2.09e-07, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 20/20 (ref 20) |  | 0.6 MB | bronze |
| mlx_gpu | 0.2223 | 0.2409 | 4,499 | 0.0845 | logits_max_err 0, value_max_err 0, h_max_err 0, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 20/20 (ref 20) |  | 0.6 MB |  |
| mlx_gpu_tf32 | 0.2185 | 0.2385 | 4,576 | 0.086 | logits_max_err 0, value_max_err 0, h_max_err 0, greedy_action_agreement 1 | yes: greedy action agreement 1.000 on the timing inputs (no task check) |  | 0.6 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 0.6519 | 0.7584 | 1,534 | 0.0288 | logits_max_err 0.00297, value_max_err 3.10e-04, h_max_err 5.65e-04, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 20/20 (ref 20) |  | 0.9 MB |  |
| mlx_gpu_compiled | 0.1957 | 0.2144 | 5,109 | 0.096 | logits_max_err 0, value_max_err 0, h_max_err 0, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 20/20 (ref 20) |  | 0.9 MB |  |
| mlx_gpu_compiled_tf32 | 0.1892 | 0.2142 | 5,285 | 0.0993 | logits_max_err 0, value_max_err 0, h_max_err 0, greedy_action_agreement 1 | yes: greedy action agreement 1.000 on the timing inputs (no task check) |  | 0.6 MB | exhibition (TF32 pass) |
| mlx_gpu_q8 | 0.2399 | 0.256 | 4,168 | 0.0783 | logits_max_err 0.0121, value_max_err 4.71e-05, h_max_err 0.00341, greedy_action_agreement 1 | yes: open-loop action agreement 0.999, escaped 19/20 (ref 20) |  | 1.1 MB |  |
| mlx_gpu_q4 | 0.241 | 0.2618 | 4,149 | 0.078 | logits_max_err 0.215, value_max_err 0.00742, h_max_err 0.046, greedy_action_agreement 1 | **no**: open-loop action agreement 0.997, escaped 18/20 (ref 20) |  | 1.3 MB |  |
| coreml_cpu | 0.04767 | 0.04958 | 20,979 | 0.394 | logits_max_err 0.0313, value_max_err 3.10e-04, h_max_err 0.00228, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 20/20 (ref 20) | CPU 32/32 |  | silver |
| coreml_gpu | 0.04738 | 0.04925 | 21,108 | 0.397 | logits_max_err 0.0313, value_max_err 3.10e-04, h_max_err 0.00228, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 20/20 (ref 20) | CPU 32/32 **(fell back to CPU)** |  |  |
| coreml_ane | 0.05358 | 0.05583 | 18,663 | 0.351 | logits_max_err 0.0313, value_max_err 3.10e-04, h_max_err 0.00228, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 20/20 (ref 20) | CPU 32/32 **(fell back to CPU)** |  |  |

### batch 512

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 0.6461 | 0.6772 | 792,468 | 1 | logits_max_err 3.81e-06, value_max_err 2.38e-07, h_max_err 2.94e-07, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 495/500 (ref 495) |  |  |  |
| mlx_cpu | 0.6026 | 0.6268 | 849,675 | 1.07 | logits_max_err 3.81e-06, value_max_err 2.38e-07, h_max_err 4.77e-07, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 495/500 (ref 495) |  | 5.0 MB |  |
| mlx_gpu | 0.2995 | 0.318 | 1,709,282 | 2.16 | logits_max_err 0, value_max_err 0, h_max_err 0, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 495/500 (ref 495) |  | 5.1 MB |  |
| mlx_gpu_tf32 | 0.2914 | 0.3 | 1,756,938 | 2.22 | logits_max_err 0.0285, value_max_err 8.76e-04, h_max_err 0.00168, greedy_action_agreement 0.998 | yes: greedy action agreement 0.998 on the timing inputs (no task check) |  | 4.5 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 0.691 | 0.7737 | 740,977 | 0.935 | logits_max_err 0.00765, value_max_err 9.43e-04, h_max_err 0.00103, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 495/500 (ref 495) |  | 3.9 MB |  |
| mlx_gpu_compiled | 0.2715 | 0.2937 | 1,885,820 | 2.38 | logits_max_err 0, value_max_err 0, h_max_err 0, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 495/500 (ref 495) |  | 4.7 MB | bronze |
| mlx_gpu_compiled_tf32 | 0.2619 | 0.2685 | 1,954,818 | 2.47 | logits_max_err 0.0285, value_max_err 8.76e-04, h_max_err 0.00168, greedy_action_agreement 0.998 | yes: greedy action agreement 0.998 on the timing inputs (no task check) |  | 3.7 MB | exhibition (TF32 pass) |
| mlx_gpu_q8 | 0.3086 | 0.3261 | 1,658,971 | 2.09 | logits_max_err 0.0266, value_max_err 0.00256, h_max_err 0.00529, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 494/500 (ref 495) |  | 11.7 MB |  |
| mlx_gpu_q4 | 0.3055 | 0.3245 | 1,676,172 | 2.12 | logits_max_err 0.483, value_max_err 0.0408, h_max_err 0.104, greedy_action_agreement 0.996 | yes: open-loop action agreement 0.996, escaped 493/500 (ref 495) |  | 11.9 MB |  |
| coreml_cpu | 0.2192 | 0.2283 | 2,335,234 | 2.95 | logits_max_err 0.0583, value_max_err 0.0042, h_max_err 0.00524, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 495/500 (ref 495) | CPU 32/32 |  | silver |
| coreml_gpu | 0.2181 | 0.2267 | 2,347,730 | 2.96 | logits_max_err 0.0583, value_max_err 0.0042, h_max_err 0.00524, greedy_action_agreement 1 | yes: open-loop action agreement 1.000, escaped 495/500 (ref 495) | CPU 32/32 **(fell back to CPU)** |  |  |
| coreml_ane | 0.2161 | 0.2318 | 2,368,775 | 2.99 | logits_max_err 0.0528, value_max_err 0.00336, h_max_err 0.00602, greedy_action_agreement 0.998 | yes: open-loop action agreement 1.000, escaped 495/500 (ref 495) | NeuralEngine 32/32 |  | gold |


## 07_terra  (FINAL)

HNet U-Net heuristic (base 32, GroupNorm, 4 downsamplings). Params 1.97e+06. One item = heuristic fields (one 96x96 map + goal). Checkpoint `../07_terra/checkpoints/hnet`.

Queries from ../07_terra/data/val.npz (generated, gitignored there; regenerable with its make_dataset.py). Core ML packages are fp16.

MLX GPU fp32 reference: true fp32 (matmul rel err 5e-07). `*_tf32` athletes come from `results/e07_terra_tf32.json` (TF32 on, timed in a separate process, not interleaved with the others; agreement vs the saved true-fp32 reference).

Run 2026-10-22 18:03:16; load avg before 2.0/1.7/2.0, after 2.5/1.9/2.1; Now drawing from 'AC Power'; noise probe before/after CPU 0.17 ms, GPU 1.53 ms / CPU 0.17 ms, GPU 1.56 ms.

### batch 1

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 23.67 | 24.33 | 42 | 1 | r_max_err 2.62e-06, h_max_rel_err 2.62e-06 | yes: A* cost above opt 3.75% vs ref 3.75%, expansions x1.000, same path cost 100%, label err 0.1075 vs 0.1075 |  |  |  |
| mlx_cpu | 15.26 | 15.42 | 66 | 1.55 | r_max_err 2.03e-06, h_max_rel_err 2.03e-06 | yes: A* cost above opt 3.75% vs ref 3.75%, expansions x1.000, same path cost 100%, label err 0.1075 vs 0.1075 |  | 86.6 MB |  |
| mlx_gpu | 1.651 | 1.676 | 606 | 14.3 | r_max_err 0, h_max_rel_err 0 | yes: A* cost above opt 3.75% vs ref 3.75%, expansions x1.000, same path cost 100%, label err 0.1075 vs 0.1075 |  | 49.1 MB |  |
| mlx_gpu_tf32 | 1.66 | 1.696 | 602 | 14.3 | r_max_err 0, h_max_rel_err 0 | yes: r max err 0 (tolerance 0.05) |  | 47.7 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 1.575 | 1.625 | 635 | 15 | r_max_err 0.00187, h_max_rel_err 0.00187 | yes: A* cost above opt 3.73% vs ref 3.75%, expansions x1.000, same path cost 78%, label err 0.1074 vs 0.1075 |  | 31.6 MB |  |
| mlx_gpu_compiled | 1.5 | 1.569 | 667 | 15.8 | r_max_err 0, h_max_rel_err 0 | yes: A* cost above opt 3.75% vs ref 3.75%, expansions x1.000, same path cost 100%, label err 0.1075 vs 0.1075 |  | 49.1 MB | bronze |
| mlx_gpu_compiled_tf32 | 1.504 | 1.57 | 665 | 15.7 | r_max_err 0, h_max_rel_err 0 | yes: r max err 0 (tolerance 0.05) |  | 47.7 MB | exhibition (TF32 pass) |
| coreml_cpu | 2.83 | 2.862 | 353 | 8.36 | r_max_err 0.0122, h_max_rel_err 0.0122 | yes: A* cost above opt 4.08% vs ref 3.75%, expansions x1.007, same path cost 38%, label err 0.1076 vs 0.1075 | CPU 237/237 |  |  |
| coreml_gpu | 1.047 | 1.117 | 955 | 22.6 | r_max_err 0.00145, h_max_rel_err 0.00145 | yes: A* cost above opt 3.75% vs ref 3.75%, expansions x0.998, same path cost 78%, label err 0.1075 vs 0.1075 | GPU 237/237 |  | silver |
| coreml_ane | 0.5494 | 0.5642 | 1,820 | 43.1 | r_max_err 0.028, h_max_rel_err 0.0284 | yes: A* cost above opt 3.87% vs ref 3.75%, expansions x1.002, same path cost 41%, label err 0.1073 vs 0.1075 | NeuralEngine 237/237 |  | gold |

### batch 16

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 419.3 | 426.1 | 38 | 1 | r_max_err 6.32e-06, h_max_rel_err 6.32e-06 | yes: A* cost above opt 3.75% vs ref 3.75%, expansions x1.000, same path cost 100%, label err 0.1075 vs 0.1075 |  |  |  |
| mlx_cpu | 230.4 | 243.2 | 69 | 1.82 | r_max_err 5.90e-06, h_max_rel_err 5.90e-06 | yes: A* cost above opt 3.75% vs ref 3.75%, expansions x1.000, same path cost 100%, label err 0.1075 vs 0.1075 |  | 1184.2 MB |  |
| mlx_gpu | 7.585 | 7.722 | 2,109 | 55.3 | r_max_err 0, h_max_rel_err 0 | yes: A* cost above opt 3.75% vs ref 3.75%, expansions x1.000, same path cost 100%, label err 0.1075 vs 0.1075 |  | 770.4 MB |  |
| mlx_gpu_tf32 | 7.65 | 7.774 | 2,091 | 54.8 | r_max_err 0.00466, h_max_rel_err 0.00467 | yes: r max err 0.0047 (tolerance 0.05) |  | 769.0 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 5.865 | 5.985 | 2,728 | 71.5 | r_max_err 0.01, h_max_rel_err 0.00997 | yes: A* cost above opt 3.80% vs ref 3.75%, expansions x1.003, same path cost 81%, label err 0.1074 vs 0.1075 |  | 396.4 MB | silver |
| mlx_gpu_compiled | 7.006 | 7.127 | 2,284 | 59.9 | r_max_err 0, h_max_rel_err 0 | yes: A* cost above opt 3.75% vs ref 3.75%, expansions x1.000, same path cost 100%, label err 0.1075 vs 0.1075 |  | 770.4 MB |  |
| mlx_gpu_compiled_tf32 | 7.036 | 7.194 | 2,274 | 59.6 | r_max_err 0.00466, h_max_rel_err 0.00467 | yes: r max err 0.0047 (tolerance 0.05) |  | 769.0 MB | exhibition (TF32 pass) |
| coreml_cpu | 36.84 | 37.23 | 434 | 11.4 | r_max_err 0.0204, h_max_rel_err 0.0205 | yes: A* cost above opt 4.13% vs ref 3.75%, expansions x1.004, same path cost 38%, label err 0.1076 vs 0.1075 | CPU 237/237 |  |  |
| coreml_gpu | 5.46 | 5.603 | 2,930 | 76.8 | r_max_err 0.00255, h_max_rel_err 0.00255 | yes: A* cost above opt 3.72% vs ref 3.75%, expansions x0.997, same path cost 78%, label err 0.1075 vs 0.1075 | GPU 237/237 |  | gold |
| coreml_ane | 6.389 | 6.457 | 2,504 | 65.6 | r_max_err 0.043, h_max_rel_err 0.0421 | yes: A* cost above opt 3.87% vs ref 3.75%, expansions x1.002, same path cost 41%, label err 0.1073 vs 0.1075 | NeuralEngine 237/237 |  | bronze |

Skipped athletes: `mlx_gpu_q8`: conv net; MLX quantization covers Linear layers only (one 128x128 Linear here); `mlx_gpu_q4`: conv net; MLX quantization covers Linear layers only (one 128x128 Linear here)


## 08_grand_prix  (FINAL)

ES car policy MLP 13-32-32-2 (tanh). Params 1.57e+03. One item = car steps (one observation -> one action). Checkpoint `../08_tiny_grand_prix/checkpoints/es_best.npz`.

Inputs: observations recorded while the numpy policy drives held-out tracks 20-27 (fixed). Task check: vendored sim, strict one-lap trial on held-out tracks 0-11, grip 1.0. q8/q4 quantize layers 2-3 only (layer 1 has in_dim 13, not a multiple of the group size 32), so the first layer stays fp32. Core ML packages are fp16. CPU fp32 engines differ from the MLX GPU reference by ~6e-3 at batch 256 because MLX GPU runs fp32 matmuls in TF32 by default here (MLX_ENABLE_TF32='' turns it off). *(Reviewer: this note predates the true-fp32 reference; this run used true fp32, and the CPU fp32 engines agree with it to ~1e-6, see the tables. The event module's note is fixed.)*

MLX GPU fp32 reference: true fp32 (matmul rel err 5e-07). `*_tf32` athletes come from `results/e08_grand_prix_tf32.json` (TF32 on, timed in a separate process, not interleaved with the others; agreement vs the saved true-fp32 reference).

Run 2026-10-22 18:04:09; load avg before 2.5/1.9/2.1, after 2.4/1.9/2.1; Now drawing from 'AC Power'; noise probe before/after CPU 0.17 ms, GPU 1.60 ms / CPU 0.17 ms, GPU 1.58 ms.

### batch 1

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 0.002833 | 0.002916 | 352,983 | 1 | action_max_err 8.20e-08, action_mean_err 5.59e-08 | yes: finished 12/12, action max err 8.2e-08 |  |  | gold |
| mlx_cpu | 0.03096 | 0.03608 | 32,301 | 0.0915 | action_max_err 1.19e-07, action_mean_err 8.20e-08 | yes: finished 12/12, action max err 1.19e-07 |  | 13.4 MB | bronze |
| mlx_gpu | 0.1849 | 0.2572 | 5,408 | 0.0153 | action_max_err 0, action_mean_err 0 | yes: finished 12/12, action max err 0 |  | 13.4 MB |  |
| mlx_gpu_tf32 | 0.1805 | 0.252 | 5,540 | 0.0157 | action_max_err 0, action_mean_err 0 | yes: action max err 0 |  | 12.0 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 0.332 | 0.4248 | 3,012 | 0.00853 | action_max_err 0.00291, action_mean_err 0.0016 | yes: finished 12/12, action max err 0.00291 |  | 13.4 MB |  |
| mlx_gpu_compiled | 0.1807 | 0.2551 | 5,534 | 0.0157 | action_max_err 0, action_mean_err 0 | yes: finished 12/12, action max err 0 |  | 13.4 MB |  |
| mlx_gpu_compiled_tf32 | 0.1804 | 0.2551 | 5,543 | 0.0157 | action_max_err 0, action_mean_err 0 | yes: action max err 0 |  | 12.0 MB | exhibition (TF32 pass) |
| mlx_gpu_q8 | 0.2167 | 0.2661 | 4,615 | 0.0131 | action_max_err 0.0048, action_mean_err 0.00435 | yes: finished 12/12, action max err 0.0048 |  | 13.4 MB |  |
| mlx_gpu_q4 | 0.2013 | 0.2611 | 4,967 | 0.0141 | action_max_err 0.0485, action_mean_err 0.0341 | yes: finished 12/12, action max err 0.0485 |  | 13.4 MB |  |
| coreml_cpu | 0.02683 | 0.02942 | 37,266 | 0.106 | action_max_err 0.00584, action_mean_err 0.00311 | yes: finished 12/12, action max err 0.00584 | CPU 7/7 |  | silver |
| coreml_gpu | 0.02667 | 0.02887 | 37,500 | 0.106 | action_max_err 0.00584, action_mean_err 0.00311 | yes: finished 12/12, action max err 0.00584 | CPU 7/7 **(fell back to CPU)** |  |  |
| coreml_ane | 0.03292 | 0.03483 | 30,379 | 0.0861 | action_max_err 0.00584, action_mean_err 0.00311 | yes: finished 12/12, action max err 0.00584 | CPU 7/7 **(fell back to CPU)** |  |  |

### batch 256

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 0.02854 | 0.03079 | 8,969,235 | 1 | action_max_err 3.28e-07, action_mean_err 5.43e-08 | yes: finished 12/12, action max err 3.28e-07 |  |  | gold |
| mlx_cpu | 0.04675 | 0.05175 | 5,475,935 | 0.611 | action_max_err 7.15e-07, action_mean_err 8.18e-08 | yes: finished 12/12, action max err 7.15e-07 |  | 13.5 MB | bronze |
| mlx_gpu | 0.1654 | 0.2057 | 1,547,604 | 0.173 | action_max_err 0, action_mean_err 0 | yes: finished 12/12, action max err 0 |  | 13.5 MB |  |
| mlx_gpu_tf32 | 0.16 | 0.1985 | 1,600,420 | 0.178 | action_max_err 0.00592, action_mean_err 6.44e-04 | yes: action max err 0.00592 |  | 12.1 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 0.3009 | 0.3487 | 850,733 | 0.0949 | action_max_err 0.00408, action_mean_err 4.54e-04 | yes: finished 12/12, action max err 0.00408 |  | 13.5 MB |  |
| mlx_gpu_compiled | 0.1641 | 0.2013 | 1,560,176 | 0.174 | action_max_err 0, action_mean_err 0 | yes: finished 12/12, action max err 0 |  | 13.5 MB |  |
| mlx_gpu_compiled_tf32 | 0.1596 | 0.1981 | 1,603,759 | 0.179 | action_max_err 0.00592, action_mean_err 6.44e-04 | yes: action max err 0.00592 |  | 12.1 MB | exhibition (TF32 pass) |
| mlx_gpu_q8 | 0.1643 | 0.2041 | 1,558,201 | 0.174 | action_max_err 0.0144, action_mean_err 0.00251 | yes: finished 12/12, action max err 0.0144 |  | 13.5 MB |  |
| mlx_gpu_q4 | 0.1656 | 0.2037 | 1,545,660 | 0.172 | action_max_err 0.0899, action_mean_err 0.0203 | **no**: action max err 0.0899 > 0.05 |  | 13.5 MB |  |
| coreml_cpu | 0.03604 | 0.0375 | 7,102,823 | 0.792 | action_max_err 0.00533, action_mean_err 7.04e-04 | yes: finished 12/12, action max err 0.00533 | CPU 7/7 |  | silver |
| coreml_gpu | 0.03629 | 0.03762 | 7,053,894 | 0.786 | action_max_err 0.00533, action_mean_err 7.04e-04 | yes: finished 12/12, action max err 0.00533 | CPU 7/7 **(fell back to CPU)** |  |  |
| coreml_ane | 0.04375 | 0.0485 | 5,851,431 | 0.652 | action_max_err 0.00533, action_mean_err 7.04e-04 | yes: finished 12/12, action max err 0.00533 | CPU 7/7 **(fell back to CPU)** |  |  |


## 09_alphazero  (FINAL)

PVNet 5x64 ResNet (policy + value). Params 3.76e+05. One item = positions. Checkpoint `../09_tiny_alphazero/checkpoints/main/gen_191`.

Inputs are positions from random legal play (0-29 plies).

MLX GPU fp32 reference: true fp32 (matmul rel err 5e-07). `*_tf32` athletes come from `results/e09_alphazero_tf32.json` (TF32 on, timed in a separate process, not interleaved with the others; agreement vs the saved true-fp32 reference).

Run 2026-10-22 18:04:37; load avg before 2.4/1.9/2.1, after 2.5/2.0/2.1; Now drawing from 'AC Power'; noise probe before/after CPU 0.17 ms, GPU 1.64 ms / CPU 0.17 ms, GPU 1.61 ms.

### batch 1

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 0.3178 | 0.3268 | 3,147 | 1 | policy_max_err 1.13e-06, value_max_err 0 | yes: policy err 1.13e-06 (tol 0.1), value err 0 (tol 0.05) |  |  | silver |
| mlx_cpu | 0.3914 | 0.4098 | 2,555 | 0.812 | policy_max_err 1.31e-06, value_max_err 0 | yes: policy err 1.31e-06 (tol 0.1), value err 0 (tol 0.05) |  | 15.5 MB | bronze |
| mlx_gpu | 0.6919 | 0.7083 | 1,445 | 0.459 | policy_max_err 0, value_max_err 0 | yes: policy err 0 (tol 0.1), value err 0 (tol 0.05) |  | 14.9 MB |  |
| mlx_gpu_tf32 | 0.6991 | 0.715 | 1,430 | 0.455 | policy_max_err 0, value_max_err 0 | yes: policy err 0 (tol 0.1), value err 0 (tol 0.05) |  | 13.6 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 0.9178 | 0.9612 | 1,090 | 0.346 | policy_max_err 0.00282, value_max_err 2.07e-04 | yes: policy err 0.00282 (tol 0.1), value err 0.000207 (tol 0.05) |  | 15.7 MB |  |
| mlx_gpu_compiled | 0.5075 | 0.554 | 1,971 | 0.626 | policy_max_err 0, value_max_err 0 | yes: policy err 0 (tol 0.1), value err 0 (tol 0.05) |  | 15.7 MB |  |
| mlx_gpu_compiled_tf32 | 0.507 | 0.555 | 1,973 | 0.627 | policy_max_err 0, value_max_err 0 | yes: policy err 0 (tol 0.1), value err 0 (tol 0.05) |  | 13.6 MB | exhibition (TF32 pass) |
| coreml_cpu | 0.05421 | 0.05825 | 18,447 | 5.86 | policy_max_err 0.0058, value_max_err 2.07e-04 | yes: policy err 0.0058 (tol 0.1), value err 0.000207 (tol 0.05) | CPU 43/43 |  | gold |
| coreml_gpu | 0.05562 | 0.059 | 17,978 | 5.71 | policy_max_err 0.0058, value_max_err 2.07e-04 | yes: policy err 0.0058 (tol 0.1), value err 0.000207 (tol 0.05) | CPU 43/43 **(fell back to CPU)** |  |  |
| coreml_ane | 0.05567 | 0.05933 | 17,964 | 5.71 | policy_max_err 0.0058, value_max_err 2.07e-04 | yes: policy err 0.0058 (tol 0.1), value err 0.000207 (tol 0.05) | CPU 43/43 **(fell back to CPU)** |  |  |

### batch 256

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 15.96 | 16.25 | 16,036 | 1 | policy_max_err 8.34e-06, value_max_err 3.19e-06 | yes: best-move agreement 1.0000 >= 0.99 |  |  |  |
| mlx_cpu | 27.01 | 27.41 | 9,477 | 0.591 | policy_max_err 6.68e-06, value_max_err 4.29e-06 | yes: best-move agreement 1.0000 >= 0.99 |  | 174.7 MB |  |
| mlx_gpu | 1.405 | 1.447 | 182,142 | 11.4 | policy_max_err 0, value_max_err 0 | yes: best-move agreement 1.0000 >= 0.99 |  | 44.8 MB |  |
| mlx_gpu_tf32 | 1.402 | 1.436 | 182,645 | 11.4 | policy_max_err 0.0106, value_max_err 0.00105 | yes: policy err 0.0106 (tol 0.1), value err 0.00105 (tol 0.05) |  | 42.7 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 1.601 | 1.679 | 159,913 | 9.97 | policy_max_err 0.0147, value_max_err 0.0153 | yes: best-move agreement 0.9994 >= 0.99 |  | 31.1 MB |  |
| mlx_gpu_compiled | 1.158 | 1.219 | 221,071 | 13.8 | policy_max_err 0, value_max_err 0 | yes: best-move agreement 1.0000 >= 0.99 |  | 45.6 MB | bronze |
| mlx_gpu_compiled_tf32 | 1.15 | 1.204 | 222,681 | 13.9 | policy_max_err 0.0106, value_max_err 0.00105 | yes: policy err 0.0106 (tol 0.1), value err 0.00105 (tol 0.05) |  | 42.7 MB | exhibition (TF32 pass) |
| coreml_cpu | 3.801 | 3.832 | 67,347 | 4.2 | policy_max_err 0.0414, value_max_err 0.0227 | yes: best-move agreement 0.9983 >= 0.99 | CPU 43/43 |  |  |
| coreml_gpu | 0.5116 | 0.5368 | 500,407 | 31.2 | policy_max_err 0.01, value_max_err 0.00329 | yes: best-move agreement 0.9994 >= 0.99 | GPU 43/43 |  | gold |
| coreml_ane | 0.7948 | 0.8103 | 322,097 | 20.1 | policy_max_err 0.0103, value_max_err 0.00461 | yes: best-move agreement 1.0000 >= 0.99 | NeuralEngine 43/43 |  | silver |

Skipped athletes: `mlx_gpu_q8`: conv-heavy net; MLX quantization covers Linear layers only; `mlx_gpu_q4`: conv-heavy net; MLX quantization covers Linear layers only


## 10_nca  (FINAL)

NCA lizard_regen (perceive 3x3 + MLP 48-128-16). Params 8.32e+03. One item = NCA steps of one 72x72 creature. Checkpoint `../10_self_healing_pixels/checkpoints/lizard_regen/model.safetensors`.

Fire mask is an input (fixed seed). Inputs are creatures grown 200 steps (MLX CPU, fixed masks). Core ML packages are fp16.

MLX GPU fp32 reference: true fp32 (matmul rel err 5e-07). `*_tf32` athletes come from `results/e10_nca_tf32.json` (TF32 on, timed in a separate process, not interleaved with the others; agreement vs the saved true-fp32 reference).

Run 2026-10-22 18:05:06; load avg before 2.5/2.0/2.1, after 2.2/2.0/2.1; Now drawing from 'AC Power'; noise probe before/after CPU 0.17 ms, GPU 1.66 ms / CPU 0.18 ms, GPU 1.57 ms.

### batch 1

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 0.6651 | 0.6909 | 1,503 | 1 | state_max_err 3.81e-06, rgba_max_err 1.19e-07, alive_cells_diff 0 | yes: regrowth MSE 0.004638 vs ref 0.004638 (0.0%), recovered 0.00 vs 0.00 |  |  |  |
| mlx_cpu | 0.7499 | 0.7764 | 1,333 | 0.887 | state_max_err 3.81e-06, rgba_max_err 1.19e-07, alive_cells_diff 0 | yes: regrowth MSE 0.004638 vs ref 0.004638 (0.0%), recovered 0.00 vs 0.00 |  | 24.4 MB |  |
| mlx_gpu | 0.299 | 0.3075 | 3,344 | 2.22 | state_max_err 0, rgba_max_err 0, alive_cells_diff 0 | yes: regrowth MSE 0.004638 vs ref 0.004638 (0.0%), recovered 0.00 vs 0.00 |  | 21.2 MB |  |
| mlx_gpu_tf32 | 0.296 | 0.303 | 3,378 | 2.25 | state_max_err 0.00903, rgba_max_err 6.36e-04, alive_cells_diff 0 | yes: one-step RGBA max err 0.00064 (tolerance 5e-3) |  | 18.3 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 0.4247 | 0.4628 | 2,354 | 1.57 | state_max_err 0.0484, rgba_max_err 0.00104, alive_cells_diff 0 | yes: regrowth MSE 0.004636 vs ref 0.004638 (0.0%), recovered 0.00 vs 0.00 |  | 19.3 MB |  |
| mlx_gpu_compiled | 0.2344 | 0.2578 | 4,267 | 2.84 | state_max_err 0, rgba_max_err 0, alive_cells_diff 0 | yes: regrowth MSE 0.004638 vs ref 0.004638 (0.0%), recovered 0.00 vs 0.00 |  | 21.1 MB | bronze |
| mlx_gpu_compiled_tf32 | 0.2322 | 0.2383 | 4,307 | 2.86 | state_max_err 0.00903, rgba_max_err 6.36e-04, alive_cells_diff 0 | yes: one-step RGBA max err 0.00064 (tolerance 5e-3) |  | 18.2 MB | exhibition (TF32 pass) |
| coreml_cpu | 0.179 | 0.1847 | 5,585 | 3.71 | state_max_err 0.0484, rgba_max_err 0.00145, alive_cells_diff 0 | yes: regrowth MSE 0.004639 vs ref 0.004638 (0.0%), recovered 0.00 vs 0.00 | CPU 21/21 |  | silver |
| coreml_gpu | 0.1789 | 0.1837 | 5,590 | 3.72 | state_max_err 0.0484, rgba_max_err 0.00145, alive_cells_diff 0 | yes: regrowth MSE 0.004639 vs ref 0.004638 (0.0%), recovered 0.00 vs 0.00 | CPU 21/21 **(fell back to CPU)** |  |  |
| coreml_ane | 0.1629 | 0.177 | 6,140 | 4.08 | state_max_err 0.0484, rgba_max_err 0.00104, alive_cells_diff 0 | yes: regrowth MSE 0.004645 vs ref 0.004638 (0.2%), recovered 0.00 vs 0.00 | NeuralEngine 21/21 |  | gold |

### batch 32

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 20.35 | 21.3 | 1,572 | 1 | state_max_err 7.63e-06, rgba_max_err 3.35e-07, alive_cells_diff 0 | yes: regrowth MSE 0.00373 vs ref 0.00373 (0.0%), recovered 0.75 vs 0.75 |  |  |  |
| mlx_cpu | 22.78 | 23.13 | 1,405 | 0.893 | state_max_err 7.63e-06, rgba_max_err 2.38e-07, alive_cells_diff 0 | yes: regrowth MSE 0.00373 vs ref 0.00373 (0.0%), recovered 0.75 vs 0.75 |  | 264.4 MB |  |
| mlx_gpu | 1.784 | 1.829 | 17,936 | 11.4 | state_max_err 0, rgba_max_err 0, alive_cells_diff 0 | yes: regrowth MSE 0.00373 vs ref 0.00373 (0.0%), recovered 0.75 vs 0.75 |  | 162.6 MB |  |
| mlx_gpu_tf32 | 1.767 | 1.817 | 18,114 | 11.5 | state_max_err 0.019, rgba_max_err 9.25e-04, alive_cells_diff 0 | yes: one-step RGBA max err 0.00092 (tolerance 5e-3) |  | 159.6 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 1.372 | 1.445 | 23,317 | 14.8 | state_max_err 0.0696, rgba_max_err 0.00107, alive_cells_diff 0 | yes: regrowth MSE 0.003728 vs ref 0.00373 (0.1%), recovered 0.75 vs 0.75 |  | 100.6 MB | gold |
| mlx_gpu_compiled | 1.689 | 1.743 | 18,950 | 12.1 | state_max_err 0, rgba_max_err 0, alive_cells_diff 0 | yes: regrowth MSE 0.00373 vs ref 0.00373 (0.0%), recovered 0.75 vs 0.75 |  | 160.5 MB |  |
| mlx_gpu_compiled_tf32 | 1.669 | 1.723 | 19,178 | 12.2 | state_max_err 0.019, rgba_max_err 9.25e-04, alive_cells_diff 0 | yes: one-step RGBA max err 0.00092 (tolerance 5e-3) |  | 157.5 MB | exhibition (TF32 pass) |
| coreml_cpu | 5.381 | 5.915 | 5,947 | 3.78 | state_max_err 0.0696, rgba_max_err 0.00242, alive_cells_diff 0 | yes: regrowth MSE 0.003729 vs ref 0.00373 (0.0%), recovered 0.75 vs 0.75 | CPU 21/21 |  |  |
| coreml_gpu | 1.645 | 1.749 | 19,456 | 12.4 | state_max_err 0.0696, rgba_max_err 0.00107, alive_cells_diff 0 | yes: regrowth MSE 0.003727 vs ref 0.00373 (0.1%), recovered 0.75 vs 0.75 | GPU 21/21 |  | bronze |
| coreml_ane | 1.472 | 1.517 | 21,734 | 13.8 | state_max_err 0.0696, rgba_max_err 0.0014, alive_cells_diff 0 | yes: regrowth MSE 0.003731 vs ref 0.00373 (0.0%), recovered 0.75 vs 0.75 | NeuralEngine 21/21 |  | silver |

Skipped athletes: `mlx_gpu_q8`: fc1 input dim 48 is not a multiple of any MLX quantization group size (32/64/128); the whole net is 8k params; `mlx_gpu_q4`: same as q8


## 11_physics_detective  (FINAL)

Deep-Sets posterior net (enc 42-192-192-128, head 129-192-192-24, 8-comp MoG). Params 1.37e+05. One item = datasets (posterior queries; each = up to 15 pushes x 20 frames). Checkpoint `../11_physics_detective/checkpoints/sbi`.

Inputs: fresh simulated datasets (eval_sbi.py recipe, fixed seed). numpy_cpu uses scipy.special.erf for exact GELU. q8/q4: first encoder layer (in 42) and first head layer (in 129) stay fp32 (in_dim not a multiple of group size 32). Core ML packages are fp16. The MLX GPU fp32 reference runs fp32 matmuls in TF32 by default on this machine (MLX 0.32.4; MLX_ENABLE_TF32='' turns it off), so CPU fp32 engines differ from it by more than fp32 rounding; see qualifies().

MLX GPU fp32 reference: true fp32 (matmul rel err 5e-07). `*_tf32` athletes come from `results/e11_physics_detective_tf32.json` (TF32 on, timed in a separate process, not interleaved with the others; agreement vs the saved true-fp32 reference).

Run 2026-10-22 18:05:49; load avg before 2.2/2.0/2.1, after 2.0/2.0/2.1; Now drawing from 'AC Power'; noise probe before/after CPU 0.19 ms, GPU 1.56 ms / CPU 0.17 ms, GPU 1.66 ms.

### batch 1

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 0.05833 | 0.06158 | 17,143 | 1 | weight_max_err 4.17e-07, loc_max_err 1.97e-06, scale_max_err 1.60e-07, post_mean_max_err 1.48e-07 | yes: median shift 2.09e-05 sd, sd ratio 1.000, RMSE 0.0425 (+0.0%) |  |  | silver |
| mlx_cpu | 0.08917 | 0.0975 | 11,215 | 0.654 | weight_max_err 4.17e-07, loc_max_err 2.68e-06, scale_max_err 1.64e-07, post_mean_max_err 2.53e-07 | yes: median shift 2.32e-05 sd, sd ratio 1.000, RMSE 0.0425 (+0.0%) |  | 17.2 MB | bronze |
| mlx_gpu | 0.2455 | 0.263 | 4,073 | 0.238 | weight_max_err 0, loc_max_err 0, scale_max_err 0, post_mean_max_err 0 | yes: median shift 0 sd, sd ratio 1.000, RMSE 0.0425 (+0.0%) |  | 17.2 MB |  |
| mlx_gpu_tf32 | 0.2614 | 0.2695 | 3,825 | 0.223 | weight_max_err 1.02e-04, loc_max_err 0.00329, scale_max_err 7.50e-05, post_mean_max_err 3.49e-04 | yes: posterior mean max err 0.000349 (no task check) |  | 14.2 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 0.6507 | 0.7078 | 1,537 | 0.0897 | weight_max_err 0.00166, loc_max_err 7.44e-04, scale_max_err 1.51e-04, post_mean_max_err 3.18e-05 | yes: median shift 0.0158 sd, sd ratio 1.000, RMSE 0.0426 (+0.1%) |  | 17.5 MB |  |
| mlx_gpu_compiled | 0.2298 | 0.2414 | 4,352 | 0.254 | weight_max_err 0, loc_max_err 0, scale_max_err 0, post_mean_max_err 0 | yes: median shift 0 sd, sd ratio 1.000, RMSE 0.0425 (+0.0%) |  | 17.5 MB |  |
| mlx_gpu_compiled_tf32 | 0.2464 | 0.2539 | 4,059 | 0.237 | weight_max_err 1.02e-04, loc_max_err 0.00329, scale_max_err 7.50e-05, post_mean_max_err 3.49e-04 | yes: posterior mean max err 0.000349 (no task check) |  | 14.2 MB | exhibition (TF32 pass) |
| mlx_gpu_q8 | 0.2472 | 0.2648 | 4,045 | 0.236 | weight_max_err 5.36e-04, loc_max_err 0.00618, scale_max_err 3.50e-04, post_mean_max_err 9.02e-04 | yes: median shift 0.087 sd, sd ratio 1.001, RMSE 0.0426 (+0.0%) |  | 17.8 MB |  |
| mlx_gpu_q4 | 0.2504 | 0.2658 | 3,993 | 0.233 | weight_max_err 0.00914, loc_max_err 0.0646, scale_max_err 0.00624, post_mean_max_err 0.0115 | **no**: median shift 0.482 sd, sd ratio 1.003, RMSE 0.0449 (+5.6%) |  | 18.0 MB |  |
| coreml_cpu | 0.04871 | 0.0525 | 20,531 | 1.2 | weight_max_err 0.00166, loc_max_err 0.00334, scale_max_err 3.37e-04, post_mean_max_err 5.21e-04 | yes: median shift 0.0273 sd, sd ratio 1.000, RMSE 0.0425 (-0.0%) | CPU 27/27 |  | gold |
| coreml_gpu | 0.04883 | 0.0525 | 20,478 | 1.19 | weight_max_err 0.00166, loc_max_err 0.00334, scale_max_err 3.37e-04, post_mean_max_err 5.21e-04 | yes: median shift 0.0273 sd, sd ratio 1.000, RMSE 0.0425 (-0.0%) | CPU 27/27 **(fell back to CPU)** |  |  |
| coreml_ane | 0.05879 | 0.06267 | 17,009 | 0.992 | weight_max_err 0.00166, loc_max_err 0.00334, scale_max_err 3.37e-04, post_mean_max_err 5.21e-04 | yes: median shift 0.0273 sd, sd ratio 1.000, RMSE 0.0425 (-0.0%) | CPU 27/27 **(fell back to CPU)** |  |  |

### batch 1000

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 36.58 | 37.06 | 27,337 | 1 | weight_max_err 2.08e-05, loc_max_err 1.25e-04, scale_max_err 2.15e-05, post_mean_max_err 3.21e-06 | yes: median shift 2.07e-05 sd, sd ratio 1.000, RMSE 0.0425 (+0.0%) |  |  |  |
| mlx_cpu | 19.39 | 19.82 | 51,564 | 1.89 | weight_max_err 2.21e-05, loc_max_err 1.34e-04, scale_max_err 2.29e-05, post_mean_max_err 3.76e-06 | yes: median shift 2.37e-05 sd, sd ratio 1.000, RMSE 0.0425 (+0.0%) |  | 52.4 MB |  |
| mlx_gpu | 0.5475 | 0.6075 | 1,826,624 | 66.8 | weight_max_err 0, loc_max_err 0, scale_max_err 0, post_mean_max_err 0 | yes: median shift 0 sd, sd ratio 1.000, RMSE 0.0425 (+0.0%) |  | 52.4 MB | bronze |
| mlx_gpu_tf32 | 0.4917 | 0.5527 | 2,033,554 | 74.4 | weight_max_err 0.0314, loc_max_err 0.266, scale_max_err 0.0388, post_mean_max_err 0.00243 | yes: posterior mean max err 0.00243 (no task check) |  | 48.6 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 0.8502 | 0.9151 | 1,176,240 | 43 | weight_max_err 0.00767, loc_max_err 0.0478, scale_max_err 0.00941, post_mean_max_err 0.00188 | yes: median shift 0.0158 sd, sd ratio 1.000, RMSE 0.0426 (+0.1%) |  | 38.0 MB |  |
| mlx_gpu_compiled | 0.5346 | 0.5713 | 1,870,470 | 68.4 | weight_max_err 0, loc_max_err 0, scale_max_err 0, post_mean_max_err 0 | yes: median shift 0 sd, sd ratio 1.000, RMSE 0.0425 (+0.0%) |  | 52.8 MB | silver |
| mlx_gpu_compiled_tf32 | 0.4772 | 0.5369 | 2,095,428 | 76.7 | weight_max_err 0.0314, loc_max_err 0.266, scale_max_err 0.0388, post_mean_max_err 0.00243 | yes: posterior mean max err 0.00243 (no task check) |  | 48.6 MB | exhibition (TF32 pass) |
| mlx_gpu_q8 | 0.5856 | 0.6395 | 1,707,577 | 62.5 | weight_max_err 0.0238, loc_max_err 0.0975, scale_max_err 0.037, post_mean_max_err 0.00402 | yes: median shift 0.087 sd, sd ratio 1.001, RMSE 0.0426 (+0.0%) |  | 55.1 MB |  |
| mlx_gpu_q4 | 0.5828 | 0.6111 | 1,715,757 | 62.8 | weight_max_err 0.472, loc_max_err 1.72, scale_max_err 0.295, post_mean_max_err 0.0789 | **no**: median shift 0.482 sd, sd ratio 1.003, RMSE 0.0449 (+5.6%) |  | 55.3 MB |  |
| coreml_cpu | 1.023 | 1.238 | 977,835 | 35.8 | weight_max_err 0.0223, loc_max_err 0.0945, scale_max_err 0.0205, post_mean_max_err 0.00372 | yes: median shift 0.0272 sd, sd ratio 1.000, RMSE 0.0425 (-0.0%) | CPU 27/27 |  |  |
| coreml_gpu | 0.5272 | 0.5619 | 1,896,932 | 69.4 | weight_max_err 0.00639, loc_max_err 0.0435, scale_max_err 0.00745, post_mean_max_err 0.00112 | yes: median shift 0.0131 sd, sd ratio 1.000, RMSE 0.0426 (+0.0%) | GPU 27/27 |  | gold |
| coreml_ane | 1.227 | 1.257 | 814,719 | 29.8 | weight_max_err 0.0866, loc_max_err 0.785, scale_max_err 0.143, post_mean_max_err 0.0529 | **no**: median shift 0.197 sd, sd ratio 0.999, RMSE 0.0419 (-1.6%) | NeuralEngine 27/27 |  |  |


## 15_object_permanence  (FINAL)

CoordConv CNN (4 conv) + FC512 + GRU(256) + Gaussian head, 3 ball slots. Params 2.99e+06. One item = tracker steps (one 64x64 frame for one stream, GRU state in/out). Checkpoint `../15_object_permanence/checkpoints/base/final`.

Timing inputs: frame 20 of held-out test_long sequences with the fp32 hidden state after frames 0-19. q8/q4 quantize the Linear layers (fc 4096x512 + head = 72% of params); convs and the GRU matmuls stay fp32. Core ML packages are fp16.

MLX GPU fp32 reference: true fp32 (matmul rel err 5e-07). `*_tf32` athletes come from `results/e15_object_permanence_tf32.json` (TF32 on, timed in a separate process, not interleaved with the others; agreement vs the saved true-fp32 reference).

Run 2026-10-22 18:06:27; load avg before 2.0/2.0/2.1, after 2.1/2.0/2.1; Now drawing from 'AC Power'; noise probe before/after CPU 0.18 ms, GPU 1.54 ms / CPU 0.17 ms, GPU 1.63 ms.

### batch 1

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 0.25 | 0.2683 | 4,000 | 1 | mean_px_max_err 1.91e-06, sigma_rel_max_err 9.54e-07, rho_max_err 1.41e-08, h_max_err 5.96e-08 | yes: 40-step track: p99 pos dev 8.16e-06 px (tol 0.5), p99 sigma dev 9.54e-07 (tol 0.1) |  |  | bronze |
| mlx_cpu | 0.2677 | 0.2793 | 3,735 | 0.934 | mean_px_max_err 3.93e-06, sigma_rel_max_err 9.54e-07, rho_max_err 1.41e-08, h_max_err 8.94e-08 | yes: 40-step track: p99 pos dev 6.62e-06 px (tol 0.5), p99 sigma dev 9.54e-07 (tol 0.1) |  | 47.0 MB |  |
| mlx_gpu | 0.3733 | 0.3841 | 2,679 | 0.67 | mean_px_max_err 0, sigma_rel_max_err 0, rho_max_err 0, h_max_err 0 | yes: 40-step track: p99 pos dev 0 px (tol 0.5), p99 sigma dev 0 (tol 0.1) |  | 45.6 MB |  |
| mlx_gpu_tf32 | 0.3739 | 0.3804 | 2,674 | 0.669 | mean_px_max_err 0, sigma_rel_max_err 0, rho_max_err 0, h_max_err 0 | yes: one step: pos err 0 px (tol 0.25), sigma err 0 (tol 0.05) |  | 40.9 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 0.6457 | 0.6761 | 1,549 | 0.387 | mean_px_max_err 0.00803, sigma_rel_max_err 0.00205, rho_max_err 5.44e-05, h_max_err 5.10e-04 | yes: 40-step track: p99 pos dev 0.0426 px (tol 0.5), p99 sigma dev 0.00401 (tol 0.1) |  | 45.5 MB |  |
| mlx_gpu_compiled | 0.3403 | 0.3494 | 2,938 | 0.735 | mean_px_max_err 0, sigma_rel_max_err 0, rho_max_err 0, h_max_err 0 | yes: 40-step track: p99 pos dev 0 px (tol 0.5), p99 sigma dev 0 (tol 0.1) |  | 45.6 MB |  |
| mlx_gpu_compiled_tf32 | 0.3365 | 0.3424 | 2,971 | 0.743 | mean_px_max_err 0, sigma_rel_max_err 0, rho_max_err 0, h_max_err 0 | yes: one step: pos err 0 px (tol 0.25), sigma err 0 (tol 0.05) |  | 40.9 MB | exhibition (TF32 pass) |
| mlx_gpu_q8 | 0.3715 | 0.3854 | 2,692 | 0.673 | mean_px_max_err 0.0381, sigma_rel_max_err 0.0108, rho_max_err 9.07e-04, h_max_err 2.70e-04 | yes: 40-step track: p99 pos dev 0.154 px (tol 0.5), p99 sigma dev 0.021 (tol 0.1) |  | 45.6 MB |  |
| mlx_gpu_q4 | 0.3716 | 0.3845 | 2,691 | 0.673 | mean_px_max_err 0.967, sigma_rel_max_err 0.154, rho_max_err 0.0369, h_max_err 0.0111 | **no**: 40-step track: p99 pos dev 2.14 px (tol 0.5), p99 sigma dev 0.39 (tol 0.1) |  | 45.6 MB |  |
| coreml_cpu | 0.09508 | 0.1019 | 10,517 | 2.63 | mean_px_max_err 0.0229, sigma_rel_max_err 0.00507, rho_max_err 4.00e-04, h_max_err 0.00202 | yes: 40-step track: p99 pos dev 0.131 px (tol 0.5), p99 sigma dev 0.0196 (tol 0.1) | CPU 38/38 |  | gold |
| coreml_gpu | 0.09483 | 0.1016 | 10,545 | 2.64 | mean_px_max_err 0.0229, sigma_rel_max_err 0.00507, rho_max_err 4.00e-04, h_max_err 0.00202 | yes: 40-step track: p99 pos dev 0.131 px (tol 0.5), p99 sigma dev 0.0196 (tol 0.1) | CPU 38/38 **(fell back to CPU)** |  |  |
| coreml_ane | 0.1645 | 0.1767 | 6,078 | 1.52 | mean_px_max_err 0.0105, sigma_rel_max_err 0.00205, rho_max_err 1.43e-04, h_max_err 0.00184 | yes: 40-step track: p99 pos dev 0.198 px (tol 0.5), p99 sigma dev 0.011 (tol 0.1) | NeuralEngine 38/38 |  | silver |

### batch 50

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 5.962 | 6.072 | 8,386 | 1 | mean_px_max_err 1.63e-05, sigma_rel_max_err 2.38e-06, rho_max_err 1.48e-07, h_max_err 3.28e-07 | yes: 40-step track: p99 pos dev 1.37e-05 px (tol 0.5), p99 sigma dev 2.38e-06 (tol 0.1) |  |  |  |
| mlx_cpu | 6.739 | 6.866 | 7,420 | 0.885 | mean_px_max_err 1.15e-05, sigma_rel_max_err 3.34e-06, rho_max_err 2.05e-07, h_max_err 3.28e-07 | yes: 40-step track: p99 pos dev 1.41e-05 px (tol 0.5), p99 sigma dev 2.38e-06 (tol 0.1) |  | 147.6 MB |  |
| mlx_gpu | 0.7764 | 0.8449 | 64,398 | 7.68 | mean_px_max_err 0, sigma_rel_max_err 0, rho_max_err 0, h_max_err 0 | yes: 40-step track: p99 pos dev 0 px (tol 0.5), p99 sigma dev 0 (tol 0.1) |  | 79.4 MB |  |
| mlx_gpu_tf32 | 0.7933 | 0.8333 | 63,032 | 7.52 | mean_px_max_err 0.0675, sigma_rel_max_err 0.0111, rho_max_err 5.32e-04, h_max_err 0.00177 | yes: one step: pos err 0.0675 px (tol 0.25), sigma err 0.0111 (tol 0.05) |  | 74.0 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 0.9728 | 1.068 | 51,401 | 6.13 | mean_px_max_err 0.0136, sigma_rel_max_err 0.00305, rho_max_err 1.75e-04, h_max_err 0.00116 | yes: 40-step track: p99 pos dev 0.0401 px (tol 0.5), p99 sigma dev 0.00324 (tol 0.1) |  | 71.7 MB |  |
| mlx_gpu_compiled | 0.7312 | 0.7941 | 68,382 | 8.15 | mean_px_max_err 0, sigma_rel_max_err 0, rho_max_err 0, h_max_err 0 | yes: 40-step track: p99 pos dev 0 px (tol 0.5), p99 sigma dev 0 (tol 0.1) |  | 79.3 MB | bronze |
| mlx_gpu_compiled_tf32 | 0.743 | 0.7825 | 67,299 | 8.03 | mean_px_max_err 0.0675, sigma_rel_max_err 0.0111, rho_max_err 5.32e-04, h_max_err 0.00177 | yes: one step: pos err 0.0675 px (tol 0.25), sigma err 0.0111 (tol 0.05) |  | 73.9 MB | exhibition (TF32 pass) |
| mlx_gpu_q8 | 0.7882 | 0.8519 | 63,438 | 7.56 | mean_px_max_err 0.135, sigma_rel_max_err 0.0204, rho_max_err 0.00276, h_max_err 8.92e-04 | yes: 40-step track: p99 pos dev 0.154 px (tol 0.5), p99 sigma dev 0.021 (tol 0.1) |  | 80.6 MB |  |
| mlx_gpu_q4 | 0.7887 | 0.8448 | 63,393 | 7.56 | mean_px_max_err 1.74, sigma_rel_max_err 0.471, rho_max_err 0.068, h_max_err 0.0176 | **no**: 40-step track: p99 pos dev 2.14 px (tol 0.5), p99 sigma dev 0.39 (tol 0.1) |  | 80.6 MB |  |
| coreml_cpu | 2.036 | 2.073 | 24,560 | 2.93 | mean_px_max_err 0.0876, sigma_rel_max_err 0.0192, rho_max_err 0.00118, h_max_err 0.00523 | yes: 40-step track: p99 pos dev 0.11 px (tol 0.5), p99 sigma dev 0.0176 (tol 0.1) | CPU 38/38 |  |  |
| coreml_gpu | 0.5152 | 0.5534 | 97,048 | 11.6 | mean_px_max_err 0.0136, sigma_rel_max_err 0.00346, rho_max_err 1.68e-04, h_max_err 7.21e-04 | yes: 40-step track: p99 pos dev 0.032 px (tol 0.5), p99 sigma dev 0.00316 (tol 0.1) | GPU 38/38 |  | silver |
| coreml_ane | 0.3991 | 0.4175 | 125,268 | 14.9 | mean_px_max_err 0.0293, sigma_rel_max_err 0.00418, rho_max_err 3.55e-04, h_max_err 0.00611 | yes: 40-step track: p99 pos dev 0.198 px (tol 0.5), p99 sigma dev 0.011 (tol 0.1) | NeuralEngine 38/38 |  | gold |


## 26_orchestra  (FINAL)

drum VAE decoder (MLP 24-256-512-864, GELU). Params 5.81e+05. One item = patterns. Checkpoint `../26_neural_orchestra/checkpoints/vae_cond_b1.0/best`.

Decoder only (the encoder is not used for generation). z ~ N(0, I), styles cycled. Outputs hit probability, velocity, offset per cell. Core ML packages are fp16. SURPRISE: the mlx_gpu fp32 reference is itself not fp32-exact at b=2000. MLX 0.32.4 GPU fp32 matmul with >= 8 rows has TF32-class error (probe: 512-term sums of N(0,1) entries, max err 0.07-0.09 vs fp64 for M = 8..2000, 1e-5 at M = 1; mlx_cpu and numpy 1e-4). So numpy/mlx_cpu disagree with mlx_gpu on 1.25% of threshold patterns while matching an fp64 decode 100%, and mlx_gpu matches fp64 on 98.75%. qualifies() still judges against mlx_gpu as pre-registered; task has *_vs_fp64 fields for the exact comparison.

MLX GPU fp32 reference: true fp32 (matmul rel err 5e-07). `*_tf32` athletes come from `results/e26_orchestra_tf32.json` (TF32 on, timed in a separate process, not interleaved with the others; agreement vs the saved true-fp32 reference).

Run 2026-10-22 18:07:19; load avg before 2.1/2.0/2.1, after 3.2/2.2/2.1; Now drawing from 'AC Power'; noise probe before/after CPU 0.17 ms, GPU 1.58 ms / CPU 0.18 ms, GPU 1.80 ms.

### batch 1

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 0.0245 | 0.02508 | 40,816 | 1 | p_max_err 9.54e-07, vel_max_err 2.98e-07, off_max_err 3.84e-07 | yes: hit-probability max error 9.5e-07 (need <= 0.01) |  |  | gold |
| mlx_cpu | 0.05588 | 0.06175 | 17,897 | 0.438 | p_max_err 1.19e-06, vel_max_err 2.98e-07, off_max_err 3.35e-07 | yes: hit-probability max error 1.2e-06 (need <= 0.01) |  | 63.6 MB | bronze |
| mlx_gpu | 0.1612 | 0.2039 | 6,203 | 0.152 | p_max_err 0, vel_max_err 0, off_max_err 0 | yes: hit-probability max error 0 (need <= 0.01) |  | 63.6 MB |  |
| mlx_gpu_tf32 | 0.1608 | 0.2011 | 6,219 | 0.152 | p_max_err 0, vel_max_err 0, off_max_err 0 | yes: hit-probability max error 0 (need <= 0.01) |  | 58.8 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 0.5598 | 0.6334 | 1,786 | 0.0438 | p_max_err 8.67e-04, vel_max_err 4.72e-04, off_max_err 4.28e-04 | yes: hit-probability max error 0.00087 (need <= 0.01) |  | 63.6 MB |  |
| mlx_gpu_compiled | 0.1591 | 0.2008 | 6,284 | 0.154 | p_max_err 0, vel_max_err 0, off_max_err 0 | yes: hit-probability max error 0 (need <= 0.01) |  | 63.6 MB |  |
| mlx_gpu_compiled_tf32 | 0.1584 | 0.1957 | 6,314 | 0.155 | p_max_err 0, vel_max_err 0, off_max_err 0 | yes: hit-probability max error 0 (need <= 0.01) |  | 58.8 MB | exhibition (TF32 pass) |
| coreml_cpu | 0.0485 | 0.05038 | 20,619 | 0.505 | p_max_err 0.0052, vel_max_err 0.00304, off_max_err 0.00283 | yes: hit-probability max error 0.0052 (need <= 0.01) | CPU 15/15 |  | silver |
| coreml_gpu | 0.04842 | 0.05046 | 20,654 | 0.506 | p_max_err 0.0052, vel_max_err 0.00304, off_max_err 0.00283 | yes: hit-probability max error 0.0052 (need <= 0.01) | CPU 15/15 **(fell back to CPU)** |  |  |
| coreml_ane | 0.052 | 0.05529 | 19,231 | 0.471 | p_max_err 0.0052, vel_max_err 0.00304, off_max_err 0.00283 | yes: hit-probability max error 0.0052 (need <= 0.01) | CPU 15/15 **(fell back to CPU)** |  |  |

### batch 2000

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 17.02 | 17.22 | 117,487 | 1 | p_max_err 2.92e-06, vel_max_err 7.15e-07, off_max_err 5.89e-07 | yes: threshold patterns identical 0.9995 (need >= 0.99), classifier agrees on sampled patterns 1.0000 (need >= 0.99) |  |  |  |
| mlx_cpu | 10.19 | 10.38 | 196,215 | 1.67 | p_max_err 3.31e-06, vel_max_err 8.64e-07, off_max_err 7.67e-07 | yes: threshold patterns identical 0.9995 (need >= 0.99), classifier agrees on sampled patterns 1.0000 (need >= 0.99) |  | 86.1 MB |  |
| mlx_gpu | 0.526 | 0.6006 | 3,802,585 | 32.4 | p_max_err 0, vel_max_err 0, off_max_err 0 | yes: threshold patterns identical 1.0000 (need >= 0.99), classifier agrees on sampled patterns 1.0000 (need >= 0.99) |  | 86.3 MB | silver |
| mlx_gpu_tf32 | 0.4028 | 0.4099 | 4,965,860 | 42.3 | p_max_err 0.00698, vel_max_err 0.00115, off_max_err 0.00105 | yes: hit-probability max error 0.007 (need <= 0.01) |  | 81.6 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 0.786 | 0.8319 | 2,544,529 | 21.7 | p_max_err 0.00302, vel_max_err 9.03e-04, off_max_err 6.52e-04 | yes: threshold patterns identical 0.9940 (need >= 0.99), classifier agrees on sampled patterns 1.0000 (need >= 0.99) |  | 76.8 MB | bronze |
| mlx_gpu_compiled | 0.5132 | 0.5558 | 3,897,374 | 33.2 | p_max_err 0, vel_max_err 0, off_max_err 0 | yes: threshold patterns identical 1.0000 (need >= 0.99), classifier agrees on sampled patterns 1.0000 (need >= 0.99) |  | 86.3 MB | gold |
| mlx_gpu_compiled_tf32 | 0.3918 | 0.397 | 5,105,297 | 43.5 | p_max_err 0.00698, vel_max_err 0.00115, off_max_err 0.00105 | yes: hit-probability max error 0.007 (need <= 0.01) |  | 81.6 MB | exhibition (TF32 pass) |
| coreml_cpu | 1.532 | 1.603 | 1,305,376 | 11.1 | p_max_err 0.0135, vel_max_err 0.00622, off_max_err 0.00449 | **no**: threshold patterns identical 0.9755 (need >= 0.99), classifier agrees on sampled patterns 1.0000 (need >= 0.99) | CPU 15/15 |  |  |
| coreml_gpu | 0.8905 | 0.9797 | 2,245,929 | 19.1 | p_max_err 0.0028, vel_max_err 8.20e-04, off_max_err 5.36e-04 | yes: threshold patterns identical 0.9950 (need >= 0.99), classifier agrees on sampled patterns 1.0000 (need >= 0.99) | GPU 15/15 |  |  |
| coreml_ane | 0.9132 | 0.931 | 2,189,981 | 18.6 | p_max_err 0.0205, vel_max_err 0.00662, off_max_err 0.00328 | **no**: threshold patterns identical 0.9650 (need >= 0.99), classifier agrees on sampled patterns 1.0000 (need >= 0.99) | NeuralEngine 15/15 |  |  |

Skipped athletes: `mlx_gpu_q8`: quantization reserved for the LLM event (brief); also the first layer's 24 inputs don't divide MLX's group size 32/64 (the project's own JS demo does use int8 per-row weights); `mlx_gpu_q4`: same as mlx_gpu_q8


## 29_image_gen  (FINAL)

Flow-matching U-Net 64x[1,2,2], attention at 8x8 (EMA). Params 6.47e+06. One item = U-Net evaluations (one 32x32 image, one Euler step). Checkpoint `../29_image_gen/checkpoints/flow`.

Timed call = one U-Net forward; a full sample = 8 calls (Euler-8). Inputs: Gaussian x, t from the Euler-8 grid (fixed seed). Core ML packages are fp16.

MLX GPU fp32 reference: true fp32 (matmul rel err 5e-07). `*_tf32` athletes come from `results/e29_image_gen_tf32.json` (TF32 on, timed in a separate process, not interleaved with the others; agreement vs the saved true-fp32 reference).

Run 2026-10-22 18:07:43; load avg before 3.2/2.2/2.1, after 2.7/2.4/2.2; Now drawing from 'AC Power'; noise probe before/after CPU 0.17 ms, GPU 1.57 ms / CPU 0.17 ms, GPU 1.76 ms.

### batch 1

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 17.6 | 18.15 | 57 | 1 | v_max_err 9.54e-06, v_mean_err 7.12e-07 | yes: attribute labels agree 100.0% (recipe 100%) on 16 Euler-8 samples; pixel mean err 2.36e-07 |  |  |  |
| mlx_cpu | 13.32 | 13.44 | 75 | 1.32 | v_max_err 7.63e-06, v_mean_err 7.08e-07 | yes: attribute labels agree 100.0% (recipe 100%) on 16 Euler-8 samples; pixel mean err 1.74e-07 |  | 126.3 MB |  |
| mlx_gpu | 3.686 | 3.82 | 271 | 4.77 | v_max_err 0, v_mean_err 0 | yes: attribute labels agree 100.0% (recipe 100%) on 16 Euler-8 samples; pixel mean err 0 |  | 125.0 MB |  |
| mlx_gpu_tf32 | 3.695 | 3.778 | 271 | 4.76 | v_max_err 8.72e-04, v_mean_err 1.01e-04 | yes: velocity max err 0.00087 (tolerance 0.05) |  | 118.8 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 3.124 | 3.251 | 320 | 5.63 | v_max_err 0.0044, v_mean_err 6.54e-04 | yes: attribute labels agree 100.0% (recipe 100%) on 16 Euler-8 samples; pixel mean err 0.000387 |  | 113.1 MB | bronze |
| mlx_gpu_compiled | 3.314 | 3.376 | 302 | 5.31 | v_max_err 0, v_mean_err 0 | yes: attribute labels agree 100.0% (recipe 100%) on 16 Euler-8 samples; pixel mean err 0 |  | 123.3 MB |  |
| mlx_gpu_compiled_tf32 | 3.3 | 3.362 | 303 | 5.33 | v_max_err 8.72e-04, v_mean_err 1.01e-04 | yes: velocity max err 0.00087 (tolerance 0.05) |  | 118.3 MB | exhibition (TF32 pass) |
| coreml_cpu | 1.775 | 1.819 | 563 | 9.91 | v_max_err 0.046, v_mean_err 0.00436 | yes: attribute labels agree 100.0% (recipe 100%) on 16 Euler-8 samples; pixel mean err 0.00139 | CPU 730/730 |  | silver |
| coreml_gpu | 1.775 | 1.816 | 563 | 9.91 | v_max_err 0.046, v_mean_err 0.00436 | yes: attribute labels agree 100.0% (recipe 100%) on 16 Euler-8 samples; pixel mean err 0.00139 | CPU 730/730 **(fell back to CPU)** |  |  |
| coreml_ane | 0.4974 | 0.5131 | 2,011 | 35.4 | v_max_err 0.0345, v_mean_err 0.00675 | yes: attribute labels agree 100.0% (recipe 100%) on 16 Euler-8 samples; pixel mean err 0.00301 | NeuralEngine 730/730 |  | gold |

### batch 64

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 1210 | 1218 | 53 | 1 | v_max_err 1.02e-05, v_mean_err 7.83e-07 | yes: attribute labels agree 100.0% (recipe 100%) on 64 Euler-8 samples; pixel mean err 2.75e-07 |  |  |  |
| mlx_cpu | 705 | 767.6 | 91 | 1.72 | v_max_err 1.07e-05, v_mean_err 7.58e-07 | yes: attribute labels agree 100.0% (recipe 100%) on 64 Euler-8 samples; pixel mean err 2.49e-07 |  | 1525.8 MB |  |
| mlx_gpu | 23.36 | 23.68 | 2,740 | 51.8 | v_max_err 0, v_mean_err 0 | yes: attribute labels agree 100.0% (recipe 100%) on 64 Euler-8 samples; pixel mean err 0 |  | 2395.7 MB |  |
| mlx_gpu_tf32 | 23.02 | 23.41 | 2,780 | 52.6 | v_max_err 0.0111, v_mean_err 9.04e-04 | yes: velocity max err 0.011 (tolerance 0.05) |  | 2380.1 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 15.12 | 15.27 | 4,233 | 80.1 | v_max_err 0.0123, v_mean_err 9.71e-04 | yes: attribute labels agree 100.0% (recipe 100%) on 64 Euler-8 samples; pixel mean err 0.000476 |  | 1248.8 MB | gold |
| mlx_gpu_compiled | 21.13 | 21.37 | 3,029 | 57.3 | v_max_err 0, v_mean_err 0 | yes: attribute labels agree 100.0% (recipe 100%) on 64 Euler-8 samples; pixel mean err 0 |  | 2441.3 MB | silver |
| mlx_gpu_compiled_tf32 | 20.84 | 21.23 | 3,071 | 58.1 | v_max_err 0.0111, v_mean_err 9.04e-04 | yes: velocity max err 0.011 (tolerance 0.05) |  | 2429.6 MB | exhibition (TF32 pass) |
| coreml_cpu | 73.54 | 74.8 | 870 | 16.5 | v_max_err 0.055, v_mean_err 0.00348 | yes: attribute labels agree 100.0% (recipe 100%) on 64 Euler-8 samples; pixel mean err 0.00114 | CPU 730/730 |  |  |
| coreml_gpu | 21.25 | 21.53 | 3,012 | 57 | v_max_err 0.00588, v_mean_err 4.74e-04 | yes: attribute labels agree 100.0% (recipe 100%) on 64 Euler-8 samples; pixel mean err 0.000248 | GPU 730/730 |  | bronze |
| coreml_ane | 28.69 | 29.33 | 2,231 | 42.2 | v_max_err 0.0539, v_mean_err 0.00534 | yes: attribute labels agree 99.7% (recipe 98%) on 64 Euler-8 samples; pixel mean err 0.00271 | NeuralEngine 730/730 |  |  |

Skipped athletes: `mlx_gpu_q8`: conv-dominated U-Net; MLX quantization covers Linear layers only; `mlx_gpu_q4`: conv-dominated U-Net; MLX quantization covers Linear layers only


## 30_nanochat  (FINAL)

Nanochat d6 pilot GPT (6x384, 3 heads, vocab 8192). Params 1.69e+07. One item = tokens. Checkpoint `../30_nanochat_mlx/checkpoints/pilot/last`.

b=1: one KV-cached decode step at position 255 (prefix cached per athlete, own precision). b=256: one 256-token prefill forward with full logits. Inputs are held-out val tokens. mlx_gpu_bf16 is extra: the pilot's own training/inference dtype. q8/q4 = nn.quantize (group 64, affine) on every Linear incl. lm_head and on the token embedding, fp32 activations. Core ML packages are fp16. Core ML decode is stateless with the KV cache as I/O: a 255-token prefix package returns K/V (own fp16 precision, kept host-side as fp32 numpy), and every timed step passes the full past K/V (2 x 6x3x255x128 fp32 = 4.7 MB) in and gets logits + new K/V out; MLX/numpy keep their cache on-device/in-process. bpb at 256-token context is higher than the README's 1.167 (1024 context, bf16). numpy and mlx_cpu differ from mlx_gpu by 0.033 in logits at b=256 (and 0.008 at b=1 through the 255-token prefix): MLX GPU fp32 matmul with >= 8 rows is TF32-class (see e26 notes), so the reference, not the CPU engines, carries that error.

MLX GPU fp32 reference: true fp32 (matmul rel err 5e-07). `*_tf32` athletes come from `results/e30_nanochat_tf32.json` (TF32 on, timed in a separate process, not interleaved with the others; agreement vs the saved true-fp32 reference).

Run 2026-10-22 18:09:39; load avg before 2.7/2.4/2.2, after 2.6/2.4/2.3; Now drawing from 'AC Power'; noise probe before/after CPU 0.17 ms, GPU 1.81 ms / CPU 0.17 ms, GPU 1.60 ms.

### batch 1

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 0.7477 | 0.7821 | 1,337 | 1 | logits_max_err 1.03e-05, argmax_agreement 1 | yes: decode next-token agreement 1.0000 over 128 contexts (need >= 0.98) |  |  |  |
| mlx_cpu | 2.771 | 2.913 | 361 | 0.27 | logits_max_err 5.25e-06, argmax_agreement 1 | yes: decode next-token agreement 1.0000 over 128 contexts (need >= 0.98) |  | 276.8 MB |  |
| mlx_gpu | 0.6717 | 0.7078 | 1,489 | 1.11 | logits_max_err 0, argmax_agreement 1 | yes: decode next-token agreement 1.0000 over 128 contexts (need >= 0.98) |  | 280.7 MB |  |
| mlx_gpu_tf32 | 0.6683 | 0.6772 | 1,496 | 1.12 | logits_max_err 0.0077, argmax_agreement 1 | yes: argmax agreement 1.0000 (need >= 0.98) |  | 266.7 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 0.6268 | 0.6805 | 1,595 | 1.19 | logits_max_err 0.0103, argmax_agreement 1 | yes: decode next-token agreement 0.9844 over 128 contexts (need >= 0.98) |  | 280.6 MB |  |
| mlx_gpu_bf16 | 0.6191 | 0.6461 | 1,615 | 1.21 | logits_max_err 0.1, argmax_agreement 1 | **no**: decode next-token agreement 0.9766 over 128 contexts (need >= 0.98) |  | 282.9 MB |  |
| mlx_gpu_compiled | 0.5988 | 0.6112 | 1,670 | 1.25 | logits_max_err 0, argmax_agreement 1 | yes: decode next-token agreement 1.0000 over 128 contexts (need >= 0.98) |  | 289.7 MB | bronze |
| mlx_gpu_compiled_tf32 | 0.584 | 0.5891 | 1,712 | 1.28 | logits_max_err 0.0077, argmax_agreement 1 | yes: argmax agreement 1.0000 (need >= 0.98) |  | 271.2 MB | exhibition (TF32 pass) |
| mlx_gpu_q8 | 0.6274 | 0.6838 | 1,594 | 1.19 | logits_max_err 0.127, argmax_agreement 1 | **no**: decode next-token agreement 0.9766 over 128 contexts (need >= 0.98) |  | 294.2 MB |  |
| mlx_gpu_q4 | 0.6144 | 0.6587 | 1,628 | 1.22 | logits_max_err 1.73, argmax_agreement 1 | **no**: decode next-token agreement 0.8672 over 128 contexts (need >= 0.98) |  | 298.7 MB |  |
| coreml_cpu | 0.3835 | 0.4095 | 2,608 | 1.95 | logits_max_err 0.0601, argmax_agreement 1 | yes: decode next-token agreement 0.9844 over 128 contexts (need >= 0.98) | CPU 441/441 |  | gold |
| coreml_gpu | 0.3809 | 0.4061 | 2,626 | 1.96 | logits_max_err 0.058, argmax_agreement 1 | yes: decode next-token agreement 0.9844 over 128 contexts (need >= 0.98) | CPU 441/441 **(fell back to CPU)** |  |  |
| coreml_ane | 0.5917 | 0.6091 | 1,690 | 1.26 | logits_max_err 0.0368, argmax_agreement 1 | yes: decode next-token agreement 0.9844 over 128 contexts (need >= 0.98) | NeuralEngine 433/441, CPU 8/441 |  | silver |

### batch 256

| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | Core ML placement | peak MLX mem | medal |
|---|---|---|---|---|---|---|---|---|---|
| numpy_cpu | 12.85 | 13.06 | 19,926 | 1 | logits_max_err 3.70e-05, argmax_agreement 1 | yes: greedy agreement 1.0000 (need >= 0.98), bpb delta +0.0000 (need <= 0.01) |  |  |  |
| mlx_cpu | 13.5 | 13.87 | 18,956 | 0.951 | logits_max_err 3.93e-05, argmax_agreement 1 | yes: greedy agreement 1.0000 (need >= 0.98), bpb delta -0.0000 (need <= 0.01) |  | 309.6 MB |  |
| mlx_gpu | 1.673 | 1.722 | 153,007 | 7.68 | logits_max_err 0, argmax_agreement 1 | yes: greedy agreement 1.0000 (need >= 0.98), bpb delta +0.0000 (need <= 0.01) |  | 359.0 MB |  |
| mlx_gpu_tf32 | 1.298 | 1.304 | 197,195 | 9.9 | logits_max_err 0.0331, argmax_agreement 1 | yes: argmax agreement 1.0000 (need >= 0.98) |  | 313.6 MB | exhibition (TF32 pass) |
| mlx_gpu_fp16 | 1.185 | 1.227 | 216,076 | 10.8 | logits_max_err 0.0576, argmax_agreement 1 | yes: greedy agreement 0.9989 (need >= 0.98), bpb delta +0.0000 (need <= 0.01) |  | 327.8 MB | gold |
| mlx_gpu_bf16 | 1.191 | 1.236 | 214,998 | 10.8 | logits_max_err 0.404, argmax_agreement 0.984 | yes: greedy agreement 0.9884 (need >= 0.98), bpb delta +0.0001 (need <= 0.01) |  | 327.8 MB | silver |
| mlx_gpu_compiled | 1.571 | 1.608 | 162,997 | 8.18 | logits_max_err 0, argmax_agreement 1 | yes: greedy agreement 1.0000 (need >= 0.98), bpb delta +0.0000 (need <= 0.01) |  | 356.8 MB | bronze |
| mlx_gpu_compiled_tf32 | 1.198 | 1.206 | 213,615 | 10.7 | logits_max_err 0.0331, argmax_agreement 1 | yes: argmax agreement 1.0000 (need >= 0.98) |  | 311.3 MB | exhibition (TF32 pass) |
| mlx_gpu_q8 | 1.636 | 1.715 | 156,477 | 7.85 | logits_max_err 0.339, argmax_agreement 0.988 | yes: greedy agreement 0.9917 (need >= 0.98), bpb delta +0.0001 (need <= 0.01) |  | 372.6 MB |  |
| mlx_gpu_q4 | 1.631 | 1.704 | 156,995 | 7.88 | logits_max_err 6.14, argmax_agreement 0.824 | **no**: greedy agreement 0.8361 (need >= 0.98), bpb delta +0.0271 (need <= 0.01) |  | 372.6 MB |  |
| coreml_cpu | 4.083 | 4.153 | 62,697 | 3.15 | logits_max_err 0.357, argmax_agreement 0.996 | yes: greedy agreement 0.9889 (need >= 0.98), bpb delta +0.0000 (need <= 0.01) | CPU 419/419 |  |  |
| coreml_gpu | 1.806 | 2.077 | 141,760 | 7.11 | logits_max_err 0.0396, argmax_agreement 1 | yes: greedy agreement 0.9976 (need >= 0.98), bpb delta +0.0000 (need <= 0.01) | GPU 419/419 |  |  |
| coreml_ane | 1.623 | 1.663 | 157,765 | 7.92 | logits_max_err 0.0942, argmax_agreement 1 | yes: greedy agreement 0.9974 (need >= 0.98), bpb delta -0.0000 (need <= 0.01) | NeuralEngine 411/419, CPU 8/419 |  |  |

