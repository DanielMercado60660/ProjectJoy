# Round 5 — Does faster inference produce better control?

**Turn 1, 2026-10-24.** Frozen models; no training or optimizer. All R5 performance
outcomes await Claude. Existing exhibits remain intact.

**Tasks.** Car: existing held-out tracks, 13 ray/speed/previous-action inputs, strict
walls, 10 Hz sensors, 40 s horizon. Original bicycle equations use ≤10 ms integration
to admit actions between observations; this differs from the old 100 ms integrator.
Score finish/crash, finish time and progress. Tracker: freshly seeded original
40-frame long-occlusion generator, 30 Hz RGB, own GRU state, 90 px/s pursuit cursors.
Score time-weighted pursuit/tracking error, hidden error and time within 5 px,
including startup and periods without actions. Truth is available only to scoring;
the target does not react to pursuit.

**Matrix.** NumPy CPU baseline and MLX GPU use identical frozen weights; Core ML
CPU and CPU+NE load the existing fp16 packages. Smoke also attempts MLX CPU,
compiled GPU and Core ML CPU+GPU. Requested Core ML units do not establish actual
placement. Missing packages, runtime failures and parity failures are recorded gaps;
no conversion or installation. No baseline tuning or fine-tuning occurs. The optional
classical predictor is omitted to isolate engines and scheduling.

Cross engines with FIFO versus newest-pending-frame replacement, and ambient versus
one concurrent 1024² MLX matmul process (~70% duty). One worker owns inference and
recurrent state. The world captures during inference; missed sensor ticks are counted,
never backfilled. In-flight work is not cancelled. Apply completed actions on pickup
(0.5 ms polling interval plus OS delay). Pending work at the horizon is censored and
counted, never drained for credit. A separate tracker matrix adds explicit 0/50 ms
service delay: stress results are not natural engine latency.

**Evidence.** Save per-action age and host preprocess, queue, dispatch/materialization,
synchronized compute, postprocess/state copy, injected wait, pickup and residual overhead.
Core ML internal dispatch remains inside `predict`; compute is not kernel-only timing.
Report p50/p95/p99 alongside coverage and misses over all scheduled frames. Car trials
retain the full horizon after termination; also save active-task age. Record sensor
jitter/loss, backlog, load-work counts, uptime, power, thermal state, versions and thread
caps per episode. Each process caps numerical-library threads at two. Ambient includes
other projects. Shuffle matched seed blocks: fresh nominal 910031+, stress 920031+;
calibration 850031–850042. Old splits are optional secondary comparisons. Show sample
counts and partial blocks; two nominal seeds support only a limited exhibit.

**Budget and custody.** Existing final batch-1 medians (car / tracker, ms): NumPy
0.00283 / 0.25000, MLX GPU 0.18490 / 0.37329, Core ML CPU 0.02683 / 0.09508
(`results/e08_grand_prix.json`, `results/e15_object_permanence.json`). These motivate
nominal plus stress conditions. The pilot measures this loop's wall/horizon ratio and
setup cost. Fill 23 nominal and 5 stress minutes with complete blocks; missing engines
buy more seeds. Expected nominal: two blocks, 64 episodes, ~22 minutes. Hard caps
25 + 6 minutes plus <2-minute smoke fit battery's 35 minutes. No under-training applies.
One selected weight generation (~11 MB), hash-checked config and replaceable small
resume cursors; no per-episode weight snapshots. Interrupted episodes restart from seed;
completed receipts survive. Smoke verifies save/kill/resume before the >20-minute run.
A process supervisor enforces wall limits. Offline HTML replays preserve failures.
