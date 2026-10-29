# Round 5 — When should a world model distrust its dream?

**Hypotheses.** Exact walls plus learned ball interactions should reduce unnecessary free-flight
error; ensemble disagreement may predict error and improve planning; extra search may improve
dreams more than reality. All three can fail. Correctly fitted physics should be very strong.

**Methods.** Five independently initialized, scene-bootstrapped MLX MLPs, two hidden layers,
width 32 or 64. The exact ten-substep gravity/wall integrator is followed by a frame-level pair
correction. A shared directed-edge network, antisymmetrization and a local travel-distance gate
make this permutation equivariant; no true collision flags enter inference. Chain contacts,
near misses and simultaneous wall contacts remain approximation risks. Ordinary planning uses
mean endpoints of independent member rollouts; risk planning adds beta × launched-ball RMS
endpoint disagreement. Both always launch. Baselines: exact walls only, and the true contact
integrator with fitted restitution/separation. The fitted baseline intentionally has a stronger
structural physics prior. All methods receive the same train/validation data and known radius,
gravity and wall law. No pretrained weights.

**Tune before testing.** Seeds 51001/51002 generate 2,000/256 train/validation scenes, 60 frames.
Each width gets five ensemble-training minutes, batch 128/member, three-frame loss, half uniform
and half ball-contact windows. Select width by validation forecast error. Physics gets a
20-point grid plus 60 random proposals over restitution [0.5,1.25], separation [0.5,1.5], then
16 local proposals (SD .04/.08); shortlist eight by training loss, choose by validation loss.
The exact-law hypothesis is included. Both use position/velocity loss scales .06/1.0. Select
beta from {0,.25,.5,1,2} using 24 independent validation planning tasks (seed 51003), 256 shots,
horizon 40: maximize real success, break ties by real distance then smaller beta. Freeze hashes,
checkpoint generation, parameters and beta. Beta zero is a legitimate negative result.

**Search pressure and fairness.** Nested random shots 64/256/1024/4096; CEM population 64,
iterations 1/2/4, shared initial shots/noise and method-specific elites, retaining the incumbent.
Match tasks, bounds and candidates. Separately give classical baselines five times the random
shots to match member × candidate × frame counts. This is not equal FLOPs or wall time; record
costs/timings and disclose shared mean/risk forecasts.

**Metrics.** Fresh scene/target/action seeds 51901/52001/51902: 64 paired tasks, horizon 40;
predeclare 200 on AC. Forecasts use 256 independent scenes, seed 52101, one window each. Report
real/dream success (<.06), fooled rate (dream hit AND real miss, all-task denominator), conditional
fooled rate, and legacy severe misses (real ≥.12). Forecast mean/median errors include collision,
no-collision, ball-collision and no-ball-collision counts; never discard nonfinite failures.
Disagreement: Spearman, error≥radius AUROC and quantile bins on unselected forecasts, separately
on selected plans. Exact paired McNemar, paired bootstrap difference intervals and Holm correction;
primary contrast is risk vs mean at 256 random shots. Coverage is 100%. Old `data/test.npz` is an
optional secondary scene split under the new protocol, never a tuning source.

**Budget and minimum version.** Battery-first guards: smoke 1.9 min, prepare .7, fit 1.5,
training 2×5, tune 2.5, evaluate 14, render .8 = 31.4 minutes plus startup margin (<35 total).
No new battery job below 50%. Minimum complete: both width trials, five members, 96 physics
proposals, 64 fully paired tasks, full pressure/cost sweeps, forecast metrics and checkpoint-reloaded
demo. Timeouts remain partial. On AC, predeclare 200 tasks; allow evaluation continuations ≤40
then ≤35 minutes and old-split secondary ≤5 minutes, total <135. Use `gpu_run`, ≤4 CPU threads,
bounded batches (expected <16 GB), and wall guards. Smoke verifies actual save/kill/resume versus
uninterrupted training before long runs. Preserve raw rows and a fooled-case clip, or largest-gap
clip if none exists. **MLX execution and experiment outcomes are unmeasured in Astra turn 1.**
