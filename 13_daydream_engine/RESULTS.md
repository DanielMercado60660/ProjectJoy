# Daydream Engine: results

Every number below comes from one main training run (seed 0) and is evaluated on held-out test
scenes (seed 1, never seen in training). Sources: `results/eval.json`, `results/plan.json`,
`checkpoints/dynamics.json` and `logs/train.log`.

> **Review correction.** The original headline ("4–16× more accurate
> than the baselines from 5 frames on") was measured only against baselines that ignore *all*
> collisions, so they fly through walls. A cross-review added a third, still cheap baseline:
> **walls-only physics** (the true integrator, gravity and wall restitution, but no ball-ball
> collisions). It is exact until the first ball-ball collision. Against it the model is tied at
> 5 frames and 2.7× / 3.9× / 2.0× better on mean error at 10 / 20 / 50 frames. On the
> *median* rollout it is worse through 20 frames, because walls-only is still exact there.
> **The model's whole advantage is ball-ball collisions.** The planner result holds: 51% vs 35%
> for walls-only (paired exact McNemar p ≈ 6e-5, 200 tasks, one seed).

Units: **mean per-ball Euclidean position error, in box units.** The box side is 1, and the ball
radius is 0.06, so an error of 0.06 means "off by one ball radius". One frame is 20 ms.

## Training
| item | value |
|---|---|
| model | MLP 12 → 4×512 SiLU → 12, residual delta, 800,780 params |
| data | 20,000 train scenes × 100 frames; 2,000 test scenes |
| loss | open-loop unrolled MSE, K ramped 1 → 25 frames over the first 30% |
| run | 20,189 steps × batch 1,024, **16.0 min** wall clock (time budget hit), `gpu_run` slot acquired immediately |
| held-out 20-frame position RMSE (256 scenes, monitor) | 0.169 at init → 0.0140 at the end |
| load during run (busy machine) | `uptime` 4.10 at start, 12.70 / 13.59 / 10.00 at the end. Throughput was about 21 steps/s averaged over the curriculum, **under load** |

## 1. Forecast error by horizon (open loop, 6,000 held-out rollouts from 2,000 scenes)
Rollouts start from the true state at frames 0, 25 and 50 of each of the 2,000 test scenes. Each
runs 50 frames (1 s) with no correction. The three windows of a scene overlap by 25 frames,
so the independent unit is the **2,000 scenes**, not the 6,000 rollouts. Means are from
`results/eval.json → by_horizon`, medians from `by_horizon_median`. No rollout of any
predictor is non-finite (`n_nonfinite_rollouts`).

| horizon (frames) | 1 | 2 | 5 | 10 | 20 | 25 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| **model** | 0.0036 | 0.0051 | **0.0074** | **0.0102** | **0.0222** | **0.0323** | **0.0454** | **0.0803** | **0.1225** |
| constant velocity | 0.0018 | 0.0069 | 0.0387 | 0.1309 | 0.3869 | 0.5274 | 0.6688 | 0.9419 | 1.1927 |
| ballistic, no collisions | 0.0012 | 0.0049 | 0.0287 | 0.1052 | 0.3620 | 0.5351 | 0.7334 | 1.1947 | 1.7473 |
| walls-only physics (review) | **0.0003** | **0.0013** | 0.0079 | 0.0279 | 0.0865 | 0.1167 | 0.1461 | 0.1995 | 0.2404 |

Median error (same rollouts):

| horizon (frames) | 1 | 2 | 5 | 10 | 20 | 25 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| model | 0.0013 | 0.0026 | 0.0046 | 0.0069 | 0.0134 | **0.0184** | **0.0255** | **0.0466** | **0.0804** |
| walls-only physics | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | 0.0543 | 0.1017 | 0.1889 | 0.2443 |
| ballistic, no collisions | 0.0001 | 0.0001 | 0.0078 | 0.0853 | 0.3370 | 0.5094 | 0.7083 | 1.1744 | 1.7288 |

- **Against walls-only physics (added in review), the gap is 2–4×, not 4–16×.** Walls-only
  uses the true integrator and the true wall restitution, so it is exact until two balls
  touch. The share of rollouts with a ball-ball collision so far is 15% by h=5, 28% by h=10,
  49% by h=20 and 84% by h=50 (from the test flags). So on the median rollout it is exactly
  right through h=20, and the model only wins the median from h=25 on (0.018 vs 0.054). On
  the mean, the two are tied at h=5 (0.0074 vs 0.0079), and the model is 2.7× better at h=10,
  3.9× at h=20 and 2.0× at h=50 (0.12 vs 0.24). Everything the model gains over known
  physics comes from learning ball-ball collisions. In free flight and at wall bounces it is
  strictly worse.

- **Against the no-collision baselines, the crossover is at about 2 frames.** At horizon 1
  ballistic beats the model by 3×, and they are tied at horizon 2. From horizon 5 on, the
  model wins by 4× (h=5), 10× (h=10) and 16× (h=20), reaching 14× at h=50. Those baselines fly
  through walls, so this mostly measures "knows about walls" (see the walls-only row).
- **Past the training unroll (25), the model degrades smoothly rather than blowing up.** Error
  goes 0.032 → 0.045 → 0.080 → 0.12 at h = 25 → 30 → 40 → 50. At 1 s the imagined balls are
  about two radii off on average. In the GIF you can watch them drift out of phase in the
  second second.
- **Soft walls.** Imagined balls dig into walls at bounces (`out_of_box`). "Penetration" means
  how far a ball's center goes past the wall-contact line, the point where the ball touches
  the wall. In 84% of model rollouts some ball penetrates by more than 0.02 at some frame. The
  median worst-case penetration per rollout is 0.031 (about half a radius), and the 99th
  percentile is 0.068. A ball center actually leaves the box in 2.5% of model rollouts. Only
  1.2% of individual ball-frames penetrate by more than 0.02, so this happens briefly, right at
  bounces. The no-collision baselines leave the box in 100% of rollouts (median overshoot of 1 to 2 box
  widths). Walls-only physics never does.

## 2. Error at bounces
**One-step error from the true state**, over all 200,000 test transitions, split by what happens
in that frame (`one_step`). Values are position / velocity error:

| | free flight (n=167,726) | wall bounce (n=25,270) | ball-ball (n=7,004) |
|---|---|---|---|
| **model** | 0.00164 / 0.0154 | 0.0127 / **0.185** | 0.0143 / **0.323** |
| ballistic, no collisions | **0.00006 / 0.0000** | 0.0071 / 0.697 | 0.0104 / 1.057 |
| constant velocity | 0.00066 / 0.0600 | 0.0073 / 0.724 | 0.0106 / 1.074 |
| walls-only physics (review) | 0.00000 / 0.0000 | 0.0000 / 0.000 | 0.0097 / 1.009 |

- In a bounce frame, the model's position error is about 8× its free-flight error, and its
  velocity error is 12× (wall) to 21× (ball-ball). **Bounces are where the imagination is
  weakest,** and ball-ball collisions are worse than wall bounces.
- Even so, on a bounce frame the model's velocity error is 3.5× smaller than the no-collision
  baseline's (wall) and 3.3× smaller (ball-ball). It gets most of each reflection right.
- **Surprise: in free flight the model is 27× worse than the trivial ballistic predictor** (one-step
  position error 0.0016 vs 0.00006). A generic MLP never learns gravity exactly. Its
  free-flight error is a small smeared bias, and it compounds slowly.

**Error aligned on the first bounce of each rollout** (`aligned_on_first_bounce`, right panel of
`media/horizon_error.png`). Values are at frame offsets relative to the first collision:

| offset | −5 | −1 | 0 (bounce) | +1 | +5 | +10 | +15 |
|---|---|---|---|---|---|---|---|
| model | 0.0033 | 0.0080 | **0.0125** | 0.0093 | 0.0114 | 0.0168 | 0.0251 |
| ballistic | 0.0004 | 0.0006 | 0.0084 | 0.025 | 0.108 | 0.243 | 0.409 |

- **Not a blow-up, a spike.** The model's error peaks on the bounce frame itself. This
  looks like it blurs the exact bounce time, part reflecting early and part late. The error
  then partly recovers (0.0125 → 0.0093), because the model does reflect the velocity and the
  ball comes back onto the right path. After the bounce, error grows only about 2.7× over 15
  frames. The no-collision baseline jumps 14× on the bounce frame, is 42× higher by the frame after, and keeps climbing.
- The error is already rising in the 5 frames before the bounce (0.0033 → 0.0080). One reading
  is that the model starts "anticipating" the wall a little early. Rollouts with a bounce
  between frames h−1 and h also have higher error at short h (0.0144 at h=1 vs 0.0101 at
  h=5), which is a selection effect of the same spike.
- With/without-collision splits by horizon are in `by_horizon_split`. Almost every window
  collides by h=20: only 159 of 6,000 are still collision-free at h=20, and 0 at h=50. So the
  "no collision" curve is only plotted where n ≥ 30.

## 3. Closed loop: choose a launch to hit a target (imagined vs realized)
There are 200 held-out scenes. The planner picks ball 0's initial velocity from the same 512
random candidates (uniform in [-1.5, 1.5]²), so that ball 0 is at a random target after 40
frames (0.8 s). Success means the final distance is under one ball radius (0.06). Each planner
scores candidates only with its own predictor, and the chosen launch then runs in the real
simulator. 96% of candidate launches make ball 0 bounce at least once, so the task is mostly
about collisions. From `results/plan.json`:

| planner | success (<0.06) | success (<0.12) | realized dist (mean / median) | imagined dist | **gap (realized − imagined)** | "fooled"* | rank corr. with truth | chosen launch bounced |
|---|---|---|---|---|---|---|---|---|
| oracle (true sim; ceiling) | 81.0% | 89.0% | 0.050 / 0.022 | 0.050 | 0.000 | 0% | 1.00 | 92% |
| **model** | **51.0%** | **67.5%** | **0.114 / 0.058** | 0.051 | **+0.063** | 17.5% | **0.86** | 92.5% |
| walls-only physics (review) | 35.0% | 43.5% | 0.214 / 0.166 | 0.067 | +0.148 | 36.0% | 0.52 | 90% |
| ballistic, no collisions | 29.0% | 47.0% | 0.218 / 0.148 | 0.093 | +0.125 | 22.0% | 0.05 | 53% |
| constant velocity | 3.0% | 8.5% | 0.360 / 0.341 | 0.059 | +0.301 | 53.0% | 0.02 | 100% |
| random candidate (floor) | 1.0% | 3.5% | 0.445 / 0.445 | n/a | n/a | n/a | n/a | 96% |

\* "fooled" = the planner believed it would hit (imagined distance < 0.06), but in reality it
missed by more than 0.12.

200 tasks, one seed: 95% Clopper-Pearson intervals are model 44–58%, walls-only 28–42%,
ballistic 23–36%, oracle 75–86%. Model vs walls-only, paired on the same tasks: the model
succeeds where walls-only fails in 47 tasks, and the reverse happens in 15 (exact McNemar
p ≈ 6e-5). Per-task realized distances are in `plan.json → planners.*.realized_dist_per_task`.

- **Exact physics without ball-ball collisions is the most fooled planner after constant
  velocity (36% vs 17.5% for the model).** It is right about walls, so it confidently picks
  bank shots whose path a later ball-ball collision ruins. It ranks candidates moderately
  (rank correlation 0.52 vs 0.86 for the model) and reaches 35%, above ballistic but well below
  the model.
- **The imagined-vs-realized gap is real but modest for the model.** It believes it will hit in
  80.5% of tasks and actually hits in 51%. Its mean realized miss is 2.2× what it imagined.
  The ballistic planner's gap is 2× bigger, and constant velocity is badly fooled: it
  "hits" in 58% of tasks in imagination and in 3% for real.
- **The planner exploits model errors, which the design doc warned about.** By selection, the
  best-looking of 512 candidates is the one where the model's error happens to flatter it.
  That is why the model's mean imagined distance (0.051) matches the oracle's real
  achievable distance (0.050), while its realized distance is 0.114. This is the optimizer's
  curse, measured: 17.5% of tasks fooled.
- **The ballistic planner partly avoids its own blind spot.** It chooses launches with no bounce
  for ball 0 47% of the time, when only 4% of candidates are bounce-free. Bounce-free shots
  are the only ones it can predict. That gives it 29% success, well above random, from a
  model that ranks candidates almost at chance overall (rank correlation 0.05).
- In `media/plan_examples.png` (model planner, the first 6 tasks), tasks 1–4 land within
  0.04 of the target through one or more bounces. Task 0 shows the failure mode: the imagined
  path (dashed) and the real one (black) agree through the first bounces, then split near the
  left wall and floor, so a 0.07 imagined miss becomes 0.24 for real.

## Demo
- `media/real_vs_imagined.gif` (100 frames = 2 s, at 0.5× speed). The columns show three
  held-out scenes in three panels each: REAL, IMAGINED (model, open loop, with thin outlines
  where the real balls are), and the no-collision baseline. The baseline's balls fall through
  the floor within about half a second. The footer text turns red after frame 25, which is
  past the training unroll.
- `media/real_vs_imagined_strip.png`: frames 0, 25, 50 and 100 as a still.
- `media/horizon_error.png`: error-by-horizon, the with/without-bounce split, and bounce-aligned error.
- `media/plan_examples.png`: closed-loop examples, imagined vs realized path.

**A failure I nearly reported, and a bug it exposed.** In the first GIF render, scene 11
seemed to show an imagined ball escaping through the top-right corner. It was a rendering bug.
Baseline balls with x < 0 were being drawn into the neighboring IMAGINED panel. `make_gif.py`
now draws each panel on its own cropped tile, and the claim was dropped. The real imagined
failure is quieter: soft walls (see above). Balls sink about half a radius into the wall
before reflecting. You can see this in the GIF as the gap between filled imagined balls and the
outlined real ones at each bounce.

## What worked
- A plain residual MLP with a multi-step (25-frame) open-loop loss gives smooth, stable 2-second
  imaginations on held-out scenes with no explosions. We used a curriculum (K ramped from 1) as a design choice. It was not ablated.
- The collision flags made the bounce analysis cheap and showed the error structure clearly:
  the error spikes at the bounce frame and does not blow up.
- The model is good enough to plan through 2+ bounces: 51% success vs 81% for the oracle and
  35% for the best cheap baseline (walls-only physics, added in review).

## What didn't / caveats
- In free flight the model is worse than exact physics (27× at one step), so a hybrid
  (analytic ballistic + learned collision correction) would probably dominate both.
- Bounce timing is blurred: the error peaks at the bounce frame.
- The ball identity order is fixed and the MLP is not permutation-invariant. Not tested with a
  different number of balls.
- The original cheap baselines ignored every collision. The review added walls-only physics,
  which is privileged (true integrator and restitution) but still ignores ball-ball contact.
  It is far from the simulator (0.24 at 1 s, 35% planner success), and it is the comparison
  that matters: the model's edge is ball-ball collisions only.
- The held-out monitor logged during training (`test_rmse20`) uses the first 256 test scenes.
  No decision used it (one run, final step saved, no early stopping), but it is test data.
- 6,000 rollouts come from 2,000 scenes with overlapping windows. Treat n = 2,000.

## Not measured
- Pixels / video encoder (go-deeper).
- Uncertainty / ensembles.
- Multiple training seeds (one run per configuration, seed 0). The one ablation run is in the
  addendum.
- Imagination throughput and latency on an idle machine. The only speed number above is the
  training rate, under load.
- A planner with model-only access that is then re-scored for exploits beyond the "fooled"
  rate. An adversarial search (CEM with many iterations) would likely widen the gap and was
  not run.

## Next steps
- Seeds for the ablation below. A middle unroll (5 or 10) to find the sweet spot.
- Hybrid physics + learned residual. Train on collision frames up-weighted.
- A graph network (per-ball, pairwise messages) for permutation invariance and generalization
  to N balls.
- Use CEM refinement instead of 512 random shots, and measure how the fooled rate grows with
  optimization pressure. This is the go-deeper "exploit the dream" experiment.

## Addendum: ablation, one-step loss vs 25-frame multi-step loss
This was run after the write-up was complete. Same model, data, batch, seed and **16-minute
wall-clock budget**, but `--unroll 1` (plain teacher-forced next-frame regression). Because
each step is cheaper, it took 224,008 steps vs 20,189. Sources: `results/eval_ablation_unroll1.json`,
`results/ablation_summary.json` and `results/plan_ablation_unroll1.json`. Checkpoint:
`checkpoints/ablation_unroll1`. Load average at the end: 4.42 / 8.84 / 10.64, **under load**.

**Median** open-loop position error over the same 6,000 held-out rollouts (the mean is useless
here, see below):

| horizon | 1 | 5 | 10 | 20 | 25 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|
| multi-step (main) | 0.0013 | 0.0046 | **0.0069** | **0.0134** | **0.0184** | **0.0255** | **0.0466** | **0.0804** |
| one-step (ablation) | **0.0010** | **0.0046** | 0.0091 | 0.0235 | 0.0360 | 0.0540 | 0.1019 | 0.1600 |

- **The one-step model is the better one-step predictor.** Its one-step error from the true
  state is lower in every event class. Wall bounce position / velocity error is 0.0017 / 0.038,
  vs 0.0127 / 0.185 for the main model, so its bounces are 5–8× sharper.
- **But it is the worse daydreamer.** From about 10 frames out it falls behind, and its
  median error at 1 s is 2× worse. **3 of 6,000 rollouts (0.05%) diverge to infinity** by
  frame 25–40. Its *mean* error is 16.5 at h=25 and inf at h=40, which is the classic
  compounding-error blow-up that the multi-step loss exists to prevent. The main model has 0
  diverged rollouts.
- **Closed loop, same 200 tasks and 512 candidates:** the one-step model reaches 33% success
  vs 51% for the main model. Its gap (realized − imagined) is +0.116 vs +0.063, and it is
  fooled in 32% of tasks vs 17.5%. Better one-step accuracy did not mean better planning.
- A bug this surfaced and fixed: NaN imagined distances made `np.argmin` pick the NaN
  candidate. `plan.py` now maps NaN/inf to +inf before choosing. The main model's planning
  numbers came out identical when rerun after the fix.

## Round 5: A world model that knows when to distrust itself

**TL;DR (turn-2 headline, fresh seeds, 200 paired tasks).** The gain comes from the ensemble
knowing when to distrust itself, not from a better model. Penalising ensemble disagreement
(β=0.5, picked on validation) raises real success from **43.5% to 54.5%** at the prespecified
256 shots (22 risk-only vs 0 mean-only successes, Holm p=4.9e-5) and from **57.0% to 75.0%** at
4,096 shots. It cuts the fooled rate **2–3×**: 26.5→12.5% at 256 shots, 29.5→10.5% at 4,096
(1.9× at 64 shots). The learned mean model is **no better than a fitted soft-contact physics
model** whose family excludes the true hard-contact law. Real success is the same within noise
(after Holm) at every effort, and the soft-contact model forecasts slightly better at horizons 20–40. Against
the member-matched control (soft contact with 5× the shots), risk planning loses at 64 shots,
ties at 256, and wins only at ≥1,024 shots. Searching harder exploits every mean model: dream
success climbs to 85–89%, but real success reaches only 53–58.5% (soft contact, hybrid mean) and
28–35% (walls only), while their fooled rates rise to about 30–44%. The new turn-2 ensemble is **not** significantly better than the frozen
turn-1 model on the same tasks. The exact-law oracle reaches 86.5%. Turn-1 numbers further down
are preliminary: their test seeds informed the turn-2 redesign.

**Turn 3 closes the open control (fresh seeds 73001–73101, 200 tasks).** A *classical* risk-aware
planner (five validated soft-contact fits, same disagreement penalty, β=2 chosen on validation) ties
the learned hybrid risk planner: **77.0% vs 78.5%** real success at 4,096 shots and 61.5% vs 63.5% at
256 (Holm p=1), with fooled rates of 8–11% for both. Classical risk beats classical mean by 20 points at
4,096 shots. So the useful ingredient is the distrust signal, which a cheap parameter ensemble provides as
well as a learned one. See "Turn 3: classical risk-aware control" below.

**Runner fix (both turns).** The first turn-1 smoke died with NaN loss at step 2.
`r5_core.pair_features` took `sqrt` of the zero self-distance, and MLX's VJP produced `0*inf=NaN`
even through a boolean gate. Claude (the runner) wrapped the gate inputs in `stop_gradient`.
Turn 2 kept that fix and also adds an epsilon inside differentiated norms.

### Turn 2 headline: setup and training (run `r5_t2_20261024T161011Z`)

Sources are under `results/r5/r5_t2_20261024T161011Z/` (abbreviated **T2**) and
`logs/r5/r5_t2_20261024T161011Z/`. The runs were on AC power on 2026-10-24, on a busy machine with other
projects training, all through `gpu_run`.

- **Smoke and tests:** `SMOKE_R5_OK seconds=17.1` ([smoke.log](logs/r5/r5_t2_20261024T161011Z/smoke.log));
  NumPy suite 40 passed ([numpy.log](logs/r5/r5_t2_20261024T161011Z/numpy.log)).
- **Soft-contact fit** ([T2 soft_fit.json](results/r5/r5_t2_20261024T161011Z/soft_fit.json)): a
  finite spring/damper with `contains_exact_hard_law=false`. Training ran 256 trials (192 grid/random
  proposals + 64 local refinements), the top 32 by train loss were validated on 128 natural 40-frame
  validation scenes, and the fit took **6.1 s**. Selected: stiffness **14,242**, damping ratio **0.0**
  (the lower bound; the true ball-ball law is elastic), contact radius **0.0644** (true R=0.06).
  Validation mean position error over frames 1–40 is **0.013205**; at frame 40 alone it is 0.0461.
- **Two learned ensembles** (5 members, width 64, depth 3, batch 256, unroll 5, 3× collision weight,
  75% contact windows, same seed) were trained for **840 s** each and stopped by the wall guard.
  The raw-feature ensemble reached **123,244 steps**, the geometry-feature ensemble **85,224**
  ([train_raw.log](logs/r5/r5_t2_20261024T161011Z/train_raw.log),
  [train_geometry.log](logs/r5/r5_t2_20261024T161011Z/train_geometry.log)). Geometry stopped before
  its 100k-step cosine finished (lr 1.19e-4 at the stop, versus the 8e-5 floor). The trials are
  matched on time, not on updates.
- **Validation-only selection** ([T2 selection.json](results/r5/r5_t2_20261024T161011Z/selection.json)).
  This is the ensemble-mean position error on the same 128 validation scenes and frames 1–40 that
  the soft-contact fit used:
  geometry **0.013416** (selected), frozen turn-1 h32 0.014121, raw 0.014494. The soft-contact fit's
  0.013205 is lower than all three, so on validation the classical model already forecasts at least
  as well. β was tuned on 64 separate validation tasks (seed 53003, 256 shots); validation real
  success for β = 0 / .25 / .5 / 1 / 2 was 53.1 / 56.3 / **59.4** / 57.8 / 50.0%. β=0.5 beat β=1 by
  one task, so the pick is noisy, but it was frozen before testing.
- **Fresh test streams:** scenes 62001, actions 62002, targets 62101, forecasts 62201
  ([manifest.json](results/r5/r5_t2_20261024T161011Z/headline/manifest.json)). `require_fresh_turn2`
  rejects every turn-1 and validation seed. Nothing was selected on these seeds. Evaluation completed
  **200/200 tasks in 1,304 s** ([status.json](results/r5/r5_t2_20261024T161011Z/headline/status.json)).
- **Frozen reference:** the turn-1 h32 ensemble with its own turn-1 β=1.0 ran on the same tasks
  (`reference_*`). So `hybrid_risk` vs `reference_risk` compares the model and β together.

### Turn 2 headline: planning (200 paired tasks, horizon 40)

Success means distance <0.06. **Fooled** means dream success and real failure, as a fraction of
all tasks. Coverage is 100% for every method, and there were 0 invalid candidates. Values are
`metrics.<key>` in [T2 summary.json](results/r5/r5_t2_20261024T161011Z/headline/summary.json), in
percent. I recomputed every success and fooled rate in it from the 200 `tasks/*.json` rows, with
0 mismatches. `hybrid_mean` scores the mean endpoint of the 5 members. `hybrid_risk` uses the same
5-member rollouts plus β·disagreement, so the two have identical compute. The classical models
run one rollout per candidate. Their `random_member_matched` rows use 5× the shots (equal member-frames,
not equal FLOPs or wall time).

| Method | 64 shots dream/real/fooled | 256 | 1,024 | 4,096 |
|---|---|---|---|---|
| Walls only | 40.0 / 18.0 / 23.5 | 65.0 / 28.5 / 37.0 | 72.0 / 28.0 / 44.0 | 73.5 / 34.5 / 40.0 |
| Walls only, 5× shots | 66.5 / 32.0 / 36.0 | 72.5 / 28.0 / 44.5 | 74.0 / 32.5 / 41.5 | 75.5 / 34.0 / 42.0 |
| Soft contact (fitted) | 42.5 / 32.5 / 13.5 | 70.0 / 47.5 / 22.5 | 81.0 / 52.5 / 29.0 | 86.0 / 53.0 / 33.5 |
| Soft contact, 5× shots | 72.0 / 50.5 / 22.0 | 82.5 / 52.0 / 31.0 | 86.0 / 53.5 / 33.0 | 88.5 / 58.5 / 30.5 |
| Hybrid mean | 48.0 / 32.5 / 17.0 | 69.0 / 43.5 / 26.5 | 78.5 / 51.5 / 27.5 | 85.5 / 57.0 / 29.5 |
| **Hybrid risk (β=0.5)** | 41.5 / 33.5 / 9.0 | 65.0 / **54.5** / **12.5** | 74.5 / **65.0** / **10.0** | 85.0 / **75.0** / **10.5** |
| Turn-1 ref. mean | 44.5 / 33.0 / 15.5 | 67.5 / 42.5 / 25.5 | 79.0 / 49.5 / 30.5 | 85.5 / 52.5 / 33.0 |
| Turn-1 ref. risk (β=1) | 37.0 / 32.5 / 7.0 | 61.5 / 52.5 / 9.5 | 74.0 / 63.0 / 11.5 | 80.0 / 70.0 / 10.0 |
| Exact-law oracle | 43.5 (all three equal, fooled 0) | 66.0 | 77.0 | 86.5 |
| Exact-law oracle, 5× shots | 69.0 | 80.5 | 87.5 | 89.0 |

CEM (population 64; 1/2/4 iterations = 64/128/256 candidates; iteration 1 *is* the 64-shot
random prefix, so those rows are identical by construction). Real success / fooled:
hybrid mean 32.5/17.0, 43.0/23.0, 49.0/25.0; hybrid risk 33.5/9.0, 49.0/13.0, **60.0/9.5**;
soft contact 32.5/13.5, 50.0/17.5, 57.5/18.5; walls 18.0/23.5, 29.5/36.0, 31.0/43.0; oracle 43.5,
61.0, 77.0 (`*/cem/{1,2,4}`).

**Paired tests** (`paired_tests` in the same summary; exact McNemar; bootstrap 95% CI; Holm across
all **183** stored contrasts, which is conservative). Differences are risk minus the other method,
in percentage points:

| Contrast | 64 | 256 (primary) | 1,024 | 4,096 | CEM×4 |
|---|---|---|---|---|---|
| Risk vs mean (same rollouts) | +1.0, Holm 1 | **+11.0 [7.0, 15.5]**, 22 vs 0, p=4.8e-7, Holm **4.9e-5** | +13.5, Holm 1.3e-4 | +18.0 [11.5, 24.0], Holm 2.1e-6 | +11.0, Holm 0.020 |
| Risk vs soft contact, equal shots | +1.0, Holm 1 | +7.0 [2.0, 12.0], p=0.013, Holm 0.70 | +12.5, Holm 0.0049 | +22.0, Holm 5.2e-8 | +2.5, Holm 1 |
| Risk vs soft contact, 5× shots | **−17.0**, Holm 1.8e-4 | +2.5 [−4.0, 9.5], Holm 1 | +11.5 [5.5, 17.5], Holm 0.038 | +16.5 [9.5, 23.0], Holm 2.9e-4 | — |
| Risk vs turn-1 reference risk | +1.0, Holm 1 | +2.0 (9 vs 5), Holm 1 | +2.0, Holm 1 | +5.0 (13 vs 3), p=0.021, Holm 1 | −3.5, Holm 1 |

Mean model vs soft contact at equal shots (soft minus mean): 0.0, +4.0 (p=0.24), +1.0, −4.0 (p=0.32),
and +8.5 at CEM×4 (p=0.012, Holm 0.66). None is significant after Holm. Hybrid mean vs turn-1
reference mean is also n.s. everywhere. The oracle beats hybrid risk by 11.5 points at 4,096 shots
(23 vs 0, Holm 2.7e-5).

**Fooled rate** (reviewer-computed from task rows; unadjusted exact McNemar on the fooled indicator):
mean→risk 17.0→9.0% at 64 shots (17 vs 1 discordant, p=1.4e-4), 26.5→12.5% at 256 (28 vs 0,
p=7.5e-9), 27.5→10.0% at 1,024, 29.5→10.5% at 4,096 (41 vs 3, p=1.6e-9), 25.0→9.5% at CEM×4. Conditional on
a dream success, the mean planner is wrong 38.4% / 34.5% of the time at 256 / 4,096 shots, versus
19.2% / 12.4% for risk (`fooled_given_dream_success`). Soft contact with 5× shots is fooled 30–33%
of the time from 256 shots up. Risk planning keeps most dream success (65.0 vs 69.0% at 256),
and it picks the same launch as the mean planner in 64% of tasks at 256 shots and 48% at 4,096.

**Effort sweep.** For every model with fooled > 0, dream success rises with search, faster than
real success does. Walls-only real success sits at 28–35% from 256 shots up while dream reaches
73.5%. Its fooled rate grows from 23.5% to 40–44%. Past 1,024 shots, soft contact and
hybrid mean gain only 0.5 and 5.5 points of real success (52.5→53.0, 51.5→57.0), while dream climbs to 86%. The risk
planner converts more of the extra search into real success (54.5→65.0→75.0) and keeps its fooled rate flat at 10–12.5%; the mean planner converts less (43.5→51.5→57.0) while its fooled rate rises. Plot:
[pressure.png](media/r5_t2_20261024T161011Z_pressure.png). Wall time per task (median
`prediction_seconds_shared`, busy machine, NumPy CPU vs MLX GPU, so this is not a fair hardware
comparison): soft contact over 20,480 shots 2.1 s, hybrid ensemble over 4,096 shots 0.35 s.

### Turn 2 headline: forecasting and uncertainty (256 fresh scenes, seed 62201)

`metrics.<method>.<h>.all` mean (median) per-ball position error, from
[T2 forecast.json](results/r5/r5_t2_20261024T161011Z/headline/forecast.json); 0 nonfinite:

| Horizon | Hybrid mean | Soft contact | Turn-1 ref. mean | Walls only |
|---:|---:|---:|---:|---:|
| 1 | 0.0000538 (1.2e-7) | 0.0000574 (0) | 0.0000488 (1.2e-7) | 0.000123 (0) |
| 5 | 0.000558 (6.3e-7) | 0.000769 (0) | 0.000521 (5.9e-7) | 0.00638 (0) |
| 10 | 0.00208 (1.6e-6) | 0.00205 (0) | 0.00221 (1.3e-6) | 0.0246 (0) |
| 20 | 0.00810 (0.00183) | 0.00729 (0) | 0.00893 (9.1e-5) | 0.0718 (0) |
| 40 | 0.0443 (0.0165) | **0.0372** (0.0141) | 0.0430 (0.0197) | 0.2007 (0.1945) |

The oracle's error is exactly 0. At horizon 40, the 199 scenes with a ball-ball collision have mean
(median) error 0.0543 (0.0243) for the hybrid and 0.0452 (0.0193) for soft contact. The 57 without one have
0.00954 vs 0.00937. Every scene has a wall bounce by horizon 40, so there is no collision-free
group. The learned correction is better only in the first ~5 frames. Soft contact's mean of ~0.009
without ball collisions, despite a median of 0, comes from rare spurious contacts: its fitted radius
0.0644 exceeds the true 0.06.

**Does disagreement predict real error? Yes, on untouched forecasts.** At horizon 20 / 40,
Spearman(disagreement, hybrid error) is **0.922 / 0.937**. AUROC for error ≥0.06 is 0.997 / **0.962**,
but horizon 20 has only 5 positives versus 65 at horizon 40 (`uncertainty.20/40`). Over
the five quantile bins at horizon 40, mean disagreement 0.0015 / 0.014 / 0.030 / 0.061 / 0.151 tracks mean
error 0.0010 / 0.009 / 0.023 / 0.052 / 0.136. It is more than a collision detector. Disagreement
separates ball-collision from collision-free scenes with AUROC 0.90, but *within* the 199
ball-collision scenes it still ranks error with Spearman 0.895 and AUROC 0.949 for error ≥0.06
(reviewer check on `forecast_raw.npz`). On the plans actually chosen, `selected_uncertainty`
gives Spearman 0.899–0.926 / AUROC 0.925–0.950 for the mean planner and 0.879–0.920 / 0.879–0.940
for the risk planner (random and CEM rows). Among the mean planner's dream successes at 256 shots
(n=138, 53 real failures), disagreement flags the real failures with AUROC 0.899 (0.919 at 4,096).
These are selection-biased populations and ranking scores, not calibrated probabilities.

### Turn 2 headline: demo and what it shows

[Demo GIF](media/r5_t2_20261024T161011Z_demo.gif) / [strip](media/r5_t2_20261024T161011Z_strip.png)
show task 1 under `hybrid_mean/random/64`, the first fooled case: dream distance **0.0533**
(just inside 0.06), real **0.0993**, disagreement **0.0690**.
The checkpoint replay endpoint delta is **0**. The third panel is the fitted soft-contact model following the same launch
([demo.json](media/r5_t2_20261024T161011Z_demo.json), [render.log](logs/r5/r5_t2_20261024T161011Z/render.log)).
Also rendered: [forecast.png](media/r5_t2_20261024T161011Z_forecast.png) and
[cost.png](media/r5_t2_20261024T161011Z_cost.png).

### Turn 2: what worked, what didn't, caveats

- **Worked:** uncertainty-aware planning. The primary contrast is decisive, the fooled rate falls
  2–3× at every effort ≥256 shots, and the advantage grows with search effort. Disagreement ranks real error well on fresh
  scenes, including within collision scenes.
- **Didn't (H2 failed):** more training, contact weighting and geometry features did not beat the
  turn-1 model on test (all reference contrasts are n.s. after Holm), and the raw-feature ensemble
  validated *worse* than turn-1 h32. The learned mean model is not better than the misspecified
  classical fit: equal planning success, worse long-horizon forecasts.
- **Cost-matched caveat:** at equal member-frames, soft contact with 5× shots beats risk planning
  at 64 shots (50.5 vs 33.5%), ties at 256, and loses only at ≥1,024. "Risk planning beats the
  fitted classical model" holds at equal candidates for ≥1,024 shots (and at 256 only before Holm),
  and at equal member-frames only for ≥1,024.
- **Missing fair classical analogue (not run):** a risk-aware classical planner. One option is a
  soft-contact *parameter ensemble* (for example, the top-5 validated `soft_fit.json` trials: ζ=0,
  k 14k–32k, r 0.060–0.064) scored with the same disagreement penalty, or a penalty on predicted
  ball contact. Within-family spread may under-represent the structural hard-vs-soft misspecification,
  so the result is not obvious either way. Until this runs, the finding is "an ensemble's disagreement
  is a useful distrust signal", not "learned beats classical".
- **Other caveats:** one trained ensemble per arm (task-level tests, not training replications);
  β chosen from 64 validation tasks with a one-task margin; the soft-contact fit used 6 s of its 5-min
  allowance (256 trials), yet already matched the learned models on validation; Holm across 183
  contrasts is conservative, so raw p-values are also given.

### Turn 1 (preliminary; its test seeds informed turn 2)

Turn 1 found that uncertainty-aware planning improves real success at 256–4,096 random shots and
reduces false optimism, but is slightly worse at 64 shots. The turn-1 `fitted` comparison recovered
the simulator's exact collision law (validation loss 4.4e-14), so it is an **exact-law oracle
reference**, not a fitted baseline. Turn-1 and turn-2 tables use different task seeds and must not
be read as a paired before/after; the only paired old-vs-new comparison is the `reference_*` rows
above.

All turn-1 numbers below come from `results/r5/r5_20261024T152040Z/` (abbreviated **T1**).
[The NumPy audit](results/r5/turn2_prepare/turn1_audit.json) recomputed both splits' planning
metrics, all paired tests/Holm adjustments, forecast summaries and uncertainty statistics from
the retained raw records. It also verified checkpoint/fit hashes and that the saved `fitted`
forecasts are exactly equal to truth. Original code was archived with matching evaluation hashes
in [the source snapshot](results/r5/turn1_source_snapshot_20261024T153713Z/audit.json).

#### Turn 1: setup, training and the oracle correction

Five independently initialized, scene-bootstrapped pair networks add a correction to known
gravity/wall physics. Two widths used the same 2,000 training scenes and 256 validation scenes;
the loss unrolled three frames. Both reached the **100,000-step cap**, rather than their five-minute
guards: width 32 took **175.375 s**, width 64 **202.228 s**. Their final balanced validation losses
were **0.00946436** and **0.00934609**. Long-horizon validation instead favored width 32:
mean position error **0.0121302** versus **0.0141728** over the 24 monitored validation scenes.
The selected penalty was **beta=1** from 24 separate validation planning tasks. Sources:
[h32 run status](checkpoints/r5_20261024T152040Z_h32/run_status.json),
[h64 run status](checkpoints/r5_20261024T152040Z_h64/run_status.json), their selected `state.json`
files recorded in [selection.json](results/r5/r5_20261024T152040Z/selection.json), and
[training logs](logs/r5/r5_20261024T152040Z/train_h32.log).

The 96-proposal physical fit selected **restitution=1, separation=1** with validation loss
**4.388684274070553e-14**. Its family included the actual simulator; the residual training/validation
loss comes from float32 stored trajectories, while fresh float64 forecast error is exactly zero.
Thus `fitted/*` in the immutable T1 JSON means **oracle over the given random candidate set**.
Its success is **84.5% at 4,096 shots**, and **85.5% at 20,480 shots** in the separately recorded
`random_member_matched/4096` control. The latter is not a 4,096-shot result. Oracle CEM uses
model-dependent proposals and is a reference search, not a universal upper bound over other
methods' different CEM candidates. Sources: [fit.json](results/r5/r5_20261024T152040Z/fit.json),
[forecast.json](results/r5/r5_20261024T152040Z/headline/forecast.json), and the corresponding
`metrics.fitted/*` entries in [summary.json](results/r5/r5_20261024T152040Z/headline/summary.json).

#### Turn 1: preliminary planning, 200 paired tasks, horizon 40

Scenes/targets/actions used seeds 51901/52001/51902. The same nested shots and velocity bounds
were shared across methods. Success means distance <0.06; **fooled** means dream success and
real failure at that same threshold, divided by all tasks. This differs from the earlier exhibit's
severe-miss definition; severe misses are separately retained. Every method commits a launch
(100% coverage). The following entries are `metrics.<method>/random/<shots>` in
[T1 headline summary](results/r5/r5_20261024T152040Z/headline/summary.json), backed by its
`tasks/*.json` records. All values are percentages.

| Method | Shots | Dream success | Real success | Fooled |
|---|---:|---:|---:|---:|
| Walls only | 64 | 50.0 | 24.0 | 26.5 |
| Walls only | 256 | 69.0 | 29.0 | 40.0 |
| Walls only | 1,024 | 76.0 | 35.5 | 40.5 |
| Walls only | 4,096 | 77.5 | 34.5 | 43.0 |
| Exact-law oracle (`fitted`) | 64 | 44.5 | 44.5 | 0.0 |
| Exact-law oracle (`fitted`) | 256 | 67.5 | 67.5 | 0.0 |
| Exact-law oracle (`fitted`) | 1,024 | 81.5 | 81.5 | 0.0 |
| Exact-law oracle (`fitted`) | 4,096 | 84.5 | 84.5 | 0.0 |
| Hybrid mean | 64 | 45.0 | 32.0 | 16.0 |
| Hybrid mean | 256 | 68.5 | 43.5 | 27.5 |
| Hybrid mean | 1,024 | 78.0 | 48.0 | 31.5 |
| Hybrid mean | 4,096 | 82.5 | 53.0 | 30.0 |
| Hybrid risk | 64 | 35.5 | 31.5 | 6.0 |
| Hybrid risk | 256 | 58.5 | 53.5 | 8.0 |
| Hybrid risk | 1,024 | 72.5 | 63.0 | 10.5 |
| Hybrid risk | 4,096 | 76.5 | 68.5 | 8.0 |

At the prespecified 256-shot comparison, risk gains **10.0 percentage points**, paired bootstrap
95% interval **[4.5, 15.5] points**. There are 28 risk-only successes and eight mean-only successes;
exact McNemar p=**0.00119324**, Holm-adjusted p=**0.0143189** across the 62 reported contrasts.
The stored key is `hybrid_mean/random/256 vs hybrid_risk/random/256`, whose difference/interval
have the reverse sign. At 4,096 shots the gain is **15.5 points**, Holm p=**3.76309e-5**. These
are task-level comparisons for one trained ensemble pair, not independent training replications.
Source: `paired_tests` in the same headline summary.

CEM population 64 gives real success (mean/risk) of **32.0/31.5%**, **44.0/46.0%** and
**53.0/62.5%** at one/two/four iterations. Four iterations evaluate 256 candidates; the
corresponding risk fooled rate is **6.5%**, versus mean **22.0%**. Walls-only CEM reaches
**75.5% dream / 31.5% real / 44.0% fooled** at four iterations. These are the `*/cem/{1,2,4}`
entries in the same summary. Random cost controls give classical methods five times as many
shots; member-frame counts are matched, not FLOPs or wall time. The headline evaluation completed
200/200 tasks in **412.320 s** ([status.json](results/r5/r5_20261024T152040Z/headline/status.json)).

#### Turn 1: forecasting and uncertainty

There are 256 independent forecast scenes, one window each (seed 52101). Error is mean per-ball
Euclidean position error in a unit box; the median is across scenes. From
`metrics.<method>.<horizon>.all` in [T1 forecast.json](results/r5/r5_20261024T152040Z/headline/forecast.json):

| Horizon | Hybrid mean error | Hybrid median | Walls mean error | Walls median |
|---:|---:|---:|---:|---:|
| 1 | 0.00004566 | 0.000000112 | 0.00023388 | 0 |
| 5 | 0.00038085 | 0.000000558 | 0.00452067 | 0 |
| 10 | 0.00186708 | 0.000001256 | 0.01815446 | 0 |
| 20 | 0.00897015 | 0.00083889 | 0.06936353 | 0 |
| 40 | 0.05395178 | 0.02575288 | 0.21385981 | 0.21349727 |

The oracle's errors are zero; nonfinite forecast counts are zero. At horizon 40, the 204 scenes
with a ball collision have hybrid mean/median error **0.0651687 / 0.0356004**, versus
**0.00994706 / 0.000004689** for the 52 with no ball collision. The latter group includes wall
bounces and rare false-contact/drift failures: the hybrid is not exact there. No scene is free
of *all* collisions by horizon 40, so that subgroup has no measured error. Source: the
`ball_collision`, `no_ball_collision`, and `no_collision` fields in the same file.

On unselected forecasts at horizons 20/40, disagreement has Spearman **0.9241/0.9315** and
AUROC **0.9738/0.9644** for error ≥0.06. At early horizons the correlation is weaker and AUROC is
undefined when no forecast exceeds the threshold. These are `uncertainty.20/40`. On selected
mean-planner shots, Spearman ranges **0.8800–0.9380** and AUROC **0.9086–0.9615**; risk-selected
plans have Spearman **0.8894–0.9160**, AUROC **0.8874–0.9284**. Those are separate, selection-biased
populations (`selected_uncertainty.* /random/*` in headline summary), not calibrated failure
probabilities.

The old-scene secondary has only 32 tasks. At 256 shots, mean/risk real success is
**59.375/65.625%**, Holm p=**1.0**; at 4,096 it is **68.75/87.5%**. It uses the original scene
array with a new planning protocol, so it is not a reproduction of the earlier 512-shot result.
Source: [old_secondary/summary.json](results/r5/r5_20261024T152040Z/old_secondary/summary.json).

#### Turn 1: what worked, failures and next steps

Hard-coded wall physics removed most unnecessary drift; uncertainty penalties helped most when
search was strong enough to offer alternatives. More mean-model search also increased real
success, but substantial false optimism remained. Walls-only search increasingly found dream
solutions that failed after real ball collisions. Bigger width did not win validation forecasting,
and the step cap prevented either run from using its full time allocation.

The first smoke failed with NaN loss at step 2: differentiating the zero self-distance square root
created a `0*inf` VJP even through a boolean gate. Claude detached the gate inputs; the retained
[failed trainer log](logs/r5/smoke_20261024T152040Z_67494/killed.log) documents the failure.
[Smoke 2](logs/r5/r5_20261024T152040Z/smoke2.log) passed in **11.8 s** and
[its MLX checks](logs/r5/smoke_20261024T152133Z_67743/mlx_checks.log) confirmed kill/resume losses
match uninterrupted training. Turn 2 preserves that fix and adds explicit zero-speed/self-edge
gradient regression checks.

The preserved [failure GIF](media/r5_20261024T152040Z_demo.gif) / [strip](media/r5_20261024T152040Z_strip.png)
show task 3 at 256 mean-model shots: dream distance **0.0323440**, real **0.0761954**, disagreement
**0.0587917**, checkpoint replay endpoint delta **0**. The original image's “FITTED” panel is
the **exact-law oracle following the same launch**; the artwork is retained unchanged.
Source: [demo receipt](media/r5_20261024T152040Z_demo.json). This is a selected failure, not a
representative success-rate sample.

Turn 2 added finite-duration spring/damper physics that cannot express the true hard-contact law,
two longer training trials, richer safe geometry features, collision weighting and a deterministic
LR schedule. Selection remains train/validation-only, including a frozen turn-1 fallback. New
scene/target/action/forecast seeds are 62001/62101/62002/62201; the frozen turn-1 model will also
run on these new paired tasks (it did: see the `reference_*` rows in the turn-2 tables). The design is in [DESIGN_R5.md](DESIGN_R5.md); the exact commands are in
[README.md](README.md).

| Turn-2 quantity | Result (details in the turn-2 sections above) |
|---|---|
| Fitted soft-contact parameters / validation error | k=14,242, ζ=0, r=0.0644; val mean error 0.013205 (`soft_fit.json`) |
| Selected model / validation gain / beta | geometry ensemble at step 85,224, val error 0.013416 vs turn-1 h32 0.014121; β=0.5 (`selection.json`) |
| New-seed real success, fooled rate and paired tests | risk vs mean at 256 shots: 54.5 vs 43.5%, fooled 12.5 vs 26.5%, Holm 4.9e-5 (`headline/summary.json`) |
| Forecast errors and uncertainty calibration | h40 hybrid 0.0443, soft 0.0372; Spearman 0.937, AUROC 0.962 (`headline/forecast.json`) |
| New smoke, training throughput and demo | smoke 17.1 s; raw 123,244 / geometry 85,224 steps in 840 s; demo replay delta 0 |

**Next steps:** run the risk-aware classical analogue (soft-contact parameter ensemble plus the
same penalty) on these tasks. Train 2–3 independent ensembles per arm to separate training noise
from method effects. Add a cost-weighted abstain option now that disagreement ranks failures well.

### Turn 3: classical risk-aware control

**Result (Claude runner, 2026-10-24 12:25–12:36 CDT, AC, run `r5_t3_20261024T172531Z`).** A classical
five-member soft-contact parameter ensemble scored with the *same* disagreement penalty **ties** the learned
hybrid risk planner, and both beat every mean planner. So in this world the benefit comes from planning with a
model that knows when to distrust itself, not from the learned dynamics.

- Members: the five lowest-validation-error distinct trials (indices 242, 232, 251, 37, 208) of the frozen
  turn-2 soft fit; all have damping 0, contact radius 0.060–0.064 (`results/r5/r5_t3_20261024T172531Z/classical_selection.json`).
- Classical beta chosen on the 64 validation tasks (seed 53003, 256 shots): real success by beta
  {'0.0': 0.484, '0.25': 0.5, '0.5': 0.5, '1.0': 0.562, '2.0': 0.578}; selected **beta = 2.0, the top of the grid** (larger values were not tried; a caveat).
- Fresh headline: 200 tasks, seeds 73001/73002/73101, horizon 40, all methods on the same tasks and nested
  candidate prefixes; 0 invalid candidates in every cell; evaluation 592 s (`results/r5/r5_t3_20261024T172531Z/headline/summary.json`,
  `logs/r5/r5_t3_20261024T172531Z/`). Ensembles and the 5× single-fit control match member-frame budgets,
  not wall time.

| method | shots | dream success | real success | fooled rate |
|---|---:|---:|---:|---:|
| hybrid_mean | 256 | 75.0% | 53.0% | 22.5% |
| hybrid_mean | 4096 | 87.0% | 62.5% | 25.5% |
| hybrid_risk | 256 | 72.5% | 63.5% | 9.5% |
| hybrid_risk | 4096 | 86.5% | 78.5% | 8.0% |
| soft_contact | 256 | 73.5% | 53.0% | 22.0% |
| soft_contact | 4096 | 87.0% | 64.0% | 23.5% |
| soft_contact_5x | 256 | 86.0% | 58.0% | 29.0% |
| soft_contact_5x | 4096 | 89.5% | 63.0% | 26.5% |
| classical_mean | 256 | 75.0% | 54.0% | 22.0% |
| classical_mean | 4096 | 87.0% | 57.0% | 30.5% |
| classical_risk | 256 | 71.5% | 61.5% | 11.0% |
| classical_risk | 4096 | 86.0% | 77.0% | 9.0% |

Paired exact McNemar tests (one Holm family of 30 within-effort comparisons; source `results/r5/r5_t3_20261024T172531Z/headline/summary.json → paired_tests`):

| shots | comparison | difference |
|---|---|---|
| 256 | hybrid_risk − classical_risk | +2.0 pts (10 vs 6 discordant, Holm p=1) |
| 256 | classical_mean − classical_risk | -7.5 pts (3 vs 18 discordant, Holm p=0.028) |
| 256 | soft_contact_5x − classical_risk | -3.5 pts (22 vs 29 discordant, Holm p=1) |
| 256 | hybrid_mean − classical_risk | -8.5 pts (8 vs 25 discordant, Holm p=0.077) |
| 4096 | hybrid_risk − classical_risk | +1.5 pts (12 vs 9 discordant, Holm p=1) |
| 4096 | classical_mean − classical_risk | -20.0 pts (4 vs 44 discordant, Holm p=4.5e-08) |
| 4096 | soft_contact_5x − classical_risk | -14.0 pts (8 vs 36 discordant, Holm p=0.0006) |
| 4096 | hybrid_mean − classical_risk | -14.5 pts (9 vs 38 discordant, Holm p=0.0006) |

What it means:
- **Learned vs classical, both risk-aware: no detectable difference** (+2.0 and +1.5 points, Holm p = 1).
  The turn-2 headline "risk-aware planning beats mean planning" holds for both model families, and the
  reviewer's caveat is now answered: the gain is the distrust signal, which a cheap parameter ensemble also provides.
- Classical risk beats classical mean by 7.5 points at 256 shots and 20 points at 4096, and cuts the fooled
  rate from 22–31% to 9–11%; harder search helps the risk-aware planners much more than the mean planners.
- More candidates for the single fit (5× shots) do not substitute for disagreement at 4096 shots (−14 points).
- Caveats: one training seed for the hybrid; beta at the grid edge; members share zero damping, so their
  disagreement can miss shared model error; turn-3 numbers use new seeds and are not a paired continuation of
  turn 2's percentages.
