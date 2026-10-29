# Tiny Grand Prix: results

All numbers come from real runs in this folder (`results/eval.json`, `results/hack.json`,
`results/diagnose_grip.json`, `results/seeds/`, `logs/*_history.jsonl`, `results/dr_seeds/`). Main tables use one
training seed per method; a second seed of ES and PPO is in "Seeds", and **three seeds of each
grip-randomized learner, under two checkpoint-selection rules**, are in "Grip-randomized runs:
3 seeds, two selection rules" (follow-up, 11:30–12:30).

## TL;DR

* **First satisfying version: reached.** A 1.6k-weight MLP that sees only 10 wall rays + its own
  speed finishes **100 / 100 held-out tracks** it never trained on (ES; PPO 99 / 100), with no
  wall-hugging. It spends 0.7% of the lap closer to a wall than 20% of the available room; the tuned
  hand-written baseline spends 0.4%.
* **Learned and tuned baseline are close at the training grip.** The tuned ray-follower also
  finishes 100% of held-out laps. ES is faster on 95 / 100 tracks, but only by 0.65 s per lap
  (−2.5%).
* **The learned policies hold up much better when grip drops.** At grip 0.8 (trained at 1.0),
  ES finishes 98% and PPO 100%, while the ray-follower tuned at grip 1.0 (same training grip as ES/PPO) finishes **3%**. Much of that cliff is CEM pushing one knob (assumed braking) to the edge of grip-1.0 physics; see the next bullet for a baseline tuned under the same grip randomization. With grip
  randomization during training, PPO still finishes **93% at grip 0.5**. That number is
  **typical, not lucky**: three seeds give 93%, 96% and 93% (mean 94%), and choosing the checkpoint
  on randomized-grip validation instead of grip-1.0 validation gives 93%, 96% and 94%.
* **How the checkpoint is chosen matters a lot for ES + grip rand.** The original rule (best
  validation score at grip 1.0) picks late ES checkpoints that drive well at high grip but are
  fragile at low grip: **21–41% at grip 0.5** over 3 seeds. Choosing on randomized-grip
  validation (same μ ~ U(0.5, 1.1) as training, validation tracks only) gives **76–90%**, and
  100% at grip 0.6 on every seed. The cost is 0.6–0.8 s per lap at grip 1.0. For PPO the
  rule barely matters; its best checkpoint comes early under either rule (4.3–14.1 M of 78 M
  steps with the grip-1.0 rule, 4.6–9.2 M with the randomized-grip rule).
* **A baseline tuned under the same grip randomization is also robust, but slow.** It finishes
  100% down to grip 0.6 by driving cautiously everywhere: 31.8 s laps at every grip, against
  26.0 s for ES+rand. and 27.3 s for PPO+rand. at grip 1.0. The learned drivers slow down only
  as much as the grip requires; this baseline drives cautiously at every grip. At grip 0.5 it finishes 24%; PPO+rand. finishes 93–96% (3 seeds) and ES+rand. 76–90% when selected on randomized grip (only 21–41% under the original grip-1.0 selection, not reliably better than the baseline: seed 1 finishes 21%). On completion alone it matches the grip-randomized learners from grip 1.0 to 0.6 (100% vs 89–100% over all seeds and both rules). On the hard tracks it beats PPO+rand. at grip 0.7–1.0 **on every seed** (88–92% vs 60–90%), while ES+rand. (randomized-grip selection) ties or beats it there (88–100%). With 3 seeds the conclusion stands: the learned advantage is lap time and the grip-0.5 extreme, not reliability in general. (The baseline is one deterministic CEM tuning run; its tuning-seed variance was not measured.)
* **Out-of-distribution tracks (narrower, twistier) are where all ray-based drivers lose
  laps.** At grip 1.0: ES 80%, PPO 72%, ray-follower 82%, ES+grip-rand. 90% (3 seeds: 90–98%; 88–96% with randomized-grip selection), PPO+grip-rand. 60%
  (3 seeds, either selection: 60–76%), cautious grip-rand. ray-follower 90% (but 36 s laps vs 29.5 s), privileged pure pursuit 100%.
  **Surprise:** the learned policies finish *more* of these tracks at grip 0.8 than at their
  training grip of 1.0 (ES 100% vs 80%). Every crash at grip 1.0 is into the *inside* wall: they
  cut the apex. Less grip makes the car run wider, which cancels the over-rotation. Two other
  explanations were tested and ruled out (below).
* **Reward-hacking exhibit.** With "soft" walls (the car slides along them instead of crashing),
  a car with *no steering at all* gets round 100% of held-out tracks. ES trained under those
  rules learns to **wall-ride**: 21.8 s laps, faster than anything legal, scraping a wall for
  20 steps per lap. Under real rules it finishes **0%**. Clip: `media/hack_wall_riding.mp4`.
* **Sample efficiency.** PPO (MLX) reaches the baseline's validation score after 2.9 M env
  steps (0.5 min). ES needs 114 M steps (13.5 min). Final quality is the same: best validation
  score 1398 (PPO) vs 1400 (ES). PPO then *degrades* if kept training (see below).

## Setup (what was actually built)

**Tracks.** Closed centerlines from a periodic cubic spline through 8–16 randomly perturbed
points around a circle. Each track is randomly stretched, rotated and run clockwise or
counter-clockwise, with half-width 5–7 m and length about 400–800 m. Tracks are
rejection-sampled: no corner can be tighter than the walls allow, and no two far-apart parts
of the track can touch. Every track comes from one integer seed:

| split | seeds | used for |
|---|---|---|
| train | 0–255 | training (ES, PPO, baseline tuning) |
| validation | train-generator seeds 5000–5127 | checkpoint selection only |
| **held-out** | 1,000,000 + 0…99 | final test, same generator, never seen |
| **hard (OOD)** | 2,000,000 + 0…49 | final test: narrower (half-width 4–5 m), 12–18 control points, more and tighter corners |

**Car.** Kinematic bicycle model with dt = 0.1 s, wheelbase 2.6 m and max steer 0.45 rad.
Throttle gives up to 6 m/s², brakes up to 12 m/s², and quadratic drag sets a top speed of about
40 m/s. There is no reverse gear. **Grip μ** caps longitudinal acceleration at μg and yaw rate
at μg/v, so a car that enters a corner too fast runs wide (understeer). Training uses μ = 1.0
unless stated; the "+ grip rand." variants draw μ ~ U(0.5, 1.1) per episode.

**Observations (13 numbers).** 10 wall-distance rays (−90°…+90°, 60 m range), speed, previous
steering and previous pedal. No pixels, no map, no heading-to-track and no grip value.

**Reward / fitness and hack guards.**
* Progress is the *signed* change of arclength along the centerline. It uses nearest-segment
  tracking with a small local search, so it cannot jump across the track.
* Reversing gives negative progress (`tests/test_smoke.py::test_reversing_loses_progress`).
* A lap counts only when cumulative signed progress reaches one track length.
* Touching a wall ends the episode. That costs 100 m of ES fitness, or −10 reward for PPO
  (progress is scaled by 1/10).
* Training episodes last 60 s and start from rest at a random point of a random training track.

**Learners.** Both use the same 13→32→32→2 tanh MLP (1,570 weights), stored as one flat vector.
* **ES:** OpenAI-style. 64 antithetic pairs × 16 shared episodes per generation, σ = 0.04,
  Adam lr 0.01, centered ranks, weight decay 0.005. 300 generations, 346 M env steps, 28 min.
* **PPO (MLX):** 256 numba envs × 128 steps per update, 4 epochs of 4096-sample minibatches,
  γ = 0.99, λ = 0.95, clip 0.2, lr 3e-4, learned state-independent log-std and a separate
  64-64 critic. It ran 25 min (122 M steps). Rollouts use a numpy copy of the actor; updates
  run in MLX.
* Each run keeps the checkpoint with the best **validation** score (60 s drives on 128 unseen
  training-distribution tracks at grip 1.0), and that is what gets evaluated. Held-out and hard
  tracks were used only by `eval.py`.

**Baselines.**
* **Ray follower (matched information).** It reads the exact same 13-number observation. It
  steers toward the ray-length-weighted open direction plus a centering term, and sets a
  target speed it could brake from given the free distance ahead. Its 7 knobs were tuned by
  the cross-entropy method (15 × 40 candidates × 64 episodes, about 23 M env steps, 7.7 min)
  on the **same training tracks and the same fitness** as ES. The untuned defaults are
  reported too. Its budget (~23 M env steps) is far smaller than ES's 346 M, so the near-tie at
  grip 1.0 is, if anything, flattering to the learners. But a 7-knob controller saturates
  quickly: CEM's best candidate score sat at 1340–1380 from iteration 3 of 15 on.
* **Pure pursuit (PRIVILEGED).** It sees the true centerline and curvature. It chases a
  lookahead point and follows a curvature speed profile that assumes grip 0.9 with 80% braking
  margin. It is a reliability reference, not a speed bound, and not a fair competitor.

**Evaluation.** One lap from a standing start on the start line, with a time limit of
length / 4 m/s.
* *strict* rules: any wall contact = DNF.
* *respawn* rules: wall contact counts as one collision; the car is put back on the centerline
  at rest and drives on. This makes "collisions per lap" meaningful for controllers that crash.

## Results

Full tables: [`results/eval_table.md`](results/eval_table.md). (Review note, 08:35: `racing/evaluate.py` used to
count a lap as finished even if the car touched a wall on the very step that completed it. That is
fixed, all results files were regenerated, and the 5 strict-rule cells that changed by 1–2 points are
updated below; no headline claim changed except PPO + grip rand. at grip 1.0, 98% → 96%.) Lap times are means over the
laps that controller finished. Paired comparisons use only tracks both controllers finished.

### Held-out tracks (100 unseen layouts), grip 1.0 = training grip, strict rules

| controller | completion | mean lap | paired vs tuned ray follower | wall-hugging check: time with abs(lat) > 0.8 / p95 abs(lat) |
|---|---|---|---|---|
| ES | **100%** | 25.2 s | −0.65 s, faster on 95/100 | 0.7% / 0.63 |
| PPO (MLX) | 99% | 25.2 s | −0.65 s, faster on 94/99 | 1.6% / 0.67 |
| ES + grip rand. | 100% | 26.0 s | +0.20 s, faster on 34/100 | 0.7% / 0.66 |
| PPO + grip rand. | 96% | 27.3 s | +1.43 s, faster on 6/96 | 6.3% / 0.80 |
| Ray follower (tuned) | 100% | 25.8 s | – | 0.4% / 0.67 |
| Ray follower (untuned defaults) | 51% | 26.2 s | | 10.6% / 0.84 |
| Pure pursuit (PRIVILEGED) | 100% | 29.1 s | | 0.0% / 0.06 |

`lat` = distance from the centerline as a fraction of the room to the wall (0 = centerline,
1 = touching). The learned cars use the track width like the tuned baseline does: they take
racing lines through corners, but they do not live on the wall. On the training tracks the
numbers are the same (ES 100% / 25.6 s, PPO 98%), so there is **no measurable generalization
gap** within the training distribution.

### Robustness to grip (held-out tracks, strict rules, completion %)

| grip → | 1.0 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|---|---|
| ES | 100 | 100 | 98 | 83 | 28 | 3 |
| PPO | 99 | 100 | 100 | 90 | 43 | 2 |
| ES + grip rand. | 100 | 100 | 100 | 100 | 95 | 41 |
| PPO + grip rand. | 96 | 97 | 100 | 100 | **100** | **93** |
| *ES + grip rand., randomized-grip selection (seed 0)* | 98 | – | 100 | 100 | 100 | 90 |
| Ray follower (tuned) | 100 | 52 | 3 | 0 | 0 | 0 |
| Ray follower (tuned w/ grip rand.) | 100 | 100 | 100 | 100 | 100 | 24 |
| Ray follower (untuned) | 51 | 49 | 46 | 1 | 0 | 0 |
| Pure pursuit (PRIVILEGED, assumes μ = 0.9) | 100 | 100 | 94 | 5 | 0 | 0 |

Grip-randomized rows are seed 0, selected on grip-1.0 validation, except the italic row (the same
seed-0 run, re-selected on randomized-grip validation; grip 0.9 not evaluated). Over 3 seeds,
PPO + grip rand. finishes 93–96% at grip 0.5 under either selection rule, and ES + grip rand.
finishes 21–41% (grip-1.0 selection) or 76–90% (randomized-grip selection). See "Grip-randomized
runs: 3 seeds, two selection rules".

Under respawn rules (collisions per lap, held-out, grip 0.6): ES 1.23, PPO 0.81, ES+rand
0.05, PPO+rand 0.00, ray follower 3.77, pure pursuit 4.17. Lap times grow smoothly as grip
drops (PPO+rand: 27.3 s at 1.0 → 33.5 s at 0.5): the policies slow down in response to what
lower grip does to the car (they have no grip input).

**Why the tuned baseline falls off a cliff.** CEM tuned its "assumed braking" knob to
9.4 m/s². That is the right value at grip 1.0, but impossible below grip 0.96 (μg < 9.4). The
baseline was optimized right up to the edge of the training physics. The learned policies
were trained under exactly the same conditions, yet they keep some margin: they brake earlier
and enter corners slower. ES finishes 98% at grip 0.8 even though nothing in its training
asked for that.

**Matched-randomization baseline.** For a fair test of the grip-randomized learners, the ray
follower was re-tuned with the *same* grip randomization: CEM, μ ~ U(0.5, 1.1) per episode,
same budget (`checkpoints/ray_follower_dr.json`). CEM's answer is to drive cautiously:
assumed braking drops from 9.4 to 4.5 m/s². That makes it as reliable as the learned DR
policies down to grip 0.6, but much slower and unable to use extra grip when it exists.

| held-out, strict: mean lap time (completion) | 1.0 | 0.8 | 0.6 | 0.5 |
|---|---|---|---|---|
| ES + grip rand. | 26.0 s (100%) | 27.6 s (100%) | 30.5 s (95%) | 32.7 s (41%) |
| PPO + grip rand. | 27.3 s (96%) | 27.9 s (100%) | 30.4 s (100%) | 33.4 s (93%) |
| ES + grip rand., randomized-grip selection | 26.8 s (98%) | 27.9 s (100%) | 30.6 s (100%) | 33.0 s (90%) |
| Ray follower tuned w/ grip rand. | 31.8 s (100%) | 31.8 s (100%) | 32.7 s (100%) | 35.8 s (24%) |

The learned policies have no grip input and no memory, yet their lap times stretch smoothly
as grip falls. The likely reason: they react to the *situations* low grip creates. A car that
runs wide sees a nearer outside wall in its rays and slows down; it does not need an explicit
grip estimate. The hand-coded structure can only pick one fixed caution level.

### Hard (out-of-distribution) tracks: narrower and twistier, 50 layouts, strict rules

| grip → | 1.0 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|---|---|
| ES | 80 | 92 | **100** | 94 | 62 | 6 |
| PPO | 72 | 88 | **100** | 96 | 72 | 6 |
| ES + grip rand. | 90 | 96 | 100 | 100 | 100 | 66 |
| PPO + grip rand. | 60 | 66 | 66 | 90 | 98 | 98 |
| Ray follower (tuned) | 82 | 64 | 10 | 0 | 0 | 0 |
| Ray follower (tuned w/ grip rand.) | 90 | 92 | 88 | 92 | 96 | 38 |
| Pure pursuit (PRIVILEGED) | 100 | 100 | 80 | 0 | 0 | 0 |

**The surprise: less grip helps the learned cars on unfamiliar narrow tracks.** At grip 1.0,
ES and PPO finish 80% and 72% of the hard tracks; at grip 0.8 they finish 100%.
`diagnose_grip.py` tested three explanations:

| hard tracks, grip 1.0 unless noted | ES done | PPO done | share of crashes into the **inside** (apex) wall |
|---|---|---|---|
| as trained | 80% | 72% | 100% / 100% |
| grip 0.8 | 100% | 100% | – (≤ 1 crash) |
| steering low-passed (EMA 0.5) | 64% | 62% | 100% / 100% |
| steering low-passed (EMA 0.3) | 78% | 54% | 100% / 100% |
| speed capped at 22 m/s | 76% | 78% | 100% / 100% |
| speed capped at 18 m/s | 80% | 74% | 100% / 100% |

* "Twitchy steering amplified by full grip" is ruled out: smoothing the steering does not help.
* "Simply too fast" is ruled out: a speed cap does not help.
* The crash side gives it away. **Every** grip-1.0 crash is into the *inside* wall. The policies
  learned to turn in tightly and cut the apex on 5–7 m-wide tracks. On 4–5 m tracks the inside
  wall is closer than they expect, so they clip it.
* Lower grip caps the yaw rate, which makes the car run a bit wide and cancels the
  over-rotation. Where the ray follower crashes at grip 0.8, 64% of its crashes are into the
  *outside* wall: that is the normal "too fast for the grip" failure.

### Reward-hacking exhibit: soft walls teach wall-riding (`hack_check.py`, 100 held-out tracks)

`train_es.py --soft_walls` uses walls that push the car back and let it slide along them,
keeping 97% of the along-wall speed. This is a common "forgiving" game physics.

| controller | soft-wall rules: done / lap / wall-contact steps per lap / time near wall | strict rules: done |
|---|---|---|
| no steering, full throttle (a brick on the pedal) | 100% / 32.4 s / 198 / 80% | 0% |
| ES trained with soft walls (`es_soft`) | 100% / **21.8 s** / 20.5 / 33% | **0%** (24% of a lap on average) |
| ES trained with real walls | 100% / 25.2 s / 0 / 0.7% | 100% |
| Ray follower (tuned) | 100% / 25.8 s / 0 / 0.4% | 100% |

Under soft walls, the walls do the steering. A car with no steering finishes every lap, and
an ES agent learns to throttle flat-out into the outside wall of every corner and ride it
round. It laps 3.4 s faster than any legal driver. The racing-line plot
`media/hack_wall_riding_lines.png` shows it hugging the outside wall through corners, and the
clip `media/hack_wall_riding.mp4` races it against the ray follower. The guard used for every
real result is simple: any wall contact ends the episode, and progress is signed and measured
along the centerline. With that guard the same ES recipe learns clean racing lines.

### Learning dynamics and compute (`media/learning_curves.png`)

| run | env steps to reach the tuned baseline's validation score (1355) | best validation | at | total run |
|---|---|---|---|---|
| PPO | 2.9 M (0.5 min) | 1398 | 19 M steps, 2.9 min | 122 M steps, 25 min |
| ES | 114 M (13.5 min) | 1400 | 346 M steps, 28.4 min | same |
| PPO + grip rand. | never (validation is at grip 1.0) | 1238 | 4.6 M steps | 78 M, 20 min |
| ES + grip rand. | never | 1345 | 304 M steps | 311 M, 20 min |
| PPO / ES + grip rand., 3 seeds, randomized-grip validation (`select_dr.py`) | – | 1195–1203 / 1198–1209 (val_rg) | 4.6–9.2 M / 256–274 M | 78 M / 305–316 M |
| (ray-follower CEM tuning) | – | 1355 | ~23 M steps | 7.7 min |

* **PPO is about 40–90× more sample-efficient than ES here** (2 seeds each) and reaches the same quality.
  Wall-clock budgets were roughly matched (25 vs 28 min on 4 CPU threads, shared machine).
* **PPO degrades after its peak.** After about 15–20 M steps its validation score falls (1398 →
  1161) and its training crash rate rises from about 1% to 8–13%, while the policy std keeps
  shrinking (0.18 → 0.06). The likely cause: with γ = 0.99 the agent sees only about 10 s
  ahead, so a crash "costs" −10 plus about 10 s of lost progress. ES sees the full 60 s. A
  learning rate that never decays may also contribute. Not ablated. Evaluated checkpoints are
  the best-on-validation ones, so this does not leak into the test numbers, but a reader
  should know `ppo_final.npz` is worse than `ppo_best.npz`.
* Both grip-randomized runs score lower at grip 1.0 (that is what validation measures) and much
  higher at low grip. Selecting DR checkpoints on grip-1.0 validation was a mismatch. The
  follow-up re-selected them on randomized-grip validation: this changes nothing for PPO, whose
  seed-0 choice is the same checkpoint, and a lot for ES (grip 0.5: 21–41% → 76–90%). See
  "Grip-randomized runs: 3 seeds, two selection rules".

## What worked, what didn't

* **Worked:** signed progress + crash-terminates as the whole reward. No shaping was needed
  and no hack appeared under these rules: no reversing, no spinning, no wall-hugging.
  Numba envs reached about 430 k steps/s on 4 threads. ES "just worked" with default-ish
  hyperparameters. The JS port in `demo.html` matches the Python sim to 1e-6 m over 120 steps
  (tested).
* **Didn't / limits:**
  * PPO's late-training decline (above).
  * The PPO+grip-rand. policy is twitchy (steer jitter 0.43 vs 0.10 for ES) and the slowest
    learner at grip 1.0.
  * No policy reaches the privileged controller's 100% on the hard tracks. Rays alone don't
    tell a car how narrow the *next* corner is.
* **Surprises:**
  * The tuned hand-written baseline is a genuinely strong competitor at the training grip.
  * The learned policies' robustness to grip they never saw.
  * The apex-cutting failure that lower grip fixes.

## Seeds (variance check)

A second seed of each learner was trained with the same config (`train_seeds.sh`). PPO
stopped after 10 min because its best checkpoint comes early. Evaluation:
`results/seeds/eval_table.md`, strict rules.

| completion (mean lap) | held-out 1.0 | held-out 0.8 | held-out 0.7 | hard 1.0 | hard 0.8 | hard 0.7 |
|---|---|---|---|---|---|---|
| ES seed 0 | 100% (25.2 s) | 98% | 83% | 80% | 100% | 94% |
| ES seed 2 | 100% (25.3 s) | 100% | 75% | 92% | 100% | 96% |
| PPO seed 0 | 99% (25.2 s) | 100% | 90% | 72% | 100% | 96% |
| PPO seed 2 | 99% (25.3 s) | 100% | 93% | 76% | 94% | 100% |

| learning | best validation | env steps to reach the baseline's 1355 |
|---|---|---|
| ES seed 0 / seed 2 | 1400 / 1390 | 114 M / 217 M |
| PPO seed 0 / seed 2 | 1398 / 1392 | 2.9 M / 2.3 M |

The headline numbers reproduce:
* about 100% held-out completion at 25.2–25.3 s;
* the grip-0.8 robustness;
* "hard tracks are easier at grip 0.8 than 1.0" holds for all 4 runs (80→100, 92→100,
  72→100, 76→94);
* PPO about 40–90× more sample-efficient than ES.

Hard-track completion at grip 1.0 moves by up to 12 points between seeds, so ES-vs-PPO
differences there are within seed noise. A seed-1 chain was also queued, but it waited more than 75 min for a shared
GPU/CPU slot and was cancelled; it was never run.

## Grip-randomized runs: 3 seeds, two selection rules (follow-up)

The cross-review flagged three things about the grip-randomized (DR) headline
"PPO + grip rand. finishes 93% at grip 0.5": it was **one seed**, its checkpoint was **very early**
(4.6 M of 78 M steps), and it was **selected on grip-1.0 validation** even though the policy was
trained for μ ~ U(0.5, 1.1). ES + grip rand. had the same problems. This follow-up addresses all three.

**What was run.**
* Seeds 1 and 2 of PPO + grip rand. and of ES + grip rand. Config is the same as seed 0, and so is
  the step budget: PPO stops at 2,386 updates (78.2 M steps), ES at 276 generations (305–316 M
  steps). Seed 0 had been stopped by its 20-min wall-clock limit on a busy machine; matching steps
  is the fairer budget here.
* A **rerun of seed 0** of both learners (`ppo_dr_r0`, `es_dr_r0`). Both are **bit-identical** to
  the original runs: same validation score at every logged step, and the update-140 / gen-269
  weights equal `ppo_dr_best.npz` / `es_dr_best.npz` exactly. Seed 0 can therefore be re-selected
  honestly, even though the original run kept only its best and final weights.
* All six runs saved a snapshot of the policy at every validation step (`--snapshots`; 238 for
  PPO, 28 for ES). `select_dr.py` scores every snapshot on **randomized-grip validation**: the same
  128 validation tracks (seeds 5000–5127, never the test tracks) and start points as before, each
  driven at 4 grips drawn once from U(0.5, 1.1), 512 episodes in all. The same draws are used for
  every snapshot and seed. It keeps the argmax as `<tag>_rg_best.npz`. The original grip-1.0
  rule's choice is still `<tag>_best.npz`. Self-check: the grip-1.0 argmax over snapshots equals
  the trainer's best file for all 6 runs.
* Every checkpoint was evaluated with the unchanged `eval.py` (100 held-out tracks and 50 hard
  tracks, 1 lap, strict rules, with the review's lap-finish fix) at grips 1.0, 0.8, 0.7, 0.6 and
  0.5 → `results/dr_seeds/` (`seeds_table.md`, `eval_table.md`, `selection.json`). Seed 0's
  grip-1.0-rule cells reproduce the main tables exactly (`ppo_dr` 96/100/100/100/93).

**Seeds × grip, held-out tracks: completion (mean lap time of finished laps); hard tracks: completion**

| run | selection rule | ckpt at | held-out 1.0 | 0.8 | 0.7 | 0.6 | **0.5** | hard 1.0 | hard 0.8 | hard 0.7 | hard 0.6 | hard 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PPO+rand. seed 0 | grip 1.0 (original) | 4.6 M | 96% (27.3 s) | 100% (27.9) | 100% (28.9) | 100% (30.4) | **93%** (33.4) | 60% | 66% | 90% | 98% | 98% |
| PPO+rand. seed 1 | grip 1.0 | 4.3 M | 92% (27.1 s) | 99% (27.9) | 100% (28.9) | 100% (30.4) | **96%** (33.3) | 66% | 66% | 88% | 98% | 100% |
| PPO+rand. seed 2 | grip 1.0 | 14.1 M | 95% (27.0 s) | 99% (27.6) | 100% (28.6) | 100% (30.2) | **93%** (33.3) | 70% | 76% | 78% | 90% | 98% |
| PPO+rand. mean (range) | grip 1.0 | | 94% (92–96) | 99% | 100% | 100% | **94% (93–96)** | 65% (60–70) | 69% (66–76) | 85% (78–90) | 95% (90–98) | 99% (98–100) |
| PPO+rand. seed 0 | **randomized** | 4.6 M (same ckpt) | 96% (27.3 s) | 100% (27.9) | 100% (28.9) | 100% (30.4) | **93%** (33.4) | 60% | 66% | 90% | 98% | 98% |
| PPO+rand. seed 1 | randomized | 9.2 M | 92% (27.4 s) | 99% (27.8) | 100% (28.7) | 100% (30.2) | **96%** (33.1) | 66% | 72% | 90% | 98% | 100% |
| PPO+rand. seed 2 | randomized | 7.2 M | 96% (27.1 s) | 100% (27.8) | 100% (28.7) | 100% (30.3) | **94%** (33.3) | 76% | 62% | 84% | 98% | 100% |
| PPO+rand. mean (range) | randomized | | 95% (92–96) | 100% | 100% | 100% | **94% (93–96)** | 67% (60–76) | 67% (62–72) | 88% (84–90) | 98% | 99% (98–100) |
| ES+rand. seed 0 | grip 1.0 (original) | 304 M | 100% (26.0 s) | 100% (27.6) | 100% (28.9) | 95% (30.5) | **41%** (32.7) | 90% | 100% | 100% | 100% | 66% |
| ES+rand. seed 1 | grip 1.0 | 298 M | 100% (25.9 s) | 100% (27.6) | 100% (29.0) | 89% (30.8) | **21%** (32.7) | 98% | 100% | 100% | 94% | 22% |
| ES+rand. seed 2 | grip 1.0 | 310 M | 100% (26.2 s) | 100% (27.7) | 100% (28.9) | 93% (30.8) | **33%** (32.2) | 98% | 100% | 100% | 100% | 50% |
| ES+rand. mean (range) | grip 1.0 | | 100% | 100% | 100% | 92% (89–95) | **32% (21–41)** | 95% (90–98) | 100% | 100% | 98% (94–100) | 46% (22–66) |
| ES+rand. seed 0 | **randomized** | 256 M | 98% (26.8 s) | 100% (27.9) | 100% (28.9) | 100% (30.6) | **90%** (33.0) | 88% | 94% | 100% | 100% | 100% |
| ES+rand. seed 1 | randomized | 262 M | 99% (26.6 s) | 100% (27.8) | 100% (28.9) | 100% (30.6) | **84%** (33.3) | 88% | 98% | 100% | 100% | 96% |
| ES+rand. seed 2 | randomized | 274 M | 99% (26.9 s) | 100% (28.0) | 100% (29.1) | 100% (30.8) | **76%** (33.4) | 96% | 96% | 100% | 100% | 94% |
| ES+rand. mean (range) | randomized | | 99% (98–99) | 100% | 100% | 100% | **83% (76–90)** | 91% (88–96) | 96% (94–98) | 100% | 100% | 97% (94–100) |
| Ray follower tuned w/ grip rand. | (one CEM run) | | 100% (31.8 s) | 100% (31.8) | 100% (31.9) | 100% (32.7) | **24%** (35.8) | 90% | 88% | 92% | 96% | 38% |

Lap times in the mean rows are means over seeds; the per-seed spread at a given grip is at most 0.5 s.

**Is "PPO + grip rand. 93% at grip 0.5" typical, lucky or unlucky? Typical, at the low end of the
range.** The two new seeds finish 96% and 93% (grip-1.0 rule) or 96% and 94% (randomized-grip rule).
The 3-seed mean is 94%, range 93–96%. Re-selecting seed 0 on randomized-grip validation picks
**the same checkpoint** (update 140), so the headline number is unchanged by the selection fix. The
rest of the PPO + grip rand. profile also reproduces: 99–100% at grips 0.8–0.6, 92–96% at grip 1.0,
and weak hard tracks at high grip (60–76% at grip 1.0).

**"Very early checkpoint": real, not a selection artifact.** Under randomized-grip selection, PPO's
best checkpoints are at 4.6, 9.2 and 7.2 M of 78 M steps. Its randomized-grip validation score
peaks around 1195–1203 in the first ~10 M steps and falls to 1010–1033 by the end of the run. This is
the same late decline the grip-1.0 PPO run shows (see "Learning dynamics"). PPO + grip rand. is
simply best early.

**Selection rule: small effect on PPO, large effect on ES.** The grip-1.0 rule picks ES's late
checkpoints (gen 269 for all 3 seeds), which score highest at grip 1.0 but have become fragile at
low grip. The randomized-grip rule picks gens 229–239. Those checkpoints are 0.6–0.8 s slower per lap at
grip 1.0 (and 1–2 laps fewer finished there) but finish 76–90% at grip 0.5 instead of 21–41%, 100% at grip 0.6
instead of 89–95%, and 94–100% of the hard tracks at grip 0.5 instead of 22–66%. **The earlier
"PPO + rand. 93% vs ES + rand. 41% at grip 0.5" gap was mostly the selection rule.** With matched
selection it is 93–96% vs 76–90%. PPO is still ahead at grip 0.5 on every seed pair. ES is ahead on
the hard tracks at grip 0.7–1.0 (88–100% vs 60–90%) and on lap time at grip 1.0 (26.6–26.9 s vs
27.0–27.4 s).

**Versus the grip-randomized ray follower (matched information, same grip randomization).**
* *Grip 0.5, held-out:* both learners beat it on every seed, provided ES is selected on randomized grip.
  PPO + rand. finishes 93–96% and ES + rand. 76–90%, against 24%. Under the original grip-1.0 selection,
  ES + rand. (21–41%) was **not** reliably better than the baseline (seed 1: 21% < 24%).
* *Grips 1.0–0.6, held-out:* a tie on completion (baseline 100%, learners 89–100%). The learners
  are 4.4–5.9 s per lap faster at grip 1.0.
* *Hard tracks:* the baseline beats PPO + rand. at grips 1.0, 0.8 and 0.7 on all 3 seeds and both
  rules (88–92% vs 60–90%). This confirms the reviewer's caveat; it is not seed noise. ES + rand. with
  randomized-grip selection ties it at grip 1.0 (88–96% vs 90%) and beats it at 0.8 (94–98% vs 88%)
  and 0.5 (94–100% vs 38%).
* Net: the learned advantage over a matched-randomization baseline is **lap time and the grip-0.5
  extreme**, and it holds across seeds. With randomized-grip selection, ES + rand. is the most
  balanced grip-randomized driver measured here. PPO + rand. is the most reliable at grip 0.5 but the
  weakest on hard tracks. The baseline is one deterministic CEM tuning run; its tuning-seed variance was not measured.

**What is unchanged.** Main tables above still show seed 0 with the original (grip-1.0) selection.
For PPO that is also the randomized-grip choice. For ES + grip rand. it is the worse of the two
choices at low grip; see this section. `media/demo.html` still contains the original `es_dr_best`
and `ppo_dr_best` checkpoints.

## Caveats

* **Two seeds per learner** (main tables use seed 0). The 0.65 s ES-vs-baseline gap holds on
  95/100 tracks and reproduces with seed 2, but small ES-vs-PPO differences are within seed
  noise. The grip-randomized variants now have **three seeds each** (follow-up section); the
  soft-wall hack demo has one. The grip-randomized ray-follower baseline is one CEM tuning run.
* The main grip-randomized rows are seed 0 with the original grip-1.0 selection. Under
  randomized-grip selection ES + grip rand. is much more robust at low grip (see the follow-up
  section). `media/demo.html` still contains the original selection.
* MLX use: PPO trains in MLX. ES needs no gradients, and its population forward pass (129
  copies of a 1.6k-weight MLP) runs in numpy, which is faster than a GPU round trip at this
  size. All evaluation uses the same numpy forward for every policy; PPO's MLX actor is
  exported to that flat-vector format.
* The demo's JS runs float64 throughout; the Python policy forward is float32. Trajectories
  agree closely but not bit-exactly over a whole lap.
* Not measured: camera/pixel observations, opponents, behavioral cloning, SAC/MPC.

## Next steps

* More seeds of the non-randomized ES/PPO runs (they have 2); a PPO ablation with γ = 0.995 and lr decay to test the
  late-decline hypothesis.
* Train on a width mixture (or add the hard generator to training) to remove the
  apex-cutting failure.
* Done in the follow-up: 3 seeds and randomized-grip validation for the grip-randomized runs.
  Still open: why ES + grip rand. loses low-grip robustness late in training while its grip-1.0
  score keeps rising.
* Go-deeper items: rain (grip that varies along the track), opponents, camera observations.

## Round 5: Discover the road's grip while driving

**TL;DR (plain words, added by the Claude reviewer; every number below was re-traced).**
- *What was tested.* The road secretly turns slippery: invisible wet patches, or a mid-lap change
  from grippy to slippery (or back). Nothing in what the car sees (wall-distance rays, speed, its
  last commands, a yaw-rate gyro) says "wet". Three drivers were compared on 384 fresh test laps
  each (same tracks, same patch positions and grip values for everyone): a PPO policy with memory
  (a small GRU), the same policy with memory switched off during training ("FF"), and a hand-built
  wall-following controller with an online grip estimator whose 13 knobs were tuned by CEM.
- *Who finished.* FF **383/384**, memory policy **370/384**, classical + estimator **382/384**.
  Freezing the classical estimator drops it to 309/384, so its online grip tracking matters.
- *Who recovered fastest after a grip change.* "Recovery" = time until the car is back near the
  middle of the road at a sensible speed for 0.5 s; misses count as 5 s. Classical averages
  **0.67 s** and recovers **98%** of changes; FF 0.95 s / 91%; memory policy 1.08 s / 89%.
  Classical wins mainly by recovering more events, not by recovering faster when both recover.
- *Why memory didn't help.* In this simulator grip is readable from a single sensor frame: when the
  car turns harder than the grip allows, the gyro shows the yaw rate pinned at grip×g/speed, so
  gyro × speed / g ≈ grip. A fixed probe applied to FF's test laps reads the new grip within
  0.01 on the first flagged frame after a change; across all flagged frames it calls wet vs dry
  correctly 99.6% of the time. The catch: this only works while the car
  is turning at the limit. The learned drivers do that almost all the time (FF steers at ≥95% of full
  lock on 38% of frames (reviewer replay) and asks for more yaw than grip allows on 88% (audit
  `sensor_probe.ff…true_saturation_fraction`)), because in this simple car model over-steering
  costs nothing. So "memory isn't needed" holds for *this* sim and these driving styles, not for
  grip estimation in general. The gyro is itself a one-step difference, so the observation already
  carries one step of history.
- *How sure.* One training/tuning seed. The memory policy *beat* FF on validation and lost on the
  fresh test, so read "memory did not help in this seed", not "memory hurts". CEM got ~5.6× more
  simulator transitions than each PPO arm; both sides start from earlier-round artifacts.

**Result (2026-10-24, seed 0).** Recurrent memory did not improve this task.
The matched feed-forward arm finished **383/384 (99.74%)**, the RNN **370/384
(96.35%)**, and the adaptive classical ray follower **382/384 (99.48%)**.
The classical controller had the best capped safety-envelope recovery time and
coverage. These are one training/tuning seed, not a seed-variance result.
Headline source: [fresh test receipt](results/r5/r5_seed0_test.json),
`methods.*.overall` and `episodes`; every aggregate was independently recomputed
from those episode rows in [the Turn-2 audit](results/r5/r5_turn2_analysis_v2.json).

**Setup and accounting.** Invisible wet patches, bidirectional mid-lap switches,
and dry controls each use 64 fresh regular and 64 fresh hard tracks: 384 episodes
per method, with 384 planned transitions. Test scenario seed is 2026102408;
validation is all six groups ×16, including hard tracks. No controller sees true
grip, track curvature, or paint. RNN, FF and classical methods share 10 wall rays,
speed, previous steering/pedal commands and measured yaw rate. Both neural arms
start from the same historical PPO+DR checkpoint; FF uses the same cell with hidden
state reset every step. The historical frozen policy lacks the added gyro and R5
fine-tuning. Historical neural pretraining and classical tuning budgets are
unmatched. See [design](DESIGN_R5.md) and receipt `comparison_notes`/`config`.

Strict wall contact ends an attempt; all failures here were crashes, with no
timeouts. Lap penalty is 60 s for a DNF. Recovery means five consecutive 0.1 s
steps within the central half of the usable road and 0.6–1.3 times a curvature/grip
reference speed. Reported latency is the **onset** of that sustained interval,
confirmed 0.4 s later. Unreached, interrupted and unrecovered events retain a 5 s
penalty and remain in the coverage denominator. This measures driving recovery,
**not time to identify grip** (`r5_env.normal_envelope`, `r5_eval.Events`).

| Method | Finishes / 384 | Crash % | Finished lap s | DNF-penalized s | Capped recovery s | Recovered / 384 |
|---|---:|---:|---:|---:|---:|---:|
| FF, matched R5 fine-tuning | 383 | 0.26 | 32.54 | 32.61 | 0.951 | 350 (91.15%) |
| RNN | 370 | 3.65 | 32.84 | 33.83 | 1.083 | 343 (89.32%) |
| RNN, memory off; not retuned | 373 | 2.86 | 33.14 | 33.91 | 1.130 | 345 (89.84%) |
| Adaptive ray + online grip estimator | 382 | 0.52 | 32.24 | 32.39 | 0.670 | 377 (98.18%) |
| Adaptive ray, estimator frozen; not retuned | 309 | 19.53 | 33.81 | 38.93 | 1.117 | 331 (86.20%) |
| Historical FF, frozen extra-data reference | 300 | 21.88 | 30.28 | 36.78 | 1.352 | 321 (83.59%) |

All cells above: fresh receipt `methods.<name>.overall`. Finished-only means have
different survivor sets. The RNN is 0.482 s slower than FF on their 369 jointly
finished episodes (`paired.rnn_minus_ff`). Turning RNN memory off gains three
finishes but worsens capped recovery and lap time; it does **not** dominate every
metric. Freezing the classical estimator costs 73 finishes, showing that its
online adaptation matters; the frozen version is an ablation, not a separately
tuned best fixed-grip controller.

**Hard groups and adaptation curves.** Cells below are finishes/64; capped
recovery seconds, from the same receipt's `methods.*.groups`. Static controls have
no surface-change events.

| Group | FF | RNN | RNN memory off | Adaptive ray |
|---|---:|---:|---:|---:|
| normal_patch | 64; 0.505 | 64; 0.585 | 64; 0.641 | 63; 0.527 |
| hard_patch | 63; 1.439 | 58; 1.684 | 60; 1.694 | 63; 0.880 |
| normal_switch | 64; 0.303 | 64; 0.288 | 64; 0.413 | 64; 0.322 |
| hard_switch | 64; 1.516 | 61; 1.670 | 62; 1.700 | 64; 0.881 |
| normal_static | 64; — | 64; — | 64; — | 64; — |
| hard_static | 64; — | 59; — | 59; — | 64; — |

All 14 RNN crashes are hard: six patch, three switch, five static. Thus its deficit
also involves ordinary hard-track driving. On hard patches, recovery coverage by
1 s / by the full window is **60.9% / 82.8% FF**, **57.0% / 79.7% RNN**, and
**75.0% / 96.1% classical**. On hard switches it is **59.4% / 82.8%**,
**54.7% / 81.3%**, and **65.6% / 100%**, respectively. Source: audit
`test.*.curves`; [recovery curves](media/r5_turn2_v2_recovery.png) retain all planned
events, rather than plotting the 5 s failure cap as a successful recovery.

The classical method is not fastest under every conditioning: among recovered
events, FF averages **0.558 s**, classical **0.589 s**, RNN **0.614 s**. Classical
wins the capped metric chiefly by recovering more events. Moreover, **32/32 FF**
and **34/35 RNN** expired recovery windows follow grip *increases*. Instrumented
replay finds FF's expired windows outside the lateral envelope on 53.4% of steps,
underspeed on 43.4%, and overspeed on 0.06%; corresponding RNN fractions are
58.9%, 38.2%, 0.34%. Categories can overlap. Increased grip raises the reference
speed, so cautious acceleration and cornering lines can fail this criterion even
without a crash. Sources: audit `test.*.direction_curves` and
`sensor_probe.*.groups.overall.expired_mean_violation_fraction`. The metric should
not be described as a pure slip-detection speed ranking.

**Why recurrence is unnecessary here.** The observation itself includes a
measurement of the preceding transition. For speed midpoint `v_mid`, the simulator
clips yaw rate to `mu*g/max(v_mid,1)`; requested yaw is
`v_mid*tan(previous_steer*0.45)/2.6`. A single packet contains previous steer,
current speed and actual yaw rate. Using current speed in place of the midpoint
therefore detects saturation and estimates grip approximately, without retaining
an earlier packet. Unexcited motion only gives a lower bound: grip is not known
in advance of a change or before it affects motion. See `racing/sim.py:step_kernel`,
`r5_env.py:SurfaceRace.step`, and `analyze_r5.py:single_packet_probe`.

A fixed, unfitted 0.05 rad/s slip threshold was applied retrospectively to frozen
policies' full test replays. All episode outcomes **and every event row** matched
the original receipts. On nonterminal driving frames with both speeds above
3 m/s, the probe flags **87.57% of FF frames**, with **99.46% saturation precision**,
grip MAE **0.0121**, and **99.62% wet/dry accuracy** on flagged frames. For RNN those
values are **78.13%, 99.73%, 0.0093, 99.77%**. The first post-change packet flags
**91.15% of all planned FF changes** (within 0.1 s); by 0.5 s it flags **98.70%**.
Its first flagged estimate has MAE **0.0079**. For hard_patch/hard_switch, immediate
FF coverage remains **85.94%/90.63%**. Source: audit `sensor_probe`; ground-truth
grip and previous speed score the probe only and are not its inputs. The first
flag's timing is measured relative to the simulator's change, which the probe is
not told. This establishes available information, not what the learned FF
internally represents. **The supplied task does not require persistent recurrent
memory for strong performance.** The classical estimator does use history, and
may still benefit from it, particularly between informative measurements.

**Training and selection audit.** [PPO history](logs/r5/r5_seed0_train.jsonl) ends
at **4,436 paired updates**, **36,339,712 environment steps and 70,976 gradient
updates per arm**, **1,076.93 s** of the 1,080 s allocation, `wall_guard`. Active
time was RNN **553.39 s**, FF **519.89 s**. Both had **39** full 96-episode
validation checks. Selection maximized the declared equal-group utility; both
selected exports exactly match the logged maxima:

| Arm | Initial utility | Selected update; samples | Selected utility; finishes/96 | Last logged utility (4332) | Final weights utility (4436); finishes/96 |
|---|---:|---:|---:|---:|---:|
| RNN | 1.20848 | 2736; 22,413,312 | 2.42795; 96 | 2.10758 | 2.04318; 90 |
| FF | 1.20848 | 2280; 18,677,760 | 2.38696; 95 | 2.25489 | 2.35930; 95 |

Final-weight validation was measured in Turn 2 on the full original validation
set, using the saved rolling state; it changes no selected checkpoint. Source:
audit `selection.arms`, selected checkpoint metadata, and
[validation curves](media/r5_turn2_v2_validation.png). RNN actually wins selected
validation utility, then loses on fresh test. That points to selection/generalization
variance, not an incorrect argmax or an obviously premature stop. Later training
did not improve the selected score. It does not prove optimal RNN training: one
seed, no learning-rate/unroll sweep, a direct feed-forward actor path, and only
16-step (1.6 s) backpropagation remain limitations. A richer recurrent model or
longer credit assignment was not tested. No test-based reselection was performed.

**Classical tuning was substantial.** [CEM history](logs/r5/r5_seed0_baseline.jsonl)
records **627 generations**, **30,096 candidate evaluations** (48 each, including
repeated incumbents), **24 scenarios per candidate**, **205,031,674 active
transitions**, and **716.44 s** of 720 s, `wall_guard`. It received more wall time
than either neural arm's active time and 5.64× as many transitions per arm; this is
not an equal-sample comparison. It had **75** full validation evaluations of
candidate winners/means, with 25% broad random exploration and a normalized
standard-deviation floor of 0.08. The historical DR follower was included as an
initial candidate. Validation utility improved from **2.30538** for that seed to
**2.44347**, selected generation **561**, with 96/96 finishes. Its 13 search ranges
and selected values are recorded in
[the selected baseline](checkpoints/r5_seed0_baseline_selected.json):

| Knob | Range | Selected |
|---|---:|---:|
| steering gain | 0.5–3 | 1.95919 |
| ray exponent | 1–8 | 1.88002 |
| centering gain | 0–2 | 1.37961 |
| brake fraction | 0.25–1.2 | 0.78658 |
| margin | 2–18 | 9.51680 |
| maximum speed | 18–40 | 26.86278 |
| minimum speed | 2–8 | 4.77626 |
| grip prior | 0.4–1.1 | 0.78661 |
| forgetting factor | 0.8–1 | 0.84608 |
| optimism rise | 0–0.15 | 0.08171 |
| grip safety factor | 0.55–1 | 0.77047 |
| slip tolerance | 0.005–0.15 | 0.05362 |
| grip steering exponent | 0–0.8 | 0.14344 |

None is at a bound; the closest is ray exponent, **12.57% of its range** from the
lower edge (audit `selection.baseline.bounds`). Selection improved late, so global
convergence is not established, but there is no boundary knob demanding an
immediate wider search. Compute history before R5 remains unmatched.

**Secondary tracks, artifacts and next step.** On the original 100 regular/50
hard layouts crossed with three surface conditions (450 episodes), FF finishes
**449**, RNN **441**, memory-off RNN **443**, adaptive ray **450**, frozen estimator
**351**, and historical frozen FF **389**. Capped recovery is respectively
**0.650, 0.878, 0.930, 0.502, 1.121, 1.032 s**. Source:
[legacy receipt](results/r5/r5_seed0_legacy.json), `methods.*.overall`. This older,
2:1 regular/hard split is secondary evidence, not a fresh holdout.

[The offline race demo](media/r5_seed0_demo.html) retains the first RNN crash per
group where present; otherwise it uses the first test case. Original simulator
and checkpoint/source hashes were verified unchanged. The earlier full-bank
track-generation failure and its deterministic R5-only replacement remain recorded
in [the track audit](results/r5/r5_turn1b_track_audit.json) and original console logs.
Turn-2 tests: **23 passed in 9.89 s**
([log](logs/r5/r5_turn2_tests_final.log)); analysis **3.77 s** with a warm plotting
cache ([log](logs/r5/r5_turn2_analysis_v2.log)); first plotting-cache build took
13.28 s. No MLX or training ran in Turn 2.

No further training is queued within the optional 35-minute follow-up budget.
Removing the gyro is a useful candidate change, but alone does not establish a
memory requirement: current motion/rays and previous commands still convey
physical feedback. The smallest convincing memory test would add an informative
forced grip probe, then present **identical decision-time observations** for wet
and dry cases while preserving the probed grip, before a bend whose safe/fast
choice differs. Reset/mask pose, speed and action-history cues at that decision;
keep a lap-time cost so universal caution cannot win. First verify the paired
observation aliasing and distinct optimal actions in NumPy; then train fresh-seed
RNN/FF arms and retune the classical estimator on new train/validation splits,
reserving another fresh test. That task and its training runtime are **not
measured**. Repeating the present easy-to-observe task would measure seed variance,
not resolve its weak test of memory.
