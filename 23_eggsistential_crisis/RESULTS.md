# Results: 23 Eggsistential Crisis

All numbers come from real runs on 2026-10-24 on a shared M5 Max (up to seven builders at once,
load average 20-29 on 18 cores during training). Every speed figure is a **busy-machine number**.

## TL;DR
- **PPO learned the narrow task and then much more.** On held-out seeds with |δ| ∈ [0.15, 0.4] rad,
  the final policy succeeds in **99.6% ± 0.4%** of 256 episodes, and **86.7% stay within the 10 N
  force budget** for the whole episode. The best cheap baseline, the force-aware CEM-tuned scripted
  finger gait, reaches **50.4%**, and all of its episodes stay under 10 N. Hold still gets 3.9%.
  (`results/eval.json`, main table below.)
- **What "success" means here (read this before the numbers).** Success is judged on the egg's
  **long axis only** (axis error < 0.1 rad at t = 4 s, not dropped). Spin about the long axis is
  ignored, because the egg is an ellipsoid of revolution (3×3×4 cm) and that spin is physically
  invisible. Under gymnasium-robotics' own full-quaternion criterion, where spin counts, PPO
  succeeds in only **27.3%** of main-range episodes, against 25.4% for the tuned gait, 13.3% for
  the force-aware gait and 0.8% for hold still (median full-quaternion error 11.5° for PPO,
  12.2-13.0° for the gaits). So the 99.6% vs 50% gap is a gap in **pointing the long axis**. In
  the gym task's own terms, PPO and the tuned gait are about level, because neither controls spin.
  PPO's success is stable, not a transient at t = 4 s: 99.6% stay under the threshold for the
  whole last 0.5 s, against 35.9-45.3% for the three gaits (`results/eval_review.json`; review table
  at the end). The two original gaits also swap places under gym's metric: the tuned gait is at
  25.4% and the force-aware one at 13.3%.
- The curriculum widened the target range to the full ±90° within 1.5M env steps. On 46-90°
  rotations, PPO still succeeds 98.4% of the time, but only 36.7% of those episodes stay within
  the budget. The baselines tuned on ≤ 0.4 rad never succeed at that range. A reviewer retune of
  the gait on PPO's full curriculum range (0.15 to π/2) reached only 1.6% there and 24.2% at
  23-46° (from 17.2%), so most of that gap is the controller class, not the tuning range. The
  retune was short and noisy, though (see Review).
- **Force was the hard part, not the rotation.** After run1 (7.5M steps, weak force penalty),
  success was 98.8% but within-budget only 45.7%, with a median peak of 10.5 N. Two chained
  force-aware fine-tunes (21.3M more steps) raised within-budget to 86.7% and lowered the median
  peak to 7.5 N without losing success.
- **Drops don't discriminate here.** No method dropped the egg at nominal friction and mass. The
  palm cradles it, and even full-range random actions didn't knock it out in our smoke test. The
  only drops seen anywhere are PPO's, out of distribution: 3.1% at 0.4× mass and 0.8% at 0.3×
  friction.
- "Target already satisfied" rate: hold still succeeds in 22.7% of episodes when near-zero
  targets are included (|δ| ∈ [0, 0.4]), against 3.9% when |δ| ≥ 0.15. Drift accounts for the
  rest.
- Robustness outside the training range (0.7-1.3×): PPO's success stays at 78-100%, but heavy
  eggs break the force budget. At 3× mass, only 5.5% of episodes are within budget: it squeezes
  harder to carry a 4.4 N egg.

## Setup in one paragraph
Shadow hand + egg from gymnasium-robotics' `HandManipulateEggRotate_ContinuousTouchSensors-v1`
assets. The narrow first-version task: the egg lies on its side in the palm, and the target is the
same pose rotated about world z by δ, with |δ| ∈ [0.15, 0.4] rad (8.6-23°) at first. The curriculum
widens the range to π/2. Success means the egg's long-axis error is < 0.1 rad (5.7°) at the end of
a 4 s episode and the egg wasn't dropped. The force budget is 10 N of total normal contact force on
the egg (about 6.8× its weight), measured every 2 ms substep in evaluation. Full definitions are in
[README.md](README.md).

## The gymnasium-robotics blocker (fixed)
- Symptom: `gym.make("HandManipulateEggRotate_ContinuousTouchSensors-v1")` raises an AssertionError in
  `mujoco_utils.get_joint_qpos` (already in `make()`, which calls `_get_obs`).
- Cause: not a wrong joint type. The hand's joints are **hinges** (type 3), but under mujoco 3.16
  `np.int32(3) in (mjtJoint.mjJNT_HINGE, mjtJoint.mjJNT_SLIDE)` evaluates to `False`, while
  `np.int32(3) == mjJNT_HINGE` is `True`.
- Fix: `gymfix.py` monkeypatches the four helpers in memory and casts the joint type to `int`.
  Nothing in site-packages is edited.
- Semantics check (`test_parity.py`, passes): the patched helpers equal raw qpos/qvel slices for
  all 26 joints. Our `egg_env.py` base observation (153-d: 24 joint pos, 24 joint vel, 6 egg vel,
  7 egg pose, 92 touch) equals the patched gym observation exactly on the same state. After one
  control step from the same state with the same action, it matches to within 1e-6 (actually 0.0),
  over 3 resets × 20 steps.

## Methods and budgets
| method | what | tuning (training seeds only) | budget |
|---|---|---|---|
| zero (hold still) | action 0 = all joints mid-range (gym's reset action) | none | - |
| posture (tuned hold) | best constant action | CEM, 20 params, 6 iters × 24 candidates × 64 eps | ~0.9M env steps |
| random | σ·U[-1,1] around 0 | grid σ ∈ {0.02, 0.05, 0.1, 0.2, 0.4, 1.0} × 64 eps → σ = 0.1 | 38k env steps |
| scripted gait (tuned) | closed loop: a = c + A·sin(s·2πf·t + φ), where s is the sign of the yaw that reduces the error; it holds c once error < 0.8·thr | CEM, 61 params, c warm-started from the tuned posture, 8 iters × 24 × 64 eps | ~1.3M env steps |
| scripted gait (force-aware retune) | same controller, warm-started from the gait above | CEM 8 × 24 × 64 eps on objective P(success ∧ end-of-step force < 7 N) − P(drop): the same margin and force signal PPO's run2/run3 penalty used | ~1.3M env steps |
| PPO (asym. critic, touch) | MLX PPO. The actor sees the 153-d gym obs (log(1+touch)), egg axis, goal, error features and the previous action. The critic also sees friction scale, mass scale, true egg contact force, t/T, δ and the initial error | 3 chained runs | run1 43 min / 7.47M env steps (912 iters) → run2 43 min / +10.46M (to 2189) → run3 40 min / +10.85M (to 3514). Total **28.8M env steps, 126 min** |

Baseline tuning objective, stage 1 (all baselines): success − drop − 0.5·P(end-of-step force > 10 N).
Total baseline tuning was 939 s (stage 1) + 363 s (gait force-aware retune), about 3.5M env steps.

PPO reward per 40 ms step:
- 10·(error decrease)
- +0.5 while error < thr
- −min(cap, w·max(0, F_end − margin)), where F_end is the end-of-step contact force
- −0.001·‖Δa‖²
- −5 on a drop

run1 used w = 0.05, margin 10 N, cap 0.5. run2 and run3 resumed the same weights and optimizer with
w = 0.2, margin 7 N, cap 2 (a force-aware fine-tune). The 7 N margin came from a measurement on the
run1 policy at iter 300: the per-substep peak exceeds the end-of-step force by a median ×1.00, p90 ×1.44 and
p99 ×2.8 per step, and the per-episode max was a median 9.95 N (substep) against 7.6 N
(end-of-step). Other settings: 128 envs × 64-step rollouts, 5 epochs × 4 minibatches, lr 1.5e-4,
clip 0.2, γ 0.99, λ 0.95, tanh-bounded mean, state-independent log-std initialised at −1. Rollouts
run the actor in numpy on CPU, and log-probs and values are recomputed in MLX before the update.
Curriculum: θmax += 0.2 when the last 256 training episodes had > 60% success.

## Main results (held-out seeds 10001-10006, deterministic PPO mean action, per-substep force)
Episodes come from per-env RNG streams `default_rng([seed, i])`. Training used seeds 0, 913000 and
2190000, baseline tuning used 999-2001, and evaluation used 10001-10006, so no episode overlaps.
"PPO before force fine-tune" is the run1 final checkpoint (`checkpoints/run1/latest.npz`).
"PPO (asym. critic, touch)" is the run3 final checkpoint (`checkpoints/run3/latest.npz`).
"Success within budget" means success **and** a per-substep peak force ≤ 10 N throughout the
episode. Figures: `media/eval_error_force.png`, `media/robustness.png`,
`media/training_curves.png`.
**Main: held-out, θ∈[0.15, 0.4] rad (8.6-23°), training friction/mass distribution** (n=256 episodes per method)

| method | success | success within budget | ever reached | drop | final err median | peak force median / p90 / max (N) | episodes over 10 N |
|---|---|---|---|---|---|---|---|
| zero (hold still) | 3.9% ± 1.2% | 3.9% | 3.9% | 0.0% | 15.9° (start 15.6°) | 1.9 / 2.9 / 4 | 0.0% |
| posture (tuned hold) | 14.1% ± 2.2% | 14.1% | 20.7% | 0.0% | 17.7° (start 15.6°) | 2.0 / 2.5 / 3 | 0.0% |
| random (sigma=0.1) | 7.0% ± 1.6% | 7.0% | 9.4% | 0.0% | 16.1° (start 15.6°) | 2.8 / 6.3 / 8 | 0.0% |
| scripted gait (tuned) | 43.0% ± 3.1% | 43.0% | 61.7% | 0.0% | 7.2° (start 15.6°) | 4.0 / 4.9 / 6 | 0.0% |
| PPO (asym. critic, touch) | 99.6% ± 0.4% | 86.7% | 100.0% | 0.0% | 0.8° (start 15.6°) | 7.5 / 10.5 / 24 | 12.9% |
| scripted gait (force-aware retune) | 50.4% ± 3.1% | 50.4% | 65.6% | 0.0% | 5.7° (start 15.6°) | 4.5 / 5.8 / 7 | 0.0% |
| PPO before force fine-tune | 98.8% ± 0.7% | 45.7% | 100.0% | 0.0% | 1.3° (start 15.6°) | 10.5 / 30.0 / 50 | 54.3% |

**Including near-zero targets, θ∈[0, 0.4] rad** (n=128 episodes per method)

| method | success | success within budget | ever reached | drop | final err median | peak force median / p90 / max (N) | episodes over 10 N |
|---|---|---|---|---|---|---|---|
| zero (hold still) | 22.7% ± 3.7% | 22.7% | 32.0% | 0.0% | 11.0° (start 10.6°) | 2.0 / 3.1 / 4 | 0.0% |
| posture (tuned hold) | 10.9% ± 2.8% | 10.9% | 45.3% | 0.0% | 14.7° (start 10.6°) | 2.0 / 2.5 / 3 | 0.0% |
| random (sigma=0.1) | 20.3% ± 3.6% | 20.3% | 38.3% | 0.0% | 12.1° (start 10.6°) | 2.8 / 6.6 / 8 | 0.0% |
| scripted gait (tuned) | 55.5% ± 4.4% | 55.5% | 79.7% | 0.0% | 5.3° (start 10.6°) | 3.8 / 4.6 / 6 | 0.0% |
| PPO (asym. critic, touch) | 100.0% ± 0.0% | 80.5% | 100.0% | 0.0% | 0.8° (start 10.6°) | 7.4 / 10.7 / 18 | 19.5% |
| scripted gait (force-aware retune) | 53.9% ± 4.4% | 53.9% | 75.0% | 0.0% | 5.5° (start 10.6°) | 4.5 / 5.5 / 7 | 0.0% |
| PPO before force fine-tune | 98.4% ± 1.1% | 57.0% | 100.0% | 0.0% | 1.6° (start 10.6°) | 8.7 / 29.0 / 48 | 42.2% |

**Wider targets, θ∈[0.4, 0.8] rad (23-46°)** (n=128 episodes per method)

| method | success | success within budget | ever reached | drop | final err median | peak force median / p90 / max (N) | episodes over 10 N |
|---|---|---|---|---|---|---|---|
| zero (hold still) | 0.0% ± 0.0% | 0.0% | 0.0% | 0.0% | 32.0° (start 33.7°) | 2.0 / 2.7 / 4 | 0.0% |
| posture (tuned hold) | 0.0% ± 0.0% | 0.0% | 0.0% | 0.0% | 32.5° (start 33.7°) | 2.0 / 2.4 / 4 | 0.0% |
| random (sigma=0.1) | 0.0% ± 0.0% | 0.0% | 0.0% | 0.0% | 32.8° (start 33.7°) | 2.8 / 6.0 / 8 | 0.0% |
| scripted gait (tuned) | 14.1% ± 3.1% | 14.1% | 17.2% | 0.0% | 20.6° (start 33.7°) | 4.2 / 4.9 / 6 | 0.0% |
| PPO (asym. critic, touch) | 100.0% ± 0.0% | 69.5% | 100.0% | 0.0% | 0.8° (start 33.7°) | 8.6 / 15.2 / 34 | 30.5% |
| scripted gait (force-aware retune) | 17.2% ± 3.3% | 17.2% | 18.8% | 0.0% | 17.8° (start 33.7°) | 4.8 / 5.8 / 7 | 0.0% |
| PPO before force fine-tune | 97.7% ± 1.3% | 9.4% | 100.0% | 0.0% | 1.4° (start 33.7°) | 24.3 / 47.1 / 57 | 90.6% |

**Widest targets, θ∈[0.8, π/2] rad (46-90°)** (n=128 episodes per method)

| method | success | success within budget | ever reached | drop | final err median | peak force median / p90 / max (N) | episodes over 10 N |
|---|---|---|---|---|---|---|---|
| zero (hold still) | 0.0% ± 0.0% | 0.0% | 0.0% | 0.0% | 67.0° (start 68.2°) | 1.9 / 2.6 / 3 | 0.0% |
| posture (tuned hold) | 0.0% ± 0.0% | 0.0% | 0.0% | 0.0% | 67.3° (start 68.2°) | 2.0 / 2.5 / 3 | 0.0% |
| random (sigma=0.1) | 0.0% ± 0.0% | 0.0% | 0.0% | 0.0% | 67.6° (start 68.2°) | 2.8 / 5.9 / 8 | 0.0% |
| scripted gait (tuned) | 0.0% ± 0.0% | 0.0% | 0.0% | 0.0% | 56.2° (start 68.2°) | 4.2 / 5.1 / 6 | 0.0% |
| PPO (asym. critic, touch) | 98.4% ± 1.1% | 36.7% | 98.4% | 0.0% | 0.8° (start 68.2°) | 11.6 / 20.8 / 47 | 63.3% |
| scripted gait (force-aware retune) | 0.0% ± 0.0% | 0.0% | 0.0% | 0.0% | 51.6° (start 68.2°) | 4.9 / 5.9 / 7 | 0.0% |
| PPO before force fine-tune | 96.9% ± 1.5% | 0.0% | 97.7% | 0.0% | 1.2° (start 68.2°) | 36.6 / 49.8 / 64 | 100.0% |

**Robustness** (θ∈[0.15,0.4] rad, n=128 per cell; training range was friction and mass scale 0.7-1.3). Cell: success / success within budget / drop

| friction, mass | zero (hold still) | posture (tuned hold) | scripted gait (tuned) | scripted gait (force-aware retune) | PPO (asym. critic, touch) |
|---|---|---|---|---|---|
| 0.3, 1.0 | 1.6% / 1.6% / 0.0% | 8.6% / 8.6% / 0.0% | 28.1% / 28.1% / 0.0% | 22.7% / 22.7% / 0.0% | 85.9% / 75.0% / 0.8% |
| 0.5, 1.0 | 2.3% / 2.3% / 0.0% | 14.8% / 14.8% / 0.0% | 40.6% / 40.6% / 0.0% | 33.6% / 33.6% / 0.0% | 100.0% / 82.0% / 0.0% |
| 1.0, 1.0 | 3.9% / 3.9% / 0.0% | 18.0% / 18.0% / 0.0% | 60.2% / 60.2% / 0.0% | 56.2% / 56.2% / 0.0% | 100.0% / 80.5% / 0.0% |
| 2.0, 1.0 | 4.7% / 4.7% / 0.0% | 15.6% / 15.6% / 0.0% | 57.8% / 57.8% / 0.0% | 66.4% / 66.4% / 0.0% | 98.4% / 92.2% / 0.0% |
| 1.0, 0.4 | 1.6% / 1.6% / 0.0% | 5.5% / 5.5% / 0.0% | 13.3% / 13.3% / 0.0% | 28.1% / 28.1% / 0.0% | 78.1% / 60.9% / 3.1% |
| 1.0, 2.0 | 3.9% / 3.9% / 0.0% | 11.7% / 11.7% / 0.0% | 39.8% / 39.8% / 0.0% | 46.1% / 46.1% / 0.0% | 93.8% / 47.7% / 0.0% |
| 1.0, 3.0 | 6.2% / 6.2% / 0.0% | 11.7% / 11.7% / 0.0% | 25.8% / 25.8% / 0.0% | 35.9% / 35.9% / 0.0% | 84.4% / 5.5% / 0.0% |

PPO checkpoint: checkpoints/run3/latest.npz, iter 3514, env steps 28786688, final policy std 0.184 (eval uses the mean action)

## What worked
- **Fixing the env in our own code.** `gymfix.py` is 60 lines, and the parity test shows our faster
  threaded env is the same MDP as the gym env (identical observations and dynamics).
- **PPO with an asymmetric critic plus a tanh-bounded mean, starting from "hold still".** The
  actor's last layer was initialised ×0.01, so the initial policy is the zero action, which holds
  the egg safely. Training success went from 0 to 60% in 0.9M steps on the first range, and the
  curriculum then went 0.4 → π/2 rad in six steps by 1.5M steps (`media/training_curves.png`).
  Approx-KL had a median of 0.024 (1st-99th percentile 0.013-0.032) and clip fraction a median of
  0.28 (0.19-0.33). In 825 of 3514 iterations, the KL early stop (0.045) cut the 20 minibatch
  updates short. Recomputing log-probs in MLX before each update kept the first ratio at exactly 1.
- **Force-aware fine-tuning by chaining runs.** Same weights and optimizer, only the force
  penalty changed (0.05/N above 10 N with cap 0.5 → 0.2/N above 7 N with cap 2). Held-out
  within-budget success rose 45.7% → 86.7%. On the 96-episode peek set it went 0.39 → 0.60
  (mid-run2) → 0.75 (end of run2), so it was still rising when it stopped. The 7 N margin came
  from measuring how far substep peaks exceed the end-of-step force the reward sees.
- **Strong cheap baselines.** The scripted gait is only a 61-number sinusoid with a sign switch,
  and CEM got it to 43-50% held-out success on the training range with peaks of 4-7 N. That makes
  it a meaningful bar and a fair "cheap dexterity" reference.
- Headless MuJoCo rendering (`mujoco.Renderer`, CGL) worked without any fallback.

## What didn't / caveats
- **The force budget is only partly met.** 12.9% of main-range episodes still exceed 10 N at some
  2 ms substep (max 24 N), and the share grows with rotation size: 30.5% at 23-46° and 63.3% at
  46-90°. PPO's median peak (7.5 N) is still about 1.7× the gait's (4.5 N). The training signal
  sees the end-of-step force, not the substep peak. Short impulsive spikes (p99 per step was
  2.8× the end-of-step value) are partly invisible to it.
- **Heavier-than-trained eggs break the budget.** At 2× and 3× mass, success stays at 94% and 84%,
  but within-budget falls to 48% and 5.5% (median peak 12.7 N at 3×). The policy doesn't know
  the mass, which only the critic sees, and squeezes harder. The tuned gaits stay under budget
  at 3× mass (35.9% success), but they rotate the egg far less often.
- **The light egg (0.4×) is the main failure mode.** Success is 78%, within-budget 61%, and there
  are 3.1% drops. `media/demo_lightegg_mass0.4_failure.mp4` shows PPO flicking a light egg up
  onto the thumb and wrist side of the palm, where it ends up 38° off. It wasn't dropped by our
  z criterion, but it isn't in a secure grasp either.
- **The comparison isn't compute-matched.** PPO used 28.8M env steps, and baseline tuning used
  about 3.5M (stage 1 and the force-aware retune). That's about **8× less** for the baselines (6× counting the review's wide retune).
  The best candidate per CEM iteration was 0.44-0.61 over the last four iterations of both
  stages. With 64 episodes per candidate (SE ≈ 0.06), that's noisy with no clear upward trend,
  not a demonstrated plateau. *(Review: the original text said "had plateaued".)*
- **The gait was tuned only on the first range (≤ 0.4 rad).** It never saw the wider targets PPO's
  curriculum reached, so its 0-17% at 23-90° partly reflects that, not just the controller class.
  *(Review: a retune on 0.15 to π/2 moved it only to 24.2% at 23-46° and 1.6% at 46-90°. See Review.)*
- **The success metric is symmetric about the egg's long axis.** Spin about the axis isn't
  controlled. Under gym's full-quaternion distance, PPO's median final error is 11.5° on the main
  range, so most episodes would fail gym's own 0.1 rad criterion. That's by design here: the
  ellipsoid is symmetric, so only the axis matters physically. But it isn't the gym task metric.
  *(Review: measured success under gym's criterion is PPO 27.3%, tuned gait 25.4%, force-aware
  gait 13.3% and hold still 0.8% on the main range. The full table is in Review.)* Treating the
  axis as a line also means a target at δ can be reached by rotating π − |δ| the other way. Since
  |δ| ≤ π/2, the shortest rotation is still |δ|, so the task never gets easier. Near |δ| = π/2,
  though, both directions are about equally short.
- **Everything is one training seed**, so there's no variance estimate across training runs.

## Surprises
- **The gymnasium-robotics bug wasn't a joint-type bug.** It was numpy-int vs pybind-enum
  membership. The hinge joints themselves trip the assert.
- **Constant postures rotate the egg.** The CEM-tuned constant action (14.1%) beats doing nothing
  (3.9%). Some postures make the egg drift in one yaw direction, which is right for half the
  targets. In the error-over-time plot, its mean error grows steadily: it rotates every egg the
  same way.
- **The force-aware gait retune barely changed the gait's forces.** Its held-out peaks were already
  4-6 N, but success went up 43.0% → 50.4%. The original tuning objective's force penalty
  apparently wasn't binding for the gait at all.
- **PPO solves the rotation in about 0.2 s and then holds it.** *(Review correction: the original
  said 0.5 s and 0.4 s. Per-episode median settle time is 5 control steps = 0.20 s on the main
  range and 0.60 s at 46-90°, from `results/eval_review.json`.)* The mean error is below the
  threshold from the 5th control step and stays there. 96% of all steps are "in success". The gait's mean error is still 8° (force-aware) to 11°
  (tuned) at 4 s and oscillates at the gait frequency. Individual gait episodes succeed (43-50%),
  but the mean curve never crosses the threshold (`media/eval_error_force.png`). The PPO hand looks odd: it splays the
  fingers and steers the egg mainly with the thumb and palm (see the demos).
- **PPO is the only method that ever drops the egg.** Hold still never does. Learning to move the
  egg is also what can throw it out of the hand when the mass or friction is unfamiliar.

## Demo (media/)
All clips are reloaded from `checkpoints/run3/latest.npz` by `render_demo.py` and rendered
offscreen with MuJoCo. They show three panels on the same seed: hold still | tuned scripted gait |
PPO. The ghost egg on the right is the target. Red zones are touch sensors registering contact.
The text shows the target rotation, axis error and contact force against the budget.
- `demo_success_0.mp4` / `.gif`, `demo_success_1.mp4`: ±22° targets. PPO reaches them in about
  0.5 s. The gait wobbles toward them, and hold still doesn't move.
- `demo_wide_success_0.mp4`, `demo_wide_success_1.mp4`: ±86° targets (curriculum range). PPO
  ends within 1-2°, the gait gets partway, and hold still stays at 84° off.
- `demo_failure.mp4`: PPO's worst episode out of the 24 demo episodes (seed 20000, nominal range). The egg ends 7.2° off, a near miss with the
  egg resting against the thumb.
- `demo_lightegg_mass0.4_failure.mp4`: an out-of-distribution failure (0.4× mass). PPO flicks the
  light egg up onto the thumb and wrist side (38° off). The gait and hold still leave it in the
  palm.
- `*_last.png`: the last frame of each clip.

## Checkpoint/resume check (before any run > 20 min)
`logs/resume_test_smoke.jsonl`: a run was killed (own PID) at iter 19 after saving a checkpoint at
iter 15. The resume continued from iter 16 with the same env-step count. Policy std continued
(0.3652 → 0.3648) and the episode return stayed in range (1.9 before, 1.4-2.3 after). The three
real runs were chained the same way. In-flight episodes aren't restored: envs restart fresh on
resume.

## Not measured
- Real eggshell fracture. The 10 N budget is a toy fragility threshold fixed before training,
  not validated fracture mechanics.
- Touch vs no-touch actor ablation: there wasn't time to train a second policy with the touch
  dims zeroed. **So this result does not show that touch helps.** Touch is in the observation, but
  its contribution is not measured.
- Vision-assisted vs touch-only policies, and a force-threshold failure rule that ends the
  episode (the "Go deeper" items).
- Full SO(3) targets, i.e. gym's Rotate task with random initial orientation. Only z-rotations of
  a lying egg were tested.
- Learning-curve variance across training seeds (one training seed).
- Real-time performance of the policy on hardware: everything is simulation.

## Next steps
1. Train the force term on the per-substep peak, or on the touch-sensor sum at every substep,
   instead of the end-of-step force. Or add the "Go deeper" failure rule: the episode ends when
   force exceeds 10 N.
2. Widen the mass and friction randomisation, e.g. mass 0.4-3×. Give the actor a short history
   (or a recurrent core) so it can infer mass from touch and joint effort.
3. Touch ablation: retrain with the 92 touch dims zeroed, with the same budget, to measure what
   tactile sensing contributes. Then try vision-assisted vs touch-only.
4. Move toward gym's full task: random initial orientation, then full-quaternion targets.
5. A compute-matched baseline: give the gait CEM the same 28.8M steps, and tune it on the same
   curriculum ranges.

## Speed (under load)
- Env alone: 5,760 env steps/s (64 envs, 4 threads, zero policy, load avg ~5). With the
  per-substep force measurement used in evaluation it was 2,190 steps/s.
- Physics only (128 envs, 4 threads) at load avg 27: 3,280 steps/s.
- PPO training throughput, median (5th-95th percentile) per iteration: run1 2,800 (2,300-4,350),
  run2 4,230 (3,200-5,120), run3 5,280 (3,220-6,080) env steps/s. Load avg was 20-29 during run1.
  Later runs got faster as other builders' jobs finished.

## Review (cross-reviewer, 2026-10-24): stability, gym metric, wide-range gait
The reviewer re-ran the four target-range cells with the same seeds (10001-10004) and the same n,
keeping per-episode error trajectories (`review_metrics.py` → `results/eval_review.json`, 269 s
under `gpu_run`). The plain success column reproduces `results/eval.json` exactly for every
method and cell, so the new columns are on the same episodes.
- **held for the last 0.5 s**: long-axis error < 0.1 rad at every one of the last 13 control
  steps, and not dropped. This rules out passing through the target at exactly t = 4 s.
- **gym full-quat success**: gymnasium-robotics' criterion, full-quaternion distance < 0.1 rad
  (spin about the long axis counts), and not dropped.
- **scripted gait (wide retune, review)**: the force-aware gait, re-tuned with the same CEM
  (8 × 24 × 64 episodes, objective P(success ∧ end-of-step force < 7 N) − P(drop), warm start)
  on targets |δ| ∈ [0.15, π/2], which is PPO's final curriculum range, on training seeds 999-2001.
  It used about 1.26M env steps and 224 s (`checkpoints/baselines_wide.json`,
  `logs/baselines_wide.log`). This CEM was noisy: the warm start scored 0.23 on seed 999, the
  iteration bests were 0.13-0.30, and the kept candidate re-scored 0.16. It is a cheap retune,
  not an exhaustive one.

**θ∈[0.15, 0.4] rad (main, n=256)**

| method | success (long axis, at 4 s) | held for the last 0.5 s | gym full-quat success | full-quat err median | long-axis err median | within budget | settle time (median, episodes below thr through 4 s) |
|---|---|---|---|---|---|---|---|
| zero (hold still) | 3.9% | 2.3% | 0.8% | 17.2° | 15.9° | 3.9% | 3.14 s |
| posture (tuned hold) | 14.1% | 14.1% | 6.2% | 21.8° | 17.7° | 14.1% | 1.40 s |
| random (sigma=0.1) | 7.0% | 4.3% | 3.5% | 18.0° | 16.1° | 7.0% | 2.96 s |
| scripted gait (tuned) | 43.0% | 35.9% | 25.4% | 12.2° | 7.2° | 43.0% | 2.16 s |
| PPO (asym. critic, touch) | 99.6% | 99.6% | 27.3% | 11.5° | 0.8° | 86.7% | 0.20 s |
| scripted gait (force-aware retune) | 50.4% | 44.1% | 13.3% | 13.0° | 5.7° | 50.4% | 2.36 s |
| PPO before force fine-tune | 98.8% | 98.0% | 18.0% | 16.2° | 1.3° | 45.7% | 0.24 s |
| scripted gait (wide retune, review) | 51.6% | 45.3% | 18.4% | 12.7° | 5.7° | 51.6% | 1.88 s |

**θ∈[0, 0.4] rad (n=128)**

| method | success (long axis, at 4 s) | held for the last 0.5 s | gym full-quat success | full-quat err median | long-axis err median | within budget | settle time (median, episodes below thr through 4 s) |
|---|---|---|---|---|---|---|---|
| zero (hold still) | 22.7% | 22.7% | 18.0% | 13.4° | 11.0° | 22.7% | 0.04 s |
| posture (tuned hold) | 10.9% | 10.9% | 7.0% | 20.0° | 14.7° | 10.9% | 1.12 s |
| random (sigma=0.1) | 20.3% | 18.8% | 10.2% | 14.6° | 12.1° | 20.3% | 0.04 s |
| scripted gait (tuned) | 55.5% | 45.3% | 33.6% | 9.7° | 5.3° | 55.5% | 1.56 s |
| PPO (asym. critic, touch) | 100.0% | 100.0% | 30.5% | 11.9° | 0.8° | 80.5% | 0.16 s |
| scripted gait (force-aware retune) | 53.9% | 49.2% | 15.6% | 11.5° | 5.5° | 53.9% | 1.88 s |
| PPO before force fine-tune | 98.4% | 98.4% | 21.9% | 16.5° | 1.6° | 57.0% | 0.16 s |
| scripted gait (wide retune, review) | 57.0% | 43.8% | 20.3% | 11.4° | 5.3° | 57.0% | 1.72 s |

**θ∈[0.4, 0.8] rad (n=128)**

| method | success (long axis, at 4 s) | held for the last 0.5 s | gym full-quat success | full-quat err median | long-axis err median | within budget | settle time (median, episodes below thr through 4 s) |
|---|---|---|---|---|---|---|---|
| zero (hold still) | 0.0% | 0.0% | 0.0% | 32.6° | 32.0° | 0.0% | never |
| posture (tuned hold) | 0.0% | 0.0% | 0.0% | 35.7° | 32.5° | 0.0% | never |
| random (sigma=0.1) | 0.0% | 0.0% | 0.0% | 33.7° | 32.8° | 0.0% | never |
| scripted gait (tuned) | 14.1% | 7.0% | 1.6% | 24.1° | 20.6° | 14.1% | 3.58 s |
| PPO (asym. critic, touch) | 100.0% | 100.0% | 29.7% | 11.4° | 0.8° | 69.5% | 0.32 s |
| scripted gait (force-aware retune) | 17.2% | 9.4% | 0.8% | 21.7° | 17.8° | 17.2% | 3.46 s |
| PPO before force fine-tune | 97.7% | 97.7% | 21.1% | 15.3° | 1.4° | 9.4% | 0.40 s |
| scripted gait (wide retune, review) | 24.2% | 14.1% | 0.8% | 21.0° | 12.3° | 24.2% | 3.40 s |

**θ∈[0.8, π/2] rad (n=128)**

| method | success (long axis, at 4 s) | held for the last 0.5 s | gym full-quat success | full-quat err median | long-axis err median | within budget | settle time (median, episodes below thr through 4 s) |
|---|---|---|---|---|---|---|---|
| zero (hold still) | 0.0% | 0.0% | 0.0% | 68.7° | 67.0° | 0.0% | never |
| posture (tuned hold) | 0.0% | 0.0% | 0.0% | 68.8° | 67.3° | 0.0% | never |
| random (sigma=0.1) | 0.0% | 0.0% | 0.0% | 68.3° | 67.6° | 0.0% | never |
| scripted gait (tuned) | 0.0% | 0.0% | 0.0% | 57.8° | 56.2° | 0.0% | never |
| PPO (asym. critic, touch) | 98.4% | 97.7% | 21.1% | 16.9° | 0.8° | 36.7% | 0.60 s |
| scripted gait (force-aware retune) | 0.0% | 0.0% | 0.0% | 53.0° | 51.6° | 0.0% | never |
| PPO before force fine-tune | 96.9% | 95.3% | 10.9% | 21.0° | 1.2° | 0.0% | 0.70 s |
| scripted gait (wide retune, review) | 1.6% | 0.0% | 0.0% | 46.9° | 45.7° | 1.6% | 3.96 s |

**Wide-retune gait, robustness grid** (θ∈[0.15,0.4], n=128; compare the Robustness table above)

| friction, mass | success / within budget / drop |
|---|---|
| 0.3, 1.0 | 25.8% / 25.8% / 0.0% |
| 0.5, 1.0 | 40.6% / 40.6% / 0.0% |
| 1.0, 1.0 | 59.4% / 59.4% / 0.0% |
| 2.0, 1.0 | 65.6% / 65.6% / 0.0% |
| 1.0, 0.4 | 32.8% / 32.8% / 0.0% |
| 1.0, 2.0 | 46.9% / 46.9% / 0.0% |
| 1.0, 3.0 | 36.7% / 36.7% / 0.0% |

What this changes:
- PPO's long-axis success is **not a transient**: 99.6% → 99.6% under the held-for-0.5 s rule.
  Its median error over the last 0.5 s is 0.9°. The gaits lose 6-7 points (43.0 → 35.9%,
  50.4 → 44.1%), because they oscillate around the target.
- **Under gym's full-quaternion criterion, PPO and the tuned gait are about level** (27.3% vs 25.4%
  on the main range). PPO's lead there comes only from the wider ranges (21-30% vs ≤ 1.6%). The
  headline holds only for the long-axis task as defined here.
- **The gait's failure on wide targets is mostly the controller, not the tuning range.** Tuned on
  the wide range, it reaches 24.2% at 23-46° (from 17.2%) and 1.6% at 46-90° (from 0%), and its
  main-range score holds (51.6%). It rotates the egg too slowly: the median settle time is about
  2 s even for the narrow targets.
- Compute stays unmatched: PPO used 28.8M env steps, and all baseline tuning together, this
  retune included, used about 4.8M.
