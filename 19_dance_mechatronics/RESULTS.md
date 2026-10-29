# RESULTS: 19 Dance Dance Mechatronics

**First satisfying version: done.** A planar Walker2d dances a 4-beat kick-dip groove at 90 BPM and
tracks it without falling. The nominally trained PPO policy had **0 falls in 160 clip loops** (16 held-out
seeds × 10 loops, 427 s of dancing), 0.45° joint RMSE and 0.26 cm mean foot-position error. Kick peaks and
dip bottoms landed 5 ms from the beat on average (mean |Δt|), which is the 50 Hz measurement floor. The clip was first made
physically plausible with an inverse-dynamics check (§1). Tuned classical controllers also track it:
the PD + feed-forward + balance baseline had 1 fall in 160 loops; its foot-relative variant had 0 falls
and slightly lower joint error.

> **Review update (cross-reviewer).** The baselines below all lacked an obvious, cheap piece: something
> that pulls the body back to its spot without the absolute-COM runaway. The review added
> `baseline_anchor` (foot-relative balance + a bounded COM-target shift and swing-foot placement on the
> root-x error, 3 extra gains, same tuning protocol and seeds; §2). It has 0 falls on every non-push
> condition, joint RMSE of 0.40° (0.405 vs 0.398 for `baseline_rel`, the best), and 0.56 cm foot error /
> 0.77 cm root drift. PPO's "holds the dance in place" edge survives but shrinks from 5.6× to about 2×
> nominal (2–6× off-nominal). The push edge survives: at 40 N the anchored baselines fall 0.04–0.05 per loop
> (PPO 0); at 50–60 N 0.18–0.39 (push-trained PPO 0–0.11). Numbers in the original text below are unchanged;
> review numbers are added in brackets and as two new table columns.

Against the hand-tuned baselines, PPO's real edges are:
1. **It holds the dance in place.** Foot error is 0.26 cm vs 1.45–1.71 cm, and root drift 0.27 cm vs
   1.6–1.8 cm, and it stays that way across every held-out tempo, friction and mass condition.
   [Review: vs the anchored baseline, 0.26 vs 0.56 cm foot and 0.27 vs 0.77 cm root drift nominal;
   0.33–0.42 vs 0.73–2.52 cm foot error off-nominal. Smaller edge, still PPO's.]
2. **It survives small pushes.** At 40 N it had 0 falls vs 0.08–0.13 per loop for the baselines, held-out for
   the nominal policy. At 50–60 N the push-trained policy had 0–0.11 falls per loop vs 0.26–0.49.
   [Review: anchored baselines 0.050 / 0.044 at 40 N; 0.18–0.39 at 50–60 N.]

Its weak spot is contact timing: its liftoff timing error is about 30 ms worse than the baseline's (56 vs 25 ms)
[review: 32 ms for the anchored baseline].
**Pushes of 70 N or more topple every controller.** That includes a third run trained on pushes up to
100 N with 2.5× more residual authority, which only moved the 70 N fall rate from 0.53 to 0.41 per loop.

All numbers come from `results/*.json` produced by the commands in "Provenance". The main tables are
`results/eval_final.json`: 7 controllers × 12 conditions × 16 seeds × 10 loops. Speed numbers were
taken under load (other builders training; `uptime` load average 15–25).

## 1. The clip, and showing it is feasible first

**Body.** gymnasium `Walker2d-v5` (planar, 23.7 kg, 6 hinge joints, ±100 Nm). It is used unchanged
except that the gear-100 motors become PD servos with the same ±100 Nm clamp (`walker2d_dance.xml`).
The body is **planar**, so the weight shift is fore-aft over the stance foot, not lateral. Its hips can only
extend (thigh range −150..0°), so the vocabulary is heel kicks and knee dips, not knee lifts.

**Clip** (`clip.py`, 90 BPM, 4 beats = 2.667 s loop). Right heel-kick peaks on beat 1, a dip bottoms on beat 2,
left heel-kick on beat 3, dip on beat 4. Kick: knee to −1.2 rad (69°) plus a 0.15 rad hip extension,
sole 37 cm off the floor at the peak. Dip: knees 0.6 rad deeper, pelvis drops 6.3 cm. Only these joint
channels are keyframed. Per frame, the stance sole is pinned flat at a fixed spot (no slip in the
reference), and the whole-body lean is **solved so that the inverse-dynamics centre of pressure sits at
mid-foot**. The solver starts quasi-static and then runs 10 cart-table correction passes, solved exactly
in the Fourier domain on the periodic signal.

**Feasibility check** (`python feasibility.py` → `results/feasibility_sweep.json`, `media/feasibility.png`).
The check runs `mj_inverse` on the finite-differenced reference with contacts off. From that it gets the
ground reaction the floating root needs, its CoP, the friction ratio, and the joint torques after
subtracting J_cᵀf of the loaded foot.

| candidate clip | feasible | CoP rel. ankle (m), sole = 0..0.2 | min normal force | max \|Fx/Fz\| | max joint torque |
|---|---|---|---|---|---|
| first draft: 100 BPM, 1-beat kicks, knee −1.45 | **no** (lean solve diverged) | −0.70 .. 0.67 | 105 N | 1.86 | 125 Nm |
| 100 BPM, 1-beat kicks, knee −1.2 | no | −0.08 .. 1.12 | 93 N | 1.12 | 138 Nm |
| 80 BPM, 1-beat kicks, knee −1.45 | no (torque) | 0.08 .. 0.17 | 100 N | 0.59 | 124 Nm |
| 80 BPM, 1-beat kicks, knee −1.2 | yes | 0.100 | 140 N | 0.40 | 90 Nm |
| 60 BPM, 1-beat kicks, knee −1.45 | yes | 0.100 | 157 N | 0.27 | 73 Nm |
| **shipped: 90 BPM, kicks spread over 1.4 beats, dip −0.6** | **yes** | 0.099 .. 0.100 | 149 N | 0.34 | **59 Nm** |
| shipped clip re-solved at 110 BPM | yes | 0.098 .. 0.101 | 143 N | 0.38 | 85 Nm |

How the clip was adjusted. The first draft (100 BPM, a full 1-beat heel kick) was not physically
plausible for this body. The heavy shank and foot (6 kg) swinging in 0.3 s create angular-momentum
changes that push the CoP out of a 20 cm foot, and the hip needs more than 100 Nm. Slowing the tempo
helps. Spreading each kick over 1.4 beats, so it overlaps the neighbouring dip, helps most: it cut the
peak torque from about 120 to 59 Nm at 90 BPM. In the shipped clip the COM swings 5–13 cm over the foot while the
CoP stays at 10 cm, so the dynamics matter (the plot shows both). Caveat: the joint torques assume the load
moves to the stance foot instantly, which causes the steps in the torque plot.

**Empirical feasibility.** All seven controllers in §3, including plain PD, track the shipped clip for
160 held-out loops with 0–1 falls. The clip is trackable, not just consistent on paper.

*Review note.* The "CoP at mid-foot" column is satisfied by construction (the lean is solved to put it
there), so the independent checks are the solve converging, normal force > 0, friction ratio and joint
torque. The empirical evidence carries the claim: plain PD tracks it for 160 held-out loops with no fall.
The sweep JSON also has a feasible milder 100 BPM row (1.5-beat kicks, knee −1.0, dip −0.2, 51 Nm) that the
table omits. The foot friction μ = 1.9 used in the check is the stock Walker2d-v5 value (both feet).

## 2. Controllers (matched interface, matched information)
All run at 50 Hz on a 500 Hz RK4 sim. Each one outputs joint targets plus a feed-forward torque into the same
±100 Nm PD servos. The reward is the DeepMimic tracking reward: 0.5 pose, 0.05 velocity, 0.25 foot
position, 0.2 root position/pitch, each `exp(-k·err²)`. Early termination on a fall: torso z < 0.8 m,
|pitch| > 1 rad, or any non-foot geom on the floor.

* **PD-only.** The baseline's tuned PD gains on the reference, with no feed-forward and no balance term.
  This is exactly the RL policy's zero action.
* **Baseline.** PD + reference inverse-dynamics feed-forward + balance (stance-ankle torque on COM
  position and velocity error, stance-hip torque on torso pitch error) + a target lead. 12 parameters:
  kp and damping ratio per hip/knee/ankle, ff scale, 4 balance gains, lead. **Tuned on the RL reward
  with early termination**, training seeds 100–107 only: 3000 log-uniform random configs, then 40
  generations of a (8, 24) evolution strategy, then the top 12 re-scored on 8 seeds × 5 loops. That is 3980
  evaluations, about 2.6 min on 4 workers per tuning run. The first pass hit the box edges (kp_a at the
  1000 ceiling, kp_h near its 2000 ceiling, damping ratios near the floor), so the ranges were widened (kp
  up to 6000/4000, damping ratio down to 0.003) and the tuning re-run. The score did not change
  (0.9885 → 0.9883), and the result is interior except k_com (−1783 of [−2000, 2000]). Final gains:
  kp hip/knee/ankle 3164/1342/847 Nm/rad, damping ratio 0.010/0.028/0.034, ff 0.92
  (`results/baseline_tuned.json`). Stage-3 scores plateaued: 8 of 12 candidates within 0.004.
* **Baseline, foot-relative (`baseline_rel`).** Identical, but the COM error is measured relative to the
  stance sole(s), not the reference's absolute spot. Re-tuned with the same search, training score 0.9848
  (`results/baseline_tuned_rel.json`). **Disclosure:** I added this variant *after* the first evaluation
  showed the absolute-COM baseline toppling after long drifts (§6). That is a design change made after
  looking at held-out results, and it favours the baseline.
* **Baseline, push-tuned (`baseline_push`).** The original design tuned under the same random training pushes
  as PPO run 2 (U(0, 60) N per run, 8 training seeds; `results/baseline_tuned_push8.json`). A 4-seed
  version (`results/baseline_tuned_push.json`) is in `eval_main.json` only.
* **Baseline, anchored (`baseline_anchor`, `baseline_anchor_push`; added in review, `baseline_anchor.py`).**
  `baseline_rel` plus two bounded uses of the root-x error (which PPO also observes): the foot-relative COM
  target is shifted by clip(k_anchor·Δx, ±delta), and the swing hip target gets k_sw·clip(Δx, ±0.1 m). No
  integrator, so no runaway. 15 parameters, tuned by `tune_anchor.py`: the same 3000 random + 40 ES
  generations + stage-3 protocol and training seeds, plus the tuned `baseline_rel` (anchors at 0) as one warm
  start. Training score 0.9931 (vs 0.9883 original, 0.9848 rel), 88 s on 4 workers
  (`results/baseline_tuned_anchor.json`). Tuned k_anchor 0.17, delta 1.2 cm, k_sw −0.93 rad/m.
  The push-tuned version (`--push 60 --train_seeds 8`, warm-started from the nominal anchor) scored 0.9605 vs
  0.910 for `baseline_push` (`results/baseline_tuned_anchor_push8.json`); its k_com sat at the −2000 edge, and a
  re-run with |k_com| ≤ 5000 returned the same solution (`results/baseline_tuned_anchor_push8_wide.json`).
* **PPO (MLX), residual on PD.** `target = q_ref(t+dt) + s · a`, with the policy mean squashed by tanh into
  [−1, 1], s = 0.1 rad, and the same PD gains as the baseline. Observation (45-d): phase sin/cos (2 harmonics),
  joint q/qd, root height/pitch/x errors and velocities, next reference targets and velocities,
  tracking error, actual and reference foot contacts. Reference-state initialisation (random start phase plus small
  noise), early termination, 6 s episodes, γ = 0.97, λ = 0.95, 2×256 tanh actor and critic, clip 0.2,
  target-KL early stop. The update runs in MLX on the CPU device (fp32, matching the numpy rollouts, so
  no TF32 ratio distortion). Approx-KL (final 0.01–0.04) and clip fraction (0.10–0.24) are logged in the
  checkpoints' `history`.
  * `ppo_pd` (run 1): nominal training only. 2033 updates, 8.3 M env steps, 19 min of training.
    **Checkpoint/resume verified:** SIGKILL at update 90 (reward 0.978), resumed from the checkpoint
    (0.977 on the next updates), then continued.
  * `ppo_push` (run 2): continued from `ppo_pd` with training pushes of U(0, 60) N per episode (0.1 s,
    random sign, about 1.5 per loop). 40 min, +23 M steps.
  * `ppo_push100` (run 3): continued from `ppo_push`. The residual range went from 0.1 to 0.25 rad (the output
    layer and std were rescaled by 0.4, so the starting policy is about the same), with training pushes U(0, 100) N.
    40 min, +20 M steps, 51.4 M in total. A first attempt at lr 3e-4 with a fresh Adam collapsed
    (approx-KL 1.47 on the first update, reward 0.97 → 0.64). It was relaunched at lr 3e-5, target-KL 0.02.

Compute: baseline tuning about 2.5 min per tuning run (4 runs) [review: +3 runs, 1.5–2.5 min each] versus PPO 19 + 40 + 40 min. The baseline got far less
wall-clock but a full search on the exact objective, and its score plateaued. PPO ran at about 9.4 k env
steps/s overall (13 k/s rollout only) on 4 workers, under load. [Review: from the checkpoints' env-step counts
and wall times, 7.7 k/s (run 1), 9.6 k/s (run 2), 8.4 k/s (run 3), 8.7 k/s over all three.]

## 3. Results on held-out seeds (16 seeds × 10 loops per cell; `results/eval_final.json`)

*Review columns* (`baseline_anchor`, `baseline_anchor_push`, added by the cross-reviewer, see §2) come from
`results/eval_review_anchor.json` and `results/eval_review_anchor_push.json`: same seeds 1000–1015, same loops, same conditions, same `eval.py`.

### Nominal clip (all Measure items)
| | PD-only | baseline | baseline_rel | baseline_push | **PPO `ppo_pd`** | `ppo_push` | `ppo_push100` | baseline_anchor *(review)* | baseline_anchor_push *(review)* |
|---|---|---|---|---|---|---|---|---|---|
| **balance:** falls / loop (falls in 160 loops) | 0 (0) | 0.006 (1) | 0 (0) | 0 (0) | **0 (0)** | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| **tracking:** joint RMSE (deg) | 1.64 | 0.71 | **0.40** | 0.50 | 0.45 | 0.72 | 1.28 | 0.40 | 0.56 |
| foot position error (cm, mean) | 14.8 | 1.45 | 1.71 | 4.06 | **0.26** | 0.39 | 1.16 | 0.56 | 0.99 |
| root x drift error (cm, RMS) | 19.0 | 1.64 | 1.83 | 4.14 | **0.27** | 0.49 | 1.17 | 0.77 | 0.95 |
| root height error (cm, RMS) | 1.13 | 0.64 | 0.22 | 0.37 | 0.26 | 0.33 | **0.20** | 0.25 | 0.22 |
| torso pitch error (deg) | 2.53 | 1.29 | 0.37 | 0.71 | **0.23** | 0.51 | 1.15 | 0.57 | 0.51 |
| mean tracking reward | 0.734 | 0.988 | 0.986 | 0.950 | **0.998** | 0.997 | 0.991 | 0.994 | 0.992 |
| **contact:** frame agreement with reference | **0.985** | 0.982 | 0.972 | 0.977 | 0.969 | 0.971 | 0.961 | 0.979 | 0.971 |
| foot slip (cm per loop) | 3.95 | 2.93 | 2.81 | **2.55** | 2.64 | 2.97 | 4.26 | 2.43 | 2.15 |
| touchdown timing error (ms; 320 events, 0 misses) | **10.9** | 22.6 | 32.8 | 40.9 | 26.2 | 14.6 | 29.1 | 23.9 | 32.4 |
| liftoff timing error (ms) | 27.6 | 25.4 | 40.5 | **20.6** | 56.4 | 51.9 | 73.9 | 31.7 | 43.5 |
| **beat:** kick peak vs beat (mean \|Δt\|, ms) | 6.5 | 8.8 | 5.2 | 15.7 | **5.2** | 7.5 | 6.0 | 5.5 | 7.5 |
| dip bottom vs beat (mean \|Δt\|, ms) | 20.7 | 6.5 | 5.0 | 5.0 | **4.9** | 5.7 | 5.5 | 5.1 | 5.2 |
| beats hit within 50 ms | 100% | 100% / 99.7% | 100% | 100% | 100% | 100% | 100% | 99.7% / 100% | 99.7% / 100% |

The beat-alignment floor is about 5 ms: the reference scored against itself gives 4.8 ms for both peaks,
because the 50 Hz control grid does not divide the 0.667 s beat. PPO and baseline_rel are at the floor.
The original baseline's kicks land about 9 ms early. PD-only's dips are about 20 ms late, because its knees sag with
no feed-forward.

### Robustness: falls per loop
Conditions not seen in training or tuning unless marked *(train)*. Pushes are 0.1 s horizontal forces on
the torso with random sign, spaced U(0.3, 1.0) loops apart (about 1.5 pushes per loop; the timer is redrawn
after a recovery reset). The same protocol was used for tuning, training and evaluation. So 0.7–0.9 falls per loop
at 70–120 N means roughly half to two-thirds of the pushes topple the body, not every push.

| condition | PD-only | baseline | baseline_rel | baseline_push | `ppo_pd` | `ppo_push` | `ppo_push100` | baseline_anchor *(review)* | baseline_anchor_push *(review)* |
|---|---|---|---|---|---|---|---|---|---|
| tempo ×0.9 (81 BPM) | 0 | **0.131** (16/16 seeds fell) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tempo ×1.1 (99 BPM) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| friction ×0.3 (feet μ 0.57) | 0 | **0.119** (16/16 seeds fell) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| all masses ×1.3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| push 40 N | 0.281 | 0.081 | 0.131 | 0.125 *(train)* | **0** | **0** *(train)* | **0** *(train)* | 0.050 | 0.044 *(train)* |
| push 50 N | 0.388 | 0.263 | 0.269 | 0.263 *(train)* | 0.125 | **0** *(train)* | **0** *(train)* | 0.269 | 0.181 *(train)* |
| push 60 N | 0.537 | 0.450 | 0.469 | 0.494 *(train)* | 0.325 | 0.106 *(train)* | **0.069** *(train)* | 0.388 | 0.362 *(train)* |
| push 70 N | 0.612 | 0.562 | 0.681 | 0.637 | 0.619 | 0.525 | **0.406** *(train)* | 0.606 | 0.550 |
| push 80 N | 0.706 | 0.712 | 0.787 | 0.806 | 0.775 | 0.781 | **0.694** *(train)* | 0.794 | 0.731 |
| push 100 N | 0.950 | 0.869 | 0.888 | 0.906 | 0.919 | 0.900 | 0.869 *(train)* | 0.900 | 0.881 |
| push 120 N | 0.987 | **0.887** | 0.969 | 0.925 | 0.975 | 0.956 | 0.925 | 0.969 | 0.956 |

Tracking under the held-out tempo, friction and mass conditions: PPO `ppo_pd` keeps foot error at
0.32–0.42 cm. The baselines drift to 1.7–7.9 cm (baseline_rel worst at 7.6–7.9 cm under tempo ×0.9 and friction ×0.3,
since nothing anchors its absolute position), while their joint RMSE stays at 0.4–2 deg.
[Review: `baseline_anchor` 0.73–2.52 cm (worst on friction ×0.3), joint RMSE 0.45–0.71°; `baseline_anchor_push`
1.14–7.89 cm.]

Root-x "drift" under pushes (not tabled; in the JSONs) is dominated by the falls themselves (the root
topples tens of cm before the reset), so it is not a drift measure there.

## 4. What worked
* **Feasibility first.** The inverse-dynamics CoP check caught that the obvious first clip was
  infeasible, and it fixed the clip by construction (lean solved against the CoP) rather than by
  trial and error. Every controller then tracked the shipped clip with at most 1 fall in 160 loops.
* **Residual-on-PD PPO learned fast.** Its stochastic training reward passed the tuned baseline's 0.988
  after 0.76 M steps (1.3 min of training; `media/training_curves.png`). It ended with 5.6× lower foot
  error and 6× lower root drift than the tuned baseline, and it hits the beat at the measurement floor.
  [Review: 2.2× / 2.8× lower than the anchored baseline, which also hits the beat floor (5.5 / 5.1 ms).]
* **Off-nominal tracking.** With no randomisation in training, `ppo_pd` never fell at tempo ×0.9 / ×1.1,
  friction ×0.3, mass ×1.3 or 40 N pushes, and kept sub-cm foot error throughout.
* **Push training works inside its range.** At 50–60 N it cut falls from 0.13–0.33 to 0–0.11 per loop, a range
  where every baseline falls at 0.26–0.49.

## 5. What didn't
* **No controller survives pushes of 70 N or more** (7 N·s, Δv ≈ 0.3 m/s). Training on pushes up to 100 N
  with ±0.25 rad of residual (run 3) moved the needle only a little: 70 N went 0.53 → 0.41, 80 N went 0.78 → 0.69,
  and 100 N stayed at 0.87–0.92. It also cost nominal precision (joint RMSE 0.45 → 1.28°, foot 0.26 → 1.16 cm,
  liftoff timing 56 → 74 ms). A rough capture-point estimate (Δv·√(h/g) ≈ 0.07–0.09 m against about 0.1 m of
  sole in front of the COM) puts 70–80 N at the edge of what an in-place ankle/hip strategy can absorb.
  Beyond that you need a step, and a residual around an in-place dance cannot learn to step.
  Clips: `media/demo_push80_failure.mp4` (both bodies topple over the stance foot, forward in this clip; the
  push-trained PPO swings its free leg forward as if trying to step, but the residual bound stops it) and
  `media/demo_push70_seed1000.mp4` (all three fall, at 70 N, including `ppo_push100`).
* **Contact timing is PPO's weakest Measure item.** Its liftoff timing error (52–74 ms) is about twice the
  baselines' (21–41 ms; review: anchored 32–44 ms), and frame-level contact agreement is lower (0.961–0.971 vs 0.972–0.985). The diagnosis
  on one seed: in 2.5–3.4% of frames the reference counts a foot as down (sole < 1 cm) while PPO already has it
  2–3 mm in the air, against 1.8–1.9% for the baseline (one-seed probe; not saved to a file). It lifts early and lands late by millimetres. The reward
  has no contact term, so this is free for it. It is a small reward gap, not a shuffle exploit: slip per
  loop is the same as the baselines' (2.6 vs 2.5–2.9 cm), and no weird strategy appeared.
* **Push-tuning the baseline did not make it push-robust** (0.125 vs 0.081 falls/loop at 40 N, about equal
  at 50–60 N). Random pushes make the tuning objective very noisy (8-seed tuning, stage-3 scores 0.66–0.91 vs 0.94–0.99 without pushes), and the search
  traded nominal precision (foot 1.45 → 4.06 cm) for a stiffer ankle (kp_a 2780 vs 847) and a much weaker COM term (k_com −327 vs −1783).
  [Review: push-tuning the anchored baseline helped a little (50 N 0.27 → 0.18, 60 N 0.39 → 0.36 falls/loop)
  and kept nominal foot error under 1 cm, but stays well behind push-trained PPO at 50–60 N.]

## 6. Surprises
* **The original baseline's failure mode is drift, not the dance.** At tempo ×0.9 and friction ×0.3 every
  seed fell, always 12–26 s in. Each time the root had first drifted about 0.85–1.06 m from the reference spot
  (forward at slow tempo, backward on low friction), and the body tipped at about 0.9 rad of pitch error. Its balance
  term acts on the *absolute* COM error with a gain of −1783 Nm/m. A slow slide that tuning never saw in its
  5–13 s episodes eventually becomes a huge ankle command and tips the body over. Measuring the COM error
  from the stance foot (`baseline_rel`) removes all of those falls but lets the body wander (7–8 cm foot
  error). PPO, trained on 6 s episodes, does both: it never falls there and holds its spot to under 1 cm over
  27 s, using the root-x-error input. [Review: a hand-written controller can do both too. `baseline_anchor`
  (bounded, integral-free use of the same root-x error) never falls there either and holds 0.8–2.5 cm.]
* **The tuner wants extreme stiffness** (kp up to 3000–4500 Nm/rad with damping ratio about 0.01) on a 500 Hz RK4
  sim. RK4 at 2 ms keeps that stable. It also means the servo saturates the ±100 Nm clamp at about 0.03 rad of error, so
  the RL residual had to be small (±0.1 rad). Probing ±0.4 rad gave a near bang-bang initial policy.
* **Slower is harder for the original baseline.** It fell at 81 BPM but not at 99 BPM. That is the drift effect
  again, since the run lasts longer in wall time and the feed-forward was computed at 90 BPM, not a dynamics limit.

## 7. Not measured
* Lateral (3-D) balance and a true side-to-side sway: the body is planar.
* Speed or real-time factor as a headline: only the busy-machine PPO throughput above, under load average 15–25.
* Re-solved tempo variants as test conditions: held-out tempos time-scale the 90 BPM reference. A re-solved
  110 BPM clip is feasible (sweep table) but was not used for tracking.
* A non-residual policy (absolute targets) and a residual on the full baseline (`--residual base` exists
  in `ppo.py` but was not trained).
* Run-to-run variance of PPO: one training seed per run.

## 8. Next steps
* Push recovery past 70 N needs stepping: a learned swing-foot placement or phase shift on top of the
  residual, with a push curriculum.
* Add a contact-timing term (or a foot-height term near touchdown and liftoff) to the reward.
* Give the baseline both anchors, foot-relative balance plus a slow absolute-position integrator, and tune
  it on longer episodes. [Review: done without the integrator or longer episodes: `baseline_anchor`.]
* Go deeper per the design: blend clips, vary tempo by re-solving the clip, and move to a 3-D body
  (`robotis_op3`) for a lateral sway.

## Demos (`media/`)
* `demo_nominal.mp4/.gif`, `demo_nominal_beats.png`: tuned baseline vs `ppo_pd`, 2 loops, nominal, seed 1000.
  Solid body over a translucent blue ghost of the reference, with a red beat light.
* `demo_push60_seed1005.mp4`: baseline (2 falls) vs `ppo_pd` (1) vs `ppo_push` (0), 3 loops at 60 N.
  **Seed selected** (from 8 checked) because it shows the aggregate ordering. `demo_push60_seed1002_allfall.mp4`
  is an unselected seed where all three fall once.
* `demo_push70_seed1000.mp4`, `demo_push80_failure.mp4`: the shared push cliff (each with a `_beats.png` contact sheet). Only the nominal demo is also a GIF (downscaled, every 3rd frame) to keep `media/` small.
* `feasibility.png`, `training_curves.png`, `robustness.png` (the plots predate the review and do not show the anchored baselines).
* `review_anchor_nominal.mp4`, `review_anchor_nominal_beats.png` (review): anchored baseline vs `ppo_pd`, 2 loops, nominal, seed 1000, both 0 falls.
* Reproduce: `python render_demo.py --ckpt checkpoints/ppo_pd.pkl --cond nominal --loops 2 --out media/demo_nominal`.

## Provenance
| result | command | output |
|---|---|---|
| feasibility sweep | `python feasibility.py` | `results/feasibility_sweep.json`, `media/feasibility.png` |
| baseline tuning | `tune_baseline.py --n_random 3000 --gens 40` | `results/baseline_tuned.json` (narrow first pass: `results/baseline_tuned_v1_narrow.json`) |
| foot-relative baseline | `... --com_rel` | `results/baseline_tuned_rel.json` |
| push-tuned baselines | `... --push 60 --train_seeds 8` / `--push 60` | `results/baseline_tuned_push8.json`, `results/baseline_tuned_push.json` |
| PPO run 1 | `ppo.py --residual pd --minutes 40 --act_scale 0.1 --init_logstd -1.6 --out checkpoints/ppo_pd` (SIGKILL at update 90) + same with `--resume --minutes 18` | `checkpoints/ppo_pd.pkl`, `logs/ppo_pd_run1*.log` |
| PPO run 2 | `ppo.py --residual pd --minutes 40 --act_scale 0.1 --push 60 --init_from checkpoints/ppo_pd.pkl --out checkpoints/ppo_push` | `checkpoints/ppo_push.pkl`, `logs/ppo_push_run2.log` |
| PPO run 3 | `ppo.py --residual pd --minutes 40 --act_scale 0.25 --push 100 --lr 3e-5 --target_kl 0.02 --init_from checkpoints/ppo_push.pkl --out checkpoints/ppo_push100` | `checkpoints/ppo_push100.pkl`, `logs/ppo_push100_run3.log` (collapsed attempt: `logs/ppo_push100_run3_failed_lr3e-4.log`) |
| final evaluation | `eval.py --seeds 16 --loops 10 --conditions nominal push40 push50 push60 push70 push80 push100 push120 tempo0.9 tempo1.1 friction0.3 mass1.3 --ckpt baseline_rel=results/baseline_tuned_rel.json baseline_push=results/baseline_tuned_push8.json ppo_pd=checkpoints/ppo_pd.pkl ppo_push=checkpoints/ppo_push.pkl ppo_push100=checkpoints/ppo_push100.pkl --out results/eval_final.json` | `results/eval_final.json`, `logs/eval_final.log` |
| earlier evaluations (same seeds, subset) | see `logs/eval_main.log` | `results/eval_main.json`, `results/eval_push_sweep.json` |
| review: anchored baseline | `tune_anchor.py --n_random 3000 --gens 40` | `results/baseline_tuned_anchor.json`, `logs/tune_anchor_review.log` |
| review: push-tuned anchored baseline | `tune_anchor.py --n_random 3000 --gens 40 --push 60 --train_seeds 8 --warm results/baseline_tuned_anchor.json --out results/baseline_tuned_anchor_push8.json` (edge check: same + `--k_com_max 5000 --warm results/baseline_tuned_anchor_push8.json`) | `results/baseline_tuned_anchor_push8.json`, `results/baseline_tuned_anchor_push8_wide.json`, `logs/tune_anchor_push8*_review.log` |
| review: anchored-baseline evaluation | `eval.py --seeds 16 --loops 10 --controllers baseline_anchor --ckpt baseline_anchor=results/baseline_tuned_anchor.json --conditions <the 12 above> --out results/eval_review_anchor.json` (and `baseline_anchor_push=results/baseline_tuned_anchor_push8.json` → `results/eval_review_anchor_push.json`) | `results/eval_review_anchor*.json`, `logs/eval_review_anchor*.log` |
| plots | `make_plots.py --eval results/eval_final.json --ckpts checkpoints/ppo_pd.pkl checkpoints/ppo_push.pkl checkpoints/ppo_push100.pkl --baseline_score 0.9883` | `media/training_curves.png`, `media/robustness.png` |
