# 21 Duck Hunt, round 6: can a learned flyer beat PD+ fairly?

R6 brief (2026-11-04): on a fresh held-out course set, beat an equally-informed, equally-budgeted
PD+ with a paired-course test at ≥ 99.8% completion under wind σ = 2 and σ = 4, or document a
clear negative. Every number below comes from a file named next to it. Nothing in `flight.py`,
`train_flight.py`, `pd_plus.py` or any existing result was changed. The R6 files are
`train_flight_r6.py`, `pd_plus_r6.py`, `eval_r6_flight.py`, `r6_*.sh` and `tests/test_r6.py`.

## Headline (fresh held-out set: 2000 new courses, wind σ = 2 and σ = 4, `results/r6_flight_eval.json`)

Each arm is listed against its own fair baseline. Δ = learned − baseline in seconds, paired over
courses (negative means the learned flyer is faster), with a 95% bootstrap CI. "Faster on" is the
fraction of courses where the learned flyer is strictly faster; ties are common because times are
multiples of 0.05 s.

| arm | fair baseline | wind | completion (3 seeds) | mean time: learned vs baseline | Δ seed-mean [95% CI] | faster on (per seed) | done bar |
|---|---|---|---|---|---|---|---|
| **Pure PPO** (no classical prior) | PD+ 1× budget | σ=2 | 100 / 100 / 100% | 6.393 / 6.400 / 6.369 vs 6.435 s | **−0.047 [−0.059, −0.035]** | 57 / 55 / 62% (slower on 35 / 37 / 30%) | met |
| | | σ=4 | 99.95% each (PD+ 99.95%) | 6.572 / 6.564 / 6.578 vs 6.498 s | **+0.073 [+0.058, +0.089]** | 42 / 44 / 47% | **not met** |
| **Learned residual on PD+** | PD+ 2× budget | σ=2 | 100 / 100 / 100% | 6.320 / 6.339 / 6.333 vs 6.444 s | **−0.113 [−0.124, −0.102]** | 74 / 68 / 72% (slower on 12 / 16 / 14%) | met |
| | | σ=4 | 100 / 99.95 / 99.95% (PD+ 99.95%) | 6.374 / 6.393 / 6.386 vs 6.501 s | **−0.117 [−0.128, −0.106]** | 74 / 68 / 73% | met |
| *context:* residual vs PD+ 1× | PD+ 1× | σ=2 / σ=4 | | vs 6.435 / 6.498 s | −0.104 [−0.110, −0.098] / −0.114 [−0.122, −0.106] | 81 / 74 / 79% at σ=2 | |
| *context:* original R0 PPO (30M steps) | PD+ 1× | σ=2 / σ=4 | 99.85% / 99.75% | 6.580 / 6.776 s | +0.146 [+0.133, +0.160] / +0.278 [+0.258, +0.300] | 26% at σ=2 | |

**Verdict.**
1. **Pure PPO, learning alone: a split result, so the done bar is not met.**
   - In the wind it was trained on (σ = 2), all three seeds beat the best PD+ on fresh courses,
     each with a CI that excludes zero. The margin is small (0.7%), and PPO is still slower on
     30–37% of courses.
   - In the stronger unseen wind (σ = 4), all three seeds are clearly slower than PD+ (+0.07 s).
   - Pure PPO is a better racer than PD+ in its training conditions and less robust outside them.
     R5 already saw this for the 30M-step PPO; it has not gone away.
2. **Learned residual on PD+, learning on top of a tuned controller: the done bar is met** at both
   wind levels against its fair baseline (PD+ with twice the tuning budget). The margin is
   0.11 s (1.8%) and it is faster on about 70% of courses for every seed. Extra PD+ tuning does
   not close the gap. The 2× PD+ gives no gain on the fresh set; if anything it is marginally slower
   than the 1× PD+ (+0.009 s, CI +0.001 to +0.019 at σ = 2; +0.003 s with a CI that includes 0
   at σ = 4).

All R6 arms and both PD+ baselines clear the ≥ 99.8% completion bar (the R0 PPO context row does not at σ = 4: 99.75%). The only failures at σ = 4 are one course each:
course 415 defeats PD+ 1×, PD+ 2×, pure seeds 0 and 2 and residual seeds 1 and 2; pure seed 1
fails course 1622 instead.

## Rules I held to

- **Same sim, same observations.** The 11-dim observation is unchanged: offset to the current
  waypoint, the next leg, velocity, lagged thrust, height, time left and waypoints left. Wind stays
  unobserved. No arm got a new input, so PD+ did not need a new input either. (The residual arm's
  policy also sees PD+'s action, which is a function of the same observation.)
- **Selection on training courses only.** Screening, recipe choice, PPO checkpoint choice and PD+
  tuning all used `flight.train_courses` (the same 128/256 courses and wind seed 11 that PD and PD+
  were tuned on). PPO checkpoints come from `best.npz`, the best `flight.score` on those 128
  courses, checked every 4M steps (2M for the residual).
- **Fresh held-out set.** The course seed and wind seed in `eval_r6_flight.py` were set only after
  both recipes and every selection were frozen (`STATUS.md` records the time). The set has 2000
  courses at wind σ = 2 and σ = 4, and was never used before.
- **Budgets.**
  - PD+ tuning: 2801 configurations, an upper bound of 2.30e8 env steps
    (`env_steps_upper_bound`, counted at 500 steps per episode).
  - Pure PPO: 2.3e8 env steps per seed. These are *actual* env steps, while PD+'s 2.3e8 is the
    upper bound counted at 500 steps per episode. Good configs finish in about 130 steps, so PPO
    used several times more simulated steps than PD+'s tuning really did. This follows the
    brief's convention and favours PPO.
  - Residual arm: it starts from the tuned PD+, so it has already spent PD+'s budget. Its fair
    baseline is **PD+ with twice the tuning budget** (`pd_plus_r6.py cont`: 2801 more
    local-refinement configurations).
- **Two arms, reported separately.** "Pure PPO" (no classical prior) answers whether learning
  alone can beat the tuned controller. "Learned residual on PD+" answers whether learning can
  improve a tuned controller. Neither stands in for the other.

## Step 1: give PD+ its best fair shot

| PD+ variant | configs | training courses (256, σ=2) | file |
|---|---|---|---|
| reviewer PD+ (R5) | 2801 | 6.389 s, score 1.74444 | `results/pd_plus_tuning.json` |
| fresh tune, widened ranges, same budget | 2801 | 6.395 s, score 1.74419 | `results/r6_pd_plus_wide_tuning.json` |
| 2× budget: +2801 local steps from the better of the two (reviewer) | 2801 + 2801 | 6.384 s, score 1.74466 | `results/r6_pd_plus_cont_tuning.json` |

- The widened search drove `v_turn` to 0, its natural floor (the reviewer had flagged it sitting at
  0.5). It gained nothing: PD+ tuning is saturated. The reviewer's PD+ stays the 1× baseline, and
  the 2× variant improves on it by only 0.005 s on the training courses.

## Step 2: screening (30M env steps each, one seed, training selection courses only)

All runs use 256 envs × 128 steps, minibatch 8192 and 4 epochs. The table shows mean time on the
128 selection courses (σ = 2, wind seed 11), all at 100% completion except `s_time_lin` (97.7 / 99.2 / 99.2% at 10 / 20 / 30M). **PD+ scores 6.359 s on the
same courses.** Logs: `results/r6_train_s_*.jsonl`, stdout `results/r6_logs/`.

| run | change | @10M | @20M | @30M | best |
|---|---|---|---|---|---|
| s_orig256 | original reward (progress to ring centre, bonuses) | 6.656 | 6.561 | 6.564 | 6.541 @24M |
| s_time | time-optimal: −dt per step, −10 on crash/timeout, potential shaping γΦ(s′)−Φ(s), Φ = −(remaining path)/10 m/s | 7.009 | 6.625 | 6.632 | 6.625 @20M |
| s_time_lin | + lr 5e-4 decaying linearly to 0 at 30M | 6.720 | 6.593 | 6.558 | 6.575 @26M |
| s_time_g995 | + γ = 0.995 | 6.980 | 6.569 | 6.482 | 6.482 @30M |
| **s_time_h256** | + 256 hidden units (2 × 256 MLP) | 6.610 | **6.403** | 6.431 | **6.403 @20M** |
| s_res | learned residual on PD+ (×0.5, zero-init, log-std −1.6), time reward | 6.303 | 6.292 | 6.296 | 6.292 @20M |

- The time-optimal reward alone was not better than the original reward at 30M; it learns more
  slowly at first (more early crashes). Network width mattered most.
- **Frozen pure recipe:** time reward, 2 × 256, γ = 0.99, lr 3e-4 decaying linearly to 0 over
  2.3e8 steps, seeds 0, 1 and 2. This combines the best single change (width) with the lr decay
  that helped `s_time_lin`. γ = 0.995 also helped, but the width + γ combination was never
  screened, so I left γ out rather than pick an untested combination.
- **Frozen residual recipe:** as `s_res`, with lr 3e-4 decaying linearly over 6e7 steps (the
  residual plateaus by about 20M), seeds 0, 1 and 2. The base is the reviewer PD+, the better of
  reviewer and wide on the training courses (`results/r6_logs/residual_base.txt`).
- **Engineering:**
  - The PPO update now optionally runs on the MLX GPU stream; rollouts stay on CPU and the old
    log-probs are recomputed on the update device. Under today's machine load (`uptime` ≈ 12)
    this gave 4.4× more throughput for the 256-wide net: 21k vs 94k steps/s in a 1.3M-step
    benchmark. The pure runs use it.
  - Save, stop and resume is bit-exact: a straight run and a stopped-then-resumed run give
    identical weights, with CPU updates (`tests/test_r6.py::test_r6_flight_resume_is_exact`) and
    with GPU updates (the same A/B check run once with `--update_device gpu --hid 256`: max weight
    difference 0.0).

## Step 3: final runs (training courses only)

| run | env steps | wall (s) | update device | best checkpoint (selection courses) | final log-std |
|---|---|---|---|---|---|
| f_pure_s0 | 2.30e8 | 1422 | GPU | 6.324 s @ 68M | −5.5 / −5.0 |
| f_pure_s1 | 2.30e8 | 2285 | GPU | 6.347 s @ 124M | −5.5 / −5.0 |
| f_pure_s2 | 2.30e8 | 1558 | GPU | 6.302 s @ 56M | −5.6 / −5.0 |
| f_res_s0 | 6.0e7 | 906 | CPU | 6.254 s @ 60M | −2.8 / −3.2 |
| f_res_s1 | 6.0e7 | 487 | CPU | 6.270 s @ 60M | −2.9 / −3.1 |
| f_res_s2 | 6.0e7 | 1129 | CPU | 6.264 s @ 54M | −2.9 / −3.0 |

Wall times vary with machine load (`uptime` load ranged from 12 to 49 during these runs). Every run
stayed under 45 minutes, so none needed its resume step. Logs: `results/r6_train_f_*.jsonl`,
`results/r6_logs/f_*.txt`.

## Held-out time vs env steps (`media/r6_flight_curve.png`)

Every 10M-step snapshot of every run, on the first 500 fresh courses at σ = 2. Mean time in
seconds; the 10M point for seed 1 completed only 97.6% of courses. On these 500 courses
PD+ 1× = 6.401 s and PD+ 2× = 6.404 s.

| run | 10M | 20M | 30M | 50M | 80M | 100M | 130M | 150M | 200M | 230M |
|---|---|---|---|---|---|---|---|---|---|---|
| pure s0 | 7.051 | 6.448 | 6.418 | 6.390 | 6.354 | 6.362 | 6.381 | 6.512 | 6.556 | 6.532 |
| pure s1 | 6.809* | 6.487 | 6.415 | 6.457 | 6.399 | 6.442 | 6.373 | 6.408 | 6.407 | 6.407 |
| pure s2 | 6.720 | 6.516 | 6.387 | 6.331 | 6.447 | 6.472 | 6.454 | 6.500 | 6.453 | 6.495 |
| residual s0 | 6.323 | 6.296 | 6.283 | 6.279 | | | | | | |
| residual s1 | 6.318 | 6.305 | 6.302 | 6.296 | | | | | | |
| residual s2 | 6.322 | 6.331 | 6.312 | 6.286 | | | | | | |

- **More steps did not keep helping pure PPO.** It reaches PD+ level by about 30M steps and its best
  region is roughly 50–130M. After that, seeds 0 and 2 get *slower* (s0 6.354 s at 80M → 6.532 s at
  230M). The same drift is visible on the training selection courses: s0 goes from 6.324 s at 68M
  to 6.516 s at 230M.
  - It coincides with the state-independent exploration std collapsing (log-std −5, std ≈ 0.006)
    while the lr decays linearly to 0. There is no entropy bonus, so nothing stops the collapse.
  - Likely mechanism (not isolated): with almost no exploration noise, PPO's advantage estimates
    stop carrying a useful signal about small trajectory changes, and the policy drifts.
  - The checkpoint selection on training courses caught the good region for every seed
    (56M, 68M, 124M), which is why the held-out rows above beat PD+ at σ = 2.
- **The residual improves steadily and stays stable.** About 0.08 s of the gain comes in the first
  10M steps, about 0.03 s more by 60M, and all three seeds are within 0.02 s of each other.

## How the flyers fly (`results/r6_flight_style.json`, first 500 fresh courses, σ = 2)

| flyer | closest approach at a pass | passes in outer half of ring | speed at pass | peak speed | thrust saturated |
|---|---|---|---|---|---|
| PD+ 1× | 0.898 m | 97.7% | 6.38 m/s | 12.18 m/s | 78.5% |
| PD+ 2× | 0.909 m | 98.3% | 6.25 m/s | 12.13 m/s | 78.3% |
| pure PPO s0 | 0.870 m | 99.5% | 6.07 m/s | 12.25 m/s | 70.4% |
| residual s0 | 0.898 m | 97.7% | 6.43 m/s | 12.35 m/s | 69.5% |

- Both learned flyers saturate the thrust **less** than PD+ (70% vs 78% of steps) but reach a
  higher peak speed. They spend less time braking and correcting, i.e. smoother arcs, not more
  bang-bang.
- The residual keeps PD+'s ring-grazing aim (identical 0.898 m) and carries slightly more speed
  through the rings. Pure PPO grazes even closer (0.870 m) but carries less speed through rings.
- Plots: `media/r6_flight_paths.png` (3 fresh courses plus each arm's worst course) and
  `media/r6_flight_diffs.png` (histograms of the per-course differences).

## Preserved failure

`media/r6_flight_paths.png`, fourth panel, fresh course 489 (σ = 2): pure PPO seed 0 is
**1.45 s slower** than PD+ (7.70 s vs 6.25 s). It approaches ring 2 on a line that just misses the
ring and has to loop back. This is the downside of grazing at 0.87 m in gusty wind; PD+'s
slightly wider margin (0.90 m) never needs the loop. The residual's worst course (1602, fifth
panel) costs 0.80 s against PD+ 2× (6.60 s vs 5.80 s); the panel shows its path through the last
rings drifting away from PD+'s line.
`r6_flight_figs.py` records the full 2000-course run and slices it. All envs share one wind RNG,
so re-running a subset would give a different wind (my first version of the figure did that and
showed the wrong times).

## What was not tried, and why

- **Wind estimate or frame history.** Fairness would require giving PD+ the same estimate and
  retuning it. On the fresh set PD+ loses only 0.06 s going from σ = 2 to σ = 4, and in R5 it lost
  0.03 s from calm to σ = 2. That bounds what wind knowledge can buy a PD+-style controller. It is
  also exactly where pure PPO is weak (+0.16–0.21 s from σ = 2 to σ = 4), so a wind input or
  training at mixed wind levels is the most promising next step for the pure arm, with PD+ given
  the same input.
- **More waypoints ahead.** Not tried: PD+ would need a redesign to use them, as well as a retune.
- **An entropy bonus, a log-std floor, or γ = 0.995 combined with width 256.** Each screened or
  diagnosed promisingly, but they were not combined in a screen, so they were not used.

## Compute (R6 flight)

| item | env steps | wall |
|---|---|---|
| screening, 6 runs | 30M each | 3.4–15 min each |
| PD+ wide retune | 2801 configs | 223 s |
| PD+ 2× continuation | 2801 configs | 133 s |
| pure PPO final, 3 seeds | 3 × 2.3e8 | 88 min |
| residual final, 3 seeds | 3 × 6e7 | 42 min |
| held-out evaluation (2000 courses × 2 winds × 9 flyers, plus 87 snapshot evaluations) | | about 1 min |

- Everything ran through `../bin/gpu_run` with at most 2 of my jobs at once, on AC power.

## Files

- **Code:** `train_flight_r6.py`, `pd_plus_r6.py`, `eval_r6_flight.py`, `r6_flight_figs.py`,
  `r6_screen.sh`, `r6_final_chainA.sh`, `r6_final_chainB.sh`, `r6_final_pure3.sh`, `tests/test_r6.py`.
- **Checkpoints:**
  - `checkpoints/r6/<run>/` holds `best.npz`, `snap_*M.npz`, `state.pkl` (resume) and
    `config.json` for every screening and final run.
  - PD+ parameters: `checkpoints/pd_plus_r6_wide_params.json`, `checkpoints/pd_plus_r6_cont_params.json`.
- **Results:** `results/r6_flight_eval.json` (all held-out numbers), `results/r6_flight_times.npz`
  (per-course times), `results/r6_flight_style.json`, `results/r6_pd_plus_*_tuning.json`,
  `results/r6_train_*.jsonl`.
- **Media:** `media/r6_flight_curve.png`, `media/r6_flight_paths.png`, `media/r6_flight_diffs.png`.

## Reproduce

```sh
export OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
../bin/gpu_run ../.venv/bin/python pd_plus_r6.py wide && ../bin/gpu_run ../.venv/bin/python pd_plus_r6.py cont
sh r6_final_chainA.sh        # residual s0-2 (after the PD+ retunes)
sh r6_final_pure3.sh         # pure PPO s0-2, 2.3e8 steps each, resumable
../bin/gpu_run ../.venv/bin/python eval_r6_flight.py   # fresh held-out set (seeds fixed in the file)
../.venv/bin/python r6_flight_figs.py
```
