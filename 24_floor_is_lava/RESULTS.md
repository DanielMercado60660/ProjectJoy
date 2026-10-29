# RESULTS: 24 The Floor Is Lava (first satisfying version)

Task: a Unitree Go1 (MuJoCo Menagerie, 12 position-servo joints) crosses six broad stepping stones
over a lava floor. Any contact between a robot geom and the floor is a failure. Stones are
0.35–0.55 m long and 0.55–0.80 m wide, with gaps of 0.05–0.20 m, lateral offsets of ±8 cm and tops
8–12 cm above the lava. The course runs about 3.7 m from the start to the finish line. Episodes are
capped at 20 s. The policy sees full state plus the next 4 stones.

Every number here comes from a real run on this machine (`results/*.json`; `results/summary.md` is
generated from them by `make_tables.py`). Every comparison uses **held-out layouts** (seeds
`10_000_000 + i`, which training and tuning never sample), with identical initial-state noise seeds
for every method. Proportions carry Wilson 95% intervals; means carry bootstrap 95% intervals.

## Headline: 500 held-out layouts, full gap range (0.05–0.20 m)

| method | env steps used | completion | lava | flip | J/m (completed) | m/s (completed) |
|---|---|---|---|---|---|---|
| open-loop CPG trot, tuning round 1 | 3.9M | 2.4% [1.4, 4.1] | 94.0% | 3.2% | 1208 | 0.47 |
| open-loop CPG trot, tuning round 2 | 11.8M total | 6.8% [4.9, 9.4] | 92.2% | 0.6% | 1358 | 0.43 |
| CPG + scripted foothold planner, round 1 | 4.4M | 7.2% [5.2, 9.8] | 87.8% | 5.0% | 1051 | 0.47 |
| CPG + scripted foothold planner, round 4 (best) | 34.5M total | **69.4%** [65.2, 73.3] | 28.8% | 1.8% | 971 | 0.44 |
| residual PPO, blind ablation (no stone obs) | 18.6M | 88.8% [85.7, 91.3] | 9.4% | 1.8% | 559 | 1.55 |
| **residual PPO** (full state + next 4 stones) | 14.4M | **98.4%** [96.9, 99.2] | 1.6% | 0.0% | 567 | 1.65 |

Reviewer addition: a planner round 5 with a widened gait range (7.9M more steps; 42.4M total) chose
the round-4 parameters again, so 69.4% stands. A planner restricted to fast gaits scored 0.4% (see
"Baselines" below).

Sources: `results/eval_run1_full.json` (PPO, CPG round 1, planner round 1), `results/eval_planner4_full.json`,
`results/eval_blind_full.json`, `results/eval_cpg2_full.json`. Planner rounds 2 and 3 scored 31.0% and
62.6% (`results/eval_planner2_full.json`, `results/eval_planner3_full.json`).

- **PPO vs the best scripted baseline** (planner round 4, same 500 layouts): completion +0.29
  [0.25, 0.33]. PPO alone succeeded on 150 layouts and the planner alone on 5.
- **Energy on layouts both completed** (n = 342; this removes the bias from comparing on
  different layout subsets): PPO 567 vs planner 970 J/m, a difference of −403 [−410, −397] J/m.
  PPO also walks 3.7× faster (1.65 vs 0.44 m/s). Against the open-loop CPG (n = 12 layouts both
  completed): 568 vs 1208 J/m. Cost of transport (E/(m g d)): PPO 4.5, open-loop CPG 9.7.
- Energy is Σ|τ·q̇|·dt over the 12 joints divided by distance; negative work counts as cost. The
  absolute values are high (≈400 W for the CPG trot on flat ground) because the Menagerie Go1's
  stiff position servos (kp = 100) do a lot of negative work. Treat them as a comparison between
  methods, not as a realistic power draw. Part of PPO's lower J/m is its speed: fixed per-second
  costs are spread over more metres.
- **Per second (added in review):** on the same 342 jointly completed layouts PPO draws **932 W**
  of mechanical power against the planner's **422 W** (2.2× more per second), and 925 vs 564 W
  against the open-loop CPG (n = 12). So PPO is cheaper per metre only because it is 3.7× faster;
  per second it costs more. `results/review_energy.json` (`review_energy.py`).

### Other held-out sets
| set | open-loop CPG r1 | open-loop CPG r2 | planner r4 | residual PPO | file |
|---|---|---|---|---|---|
| gaps 0.05–0.125 m (500) | 20.8% [17.5, 24.6] | 46.6% [42.3, 51.0] | 92.0% [89.3, 94.1] | 99.6% [98.6, 99.9] | `eval_run1_gap0125`, `eval_cpg2_gap0125`, `eval_planner4_gap0125` |
| gaps 0.05–0.25 m, wider than any seen in training (300) | 0.3% [0.1, 1.9] | not run | 24.7% [20.1, 29.8] | 94.0% [90.7, 96.2] | `eval_ood_gap025`, `eval_planner4_ood_gap025` |
| flat walkway, no gaps (200) | 92.5% [88.0, 95.4] | 98.5% [95.7, 99.5] | 100% [98.1, 100] | 100% [98.1, 100] | `eval_run1_flat_push`, `eval_cpg2_flat_push80`, `eval_planner4_flat_push80` |

The open-loop CPG's flat-walkway failures (7.0% lava) are lateral drift off the side of the
walkway. Its trunk had drifted 0.17–0.34 m to the right when it fell.

## Push recovery (lateral force on the trunk for 0.1 s once it passes the start of stone 3)
| condition | open-loop CPG r1 | planner r4 | residual PPO |
|---|---|---|---|
| stones, 40 N: completion | 1.2% [0.6, 2.6] (r2: 5.2% [3.6, 7.5]) | 64.6% [60.3, 68.7] | 97.8% [96.1, 98.8] |
| stones, 40 N: fail ≤2 s after push (episodes that reached the push) | 63.7% (n=182) | 12.3% [9.6, 15.6] (n=455) | 1.6% [0.8, 3.1] (n=497) |
| stones, 80 N: completion | 1.2% [0.6, 2.6] | 43.6% [39.3, 48.0] (added in review; 36.5% fail ≤2 s after push, n=455) | 86.2% [82.9, 88.9] |
| flat walkway, 40 N | 79.0% [72.8, 84.1] | 100% [98.1, 100] | 100% [98.1, 100] |
| flat walkway, 80 N | 59.5% [52.6, 66.1] (r2: 74.5% [68.0, 80.0]) | 83.0% [77.2, 87.6] | 98.5% [95.7, 99.5] |
| flat walkway, 120 N | 31.5% [25.5, 38.2] | 2.5% [1.1, 5.7] (97.5% flips) | 42.0% [35.4, 48.9] (0.5% flips; the rest stepped off the walkway's edge into lava) |

40 N for 0.1 s is a ~0.3 m/s lateral velocity kick on the 12.7 kg robot; 120 N is ~0.95 m/s.
Files: `eval_run1_full`, `eval_planner4_full`, `eval_push80_full`, `eval_run1_flat_push`,
`eval_flat_push80`, `eval_flat_push120`, `eval_planner4_flat_push{40,80,120}`, `review_eval_planner4_push80_full`. On stones, the
open-loop CPG's push numbers mostly measure its foothold failures: it rarely gets past stone 3. The
tuned planner stands taller (0.29 m) and flips under a 120 N push. PPO never flips, but it gets shoved
off the edge of the walkway.

## Does the policy use the stones? (perception vs motor control)
- **Blind ablation** (`ppo.py --blind`: same config and wall-clock budget, with the stone block and the
  height-above-stone observation zeroed; the base CPG's heading term still steers toward the stone
  centre line): **88.8%** held-out completion vs 98.4% with stone observations
  (`results/eval_blind_full.json`). Under the 40 N push, 13.9% of blind episodes fail within 2 s,
  against 1.6% with stone observations. Training curves: `media/training_curves_blind.png` (83.5%
  training completion over its last 100 iterations, vs 95.6%). It ran 18.6M steps in the same 42
  minutes because the machine was less loaded.
- So **most of PPO's gain over the baselines is a better gait**, not foothold perception: fast trotting
  with long strides that crosses 0.05–0.20 m gaps without looking. The stone observations take
  it from 89% to 98% and cut the lava rate from 9.4% to 1.6%.
- Zeroing the stone observations of the full-obs policy at test time drops it to 0.4%
  (`results/eval_mask_stones.json`). That is an out-of-distribution input, so it only shows the
  policy depends on those inputs. The blind-trained policy is the fair ablation.

## Diagnostics and checks for exploits
- **No jumping or bounding.** Control steps with no foot in contact: PPO 0.5%, open-loop CPG 4.6%,
  planner r4 0.1% (`airborne_frac`). Peak trunk height above the stone top: PPO 0.277 m, CPG 0.302 m.
- **No edge sliding.** Mean horizontal speed of feet in contact: PPO 0.41 m/s, CPG 0.32,
  planner 0.23. The metric includes touchdown and lift-off transitions, which scale with speed, and
  PPO walks ~3.5× faster. So the slip is comparable, not an exploit. `media/demo_heldout_ppo_0.mp4`
  shows an ordinary fast trot.
- **Edge use (added in review, `review_probe.py`, 40 held-out layouts, `results/review_probe.json`):**
  PPO lands closer to stone edges than the planner. 14% of its foot–stone contacts are within 2 cm
  of a stone's front or back edge (planner r4: 6%), and contacts whose normal is more than 45° from
  vertical (a stone's side face or corner) carry 2.6% of its foot support force (worst episode 7%;
  planner 0.4%). That is a riskier foothold style, not an exploit: the top surfaces carry ~97% of the
  load and no completed episode skipped a stone.
  The Menagerie foot is soft (`solimp 0.015 1 0.023`): contact penetration reaches ~3 cm for
  **both** methods (PPO worst 3.2 cm, planner 3.3 cm), more than the 2.3 cm foot radius. That is the
  same for all methods; it would matter for sim-to-real, not for this comparison. Max |qvel| 17.7
  (PPO) vs 15.5 (planner); no NaNs or blow-ups.
- **Sim stability:** 4 ms timestep with 5 substeps per 50 Hz control step. No NaNs or contact
  explosions in any rendered clip. Stones are moved and resized at runtime by rewriting body pos,
  geom size, rbound, aabb and the body's BVH box. `tests/test_env.py` checks that a resized stone
  holds the robot up and that a missing stone means lava.
- **PPO's 8/500 held-out failures** were diagnosed from which geom touched the lava and where.
  All 8 are lava touches (no flips, collapses or stalls). Six are a foot dropping into one of the
  widest gaps (0.145–0.195 m). Two are a foot stepping off the side edge of a stone or the goal
  platform after lateral drift. Clip: `media/demo_ppo_failure_side_by_side_140.mp4`. On held-out
  layout 140 the rear-left foot of PPO lands in a 0.167 m gap at a stone's side edge. On the same
  layout the tuned planner crosses slowly and the open-loop CPG falls in at the first gaps.

## Demo (`media/`)
- `demo_heldout_side_by_side_0.mp4` / `.gif`: held-out layout 0, picked mechanically as the lowest
  held-out index where PPO completes. Left to right: residual PPO crosses in 2.7 s, the open-loop
  CPG touches lava at 5.2 s, and the planner r4 crosses in 9.7 s. Individual clips:
  `demo_heldout_{ppo,cpg_openloop,cpg_planner}_0.mp4`.
- `demo_ppo_failure_side_by_side_140.mp4` / `.gif`: PPO's first held-out failure (layout 140).
- `training_curves.png`, `training_curves_blind.png`.
- `python render_demo.py --policy checkpoints/policy.npz --planner checkpoints/planner4_params.json`
  reloads the checkpoint and reproduces the videos and `demo_heldout_0.json`.
- The open-loop CPG in the clips is the **round-1** CPG (`checkpoints/cpg_params.json`), the gait
  under the residual policy. Round 2 (6.8% held-out) finished after PPO training and is not in the
  videos.

## Training (residual PPO; `logs/main_train.csv`, `logs/main_level.csv`, `media/training_curves.png`)
- One 42-minute run: 1753 iterations × 8192 steps = 14.4M env steps, from 4 worker processes ×
  16 envs. That is about 5k env steps/s **under load** (load average 18–24 on 18 cores while other
  builders trained). The orchestrator paused the job for ~30 s when the charger was unplugged; the
  pause counts against the 42 minutes. No resume run was needed: held-out completion was already
  98.4%. A second training seed was **not run**, so seed-to-seed variance is not measured. The blind
  ablation is a second, independent training run, and it also learned (88.8%).
- Gap curriculum: 80% of episodes at the current level and 20% at full difficulty. The level rises by 0.1
  when ≥50% of an iteration's current-level episodes (≥20 of them) succeed. Full difficulty was
  reached at iteration 286 (2.3M steps). The log shows "level -> 1.0" twice (iterations 286 and
  306): 0.9 + 0.1 accumulates to 0.9999… in floating point, so the first step lands just under 1.
  Harmless.
- Completion on full-difficulty training episodes: 1.6% at ~0.8M steps, 23% at 2.5M, 65% at 4.1M,
  88% at 8.2M, 94% at 10.2M and 95% at the end. Lava touches fell from ~70% to ~3%; training energy
  went from 2057 to 591 J/m.
- PPO health: approx-KL 0.015–0.023 (target 0.03; the early stop fires in many iterations). The
  clip fraction is 0.21–0.29: high, because the action std is small (0.20 → 0.15), which makes
  ratios sensitive. The action mean is tanh-bounded. Old log-probs and values are recomputed in
  MLX before each update, so ratios start at exactly 1 (no CPU-fp32 vs GPU-TF32 mismatch).
- Checkpoint/resume was verified before the long run: save at iteration 3, kill the PID, resume.
  It continued at iteration 4 with an identical policy std (0.49281), and the value loss stayed in
  range. `smoke.sh` repeats the save/resume check at tiny size.

## Baselines: what was tuned, and how hard
Both baselines are scripted, with no learning, and use the policy's exact episode loop
(`LavaTask.step(zeros)` *is* the open-loop baseline). The **planner** uses the policy's stone
positions (it reads the full layout, but only the stone at the predicted touchdown matters): at each foot's lift-off it predicts the touchdown point (hip x + trunk
velocity × gain × swing time + half stride). If that point lands in a gap or within `margin` of an
edge, it shifts the step by the smallest amount (≤ `dmax`) that puts the foot on a stone. The
open-loop CPG uses stone information only in its heading term, which steers toward the next stone's
centre line. The residual policy inherits that term too.

All tuning used **training layouts only**: random search, then CEM, then re-evaluation of the top
candidates on 128–192 fresh training layouts, keeping the best. Objective: completion + 0.3 × mean
progress. Search ranges (`task.py: CPG_RANGES, PLAN_RANGES`): frequency 1.2–3.5 Hz, stride 0–0.30 m,
stance height 0.20–0.31 m, swing clearance 0.02–0.13 m, duty 0.40–0.75, front and rear foot x offset
±0.06 m, abduction −0.15–0.25 rad, front-rear height difference ±0.04 m, three phase offsets in 0–1
(trot, pace, bound and walk are all reachable), heading gain 0–3, look-ahead 0.2–1.2 m. Planner only:
edge margin 0–0.09 m, max step shift 0–0.16 m, velocity gain 0.5–1.5.

| round | configs | env steps | wall | training completion of the winner | held-out completion |
|---|---|---|---|---|---|
| CPG r1 | 1182 | 3.9M | 414 s | 2.3% (128 layouts) | 2.4% |
| CPG r2 (warm start from r1, fixed CEM) | 1031 | 7.9M | 582 s | 7.8% (192) | 6.8% |
| planner r1 | 1183 | 4.4M | 619 s | 10.9% (128) | 7.2% |
| planner r2 | 1263 | 8.0M | 1126 s | 33.6% (128) | 31.0% |
| planner r3 | 1031 | 9.7M | 774 s | 66.7% (192) | 62.6% |
| planner r4 | 839 | 12.4M | 890 s | 71.9% (192) | 69.4% |

Open-loop CPG round 2 (a warm-started CEM with the bug fixed, 7.9M more steps; 11.8M total) moved
the open-loop gait from 2.4% to 6.8% held-out (46.6% on gaps ≤ 0.125 m). It chose 3.14 Hz, duty
0.64 and phase offsets (0, 0.73, 0.89, 0.34). Its training score was still noisy between generations
(best 0.07–0.17), so it may also have headroom. On the full gap range, though, open-loop footholds
land wherever the gait puts them, and that caps it far below the planner. The residual policy still
sits on the round-1 CPG; round 2 finished after PPO training.

**The baseline was under-tuned at first.** Planner round 1's CEM looked flat, but more CEM
generations kept improving it: 7% → 31% → 63% → 69% held-out. A bug made rounds 2 and 3 fresh,
longer CEM runs rather than warm-started ones: the CEM mean was initialised from the elite of the
random stage, so the warm start was dropped. It was fixed for round 4 (mean = warm parameters, spread =
8% of each range). The planner's total budget, 34.5M env steps and 57 min of tuning, is **2.4× PPO's
14.4M steps**. It was still improving slowly at the end (training completion 66.7% → 71.9% in
round 4), so a longer search might narrow the gap further. Round 4 chose a near-trot (phase offsets
0, 0.50, 0.58, 0.08) at 2.96 Hz, a 0.145 m stride, a 0.29 m stance height, a 6 cm edge margin and
step shifts of up to 11.8 cm.

**Reviewer check: is the planner stuck in a slow gait class?** (`review_tune_wide.py`,
`logs/review_planner5.sh`.) The residual policy's mean modulations put its CPG at 3.44 Hz and a
0.256 m commanded stride (`results/review_probe.json`), which is *inside* the tuner's box
(≤ 3.5 Hz, ≤ 0.30 m), so the range did not forbid PPO's gait parameters. Its real speed (1.65 m/s)
is about twice what that freq × stride commands (~0.88 m/s); the extra comes from the 12 joint
residuals. Two checks:
1. *Residual-free point:* PPO's base CPG (round 1) at PPO's mean frequency and stride, with no joint
   residuals, alone and with the planner r4's planner settings: **0.0%** on the 500 held-out layouts
   (97% lava) and 0/100 even on the flat walkway (`results/review_eval_ppogait_full.json`,
   `review_eval_ppogait_flat.json`). PPO's own mean gait settings, without its residuals, can't walk.
2. *Planner round 5 with a widened gait range* (freq 1.2–5.0 Hz, stride 0–0.45 m, warm start from
   round 4, 120 random configs then 10 CEM generations; 611 configs, 7.9M env steps, 548 s): it
   returned **exactly the round-4 parameters**, so the held-out result is unchanged at **69.4%**
   (`results/review_eval_planner5_full.json`). Across its 611 training-layout evaluations, no config
   with a nominal speed (freq × stride) above 0.6 m/s completed more than 2.5% of layouts (62
   configs), and none above 0.8 m/s completed any (40 configs).
   That was a local search, though: its CEM population never went above ~0.5 m/s nominal after
   generation 2.
3. *Fast gaits only* (`review_fast_corner.py`, then `review_tune_wide.py --fast_corner`;
   `logs/review_planner_fast.sh`): the box is restricted to freq 3.2–5.0 Hz and stride 0.22–0.45 m.
   First, open-loop on flat training walkways (0.65M steps, 43 s): fast open-loop gaits **do** exist.
   The best walks **100% of 100 held-out flat walkways at 0.97 m/s** (`results/review_eval_fastgait_flat.json`),
   2× the planner r4's speed. Then the CPG + planner was tuned from that gait on stones, restricted to
   the fast corner (611 configs, 2.4M steps, 163 s): **0.5% training, 0.4% [0.1, 1.4] held-out**, with
   93% lava (`results/review_eval_planner_fast_full.json`). A fast scripted gait can't place its feet
   with this lift-off-time planner. The slow 0.44 m/s gait is the planner's best operating point,
   not an artefact of the search range.
   Planner tuning in total: **42.4M env steps for rounds 1–5 (2.9× PPO's 14.4M)**, plus 2.4M for the
   fast-gait variant (PPO's base gait cost a further 3.9M of CPG round-1 tuning).

The round-1 open-loop CPG chose a lateral-sequence walk/trot mix
(0, 0.68, 0.70, 0.31) at 2.64 Hz with a 0.21 m stride. That round-1 CPG is also the base gait under
the residual policy.

## What worked
- **Building from a working gait.** A Cartesian CPG plus 2-link IK walked on flat ground on the first
  try. The residual policy started from the CPG's ~60% completion at the easiest curriculum level and
  learned in one 42-minute run.
- **Gait-level residual actions.** Besides 12 joint offsets, the policy controls stride and frequency
  multipliers. It used them to walk 3.5× faster with longer strides, and it does use the stone
  observations to place its feet (89% → 98%).
- **Runtime terrain edits** (no recompile) and a 4 ms timestep gave ~4–5k env steps/s per core
  under load.
- **A scripted foothold planner is a real baseline once tuned properly**: 69% held-out, 92% on
  gaps ≤ 0.125 m.

## What didn't work (aborted runs, logs kept)
1. **Initial action std 0.5** (log std −0.7): ±0.12 rad of noise on every joint plus ±20% frequency
   and ±25% stride noise wrecked the gait. Completion stayed at 0% for 25 iterations, where the
   bare CPG gets ~60% at that curriculum level. Fixed with std 0.2. No file exists for that run: its CSV and log were deleted
   before the relaunch. The two numbers are from the session console (iteration 5: success 0.000, lava 0.64; iteration 25:
   success 0.000, lava 0.63).
2. **Energy penalty scaled ~40× too high** (0.02 per J; a control step costs ~10 J, so the penalty
   was twice the progress reward). Dying early paid: training success went from 0.34
   (iterations 21–30) to 0.07 (iterations 71–80) while energy per metre fell from 1272 to 819
   (`logs/aborted_run_energy_bug_train.csv`). Fixed to 0.0005 per J, and the action-rate penalty
   dropped from 0.02 to 0.005, so a normal walking step nets a positive reward.
3. **The CEM warm-start bug** in the baseline tuner (above). It cost tuning efficiency, not
   correctness, and every round was selected on training layouts.

## Surprises
- PPO's biggest change was **speed**: 1.65 m/s against the baselines' 0.44–0.47 m/s. Long, fast
  strides cross a gap in fewer footfalls, and the stride multiplier steps over gaps. It generalises
  to gaps 25% wider than anything in training (94.0% completion at gaps up to 0.25 m, where the
  planner falls to 24.7%).
- **The blind policy gets 89%**, so the gait carries most of the result.
- The baseline kept improving with more CEM generations long after the first tuning run looked
  converged: the wave-2 lesson again.
- Push recovery fails differently by method: the tall-stance planner flips, while PPO never flips but
  gets pushed off the edge.

## Not measured
- A second PPO training seed (seed variance): not measured.
- Real-robot transfer, sensor noise, latency, motor heat and electrical energy: not measured. Energy is
  the mechanical Σ|τ·q̇| proxy.
- Pushes timed to land mid-step on a gap edge (the push point is fixed at stone 3): not measured.
- 80 N push on stones for the planner round 4: ~~not run~~ run in review: 43.6% [39.3, 48.0] (`results/review_eval_planner4_push80_full.json`).
- Depth or height-scan perception (the "Go deeper" item): not attempted.

## Next steps
- Replace the stone list with a local height scan, keeping the privileged policy as a teacher. Report
  perception and motor failures separately; the blind ablation gives the "no perception" floor (89%).
- Harder terrain where the gait alone can't win: narrower stones, stones narrower than the stance
  width, height changes, moving platforms.
- A planner that also chooses body speed and gait timing (e.g. sampling-based MPC over CPG
  parameters), tuned to convergence, and a second PPO seed.

## Review addendum (cross-reviewer, 2026-10-24)
Headline numbers unchanged; everything above traces to `results/*.json`.
- Baseline strength: none of three stronger-baseline attempts beat planner r4. A widened-range
  re-tune (round 5) returned r4 (69.4%). A residual-free "PPO gait" point scored 0.0%. A planner
  restricted to fast gaits scored 0.4%, even though its open-loop gait walks flat ground at 0.97 m/s.
  So fast scripted gaits exist but can't hit the stones with this planner. The residual policy's gain
  is a fast gait whose learned feedback places the feet, plus foothold use (blind 88.8% → 98.4%). A
  feedback-stabilised scripted gait with per-step foothold control (e.g. a Raibert-style or MPC
  controller) was not tried; it would be the next stronger baseline.
- The comparison puts PPO's residual on the round-1 gait and the planner on its own round-4 gait.
  That asymmetry favours the baseline (it re-tuned its gait; PPO did not), so it doesn't inflate PPO.
- Energy: PPO is cheaper per metre (567 vs 970 J/m) but costs 2.2× more per second (932 vs 422 W).
- PPO places feet nearer stone edges (14% of contacts within 2 cm vs 6%); side/corner contacts carry
  2.6% of its support force. Not an exploit, but a riskier style that explains its rare edge failures.
