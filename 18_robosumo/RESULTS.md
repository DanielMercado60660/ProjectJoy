# Results: RoboSumo, first satisfying version

## Headline (round robin, 200 bouts per pair, eval seeds 5000+, training ring/friction ranges, sides alternated)
All numbers from `results/tournament.json` (tables in `results/tables.md`). PPO = final checkpoint of run 3
(`checkpoints/main.pkl`), deterministic mean action. Brackets are 95% Wilson intervals.

| PPO vs | PPO win | PPO loss | draw | score (W−L)/n | best script vs the same opponent (score) |
|---|---|---|---|---|---|
| circler (trained against) | **81.0%** [75.0, 85.8] | 1.0% [0.3, 3.6] | 18.0% | **+0.80** | charger +0.355 (36.5% W); *review: corner +0.33 (38.0% W)* |
| counter (**held out**, never seen in training) | 16.5% [12.0, 22.3] | 7.5% [4.6, 12.0] | 76.0% | +0.09 | charger +0.19 (19.0% W, 0% L); *review: corner script +0.215 (23.0% W, 1.5% L)* |
| charger (trained against) | 2.5% [1.1, 5.7] | 2.5% [1.1, 5.7] | 95.0% | 0.00 | flanker −0.025 |
| flanker (trained against) | 1.5% [0.5, 4.3] | 6.5% [3.8, 10.8] | 92.0% | −0.05 | charger +0.025 |
| random | 75.5% [69.1, 80.9] | 6.5% [3.8, 10.8] | 18.0% | +0.69 | flanker +0.965 |

Second measurement for the held-out counter: at fixed centre conditions (R 1.0, friction 1.0; robustness table,
n = 100), PPO goes 13 W / 11 L / 76 D, which is about even. Pooled over both cells it is 46 W / 26 L / 228 D,
score +0.07. A two-sided sign test on the 72 decisive bouts gives p ≈ 0.025, so "slightly ahead, marginally
significant" is the honest summary.

Standings (mean score over the other five entrants): **PPO +0.306**, charger +0.289, flanker +0.213,
counter +0.075, circler −0.082, random −0.801. PPO tops the table, but by a margin inside the noise.

> **Review update (2026-10-24):** the reviewer added a seventh entrant, `corner` (an angled-hit script, `review_scripts.py`,
> tuned only against the three training scripts). With it, the standings over six opponents are **charger +0.253, PPO +0.253
> (an exact tie), corner +0.239**, flanker +0.173, counter +0.027, circler −0.123, random −0.821 (SE about ±0.01 each).
> PPO no longer tops the table on its own; it ties for first. Corner vs PPO itself: 9 W / 6 L / 185 D (+0.015, even).
> See the review addendum at the end of this file.

The short version:
- **The learned policy beats one scripted opponent decisively and is only slightly ahead of the held-out one.**
  It wins 81% against the circler (the best script manages 36.5% there). Note that the circler is the
  **weakest** script: it ranks last among the scripts (−0.082), and the other scripts beat it or draw it. Against the held-out counter-pusher it
  wins more than it loses (16.5% vs 7.5%; the intervals just touch), but the plain charger script does better
  against the counter (+0.19 vs +0.09).
- **Against the two head-on scripts (charger, flanker) almost every bout is a draw.** Equal robots that meet
  nose to nose wedge and stall for the full 10 s (clip: `media/demo.mp4`, bouts 3–4). The scripts do the
  same to each other (charger vs flanker: 97.5% draws). With matched robots, a head-on push has no winner
  unless someone gets an angle. Neither the learner nor the scripts learned or were written to reliably get one.
- So the answer to "do learned pushing policies beat scripted opponents?" is: **yes against the circler, marginally
  against the held-out counter, no against the charger and flanker (draw / slightly negative)**. It is the
  best entrant overall by a small, not significant, margin. *(Review: with the reviewer's corner script added it ties the
  charger for first, +0.253 each; and two scripts, charger +0.19 and corner +0.215, beat the held-out counter by more than PPO's +0.09.)*
- It took three runs to get here. Runs 1 and 2 collapsed to a passive "turtle" (see below).
- Single training seed, one main run, 20 minutes on battery.

## Setup
- **Physics:** MuJoCo 3.16, dt 5 ms, control at 25 Hz, 10 s bouts. Two identical ≈2.6 kg differential-drive
  robots (box chassis, two velocity-servo wheels with ≤ 0.8 N·m torque, two frictionless casters). Lose by chassis
  centre leaving the ring or by tipping (up-vector z < 0.35). Both at once, or a timeout, is a draw.
- **Training distribution:** ring radius uniform 0.9–1.1 m, sliding-friction scale 0.8–1.2, random start angle
  around the ring, 0.3–0.5 R from the centre, headings roughly facing ±0.8 rad.
- **Learner:** PPO in MLX. Separate 64×64 tanh actor and critic. Gaussian actions with **tanh-bounded mean** and
  a state-independent log-std clamped to [−2.5, 0]. GAE λ 0.95, γ 0.99, clip 0.2, 16 envs × 128 steps per update,
  4 epochs of minibatch 512, Adam lr 3e-4, grad-norm clip 0.5, no entropy bonus. Old log-probs are **recomputed in
  MLX** before each update, so the ratio starts at exactly 1 on the update device (lesson from 20).
  Logged approx-KL: median 0.0010, 99th percentile 0.0051, max 0.030. Clip fraction: median 1.1%,
  99th percentile 5.1%, max 15.9% (`checkpoints/main_log.jsonl`, `media/train_curve.png`).
- **Reward (run 3):** +1 win / −1 loss. A timeout draw costs **−0.5**. Potential-based shaping with
  Φ = (own edge margin − opponent edge margin − 1.0·distance to opponent), in metres.
- **Opponents per bout (run 3):** 25% current self, 25% uniform from the archive (a snapshot every 20 updates,
  203 snapshots by the end), 15% warm-up (idle or uniform-random actions), 35% one of the three tuned training
  scripts. The counter-pusher is never used in training.
- **Run 3 (main):** 20.0 min wall clock, 4,063 updates, **8.32 M learner env steps**, about 6,900 steps/s.
  That speed was measured **under load** (on battery, `uptime` load average 2.2–3.2 during the run). Single CPU process for MuJoCo plus MLX updates, `OMP_NUM_THREADS=4`.

## Baseline: scripted opponents, tuned on training seeds only
`tune_scripts.py` → `results/tuned_scripts.json`. Coordinate ascent, two rounds. Each candidate was scored
by mean (+1/0/−1) over 16 bouts against each of the other three scripts, with sides alternated, on tuning seeds
100 and 101 (disjoint from eval seeds 5000+). Round 1 played the defaults; round 2 played the round-1 winners.

| script | grid (values tried) | chosen | round-2 score: chosen / default |
|---|---|---|---|
| charger | k {1, 2, 3.5} × speed {0.7, 1} × slow-turn angle {0.6, 1.2} | k 1, speed 1, slow-turn 1.2 | +0.31 / +0.02 |
| flanker | standoff {0.25, 0.4} × k {2, 3.5} × push dist {0.35, 0.6} × speed {0.8, 1} | 0.25, 3.5, 0.35, 1.0 | +0.10 / +0.06 |
| circler | orbit {0.3, 0.5} × expose {0.3, 0.6, 0.85} × k {2, 3.5} | 0.3, 0.85, 3.5 | +0.10 / −0.58 |
| counter (held out) | hold {0.15, 0.35} × trigger {0.25, 0.5} × k {2, 3.5} × dodge steps {4, 8} | 0.15, 0.25, 2.0, 4 | 0.00 / +0.04 |

The tuning objective is noisy: about ±0.15 with 48 bouts per candidate. The charger's winner flipped between
rounds (k 3.5 → 1.0), and for the counter the chosen setting scored below the default on the round-2 seed
(winner's curse). Tuning used 2.5 min of compute (56 candidates × 2 rounds). The learner used 20 min plus
two aborted runs (≈4.6 min). So compute is **not** matched: the scripts got far less search. Information is matched,
because every script reads the same 15-number egocentric observation the policy sees, and every script drives
the same robot.

## Cross-play against archived snapshots (100 bouts per pair; `results/tables.md`)
| final vs snapshot at update | 20 | 820 | 1640 | 2440 | 3260 | 4060 |
|---|---|---|---|---|---|---|
| final's score (W/L) | +0.16 (16/0) | +0.16 (20/4) | +0.15 (16/1) | +0.11 (13/2) | +0.01 (5/4) | −0.01 (0/1) |

- The final policy beats its early selves, but mostly through draws (76–99% of bouts). The snapshot-vs-snapshot
  matrix is close to transitive: later beats earlier by +0.01 to +0.19 (review fix: was "+0.02"; snapshot 1640 vs 820 is +0.01). There is one small reversal
  (update 2440 beats update 3260 by +0.10, n = 100, about 2 SE). No strong strategy cycling was seen.
- **Snapshots vs scripts** shows where the skill came from. The score vs circler climbs from −0.29 (update 20) to
  +0.66 (1640), +0.85 (2440), +0.92 (3260), and then **drops to +0.71 at the end** (4060). Counter peaks at +0.20
  (update 2440), and the final is +0.10. The best checkpoint against the held-out script was a middle one. The
  headline uses the final one anyway; it was not picked.

## Falls (tips) and ring-outs
- **Under training conditions: 0 tips** in the 3,000 round-robin bouts, so every decisive round-robin bout
  was a ring-out. During training, the learner's self-tip rate (over the last 300 bouts) peaked at 0.7% and
  was usually 0.
- Across all 14,500 evaluation bouts there were **23 tips**. Three came from snapshot cross-play. The other
  20 came at **friction ×1.6 (outside training), and the PPO robot was the one that tipped every time**: 10
  of its 12 losses vs counter at R 1.0, 7 at R 1.4, 3 vs charger. With higher grip, its full-throttle ram
  climbs onto the opponent's chassis and flips itself over. Clip: `media/exploit_tip_highfriction.mp4`
  (seed 8021, the first tipping seed found in a search from 8000, so it is a selected example).

## Robustness to ring size and friction outside training (100 bouts per cell)
Training: R 0.9–1.1 m, friction ×0.8–1.2. The "decisive %" column is the same shift applied to the six
script-vs-script pairs, as a reference for how much the game itself changes.

| R | friction | vs charger W/L | vs flanker W/L | vs circler W/L | vs counter W/L | PPO mean score | script pairs decisive % |
|---|---|---|---|---|---|---|---|
| 1.0 | 1.0 (in range) | 1/3 | 3/3 | 89/2 | 13/11 | +0.217 | 21.7% |
| **0.7** | 1.0 | 8/13 | 14/22 | 73/19 | 46/26 | +0.153 | 65.5% |
| 1.4 | 1.0 | 1/2 | 2/0 | 86/0 | 16/4 | +0.247 | 8.3% |
| 1.0 | **0.5** | 0/2 | 6/6 | 95/5 | 15/14 | +0.223 | 23.5% |
| 1.0 | **1.6** | 2/4 | 0/3 | 39/1 | 11/12 | +0.080 | 6.2% |
| 0.7 | 0.5 | 3/14 | 4/28 | 78/21 | 35/34 | +0.057 | 62.8% |
| 1.4 | 1.6 | 0/0 | 0/0 | 26/0 | 2/7 | +0.053 | 1.8% |

- **Small ring (0.7 m):** bouts become decisive for everyone. PPO's losses rise sharply (flanker 22–28% L, circler
  19–21% L), but its mean score stays positive, because it also wins 46% against the counter.
- **Low friction (×0.5):** essentially unchanged from in-range.
- **High friction (×1.6):** the circler win rate falls from 89% to 39% (mostly draws), and PPO starts tipping
  itself over (above). This is the clearest out-of-distribution failure.
- **Large ring (1.4 m):** slightly better for PPO, with fewer ring-outs for everyone.

## Policy diversity across the archive
Two measures (`results/tournament.json` → `behaviour`, `action_disagreement`):
1. **Action disagreement** is the mean |Δ action| on a fixed probe set of 400 observations drawn from
   script-vs-script bouts. Consecutive snapshots: update 20→820: 0.456; 820→1640: 0.015; 1640→2440: 0.024;
   2440→3260: 0.107; 3260→4060: 0.152. Final vs first: 0.485. For reference, unrelated uniform actions would
   give about 0.67.
2. **Behaviour descriptors** come from 20 bouts vs the charger, with the same starts for every snapshot. Mean forward
   command +0.54 at update 20, then +0.91 to +0.99 for every later snapshot. The turn command is 0.01–0.09, and
   no snapshot ever reverses.

So diversity is **low**. After the first ~800 updates (~4 min), the archive is a family of near-identical
"drive at full speed, steer a little" policies, and the later drift (0.1–0.15) is fine steering detail. This is
the "collapse to a single strategy" risk the shelf notes warned about. Here it is a charger-like rammer, not an
exploit. Against the circler that ram is far better than the charger script itself. Against true head-on
opponents it stalls.

## What didn't work: two aborted runs that turtled
- **Run 1** (`checkpoints/run1_passive/`, `logs/run1_passive_log.jsonl`): no draw penalty, approach weight 0, no
  warm-up opponents. It was killed at 2.1 min (375 updates saved). It had 0 wins against every opponent
  type. **Self-play bouts were 100% draws**, and its losses to scripts fell from ~25% to ~5%. With draws worth 0
  and losses −1, it learned to do as little as possible. Clip: `media/run1_passive_turtle.mp4`.
- **Run 2** (`checkpoints/run2_passive/`): +0.5 draw penalty and 0.3 approach shaping. Killed at 2.4 min:
  still 0 wins, and the deterministic action barely moved the robot (it crept ~0.1 m in 10 s against an idle
  opponent).
- **Run 3** added 15% idle/random warm-up opponents and approach weight 1.0. Wins against idle/random appeared
  within ~1 min, and wins against the circler by ~3 min. Runs 1–2 were stopped early on a training-log judgement
  (not evaluated on held-out seeds), and run 3 was launched once. The fixes were not ablated separately.

## Surprises
- **Head-on stalemate.** Two equal robots pushing nose to nose is a stable equilibrium in MuJoCo. Ring-outs
  need an angle, and nothing on this shelf gets one reliably. Draws dominate every matchup except those
  involving the circler (which exposes its side) and random.
- **PPO is worse than three of the four scripts at beating the random mover** (75.5% win vs charger 87.5%, counter 81.5%, flanker 96.5%; circler 66%). The
  ram is aimed at where the opponent is, and a jittering random robot often slips past.
- **Self-tipping at high friction**, as described above. The physical tipping that the design hoped to see
  appears only outside the training friction range, and it is the learner's own failure.
- The best checkpoint against both the circler and the held-out counter was a middle one (update 2440–3260),
  not the final.

## Not measured
- Multiple training seeds or reruns (single seed 0). The variance of the headline across seeds is unknown.
- A fully matched compute budget between script tuning (2.5 min) and PPO (20 min + 4.6 min of aborted runs).
- Population training / league play. Only latest-self + uniform archive sampling was used.
- Quadrupeds/humanoids ("Go deeper").
- Speed numbers on an idle machine. The only speed number (~6,900 learner steps/s) is a busy-machine, on-battery
  number.

## Next steps
1. Break the head-on stall: give a reward or observation cue for contact angle, or add opponent-relative
   heading/bearing features that make flanking easier to learn. Then check whether the charger/flanker draws
   turn into wins.
2. Train with friction up to ×1.6 and add a small tip penalty for being high-centred, to remove the self-tipping.
3. Sample the archive by win rate (prioritised fictitious self-play) and select the checkpoint on a validation
   script set. Also try a second seed to put an error bar on the headline.
4. Raise the circler's and counter's tuning budget (more bouts per candidate) to remove the winner's-curse noise.

## Battery and compute log
- Battery 88% at start (23:46), 74% after the main run, tournament and renders (00:21). The Mac was on
  battery throughout, and this was the only builder running.
- Heavy compute: script tuning 2.5 min, run 1 2.1 min, run 2 2.4 min, run 3 20.0 min, tournament 1.7 min,
  smoke tests and renders ~2 min. Total about 31 min.

## Cross-review addendum (2026-10-24, reviewer)
Everything above this section is the builder's text; numbers the review changed are
marked inline. New files: `review_scripts.py`, `review_eval.py`, `results/review_*.json`, `logs/review_*.log`.

**Stronger baseline attempt.** Reviewer's probe (robots 0.4 m apart, robot A at full throttle for 4 s): a square
head-on push drives an idle robot out of the ring in 3.2 s, but against a robot pushing back it moves it 0.03 m (the stall).
The same push against a pushing robot, offset 8 cm sideways, turns it about 50° and moves it 0.10 m. So an angle helps a
little, and the reviewer wrote four angle-seeking scripts and grid-searched them against the three *training* scripts only (charger,
flanker, circler; the counter stays held out for these scripts too), on tuning seeds 300 and 302. Three were dropped
(best score vs the training scripts on seed 300, 24 bouts per opponent): `leaner` +0.10, `hooker` +0.07, `rehit` −0.28
(backing off to re-hit exposes the robot). `corner` (aim at a point 0.2 m, then 0.1 m inside 0.45 m, beside the opponent's
centre, perpendicular to the line between the robots; gain 2.5) won round 2 (seed 302, 48 bouts per opponent, +0.153;
`results/review_corner_select.json`).

Corner vs the field, 200 bouts per pair, eval seeds 6000–6005, sides alternated (`results/review_corner_rr.json`):

| corner vs | W | L | D | score | 95% CI win | 95% CI loss |
|---|---|---|---|---|---|---|
| PPO | 9 | 6 | 185 | +0.015 | [2.4, 8.3] | [1.4, 6.4] |
| charger | 0 | 14 | 186 | −0.070 | [0.0, 1.9] | [4.2, 11.4] |
| flanker | 10 | 5 | 185 | +0.025 | [2.7, 9.0] | [1.1, 5.7] |
| circler | 76 | 10 | 114 | +0.330 | [31.6, 44.9] | [2.7, 9.0] |
| counter (held out) | 46 | 3 | 151 | +0.215 | [17.7, 29.3] | [0.5, 4.3] |
| random | 184 | 0 | 16 | +0.920 | [87.4, 95.0] | [0.0, 1.9] |

What moves: **(1)** the standings: PPO now ties the charger for first (+0.253 each over six opponents) with corner just
behind (+0.239), all within about one standard error (±0.01). **(2)** Against the held-out counter, PPO's +0.09 is now
third behind corner +0.215 and charger +0.19. **(3)** PPO vs corner is even (9 W / 6 L / 185 D). What does not move: PPO's
81% vs the circler is still far above any script (best: charger 36.5%, corner 38%), and nothing cheap beats the charger
(every angled variant lost to it or drew). The headline "PPO beats one scripted opponent decisively and is level with the
rest" stands; "PPO tops the table" does not.

**Wins are real pushes.** Replaying all 3,000 round-robin bouts (bit-identical W/L/D, `results/review_contact.json`) with
MuJoCo's contact list: all 1,298 decisive bouts had robot-robot contact, 1,297 within 1 s of the ring-out and 1,294 within
0.5 s (the exceptions are 4 circler wins over random). No entrant wins by the opponent driving itself out. In draws the robots are in
contact about 90% of the time (the head-on wedge).

**Physics sanity** (`results/review_physics.json`, 1,100 bouts): no MuJoCo warnings, no non-finite states, top speed
1.02 m/s (wheel limit 1.0). Under training conditions the chassis never rises above 0.11 m (rest height 0.06; 1–2% of bouts
show a small ride-up on the opponent's nose) and never tilts past up-z 0.87, far from the 0.35 tip rule. At friction ×1.6
it rises to 0.15 m and up-z 0.20 in 4–5% of PPO bouts: the self-tip the builder already reported, nothing new.
