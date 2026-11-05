# Round 6: a learned attacker that generalizes to opponents it has never seen

**Verdict.** Population training fixes most of round 4's overfitting, but not on every seed. All three seeds of
the frozen recipe beat the hand-written striker by a wide margin vs the **presser** (+47 to +54 points) and the **zone**
defender (+9 to +11). That is two of the three opponents nobody trained or tuned against. Round 4 was below the
scripted striker on all three. The **interceptor** splits the seeds: seeds 0 and 2 beat the scripted striker by
+22 / +22 points, seed 1 is 8 points *below* it. Only **seed 0** also keeps its game vs the original goalie (89%,
better than round 4's 75%); seeds 1 and 2 collapse to 13% and 5%, below the CEM-tuned scripted striker's 18%. So the
pre-registered done bar is met by **1 of 3 seeds** (seed 0). Part (a) alone (beat the scripted striker vs every new
opponent) is met by 2 of 3. Against the **CEM-tuned** scripted striker (the fair-shot baseline), seeds 0 and 2 are
also ahead on all three new opponents (heldout set, paired CIs > 0: interceptor +13.7 / +12.8, zone +2.4 / +3.3,
presser +15.4 / +15.5); seed 1 only on zone and presser. Numbers below are from the frozen recipe on fresh held-out
seeds drawn after the freeze.

**Two extra seeds (3, 4), pre-registered after the 3-seed result, same frozen recipe:** seed 3 meets (a) but not (b)
(goalie 11.7%); seed 4 fails both (interceptor −9.4 points, goalie 22.1% [19.6, 24.7], overlapping the tuned
striker's CI). Over all **5 seeds**: zone and presser beaten by **5/5**, interceptor by **3/5**, goalie kept by
**1/5**; full done bar **1/5**, part (a) **3/5**. A pre-registered post-hoc goalie fine-tune of seeds 1 and 2 did
not fix the goalie (15.6% / 6.7%). See "Extension" and "Post-hoc" below.

## Setup

### What changed from round 4
- **Opponent population** (`r6_common.py`, `r6_opponents.py`, `r6_train.py`). Every training episode draws one slot-1 opponent:
  the legacy `goalie` / `keeper` / `chaser`, five randomized scripted families (`goalie_r`, `keeper_r`, `chaser_r`,
  `marker`, `wall`), or **self-play**: a frozen past snapshot (`self_arch`, archive of up to 24, one every 8 M steps)
  or the current snapshot (`self_cur`) of the attacker itself driving slot 1 through a 180°-mirrored observation.
  The learner's own snapshots attack the learner's goal, so self-play also trains defence.
- **Per-episode randomization** of scripted opponents: speed ×U[0.75, 1], reaction delay 0–6 steps (it sees the ball as it
  was up to 0.3 s ago), Gaussian action noise σ~U[0, 0.3], plus each family's own parameters (standoffs, rush radii, line
  depth, prediction horizon, marking distance, blocker position). 35% of scripted episodes are **nominal** (speed 1, no
  delay, no noise): see "Pilot 1" for why.
- **Observation**: 27-d = round 4's 23 + opponent velocity (ego frame) + own previous action.
  Network 128×128 tanh MLPs, tanh-squashed mean, on-device log-prob recompute (round-4 fixes kept).
- **Step-based schedule**: curriculum ramps over the first 25% of the step budget and lr anneals to 0 at the budget
  (the round-3 pool run was ruined by a wall-clock schedule on a loaded machine).
- **Checkpoints** (`checkpoints/r6_*.pkl`): weights, Adam state, obs normaliser, archive, pickled env with its RNG,
  trainer RNGs, counters, config. Resume verified bit-exact (SIGKILL after a checkpoint, resume: identical losses and
  max |ΔW| = 0; `tests/test_r6.py::test_resume_is_bit_exact`).

### Held-out opponents (new, eval-only)
Specified in STATUS.md at 18:08 before any round-6 code existed; never trained or tuned against; enforced by
`tests/test_r6.py` (no reference in training code) and by `r6_eval.py --mode dev`, which refuses them. All defend
from slot 1 at the opponent's nominal speed (0.56 m/s) with no delay or noise.
- **interceptor**: rolls the ball 2 s ahead with the real friction and wall bounces, drives to the earliest
  point of the ball's path it can reach first (own half only), else guards 0.30 m out; clears toward the near touchline.
- **zone**: high-line zonal defender at x = 0.62 tracking 0.6·ball y; steps up and clears diagonally when the ball
  enters its zone; when beaten, runs back (around the ball) to its goal line.
- **presser**: charges the ball by the shortest path whenever the attacker is within 0.30 m of it or the presser is
  closer; otherwise shadows 0.25 m goal-side of the ball; after winning the ball it counter-attacks toward the
  attacker's goal.

Validity check before training (untuned scripted attacker + random only, `results/r6_check_heldout_opponents.json`):
two revisions were made and logged in STATUS (a zone bug where its recovery run carried the ball into its own goal;
the interceptor and zone conceding ~80% to the scripted striker). Then they were frozen. The zone defender is still
easy for the scripted striker (~89%), so the "beat the scripted attacker" bar against it sits near the ceiling.

The old `goalie`, `keeper` and `chaser` are in the training pool this round, so they are **dev-visible**, not held out.

## Every attempt, in order (development used only training-visible opponents)
Dev evals: `r6_eval.py --mode dev`, seeds 1501+, 300 matches per cell, `heldout` start set (training
distribution from an unseen seed), deterministic mean action. Training goal rates are over the last 3000
training episodes (stochastic policy, randomized opponents).

| run | what | steps | outcome |
|---|---|---|---|
| pilot 1 (seed 10) | attacker-only self-play, pilot mix, no nominal episodes | 60.0 M | dev goal: keeper 99.7%, chaser 98.0%, pool families 69–100%, **exact goalie 28.0%** (run 4: 74.0%) — `results/r6_dev_pilot_s10.json` |
| pilot 2 (seed 11) | + learned defender (`--learn-def 1`, 30% of episodes vs current/archived learned defender) | 15.2 M (stopped) | training goal rate fell to 7–8% (7.4% at 14.4 M; 81% timeouts overall, 85–99% vs most kinds; vs the chaser families mostly conceded instead); pilot 1 had 70% at 14.4 M — `logs/r6_pilot_def_s11.jsonl` |
| diag (seed 11) | attacker-only, same seed as pilot 2 | 15 M (planned stop) | 74% training goal at 12 M → the seed is fine — `logs/r6_diag_s11_nodef.jsonl` |
| diag (seed 11) | learn-def code path with def_* at 4% only | 15 M (planned stop) | 73% at 14 M → no code bug; the collapse needs ~30% learned-defender episodes — `logs/r6_diag_s11_def_tiny.jsonl` |
| pilot 3 (seed 12) | pilot 1 + 35% nominal scripted episodes | 60.0 M | dev goal: keeper 99.0%, chaser 96.3%, families 71–100%, **exact goalie 12.7%**; reverses on 63–89% of steps — `results/r6_dev_pilot3_s12.json` |
| CEM (pilot mix) | scripted-attacker tuning on the pilot pool | 152.4 M | pool goal 79.7% (default 55.8%); dev goalie 15.7%, keeper 58.0% — `results/r6_tuned_attacker_pilotmix.json` |

**Pilot 1 lesson (randomization box).** Vs the exact goalie pilot 1 scored 28%, but vs the same goalie at 0.85× or
0.7× speed it scored 82% / 79% (stochastic actions did not help: 25%). Every training episode had the opponent's
speed drawn from U[0.7, 1] plus delay and noise, so the nominal opponent, which is also how every held-out
opponent plays, sat at the corner of the randomization box and was almost never seen. Fix: 35% of scripted
episodes nominal. It did not raise pilot 3's goalie number (12.7%; one seed each, so this is not separable
from seed variance), so for the final recipe I also raised the goalie's share of the mix (below).

**Pilot 2 lesson (learned defender → stalling).** The learned defender is weak: the scripted attacker scores 97%
against it (`results/r6_diag_learndef.json`). The attacker trained with it learned to win the first touch
(80–86%) and then stay ~0.7 m away from the ball (pilot 1's attacker stays 0.13–0.16 m away), so 89–98% of its
matches vs any defender are timeouts. That is the round-3 "win the race, then park" failure again: the +0.5
first-touch bonus is collected and nothing else is attempted. Why ~30% learned-defender episodes push it there and
the scripted pool does not is not established. With the pre-registered rule, the final recipe drops the
learned defender. The code stays (`--learn-def 1`) and the stopped checkpoint is kept as
`checkpoints/r6_pilot_def_s11.pkl`.

### Final recipe (frozen 18:51, before any held-out evaluation)
Attacker-only self-play, 35% nominal scripted episodes, mix goalie .14, goalie_r .14, keeper .05, chaser .05,
keeper_r .10, chaser_r .08, marker .06, wall .04, self_arch .22, self_cur .12 (the goalie share was raised from
.08/.12 on dev evidence, untested), 110 M steps, 512 envs × 128 steps per update, 128×128 MLPs, lr 3e-4 linear to
0, ent 0, archive snapshot every 8 M steps. Seeds 0, 1, 2. Held-out eval seed base drawn from /dev/urandom at the
freeze (`results/r6_heldout_seed_base.txt`). The done bar was pre-registered in STATUS at the same time.

## Backwards dribbling: diagnosed (and preserved)
Round 4's run 4 pushes the ball with its **rear**. Probes on dev seeds, 500 matches (`r6_probe_reverse.py`,
`results/r6_probe_reverse_run4.json`): vs the goalie, 78% of its near-ball steps have the ball behind the robot
(96% vs the keeper). The scripted striker, which never reverses, has 6%.

**It is a fixed convention, not a reaction to the start pose.** Split by start heading: when run 4 starts *facing*
the ball, the ball is still behind it in 75% of near-ball steps (81% when it starts facing away). In the first
second it reverses 31% of steps when facing the ball and 84% when facing away: it turns its back to the ball even
when it doesn't need to. Goal rate is the same either way (75% facing vs 76% away), so the convention costs nothing.

**Why: a symmetry the network has to break.** The robot is a disc with equal forward and reverse speed, so
"heading θ, wheels (l, r)" and "heading θ+π, wheels (−r, −l)" are exactly the same motion. Neither end is
better. The MLP is not built to respect that symmetry, so early in training it commits to one end and sticks with it.
**The mirror-twin test** (`r6_mirror_twin.py`, `results/r6_mirror_twin.json`) checks this directly. It runs run 4
as if the robot were turned round and maps the action back, i.e. the same strategy using the other end. The twin
pushes with its **front** (ball behind in 22% of near-ball steps vs 78%) and scores the same: 75.0% [71.2, 78.8]
vs 75.6% [71.6, 79.2] against the goalie. Backwards dribbling is a free choice between two equivalent ends,
not an exploit and not a skill.

**It varies by seed, and the final seeds span the whole range** (`r6_probe_reverse.py`, dev seed 1501, 500 matches;
`results/r6_probe_reverse_s01.json`, `results/r6_probe_reverse_s2.json`; reverse-fraction curves in
`media/r6_training_curves.png`):

| policy | steps reversing (goalie / keeper / interceptor) | ball behind robot, near-ball steps | goal vs goalie (held-out) |
|---|---|---|---|
| run 4 | 48% / 46% / – | 78% / 96% / – | 74.7% |
| r6 seed 0 | 53% / 53% / 49% | 56% / 69% / 67% (mixed: uses both ends) | 89.0% |
| r6 seed 1 | 73% / 83% / 85% | 88% / 99% / 99% (committed rear dribbler) | 12.8% |
| r6 seed 2 | 6% / 2% / 2% | 5% / 4% / 4% (forward dribbler) | 5.4% |

The convention is set early: seed 1 locks into reverse by ~35 M steps; seed 2 reversed ~30% early on and then
switched to forwards after ~17 M steps. It does not predict skill: the best seed vs the goalie (seed 0) uses both ends,
and the forward-only seed is the worst vs the goalie. The mirror twin of round-6 seed 1 confirms the symmetry again
(`results/r6_mirror_twin.json`): it pushes with its front (ball behind 2–12% vs 88–99%) and scores the same vs the
keeper (99.0% vs 99.2%) and within noise vs the goalie (11.4% [8.6, 14.2] vs 15.0% [11.8, 18.2]). One catch I hit:
round 6's observation includes the robot's previous action, which also has to be mapped (l, r) → (−r, −l) in the
twin; without that the twin collapsed (1.4% vs the goalie), a twin bug, not a broken symmetry.

**Clips (preserved):** `media/r6_reverse_demo.html` shows `r6_reverse_r6_s1_vs_keeper.gif` (rear dribbler) next to
`r6_reverse_twin_r6_s1_vs_keeper.gif` (its forward twin, same starts, both 4/4 goals), and the same pair for run 4
vs the goalie (`r6_reverse_run4_vs_goalie.gif`, `r6_reverse_twin_run4_vs_goalie.gif`). Dev seed 1501, first 4
starts, unselected. The round-4 clips in `media/ppo_*.gif` also show it (white heading tick points away from the ball).


## Held-out results (fresh seeds drawn after the freeze: base 809457; 1000 matches per cell)
Files: `results/r6_heldout_a.json` (scripted, tuned, run4, r6_s0, r6_s1), `results/r6_heldout_b.json` (r6_s2; same
seeds, so the same 1000 start states for every attacker). Tables: `results/r6_heldout_tables.md` (all three start sets,
from `r6_report.py`). Done bar: `results/r6_donebar.json` (`r6_donebar.py`). Deterministic mean-action policies.
goal % [95% bootstrap CI] / conceded+own-goal %. Δ = goal minus the untuned scripted striker on the **same starts**
(paired bootstrap, 95% CI). **interceptor / zone / presser are the held-out opponents. goalie / keeper / chaser are
in the round-6 training pool (dev-visible)**; for run 4, keeper and chaser were never seen.

`heldout` start set (training start distribution from an unseen seed):

| attacker | **interceptor** | **zone** | **presser** | goalie (dev-visible) | keeper (dev-visible) | chaser (dev-visible) |
|---|---|---|---|---|---|---|
| scripted striker | 64.7 [61.8, 67.7] / 5.6 | 89.1 [87.1, 91.0] / 0.3 | 36.4 [33.3, 39.2] / 11.9 | 1.3 [0.6, 2.0] / 0.2 | 49.3 [46.1, 52.3] / 0.0 | 31.4 [28.5, 34.1] / 4.0 |
| CEM-tuned striker | 73.4 [70.7, 76.2] / 3.3 · Δ +8.7 | 95.8 [94.6, 97.0] / 0.3 · Δ +6.7 | 68.2 [65.3, 71.0] / 4.2 · Δ +31.8 | 18.1 [15.7, 20.4] / 1.0 | 58.3 [55.3, 61.5] / 0.1 | 78.9 [76.3, 81.4] / 4.2 |
| run 4 (round 4, goalie only) | 18.7 [16.3, 21.2] / 6.4 · Δ −46.0 | 29.0 [26.2, 31.9] / 2.1 · Δ −60.1 | 30.2 [27.5, 33.2] / 31.9 · Δ −6.2 | 74.7 [72.0, 77.4] / 0.1 | 1.1 [0.5, 1.8] / 0.0 | 9.8 [8.0, 11.7] / 59.0 |
| **r6 seed 0** | **87.1** [85.0, 89.1] / 1.1 · Δ **+22.4** [+18.8, +26.0] | **98.2** [97.3, 99.0] / 0.2 · Δ **+9.1** [+7.0, +11.3] | **83.6** [81.2, 85.9] / 5.2 · Δ **+47.2** [+43.4, +50.9] | 89.0 [87.0, 90.8] / 0.1 | 98.6 [97.8, 99.3] / 0.1 | 94.7 [93.3, 96.1] / 3.1 |
| **r6 seed 1** | 56.8 [53.9, 59.7] / 0.7 · Δ **−7.9** [−12.0, −3.9] | **99.7** [99.3, 100.0] / 0.0 · Δ **+10.6** [+8.7, +12.6] | **90.8** [89.0, 92.5] / 2.8 · Δ **+54.4** [+51.0, +57.9] | 12.8 [10.8, 14.9] / 0.0 | 99.7 [99.3, 100.0] / 0.1 | 96.0 [94.7, 97.2] / 1.0 |
| **r6 seed 2** | **86.2** [83.9, 88.3] / 4.7 · Δ **+21.5** [+18.0, +24.9] | **99.1** [98.5, 99.6] / 0.2 · Δ **+10.0** [+8.0, +12.2] | **83.7** [81.4, 85.8] / 5.0 · Δ **+47.3** [+43.8, +50.9] | 5.4 [4.1, 6.8] / 0.7 | 99.9 [99.7, 100.0] / 0.1 | 88.2 [86.1, 90.1] / 9.8 |

The `corner` (ball starts in the corners, a region no training reset uses) and `kickoff` sets tell the same story
(`results/r6_heldout_tables.md`). Seeds 0 and 2 beat the scripted striker on all 9 held-out opponent × start-set cells.
Seed 1 beats it on 6 of 9: it loses all three interceptor cells (−7.9 / −12.2 / −2.2 points, the last CI [−6.1, +1.8]).
Against the **CEM-tuned** striker (paired CI > 0), seed 0 is ahead in 7/9 cells (not significant: zone/corner,
presser/kickoff), seed 2 in 8/9 (not significant: presser/kickoff), seed 1 in 5/9.

### Done bar (pre-registered in STATUS at the freeze)
| seed | (a) beats scripted vs interceptor, zone, presser (heldout set, paired CI > 0) | (b) goalie above CEM-tuned striker, non-overlapping CIs | both |
|---|---|---|---|
| 0 | PASS (+22.4, +9.1, +47.2) | PASS (89.0 [87.0, 90.8] vs 18.1 [15.7, 20.4]) | **PASS** |
| 1 | FAIL (interceptor −7.9 [−12.0, −3.9]) | FAIL (12.8 [10.8, 14.9]) | FAIL |
| 2 | PASS (+21.5, +10.0, +47.3) | FAIL (5.4 [4.1, 6.8]) | FAIL |

Calibration caveat: before the freeze the interceptor and zone were revised (logged in STATUS) because the *untuned*
scripted striker scored ~80% against them; the revisions lowered its rate (interceptor 78 → 67%; zone 80 → 92 → 89%).
The learned seeds' margins (+9 to +54 points, except seed 1 vs the interceptor) are larger than any shift that
calibration could have produced, and the revisions never involved a learned policy.

Conceded + own-goal rates with 95% CIs for every cell are in `results/r6_heldout_tables.md`. On the heldout start set the r6 seeds concede 0–10% (worst: seed 2 vs the chaser, 9.8%; on kickoff starts that cell is 13.5%). Run 4 conceded
59% vs the chaser and 32% vs the presser on the heldout starts (more on kickoff starts). Learning to defend came from the chaser families and the self-play strikers.

### Elo round robin (`r6_elo.py`, `results/r6_elo.json`)
15 attackers (3 final seeds, 3 archive snapshots per seed at 32/72/104 M steps, CEM-tuned and untuned scripted
strikers, run 4) × 18 slot-1 opponents (the 3 held-out and 3 legacy scripted opponents, plus the final checkpoint and the
same 3 snapshots of every seed driving slot 1 as a striker). 200 matches per pair on the `heldout` start set (seed base
809957). Score: goal 1, timeout ½, conceded/own goal 0. Bradley–Terry fit with the scripted striker anchored at 1000.
95% CIs from 200 bootstrap resamples of matches. Caveat: the learned slot-1 opponents are each seed's own archive
(training opponents for that seed), and goalie/keeper/chaser are dev-visible, so this is a strength ladder, not a
held-out test.

| attacker | Elo [95% CI] |
|---|---|
| r6 seed 0 (final) | 1322 [1305, 1342] |
| r6 seed 1 (final) | 1302 [1286, 1317] |
| r6 seed 2 (final) | 1209 [1190, 1223] |
| r6 snapshots @104 M | 1203–1323 |
| r6 snapshots @72 M | 1172–1303 |
| r6 snapshots @32 M | 1078–1176 |
| CEM-tuned striker | 1163 [1148, 1176] |
| scripted striker | 1000 (anchor) |
| run 4 | 799 [788, 810] |

Opponent side (higher = harder to score on): goalie 1074, learned slot-1 strikers 832–977 (later snapshots harder),
presser 875, interceptor 859, chaser 850, keeper 697, zone 586. Every seed rises from 32 M to 72 M (seed 0 1176 → 1303,
seed 1 1156 → 1303, seed 2 1078 → 1172) and is flat or rising after; each final checkpoint is within CI of its 104 M snapshot.

### Baselines
- **Scripted striker** (round 1's `scripted.attacker`), untuned.
- **CEM-tuned striker** (`r6_tune_attacker.py`, `results/r6_tuned_attacker.json`): 7 knobs (standoff, off-centre aim,
  alignment tolerance, push-through, detour, steering gain, reverse allowed). CEM with 48 candidates × 96 matches for
  200 generations against the **scripted part of the frozen training pool** with the same randomization: 160.1 M env
  steps in ~8 min (one PPO seed: 110.0 M env steps in 31–33 min). On the pool it goes from 52.0% to 74.1% goal rate
  (2000 fresh matches). It never saw the held-out opponents. It is the strongest non-learned attacker here: +7 to
  +32 points over the untuned striker on the new opponents. A first CEM run on the pilot mix is kept as
  `results/r6_tuned_attacker_pilotmix.json` (pool 55.8% → 79.7%).
- **Run 4** on the new opponents: 18.7 / 29.0 / 30.2%, below the untuned scripted striker on all three. That
  confirms the round-4 overfitting finding on opponents designed after the fact.

## Extension: seeds 3 and 4 (pre-registered in STATUS after the 3-seed result; recipe and code path unchanged)
`results/r6_heldout_c.json`, same held-out seeds; tables for all 5 seeds in `results/r6_heldout_tables_5seeds.md`;
bar `results/r6_donebar_5seeds.json`. heldout start set, goal % [95% CI] / conceded+own %, Δ vs untuned scripted:

| seed | interceptor | zone | presser | goalie (dev-visible) | keeper (dev-visible) | chaser (dev-visible) | bar (a) / (b) |
|---|---|---|---|---|---|---|---|
| 3 | 80.2 [77.7, 82.6] / 0.9 · Δ +15.5 | 94.9 [93.5, 96.2] / 0.1 · Δ +5.8 | 91.4 [89.6, 93.0] / 2.4 · Δ +55.0 | 11.7 [9.7, 13.8] | 69.9 [67.1, 72.7] | 93.7 [92.1, 95.1] / 3.0 | PASS / FAIL |
| 4 | 55.3 [52.2, 58.4] / 2.4 · Δ **−9.4** | 98.7 [98.0, 99.3] / 0.2 · Δ +9.6 | 82.2 [79.9, 84.5] / 7.1 · Δ +45.8 | 22.1 [19.6, 24.7] | 97.5 [96.5, 98.4] | 94.0 [92.6, 95.4] / 2.5 | FAIL / FAIL |

Seed 3 is the only seed not ahead of the CEM-tuned striker vs the zone (−0.9 [−2.8, +0.9]); it is also oddly weak vs
the training-visible keeper (69.9%). Seed 4's goalie CI overlaps the tuned striker's, so it fails (b) by the
non-overlap rule even though its point estimate is higher. Neither seed is in the Elo round robin (it was run before
the extension). **Across 5 seeds the interceptor failure (seeds 1 and 4) and the goalie failure (seeds 1–4) are the
two recurring weaknesses.** Both interceptor failures are rear dribblers (reversing 80–85% of steps), but so is
seed 3 (81%), which passes, so dribbling direction does not explain it.

## Post-hoc follow-up: goalie fine-tune (pre-registered in STATUS; not part of the headline)
`--init` from the final seed 1 / seed 2 checkpoints, 30 M more steps, goalie share .30 (+ goalie_r .15), 50% nominal
scripted episodes, lr 1e-4 → 0 (`checkpoints/r6_s{1,2}_goalieft.pkl`, `results/r6_heldout_posthoc_goalieft.json`,
`results/r6_donebar_posthoc_goalieft.json`). Evaluated once on the same held-out seeds:

| run | interceptor | zone | presser | goalie |
|---|---|---|---|---|
| seed 1 → fine-tuned | 56.8 → 66.0 [63.2, 69.0] (Δ vs scripted +1.3 [−2.6, +5.2]) | 99.7 → 99.8 | 90.8 → 90.3 | 12.8 → 15.6 [13.3, 17.8] |
| seed 2 → fine-tuned | 86.2 → 89.4 | 99.1 → 99.2 | 83.7 → 82.6 | 5.4 → 6.7 [5.3, 8.3] |

The goalie barely moves (training goal rate vs the goalie kind stayed at 37–49% over the last 60 updates), so more goalie exposure on top of a
converged policy does not find the manoeuvre seed 0 found. Not tried: training from scratch with a goalie-heavy
early phase, or a different policy class (recurrent / larger).

## What did not work / open problems
- **The goalie is a seed lottery.** Seed 0 learns to beat the exact training goalie (training rate ~90%, held-out 89%).
  Seeds 1 and 2 plateau at ~50% *training* goal rate vs the goalie kind and score 13% / 5% vs the nominal goalie on
  held-out starts (`media/r6_training_curves.png`). The pilots were low too (28%, 13%). The goalie guards on the
  ball–goal line and rushes inside 0.45 m. Beating it seems to need a specific manoeuvre that 1 of 7 population runs
  found (5 final-recipe seeds + 2 pilots; seed 4's 22% is the next best); round 4, trained only against it, found it. Raising the goalie's mix weight at the
  freeze (.08 → .14) did not make it reliable. Not tried: a goalie fine-tune phase, prioritized sampling of opponents
  the learner currently loses to, longer runs.
- **Seed 1 vs the interceptor.** In its timeouts (41% of matches, dev seeds) the ball ends near the halfway line (mean
  x ≈ −0.03). The robot stays on the ball (mean 0.16 m away) and the interceptor stays far off (0.67 m): it keeps
  possession at midfield and never attacks the goal, and the interceptor only engages in its own half. Seed 0's
  timeouts end further up the pitch (x ≈ 0.24). Hypothesis, untested: seed 1 learned to wait for defenders to come
  out (the goalie, presser, chaser and self-play strikers all do), and the interceptor never does.
- **Learned defender (pilot 2)** collapsed the attacker into "win the race, then park" (see above). Mechanism not established.
- **Throughput** under load: ~58–60 k env steps/s per job for seeds 0/1 run together (seeds 2–4: 72–80 k) (512 envs, 128×128 MLPs, MLX on
  CPU for rollouts). Idle throughput not measured.

## Not measured
Cross-play with unfamiliar *teammates* (1-v-1 only). More than 5 seeds. Elo for seeds 3–4. Ablations: pool without self-play, pool
without the chaser families, attacker-only vs learned-defender at matched seeds beyond the 15 M-step diagnostics. A
held-out *striker* style (all three new opponents defend; only the presser counter-attacks).
