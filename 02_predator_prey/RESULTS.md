# 02 · Predator–Prey Arms Race: results

Main training run (seed 0): 710 self-play PPO iterations, 54.5 M agent-steps (both sides together), 1082 s
of wall time, MLX on CPU, under load (`uptime` 1-minute load 7.8–16.8 during the run, 5-minute ≈ 12). Settings are in
`results/train_config.json`. All evaluation numbers come from `evaluate.py` with the held-out seed 12345
and 500 episodes per matchup (`results/eval.json`, `results/eval_tables.md`). Policies sample from their
action distribution unless the row says `argmax`. ± is one binomial standard error.

A first attempt at the main run was killed from outside at iteration 230, after 8 min (exit code 144, cause not confirmed). Later, when I stopped my own seed-1 job with `pkill -f`, the job
reported the same exit code 144. So a `pkill -f train.py` from another process on this shared machine is a
plausible cause. At that point `train.py` only saved at the end, so nothing was lost except time. That
run's logs are kept in `results/run1_killed/`. The script now checkpoints every 10 iterations and stops
cleanly on SIGTERM/SIGINT/SIGHUP. Run 2 used the same seed, and its in-training
eval matches run 1 exactly up to iteration 230 (`train_eval.csv` rows are identical). So training is
deterministic for a fixed seed on CPU. That is a reproducibility check, not a second seed.

## Headline

> **Reviewer correction (2026-10-22).** The original `lead` baseline was weak. A scripted
> "herding" variant (aim at a point 0.6×min(dist,1) on the arena-centre side of the evader's predicted
> position, sprint only inside 0.35; tuned on seed 777, not the held-out seed) catches smartflee
> **0.932 ± 0.011** on the held-out seed, against 0.966 for the learned pursuer, at the same energy per
> capture (10.4 vs 10.2). It also catches naive flee 0.978, where the learned pursuer gets 0.718. So the
> learned **pursuer only roughly matches a tuned scripted pursuer**. The **evader result gets stronger**:
> the herding pursuer catches the learned evader only 0.022 ± 0.007 (seed-1 evader: 0.024), while smartflee
> escapes it only 0.068. Source: `review/stronger_pursuers.json`.

| matchup (pursuer vs evader) | capture rate | source |
|---|---|---|
| **learned P vs smartflee** (hardest scripted evader) | **0.966 ± 0.008** | eval_tables.md, row learned |
| original scripted P (lead) vs smartflee (smartflee escapes 0.824) | 0.176 ± 0.017 | row lead |
| **reviewer's herding lead vs smartflee** (tuned on seed 777) | **0.932 ± 0.011** | review/stronger_pursuers.json |
| herding lead vs learned E (learned evader escapes 0.978) | 0.022 ± 0.007 | review/stronger_pursuers.json |
| herding lead vs naive flee | 0.978 ± 0.007 | review/stronger_pursuers.json |
| greedy vs smartflee | 0.034 ± 0.008 | row greedy |
| **greedy vs learned E** (learned evader's escape rate 0.992) | **0.008 ± 0.004** | column learned |
| **lead vs learned E** (escape 0.964) | **0.036 ± 0.008** | column learned |
| learned P vs learned E (final self-play) | 0.090 ± 0.013 | |
| **learned P vs naive flee**: transfer failure, see below | **0.718 ± 0.020** | |
| greedy vs naive flee | 0.998 ± 0.002 | |

Neither side ever trained against a scripted opponent, so the whole suite is held out. On the hardest
scripted opponent for each role, both learned policies beat the builder's scripted baselines by a wide margin, but
only the evader's margin survives a stronger scripted pursuer (see the correction above): against smartflee the
learned pursuer beats the herding pursuer by about 3 points (≈2 SE), and that herding pursuer had 4 parameters tuned
against smartflee itself, which the learned pursuer never saw. The comparison is matched on
everything except training: same simulator, same start states (seed 12345), same 500 episodes. The
scripted baselines have no training budget by design (cheap baseline rule). The one striking hole is
the naive `flee` evader, explained below.

## Measure items

### Capture and escape rates against a fixed opponent suite
The full 5×5 matrix is in `results/eval_tables.md`. Learned pursuer: 1.000 vs random, 0.718 vs flee,
0.966 vs smartflee. Learned evader, as escape rates: 1.000 vs random, 0.992 vs greedy, 0.964 vs lead.
Argmax versions behave about the same, except that argmax-P vs flee is worse (0.550), and argmax-E vs
the learned P escapes 0.890 against 0.910 for sampled.

### Energy efficiency
The energy proxy is effort per second = thrust-seconds + sprint-seconds per second of episode (max 2.0).
"Energy per capture" is the pursuer's total effort across all 500 episodes divided by its captures. In
`eval_tables.md`, cells with zero captures (the random pursuer) show total effort, not a ratio.

| pursuer vs smartflee | capture | P effort/s | P energy per capture |
|---|---|---|---|
| learned | 0.966 | 1.39 | **10.2** |
| lead (scripted) | 0.176 | 1.50 | 117.8 |
| herding lead (reviewer, scripted) | 0.932 | not measured | 10.4 |
| greedy (scripted) | 0.034 | 1.49 | 645.4 |

- ~~11× less energy per capture than the best scripted pursuer~~ (reviewer): this ratio was almost entirely
  the capture-rate gap. Every failed episode adds 15 s of effort to the numerator, so energy per capture
  tracks 1/capture-rate. Against the herding pursuer it is parity (10.2 vs 10.4). Effort per second is
  slightly lower for the learned pursuer than for lead (1.39 vs 1.50).
- Against easy prey it is less efficient. Versus naive flee: 14.1 per capture against greedy's 5.2.
- The learned evader escapes cheaply. Against the learned pursuer it spends 1.14 effort/s, and its
  argmax version only 0.70–0.81 effort/s. Smartflee spends 1.42–1.46 effort/s against lead or greedy,
  and still escapes less often.
- Over training, the effort per tick of the current-vs-current batch fell from 1.36 to 1.18 (pursuer)
  and from 1.35 to 1.16 (evader). Source: `train_log.csv`, columns P_effort/E_effort at iterations 1
  and 710. Both sides learned to coast.

### Cross-play over archived checkpoints
Capture rate of archived pursuers (rows) against archived evaders (columns), 200 episodes per cell
(`media/crossplay.png`):

| P \ E | 0 | 100 | 200 | 300 | 410 | 510 | 610 | 710 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 1.00 | 0.98 | 0.67 | 0.28 | 0.16 | 0.04 | 0.05 | 0.03 |
| 300 | 1.00 | 0.97 | 0.97 | 0.61 | 0.33 | 0.14 | 0.10 | 0.05 |
| 510 | 1.00 | 0.99 | 0.98 | 0.87 | 0.68 | 0.18 | 0.08 | 0.04 |
| 710 | 1.00 | 0.99 | 0.99 | 0.98 | 0.95 | 0.53 | 0.23 | 0.10 |

(The rows for 200, 410 and 610 are in `eval_tables.md`.)

- **The arms race is mostly transitive, not cyclic.** Every row rises from right to left, and every
  column rises from top to bottom, with only a few small inversions (for example P410 vs E510 at 0.12
  against P300 vs E510 at 0.14). Each new pursuer beats every older evader, and each new evader beats
  every older pursuer. There is no rock-paper-scissors in this run.
- **The evader is winning the race.** The diagonal (same-age matchups) falls from 0.98 (iter 100) to
  0.10 (iter 710). In the training batches, current-vs-current capture rate goes from about 0.96 near
  iteration 100 to about 0.08 at the end (`media/arms_race_curves.png`, right panel). The pursuer still
  improves: P710 beats E410 0.95 of the time, while P410 managed only 0.57 against it. But it can't keep
  up.
- **Competence does transfer to non-training opponents**, mostly. The table "Each snapshot vs the
  scripted suite" shows that the learned evader's escape rate against greedy and lead climbs almost
  monotonically (against lead at iterations 0/100/200/300/410/510/610/710: 0.00, 0.31, 0.39, 0.57, 0.80, 0.93, 0.92, 0.99). The pursuer against smartflee jumps
  to 0.92 by iteration 100 and stays between 0.89 and 0.98.

## Seed 1 replicate (matched: 710 iterations, 54.5 M agent-steps)
`checkpoints/seed1/`, `results/seed1/eval_tables.md` and `results/seed1/eval.json`, using the same held-out
eval (seed 12345, 500 episodes). The machine was quieter for this run, so it took 407 s (1-minute load about
4–7) against 1082 s for seed 0. The iteration and sample counts are the same.

| matchup | seed 0 | seed 1 |
|---|---|---|
| learned P vs smartflee | 0.966 ± 0.008 | 0.940 ± 0.011 |
| learned P vs naive flee (the hole) | 0.718 ± 0.020 | 0.846 ± 0.016 |
| greedy vs learned E | 0.008 ± 0.004 | 0.014 ± 0.005 |
| lead vs learned E | 0.036 ± 0.008 | 0.030 ± 0.008 |
| learned P vs learned E (final self-play) | 0.090 ± 0.013 | 0.074 ± 0.012 |
| learned P energy per capture vs smartflee | 10.2 | 11.6 |
| reviewer's herding lead vs learned E | 0.022 ± 0.007 | 0.024 ± 0.007 |
| cross-play diagonal, iteration 100 → 710 | 0.98 → 0.10 | 0.97 → 0.07 |
| evader near wall / in corner (fraction of ticks) | 0.63 / 0.032 | 0.56 / 0.016 |

All the seed-0 findings hold in 2 of 2 seeds:
- the evader wins the arms race, and the cross-play matrix is near-transitive (no cycles);
- both learned sides beat the builder's scripted baselines on the hardest scripted opponent (only the evader
  also beats the reviewer's herding pursuer clearly);
- the learned pursuer has a hole against naive flee, smaller in seed 1;
- the learned evader hugs walls and avoids corners.

Seed 1's capture rate against flee also swings across snapshots (0.85 at iteration 300, 0.73 at 510,
0.86 at 710).

## The good failures (clips in `media/`)

1. **Pursuer stalls at the corner against the dumbest evader** (`failure_learned_P_vs_naive_flee.mp4/.gif`).
   The naive `flee` evader runs straight away from the pursuer and ends up pinned in a corner. In the
   29.5% of episodes the learned pursuer fails to finish (200-episode diagnostic run), it drives up to
   that corner and then parks about 0.30 away, just outside the 0.20 capture radius. Its mean speed is
   0.14 over the last 3 s, and it stays there until time runs out. In every one of those episodes the
   evader is in a corner (|x|, |y| ≈ 1.92). Greedy catches the same evader 99.8% of the time.
   Explanation: the learned evader almost never sits in a corner (3.2% of ticks, against 6.25% for
   uniform area use), so the pursuer never trained on a cornered, motionless target. This state is out
   of distribution for it. Capture rate against flee is also not monotone over snapshots: 0.42, 0.59,
   0.78, 0.89, 1.00, 0.93, 0.73. The pursuer forgot how to finish easy prey while specializing against
   its real rival. It is a small, concrete case of "competence against the latest rival is not general
   competence".
2. **Wall-hugging, standoffs and obstacle orbiting** (`selfplay_standoff_and_orbit.mp4/.gif`, and the
   top-left tile of `matches.mp4`). The final evader spends 63% of ticks within 0.35 of a wall, against
   32% for uniform use and 17% for smartflee (`media/occupancy.png`). It slides along the walls but
   avoids corners. The pursuer often parks next to the small obstacle near the bottom wall, which looks
   like a standoff. When it commits, the evader leads it to an obstacle and circles it. The evader
   accelerates 6.0 against the pursuer's 2.5, so it can hold a tighter orbit, and the pursuer can't cut
   inside. Evader policy entropy rose late in training (1.37 → 1.76 nats between iterations ~100 and
   710), so the evader became more random, not less. A mixed strategy is the natural answer to a
   pursuer that predicts.

## What worked
- A tiny numpy sim (about 0.6 s for 9 scripted matchups × 400 envs × 300 ticks) plus 64-unit MLX MLPs
  was plenty. Learning was visible within 10 iterations.
- Three-batch self-play (current/current, current vs archive, archive vs current) with uniform archive
  sampling. Cross-play shows no forgetting of older opponents: P710 still beats E0–E300 at 0.98 or
  better. The forgetting that did show up was against an opponent type that was never in the pool
  (finding 1).
- Holding the scripted suite out of training made the transfer question answerable.

## What didn't / caveats
- **The scripted pursuer baseline was under-built** (reviewer). `lead` had no herding: smartflee slides
  tangentially toward the centre when cornered, and a straight interceptor follows it out. Better sprint timing
  alone does nothing (sprint only inside 0.35: 0.186); aiming at the centre side of the evader does (0.932).
- **Two seeds only**, at one balance setting. Whether the evader wins may depend on the balance.
  The parameters were tuned once, before training, using scripted-vs-scripted matches only: pursuer
  acceleration 3.0 → 2.5, evader cruise 0.85 → 0.90 and sprint 1.30 → 1.35. The reason was that a
  60-second smoke run had the pursuer winning self-play almost 100% of the time.
- The final pursuer vs the final evader (0.09) is roughly a stalemate in the evader's favour, so at
  this balance the game may drift toward "evader always wins" with more training.
- Speed numbers are under load. Under the load at the time, MLX on CPU did about 1.5 s per iteration
  across the full run. In the 3-iteration smoke, the shared GPU took about 4.5 s per iteration against
  about 0.9 s on CPU. The network is tiny and its batches are 128–256 rows, so launch overhead
  dominates. Not measured on an idle machine.
- Limited perception, recurrent policies, teams and refuges were not built (Tinker / Go deeper).
  Both agents see everything, including the opponent's stamina.
- Archived opponents are swapped every iteration (300 ticks), so B/C batches see K=2 frozen snapshots
  per iteration.

## Surprises
- The pursuer has the higher top speed, and it was clearly the stronger side for the first ~250
  iterations. The evader still ended up winning, by playing the geometry: walls, obstacle orbits,
  stamina.
- The pursuer's worst scripted matchup is the *simplest* evader.
- The cross-play matrix is close to perfectly transitive. I expected cycles; the uniform opponent
  archive may be what suppresses them.

## Next steps
1. Add scripted or cornered-start opponents to the archive (or randomized start states that pin the
   evader), and check that the corner stall disappears without hurting self-play.
2. Run more seeds, plus one rebalanced setting (for example pursuer acceleration back to 3.0), to see
   whether the evader always wins the race.
3. Limited perception (view cone or occlusion by obstacles) and a GRU policy, which is where hiding
   behind obstacles should start to pay.
4. Recency-weighted vs uniform archive sampling, measured by cross-play cycles.

## Files behind each number
- Suite matrix, energy, time to capture, cross-play, snapshot transfer, occupancy:
  `results/eval.json` and `results/eval_tables.md`
- Training curves: `results/train_log.csv` (training batches) and `results/train_eval.csv`
  (in-training check against the scripted suite: 128 episodes, seed 999, different from the final eval)
- Figures: `media/crossplay.png`, `media/arms_race_curves.png`, `media/occupancy.png`
- Clips: `media/matches.mp4/.gif` (2×2: learned vs learned, learned P vs smartflee, lead vs learned E,
  P@70 vs final E), `media/selfplay_standoff_and_orbit.*`, `media/failure_learned_P_vs_naive_flee.*`
  (left tile: learned P stalls at the corner; right tile: greedy catches the same kind of evader)
- Checkpoints: `checkpoints/final.npz` (iteration 710), `checkpoints/archive.npz` (72 snapshots)
