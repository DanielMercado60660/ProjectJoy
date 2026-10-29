# RESULTS: 05 · The Dungeon That Hates You

All numbers below come from real runs on 2026-10-22 and are produced by `pipeline.sh` (round 1),
`pipeline2.sh` (rounds 2-3 and ablations) and `pipeline3.sh` (11:30-12:00 follow-up: five more
training-seed chains for curriculum rounds 1-3). The full tables are in `results/eval_tables.md`
(round 1), `results/eval_r2_tables.md` (round 2+) and `results/eval_seeds_tables.md` (all 6 seed
chains × 3 rounds × 2 arms). Seeds are fixed throughout. Learned solvers
are evaluated with **sampled** actions: 4 rollouts per level, 500 levels per family. Each table
also has a greedy column for the first solver. Standard errors are binomial (± s.e. over episodes).

## TL;DR

* **The learned navigator works under partial observation and transfers to held-out maze
  families.** PPO + GRU sees a 5×5 window and trains 12M steps (≈2 min of GPU) on dfs, braid and
  rooms. It escapes 96.6% / 95.0% / 99.4% of test levels in those families, and **99.8% (caves)
  and 93.6% (spiral) in the two families it never saw**. Its path efficiency on successful
  episodes is 0.56-0.63, about the same as the scripted explorers (0.48-0.66).
* **Baseline comparison (same 5×5 view, same 200-step limit):** Trémaux (depth-first exploration
  with perfect memory) escapes 100% everywhere and is the stronger solver on success. The learned
  agent has a better SPL than Trémaux in open layouts (rooms 0.63 vs 0.48, caves 0.61 vs 0.48)
  and a worse one in corridors (dfs 0.61 vs 0.64, spiral 0.52 vs 0.59). The wall follower falls
  apart where there are loops (caves 37.9%, rooms 59.7%). The random walk escapes 11.8% of dfs
  levels.
* **The designer breaks every learned solver in about 5 generations, and never cheats:** every
  one of its levels is solvable by construction. Starting from family levels where the solver
  fails 0.5-2.4% of the time, mean failure in the population passes 50% by generation 4-6. After
  100 generations it is **84-87% on fresh rollouts with unseen seeds** in the round-1 runs (73-92% across all 15
  constrained runs against learned solvers; designer seed alone moves it by ~12 points: `v1_s0`/`v1_s1` 0.85/0.87 vs
  `expl_v1_s8` 0.73). The same budget with
  selection switched off (neutral drift) stays at 2.5%, so the hardness comes from selection
  pressure and not from mutation noise.
* **Without the solvability constraint the designer just walls off the exit.** Only 4.7% of its
  final population is still solvable, and only 12% of its raw proposals were.
* **The designer's dungeons are adversarial examples for one policy, not generally hard
  dungeons.** Levels evolved against solver_v1 stop v1 (12.5% success), but a copy of v1
  fine-tuned 6M more steps on the same families escapes 75% of them. The random walk escapes
  86% and Trémaux 100%.
* **The curriculum helps against unseen designer levels aimed at the same target, but the
  original single-seed headline overstated the gap, and it does not make the solver hard to break.**
  * **With 6 training seeds per arm (seed-0 chain plus 5 new chains), after 3 rounds the
    curriculum arm escapes 0.86 (0.66-0.94) of the held-out `v1_s1` designer levels, against
    0.39 (0.23-0.77) for the control.** The gap is +0.47 (95% t-interval over the 6 seed pairs:
    +0.15 to +0.79). The curriculum wins in 5 of 6 seeds. In seed 13 the control wins, 0.77 vs 0.66.
  * **The old headline, 0.94 vs 0.23, is the seed-0 chain, whose +0.70 gap is the largest of the
    six.** Read the effect as "about +0.5, varying from -0.1 to +0.7 with the
    seed", not as +0.7.
  * **Round 1 is weaker than it looked.** With 3 seeds every curriculum seed beat every control
    seed. With 6 seeds it is 0.80 (0.69-0.93) vs 0.63 (0.20-0.83), and the curriculum wins only
    4 of 6. The interval for the gap includes zero.
  * **Rounds 2 and 3 are where the curriculum's advantage is consistent.** In round 2 it wins
    6 of 6 seeds (+0.31).
  * **The round-2 control spike did not recur.** Seed 0's 0.82 is the highest of 6 round-2
    controls; the other five scored 0.34-0.65. The control is still very noisy across rounds
    within a seed: seed 13 went 0.83 → 0.50 → 0.77. Its mean drifts down, 0.63 → 0.55 → 0.39.
  * On the 12-step-minimum designer run (`v1_md12_s7`), round 3 is 0.72 (0.31-0.97) vs
    0.27 (0.16-0.60): 6 of 6 wins, but two seeds win only by +0.03 and +0.12.
  * On the ctl1 limit-cycle trap set, neither arm is better (0.59 vs 0.49, 4 of 6).
  * Matching: fine-tune steps, seeds and hyperparameters match between arms. The designer runs
    that built the curriculum archive are extra compute the control never gets.
  * These evaluation levels share their target policy (v1) with the training archive, so they are
    close to in-distribution for the curriculum arm (see the caveat in the round-2 section).
  * The cost is about 2 points on the held-out spiral family (0.961 vs 0.981 at round 3) and
    about 1 point on dfs and braid.
  * **A fresh designer still breaks every solver, curriculum or not.** On the original solvers it
    took 3-13 generations. The 10 new designer runs against the new chains' curriculum solvers
    reached 50% failure in 4-10 generations, and 62-93% final failure.
* **Memory matters less than expected.** The same PPO with the GRU state zeroed each step
  reaches 0.954 on the train families (GRU: 0.970) and is just as good on the held-out families
  (one training seed each, so training-seed variance on this 1.6-point gap is unmeasured).
  Reactive stochastic exploration does most of the work at this size.
* **The best failure: a limit-cycle trap.** The designer converged on a spawn pocket where the
  fine-tuned solver dances S, N, bump-W, E, S, … for 200 steps, with the exit visible two cells
  away. It is in `media/nightmares.png` and `media/rogues_gallery.png`, and you can replay it in
  `media/demo.html`.

Demos: `media/rogues_gallery.png`, `media/nightmares.png`, `media/escape.gif`,
`media/before_after.gif`, `media/curriculum.png`, `media/curriculum_rounds.png`, `media/curriculum_rounds_seeds.png`,
`media/designer_evolution.png`, `media/training_curves.png`, and `media/demo.html`. The demo is
an offline dungeon editor with a BFS validity check. It replays the GRU policy (exported weights;
the JS port matches the Python greedy trajectories 20/20 on trained policies), the memoryless
ablation, and the three scripted solvers. Checkpoints are in `checkpoints/solver_*/`.

## Setup in one paragraph

Levels are 13×13 grids with the border always wall. Start and exit are on floor, at least 6 BFS
steps apart. Every level of every family is BFS-verified, and `tests/test_core.py` asserts this.
Observations are a 5×5 wall window, a 5×5 exit window, the previous action and a bump flag.
Actions are N/E/S/W. Reward is `1 - 0.9·t/200` on escape, plus a 0.01 bonus for each new cell
visited during training only. The solver is MLP(55→128→128) → GRUCell(128) → tanh-MLP
policy/value heads, trained with PPO (512 envs × 64-step rollouts, BPTT over each 64-step
chunk, 4 epochs, 2 minibatches, lr 3e-4, γ 0.99, λ 0.95, clip 0.2, entropy 0.01). The designer
is a (16 elite + 112 tournament children) evolution over 128 levels for 100 generations. Its
mutations flip 1-6 interior cells or move the start or exit. Fitness is the regret against BFS
in env-return units, averaged over 8 sampled solver rollouts per level per generation:
`R(BFS) - R(solver)`, where `R = 1 - 0.9 t/T` on escape and 0 on failure. A failure scores
≈0.9. Every child is BFS-checked: start-to-exit distance must be ≥ 6, and up to 20 redraws are
allowed, otherwise the parent is copied.

## Measure 1: success on held-out maze families (round 1)

Success rate / SPL (SPL = success × BFS/steps). 500 levels × 4 rollouts per family.

| family | solver_v1 (PPO-GRU) | v1 greedy | Trémaux | wall follower | random walk | BFS oracle |
|---|---|---|---|---|---|---|
| dfs (train family) | 0.966 / 0.61 | 0.966 / 0.61 | 1.000 / 0.64 | 1.000 / 0.64 | 0.118 / 0.03 | 1 / 1 |
| braid (train family) | 0.950 / 0.59 | 0.932 / 0.58 | 1.000 / 0.60 | 0.782 / 0.51 | 0.285 / 0.07 | 1 / 1 |
| rooms (train family) | 0.994 / 0.63 | 0.990 / 0.62 | 1.000 / 0.48 | 0.597 / 0.38 | 0.617 / 0.15 | 1 / 1 |
| **caves (held out)** | **0.998 / 0.61** | 0.998 / 0.61 | 1.000 / 0.48 | 0.379 / 0.24 | 0.929 / 0.28 | 1 / 1 |
| **spiral (held out)** | **0.936 / 0.52** | 0.924 / 0.52 | 1.000 / 0.59 | 0.922 / 0.55 | 0.229 / 0.06 | 1 / 1 |

Standard errors are ≤ 0.011 everywhere. The train-family rows use fresh test levels (seeds
9000+), not training levels (seeds 1000+). Training curve: `media/training_curves.png`.
Training success first passed 90% at 1.6M env steps.

## Measure 2: path efficiency (BFS length / steps, successful episodes only)

| family | solver_v1 | Trémaux | wall follower | random walk |
|---|---|---|---|---|
| dfs | 0.632 | 0.642 | 0.638 | 0.257 |
| braid | 0.617 | 0.595 | 0.657 | 0.236 |
| rooms | 0.629 | 0.483 | 0.644 | 0.245 |
| caves | 0.614 | 0.485 | 0.638 | 0.303 |
| spiral | 0.559 | 0.586 | 0.601 | 0.246 |

With a 5×5 view and no map, roughly 0.6 of BFS-optimal is what a sensible explorer achieves in
13×13 dungeons. The learned agent sits right in that band.

## Measure 3: proportion of generated levels that are valid

| generator | valid |
|---|---|
| maze families (all 5) | 100% (asserted by `make_level` and `tests/test_core.py`) |
| designer raw mutation proposals, constrained runs | 69-87% of proposals with the 6-step floor (e.g. v1_s0: 11,200 of 15,076 = 74.3%); 62% with the 12-step floor (`v1_md12_s7`) |
| designer output (population + archive), constrained | **100%**: an invalid proposal is redrawn, never kept |
| designer **without** the constraint (30 gens vs v1) | 12.1% of proposals valid; **4.7% of the final population solvable** |

"Valid" means the start and exit are on floor, the exit is reachable, and the BFS distance is
≥ 6. Without the constraint the designer learns, in 3 generations, that walling off the exit is
the cheapest way to get 0.95 "failure".

## The designer: how its dungeons differ

Structure of the designer's final populations next to the families. These are means;
`media/designer_evolution.png` shows them over generations.

| level set | floor | BFS path | dead ends | junctions | loops | open 2×2 |
|---|---|---|---|---|---|---|
| dfs / braid / rooms (train) | 71 / 74 / 46 | 30.5 / 22.8 / 10.8 | 5.4 / 2.5 / 0.0 | 3.4 / 5.7 / 26 | 0 / 2.6 / 17 | 0 / 0 / 17 |
| caves / spiral (held out) | 54 / 73 | 8.0 / 26.0 | 0.8 / 4.5 | 41 / 7.5 | 35 / 2.8 | 35 / 2.2 |
| designer vs v1 (seed 1) | 63 | **6.3** | **16.3** | 13.8 | 4.5 | 4.1 |
| designer vs ctl1 | 69 | 6.2 | 15.1 | 21.7 | 10.2 | 9.8 |
| designer vs wall follower | 63 | 6.2 | 13.9 | 24.7 | 14.3 | 13.7 |
| designer vs Trémaux | 82 | 8.3 | 9.9 | 39.5 | 21.9 | 20.0 |
| neutral drift (no selection) | 65 | 10.0 | 16.1 | 19.3 | 8.4 | 8.0 |

* **Exit near, path hidden, many dead ends.** Against the learned solvers the designer drives the
  BFS path down to the 6-step minimum it is allowed. It then hides the exit just outside the
  view, or behind a wall, and surrounds the start with dead-end side corridors (≈16 dead ends,
  versus 2-5 in the families). The solver heads off into the maze and never comes back within
  200 steps.
* **Each solver gets a different nightmare** (`media/nightmares.png`):
  * Against the wall follower, the designer puts the start next to a free-standing 2×2 pillar,
    so the left-hand rule circles it forever. Failure is 94% after 100 generations and already
    50% at generation 2.
  * Against Trémaux it never produced a single failure in 100 generations, because DFS with
    memory always finishes inside 200 steps on 13×13. The best it could do was inflate the
    path: regret 0.32, with open, loopy layouts (39 junctions, 22 loops).
  * Against the PPO solvers it found policy-specific traps; see the limit cycle below.
* **Designer levels are specific to the policy they were evolved against.** Success on each
  designer set (round 1, 500 levels × 4 rollouts):

| designer run evolved against → | v1 (seed 1) | ctl1 | cur1 | wall follower | Trémaux |
|---|---|---|---|---|---|
| solver_v1 | **0.125** | 0.143 | 0.842 | 0.998 | 0.984 |
| solver_ctl1 (v1 + 6M families) | 0.754 | **0.119** | 0.951 | 0.992 | 0.977 |
| solver_cur1 (v1 + 6M families+designer) | 0.898 | 0.064 | **0.166** | 0.988 | 0.951 |
| Trémaux | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| wall follower | 0.977 | 1.000 | 0.811 | **0.059** | 0.717 |
| random walk | 0.859 | 0.996 | 0.945 | 0.869 | 0.555 |

The diagonal is the designer's target. Off the diagonal, the levels are mostly easy. The
exception is the ctl1 trap, which also catches v1 and cur1: all three share v1's weights as a
starting point.

## Before / after the curriculum (round 1, matched compute)

Both arms start from solver_v1 and train another 6M env steps with fresh Adam state, the same
hyperparameters and the same seed. The only difference is the level mix: the **curriculum** arm
draws 50% of its levels from the archive of designer run `v1_s0` (8,729 levels that reached
regret ≥ 0.5). The **control** arm draws 100% from the families. Evaluation uses different
designer runs (`v1_s1`, `cur1_s2`, `ctl1_s2`) from the one that produced the training archive.

| level set | v1 | control (ctl1) | curriculum (cur1) |
|---|---|---|---|
| train families (mean of 3) | 0.970 | 0.988 | 0.981 |
| held-out families (caves, spiral) | 0.998, 0.936 | 1.000, 0.979 | 0.999, 0.940 |
| designer levels vs v1 (held-out run `v1_s1`) | 0.125 | 0.754 | **0.898** |
| fresh designer vs this solver: gens to 50% failure | 6 (v1_s0), 5 (v1_s1) | 6 | 4 |
| fresh designer vs this solver: final failure (fresh seeds) | 0.85-0.87 | 0.87 | 0.84 |

The curriculum transfers to *new* designer levels aimed at the same target policy: 0.898 vs
0.754. That is seed 0 on the final population only. With 6 seeds on the population plus archive
set it is 0.80 vs 0.63, and the curriculum wins only 4 of 6 (see the seed replication below). It cost a little on the held-out spiral family (0.940 vs 0.979). It did **not** make the
solver harder to break: a fresh designer run against cur1 reached 50% failure even faster. Plot:
`media/curriculum.png`. Round 2+ and seed replicates are below.

## Round 2+: seed replicates, three curriculum rounds, ablations (`pipeline2.sh`)

`results/eval_r2_tables.md` covers 12 learned solvers on 12 level sets. Each designer level set here
is that run's 128-level final population **plus 372 levels sampled from its archive**, which spans
all generations and so has more than one motif. That is why the numbers differ from the
final-population-only tables in round 1. The table shows success rates. For the three seeds of
round 1, it gives the mean and (min-max). **The round-1 rows here are the first 3 seeds (0, 11,
12), and rounds 2 and 3 are the seed-0 chain only. The 6-seed replication below supersedes all of
these curriculum and control rows.**

| solver | train families (mean) | caves* | spiral* | exit-in-view probe | designer vs v1 (`v1_s1`) | designer vs v1, exit ≥ 12 (`v1_md12_s7`) | ctl1 trap set (`ctl1_s2`) |
|---|---|---|---|---|---|---|---|
| v1 | 0.970 | 0.998 | 0.936 | 0.957 | 0.086 | 0.070 | 0.340 |
| control, round 1 (3 seeds) | 0.985 (0.98-0.99) | 0.999 | **0.969** (0.96-0.98) | 0.976 | 0.479 (**0.20-0.72**) | 0.304 (0.28-0.33) | 0.268 (0.15-0.40) |
| curriculum, round 1 (3 seeds) | 0.983 (0.98-0.99) | 0.999 | 0.946 (0.94-0.96) | 0.963 | **0.803** (0.77-0.84) | **0.777** (0.74-0.81) | 0.230 (0.15-0.28) |
| control, round 2 | 0.991 | 1.000 | 0.986 | 0.990 | 0.821 | 0.339 | 0.427 |
| curriculum, round 2 | 0.983 | 0.998 | 0.957 | 0.968 | **0.938** | **0.729** | 0.407 |
| control, round 3 | 0.992 | 1.000 | 0.988 | 0.990 | 0.231 | 0.162 | 0.529 |
| curriculum, round 3 | 0.991 | 0.999 | 0.969 | 0.979 | **0.935** | **0.939** | 0.435 |
| memoryless PPO (ablation) | 0.954 | 0.999 | 0.927 | 0.958 | 0.140 | 0.214 | 0.442 |
| Trémaux / wall follower / random | 1.0 / 0.79 / 0.34 | 1.0 / 0.38 / 0.93 | 1.0 / 0.92 / 0.23 | 1.0 / 0.98 / 0.66 | 1.0 / 0.91 / 0.81 | 1.0 / 0.91 / 0.74 | 1.0 / 0.83 / 0.87 |

\* held-out family. The curriculum arm of round *k* fine-tunes from round *k-1*'s curriculum
solver. Its level mix is 50% families and 50% the union of archives from designer runs
`v1_s0`, then `cur1_s3`, then `cur2_s5` (8.7k → 17.1k → 26.1k levels). The control arm fine-tunes
from the previous control with 100% family levels, using the same steps, seeds and
hyperparameters. None of the evaluation designer runs (`v1_s1`, `v1_md12_s7`, `ctl1_s2`) ever
fed a training archive.

**Caveat:** `v1_s1` and `v1_md12_s7` share their designer target (solver_v1) with the round-1
training archive (`v1_s0`). They differ only in seed, and in the constraint for md12. So they
are close to in-distribution for the curriculum arm and out-of-distribution for the control.
The real out-of-distribution test is the exploitability runs below, and they show no gain.
Plot: `media/curriculum_rounds.png` (seed-0 chain only) and `media/curriculum_rounds_seeds.png`
(all 6 seed chains).

### Seed replication of rounds 1-3 (`pipeline3.sh`, 6 seed chains per arm)

The review flagged that rounds 2-3 had one training seed per arm. `pipeline3.sh S` repeats
pipeline2's round-2/3 procedure for an independent seed chain S:

1. **Round 1.** `cur1_rS` and `ctl1_rS` are fine-tuned from v1. They already existed for S = 11
   and 12; they were trained here for S = 13, 14 and 15.
2. **Round 2.** A designer run (seed 3, 100 generations) attacks this chain's `cur1_rS`. Its
   archive is added to `v1_s0`'s to make `curriculum_r2_rS`. Then `cur2_rS` (from `cur1_rS`) and
   `ctl2_rS` (from `ctl1_rS`) are fine-tuned.
3. **Round 3.** A designer run (seed 5) attacks `cur2_rS`, giving `curriculum_r3_rS`. Then
   `cur3_rS` and `ctl3_rS` are fine-tuned.

So each chain's curriculum is bred against its own policies, exactly as in the seed-0 chain.

* Every fine-tune is **5,996,544 env steps**, checked in all 30 new `meta.json` files.
* The hyperparameters, the 50/50 level mix and the 512 envs are the same as in pipeline2.
* Both arms share the fine-tune seed in each round: S for round 1, 10·S + k for round k.
* The machine was idle, so each chain took 5-9 minutes. That left room for 5 new chains instead
  of the 2 the review asked for.

Evaluation is `evaluate.py --out eval_seeds --designer-runs v1_s1 v1_md12_s7 ctl1_s2 --n-arch 372`.
This is the same protocol as `eval_r2`: the same eval seed 4242, the same 500 levels per family,
K = 4, and the same archive sample. It re-evaluated the seed-0 solvers too. **All seed-0 and
round-1-replicate numbers reproduce `eval_r2.json` exactly** (0 mismatches).

**Success on held-out designer levels aimed at v1 (`v1_s1`; this is the headline claim):**

| round | arm | seed 0 | 11 | 12 | 13 | 14 | 15 | mean | range |
|---|---|---|---|---|---|---|---|---|---|
| 1 | curriculum | 0.841 | 0.800 | 0.769 | 0.932 | 0.754 | 0.690 | **0.798** | 0.69-0.93 |
| 1 | control | 0.717 | 0.518 | 0.203 | 0.834 | 0.814 | 0.721 | **0.634** | 0.20-0.83 |
| 1 | gap | +0.12 | +0.28 | +0.57 | +0.10 | −0.06 | −0.03 | **+0.16** [−0.08, +0.41] | wins 4/6 |
| 2 | curriculum | 0.938 | 0.906 | 0.903 | 0.684 | 0.947 | 0.735 | **0.852** | 0.68-0.95 |
| 2 | control | 0.821 | 0.476 | 0.479 | 0.498 | 0.650 | 0.344 | **0.545** | 0.34-0.82 |
| 2 | gap | +0.12 | +0.43 | +0.42 | +0.19 | +0.30 | +0.39 | **+0.31** [+0.17, +0.45] | wins 6/6 |
| 3 | curriculum | 0.935 | 0.929 | 0.790 | 0.662 | 0.879 | 0.931 | **0.855** | 0.66-0.94 |
| 3 | control | 0.231 | 0.320 | 0.374 | 0.770 | 0.360 | 0.258 | **0.386** | 0.23-0.77 |
| 3 | gap | +0.70 | +0.61 | +0.42 | −0.11 | +0.52 | +0.67 | **+0.47** [+0.15, +0.79] | wins 5/6 |

The brackets are 95% t-intervals over the 6 per-seed gaps. The pairing is only nominal: the two
arms of a chain share seeds but are independent runs. An unpaired Welch test gives p = 0.16, 0.005
and 0.001 for rounds 1, 2 and 3. Each cell is 500 levels × 4 rollouts, so the per-cell s.e. is
≤ 0.011. Nearly all of the spread is training-seed variance, not evaluation noise.

**The other level sets, as 6-seed mean (min-max):**

| level set | round | curriculum | control | gap | curriculum wins |
|---|---|---|---|---|---|
| designer vs v1, exit ≥ 12 (`v1_md12_s7`) | 1 | 0.679 (0.46-0.81) | 0.323 (0.24-0.51) | +0.36 [+0.14, +0.58] | 5/6 |
| | 2 | 0.817 (0.73-0.93) | 0.343 (0.19-0.45) | +0.47 [+0.35, +0.60] | 6/6 |
| | 3 | 0.718 (**0.31**-0.97) | 0.265 (0.16-0.60) | +0.45 [+0.09, +0.82] | 6/6 |
| ctl1 limit-cycle trap (`ctl1_s2`) | 1 | 0.239 (0.15-0.32) | 0.324 (0.15-0.44) | −0.09 [−0.22, +0.05] | 2/6 |
| | 2 | 0.425 (0.38-0.45) | 0.432 (0.32-0.53) | −0.01 | 3/6 |
| | 3 | 0.587 (0.38-0.78) | 0.487 (0.41-0.68) | +0.10 [−0.09, +0.30] | 4/6 |
| spiral (held-out family) | 1 | 0.955 (0.94-0.97) | 0.968 (0.96-0.98) | −0.013 | 1/6 |
| | 3 | 0.961 (0.89-0.98) | 0.981 (0.97-0.99) | −0.021 | 1/6 |
| dfs / braid (train families) | 3 | 0.977 / 0.979 | 0.989 / 0.989 | −0.012 / −0.011 | 0/6 |
| caves / rooms | 1-3 | 0.998-1.000 | 0.998-1.000 | tied | |

Per-seed values for every set and round are in `results/eval_seeds.json` / `_tables.md`.

What this shows (6 seeds per arm unless noted):

* **"Curriculum beats control after 3 rounds" holds across seeds, but the gap is smaller and
  much less certain than the headline.**
  * The 6-seed means are 0.86 vs 0.39 on `v1_s1` (+0.47), with the curriculum winning in 5 of 6
    seed chains.
  * The seed-0 chain (0.94 vs 0.23, +0.70) had the largest gap of the six.
  * One chain (seed 13) reverses it: control 0.77, curriculum 0.66.
  * On `v1_md12_s7` it is 6/6, but per-seed gaps range from +0.03 to +0.78.
* **The curriculum arm itself is not stable from round to round.**
  * Three chains lost ground on `v1_md12_s7` in round 3: seed 11 dropped from 0.76 to 0.31,
    seed 12 from 0.82 to 0.50, and seed 13 from 0.82 to 0.63. Seed 12 also dropped on `v1_s1`,
    from 0.90 to 0.79.
  * Each new round's designer archive targets the previous round's solver, not v1. So later
    rounds can move away from the v1-specific evaluation sets.
* **The round-2 control spike did not recur.**
  * Seed 0's round-2 control (0.82) is the highest of the six. The other five round-2 controls
    score 0.34-0.65.
  * The control is still noisy within a chain: seed 13 went 0.83 → 0.50 → 0.77, and seed 0 went
    0.72 → 0.82 → 0.23.
  * Across seeds the control's mean drifts down with more family-only fine-tuning:
    0.63 → 0.55 → 0.39. Plain fine-tuning changes *which* quirks the policy has; it does not
    remove them.
* **Round 1 alone is not a reliable win.** With 3 seeds, every curriculum seed beat every
  control seed. With 6 seeds it is 4 of 6, the mean gap is +0.16, and the interval includes zero.
  The advantage becomes consistent only from round 2 (6/6).
* **Cost:** the curriculum gives up about 1-2 points on the held-out spiral family against the
  control: 0.955 vs 0.968 in round 1, 0.961 vs 0.981 in round 3. One curriculum seed fell to
  0.892. It also gives up about 1 point on dfs and braid at round 3. Caves and rooms are tied.
* **It does not fix everything.** On the limit-cycle trap set, neither arm is reliably better at
  any round.
* **Exploitability did not improve.** A fresh designer (seed 8, 100 generations, same budget)
  was run against every solver:

| solver attacked | gen-0 failure | mean failure, gens 0-20 | gens to 50% failure | final failure (fresh seeds) |
|---|---|---|---|---|
| v1 | 0.028 | 0.536 | 5 | 0.732 |
| control r1 / r2 / r3 | 0.014 / 0.017 / 0.001 | 0.564 / 0.531 / 0.345 | 5 / 6 / 13 | 0.821 / 0.803 / 0.840 |
| curriculum r1 / r2 / r3 | 0.020 / 0.006 / 0.009 | 0.618 / 0.702 / 0.427 | 5 / 5 / 9 | 0.903 / 0.906 / 0.857 |
| memoryless | 0.058 | 0.587 | **3** | 0.732 |

Every solver falls to a fresh designer within 3-13 generations. The curriculum solvers are, if
anything, *slightly easier* to attack early on. A plausible reason is that their training mix
contains many near-exit levels, which makes them confident near exits in a way the next designer
can bend. The most-trained solvers (round 3 of either arm) took longest to break, which looks
like a training-budget effect rather than a curriculum effect, given a single designer seed.
Levels evolved against one round-3 solver are easy for the other: the round-3 control escapes
0.965 of the levels aimed at the round-3 curriculum solver, and the curriculum solver escapes
0.989 of the levels aimed at the control.

The seed replication adds more evidence of the same kind. The ten round-2/3 designer runs in the new
chains each attack a fresh curriculum solver (`cur1_rS` or `cur2_rS`), with designer seeds 3 and 5.
From a gen-0 failure of 0.6-2.0%:

* they reached 50% failure in **4-10 generations**;
* they ended at **62-93% failure on fresh rollouts** (seed-0 chain: 7 and 7 generations, 0.83 and
  0.91).

So curriculum solvers in every chain stay just as breakable. These runs attack only curriculum
solvers, and no new control solvers were attacked. The seed-8 table above is still the only direct
curriculum-vs-control exploitability comparison, and it has one designer seed.

**Memory ablation (GRU state zeroed every step, same 12M steps).** The memoryless policy still
sees the previous action and the bump flag, so it is a reactive stochastic explorer.
* It reaches 0.954 on the train families versus 0.970 for the GRU, with braid the hardest case
  (0.901 vs 0.950). Held-out: caves 0.999 vs 0.998, spiral 0.927 vs 0.936.
* Its path efficiency is the same (0.57-0.64).
* So in 13×13 dungeons with a 5×5 view and a 200-step limit, **most of the work is done by
  stochastic reactive exploration; recurrence adds a few points**, mainly in looped mazes where
  revisiting matters. It is also the easiest solver for the designer to break (50% failure by
  generation 3).

**Longer minimum path (`--min-dist 12`).** The designer still reached 92% failure against v1.
Its final population sits at the new floor (BFS 12.1 steps, 16.6 dead ends). So the 6-step
levels in round 1 are a preference, not a requirement: the designer can trap v1 at any distance
it is allowed.

## Preserved failures (the good kind)

1. **The limit-cycle trap (designer vs ctl1).** In 97% of the final population of designer run
   `ctl1_s2`, the exit is already inside the solver's 5×5 view at spawn. That happens in only
   3% of the training pool. Every top level shares the same local motif: the start sits in a
   3-cell T-shaped pocket and the exit is two cells away around a corner. The solver executes S,
   N, bump-W, E, S, N, … with near-zero entropy, for all 200 steps, in 32 of 32 attempts. The
   random walk escapes 99.6% of these levels at 0.98 efficiency. The rest of each map is
   irrelevant, which makes this a *local adversarial patch*.
   * It is **not** simply "exit in view at spawn". I built a probe set: the family test levels
     with the start moved so the exit is in view and still ≥ 6 BFS steps away (1,338 levels).
     The solvers escape 95-99% of it.
   * It is this particular wall configuration that puts the policy into a periodic orbit.
2. **The PPO run that learned nothing (a bug).** The first training runs stayed at chance (~10%
   success, entropy pinned at ln 4) for 650k steps. The rollout policy was compiled with
   `mx.compile(_act)`, and a closure compiled that way bakes in the arrays it captures. So the
   rollouts always used the *initial random weights*, while PPO updated a different set of
   weights against stale `old_logp`. The fix was `mx.compile(_act, inputs=model.state)`, after
   which the same smoke run went from 9% to 78% success in 650k steps. The same fix is applied
   in `PolicyRunner`.
3. **The designer fitness that got gamed.** The first fitness was `1 - SPL`, i.e. one minus
   success × BFS/steps. Within 30 generations against Trémaux, the designer dropped the mean BFS
   path from 22 steps to 6 and raised fitness from 0.42 to 0.70 without causing a single
   failure. Making the exit 4 steps away, behind the wrong turn, maximises the *ratio*
   steps/BFS. I switched to regret against BFS in env-return units, which charges for wasted
   steps in absolute terms.
   * Regret still mildly prefers short paths, because a failure's regret is
     `1 - 0.9·opt/200`: 0.97 at opt = 6 versus 0.87 at opt = 30.
   * The designer does push every run to the 6-step floor.
   * With the floor raised to 12 steps, the designer still reaches 92% failure (round-2
     section), so short paths are a preference, not a requirement.
4. **Fast collapse to one lineage.** Each designer run converges on one motif, so the final
   population is close to a single family of near-identical levels. That is why the gallery
   samples the archive across several runs, and why the curriculum uses the whole archive (all
   generations) rather than the final population.

## What worked, what didn't

* Worked:
  * Vectorised numpy envs plus a compiled MLX GRU gave about 100k env steps/s when the GPU was
    free, so the whole round-1 pipeline took 6 minutes.
  * The hard BFS constraint was simple and airtight.
  * The fresh-seed re-evaluation of designer populations matched selection-time fitness within
    0.01. With K = 8 there was no large winner's curse.
* Didn't:
  * Curriculum against an adaptive designer did not produce a solver that is hard to exploit. The
    designer's search space contains a policy-specific adversarial level for every solver we
    tried, and it finds one in about 5 generations.
  * A learned solver did not reach Trémaux's 100% on dfs, braid or spiral.

## Next steps

* Evaluate the 6 seed chains on 2-3 more held-out designer runs per target. Evaluation-set
  (designer-seed) variance is unmeasured, and the curriculum's gap varies a lot between `v1_s1`
  and `v1_md12_s7` within a seed.
* Run fresh designers against the new control chains too, for a multi-seed exploitability
  comparison.
* A designer with a diversity/novelty term (or MAP-Elites over dead ends × path length), so the
  curriculum sees more than one motif per run.
* PAIRED-style antagonist: score levels by the gap between two learned agents rather than
  against BFS, so the designer can't win with levels that no partial-view agent could solve
  efficiently.
* Larger grids (21×21), where Trémaux's memory matters more and a 200-step limit bites.
* Keys and doors (the "go deeper" stage).

## Gaps / honesty notes

* Not measured: wall-clock robustness to different GPU loads, or any grid size other than 13×13.
* The 0.01 novelty bonus is training-only shaping. All reported metrics are pure escape
  success and steps.
* There is no occlusion in the view: the agent sees through walls within the 5×5 window. The
  scripted baselines see exactly the same cells.
* The scripted baselines keep their map in absolute coordinates, so Trémaux and the wall follower
  get perfect odometry and a perfect visited-cell map for free. The GRU has to infer its position
  from its action and bump history. The sensor is matched; the memory is not, and it favours the
  baselines.
* "Matched compute" for curriculum vs control means matched fine-tuning (5,996,544 env steps each,
  same seed, init, lr and envs, checked in `checkpoints/*/meta.json`). The designer runs that
  produced the curriculum archive are extra compute that only the curriculum arm benefits from.
* Curriculum-vs-control numbers are now 6 training-seed chains per arm. The 95% intervals on the
  per-seed gaps are wide: round 3 on `v1_s1` runs from +0.15 to +0.79. The two arms in a chain
  share seeds but are independent runs, so per-seed pairing carries little information. The
  evaluation designer sets are still one designer run per target (`v1_s1`, `v1_md12_s7`,
  `ctl1_s2`), so designer-seed variance in the *evaluation* sets is not measured.
* Seed-chain round-1/round-2 checkpoints and per-chain curriculum files from `pipeline3.sh` are
  kept on disk but git-ignored (see `.gitignore`). Only each new chain's round-3 weights are
  meant to be committed. The curriculum files can be rebuilt with `make_curriculum.py` from the
  committed designer `.npz` files.
* The memoryless ablation is still one training seed per arm.
* `train_solver.py` logs success on all five families, including the held-out ones, every 50
  updates. This is monitoring only: the final checkpoint is always the last one, never the best one
  on held-out families.
* Nothing was installed; everything used the shared venv.
