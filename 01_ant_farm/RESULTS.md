# The Ant Farm: results

**Status: done.** First satisfying version reached: one nest, two food sources, and a learned
colony that brings dinner home without being given a route. Every number below is from a real
run, printed by `tables.py` from `results/eval*.json`.

> **Cross-review corrections (2026-10-22).** The numbers below reproduce exactly,
> but two conclusions in the original write-up did not survive review. Both are corrected in
> the bullets below; the original wording is gone.
> 1. **The hand-written baseline was untuned, and a light tune changes the comparison.**
>    Changing three weights of the compass colony (picked by a 1-minute grid on the
>    *validation* seeds 900–915; `compass_tuned` in `handwritten.py`,
>    `results/review_hw_tuned.json`) gives 1006 food at 4.30 food/energy (100% alive)
>    in-distribution, **579 food on the held-out family (more than ppo1's 542 and ppo2's 390)**,
>    and route-block recovery to 50% in **107 steps (91% of worlds)**, faster than any PPO run.
> 2. **"Survival is fixed by the 3× starvation penalty" isn't supported.** ppo1's 38% survival
>    comes from checkpoint selection: `model_best` is the *first* validation checkpoint to reach
>    1040 deliveries (episode 60). The selection rule is a strict `>` on deliveries only, so
>    later checkpoints that also deliver 1040 with 83–100% of ants alive never replace it.
>    ppo1's **final** weights, which ppo2 started from, already get 1034 food, 4.78
>    food/energy and **100% survival** in-distribution (`results/review_ppo1_final.json`).
>    That is the same as ppo2, so the penalty has no measurable survival effect here.

## Headline

* **Efficiency.** On 32 unseen worlds the shared-policy PPO colony (`ppo1`, best-on-validation
  checkpoint) delivers all 1040 food units in 31 of 32 worlds (mean 1039.97), at 5.18 food per
  unit energy. That 5.18 is flattered by starvation: 62% of its ants are dead by the end and
  spend no more energy. The survival-healthy PPO policies get 4.78–4.95 (ppo1 final weights,
  seed 2, ppo2). The hand-written + compass colony gets 836 at 3.57 as written, and **1006 at
  4.30 after a light tune**. ES gets 867 at 4.09. So PPO still leads on in-distribution
  efficiency and throughput, but by about 11–15%, not 45%. On the held-out family (food
  farther away, different obstacles), PPO delivers 390–587 depending on run and checkpoint,
  vs. 415 for the hand-written colony as written and **579 tuned** (at 100% survival, vs.
  53–72% for PPO). ES collapses there (23).
* **Recovery after a route is blocked.** PPO runs at about 3.0–3.2 food/step before the wall
  drops, and its delivery rate is back to 50% of that 348 / 121 / 140 / 184 steps after the
  block (ppo1 / seed 2 / ppo2 / ppo1 final weights; 62–88% of worlds individually). The
  hand-written colonies *as written* never get back to 50% on average (31–34% of worlds do).
  The **tuned** compass colony recovers to 50% in 107 steps (91% of worlds), but from a lower
  pre-block rate (1.92 food/step). In absolute terms PPO still delivers more after the block
  (ppo2's last 200 steps: about 1.8 food/step vs. about 1.06 for tuned hand-written). No colony
  returned to 80% within 1000 steps.
* **Survival.** The reported ppo1 checkpoint starves after harvesting (38% alive), but this is a
  checkpoint-selection artifact (see the correction box above). ppo1's final weights, ppo2 and
  the hand-written colonies all keep about 100% of their ants alive in-distribution.
* **Demo:** `media/side_by_side_ppo2.mp4`, the hand-written colony (untuned) vs. ppo2 on **one**
  held-out world (seed 50001), with a wall dropped across the trail at step 500. The learned
  colony re-forms its trail around the end of the wall (973 vs. 272 food delivered by step
  1052). The gap in this one world is much larger than the 32-world average. Also
  `media/demo.html`, the interactive terrarium.

## Setup (decisions made overnight, without the user)

**World.** 64×64 grid; nest = disc of radius 2; two food discs of radius 2 (40 units per cell,
about 520 per source); rocks; out-of-bounds counts as rock. Pheromone has two channels ("home"
and "food"). Each step it diffuses (10% exchanged with the 4 neighbours, reflecting at rocks)
and evaporates (0.6%/step for home, 1.2%/step for food). Deposits add 1.0, capped at 30.

**Ants.** 64 per colony. Moves are heading-relative: turn −90/−45/0/+45/+90 then step, U-turn,
or stay. Moving costs 1/300 energy, staying costs 1/600 (basal metabolism), and a deposit adds 1/3000. Pick-up and
drop are automatic contact events: an empty ant on a food cell picks up 1 unit, and a carrying
ant entering the nest delivers it. A delivered unit adds 1.0 energy to the colony store. At
the nest an ant refuels to full from the store, which starts at 0.5 energy per ant. Energy 0
means death, and any load the ant carried is dropped where it falls.

**Observations, all local (59 numbers).** For 5 forward directions × distance 1 and 2: home
pheromone, food pheromone, rock, food, nest. Also pheromone on the ant's own cell, carrying
flag, energy, an internal clock (steps since last pick-up or nest visit, capped), and the
**path-integration compass**: unit bearing to the nest in the ant's own frame, distance to
the nest, and an at-nest flag. Real foraging ants (*Cataglyphis*) keep exactly this home vector
by path integration, so I included it. It points home only. It carries no information about
where food is, and a test (`test_observation_is_local`) checks that changing anything more
than 2 cells from every ant leaves every observation unchanged. Because the compass could be
read as "giving a route home", I also trained and evaluated a **no-compass** PPO ablation, and
the classic hand-written colony doesn't use the compass either.

**Same interface for everyone.** The hand-written colonies, random, PPO, and ES are all
functions `obs -> (move probs, deposit probs)` over the same observation vector and action
space. Actions are sampled from the world's own seeded RNG, so a world plus a policy replays
bit-for-bit.

**Reward (PPO), per ant per step.** +1 for each food unit delivered by this ant, +0.3 for each
pick-up (shaping), +0.5 × (colony deliveries this step) / 64 as a shared team term, −energy
spent, and −1 on starving. The pick-up bonus is a local contact event: the ant is already
standing on the food, so it says nothing about where food is or how to get there. The team
term exists because laying pheromone is a public good: without it, the deposit head would get
almost no gradient. The critic (training only) also sees time-left and the colony store;
actors never do.

**ES fitness.** The same quantity summed over the colony: delivered + 0.3·picked − energy −
deaths. The team term is only a redistribution, so it drops out at colony level.

**Evaluation sets.** Training layouts use seeds from 1,000,000 upward (ES from 3,000,000).
* `indist`: same layout family as training, seeds 0–31 (never trained on).
* `heldout`: a different family. Food sits 24–30 cells from the nest (training: 10–22) and
  the obstacles are long thin walls instead of rectangles. Seeds 50,000–50,031.

Every colony gets the same 32 worlds, 64 ants, and 1000 steps.

**Route-block recovery.** Food is made abundant (250 per cell) so that depletion can't look
like a blocked route. After 500 steps, a 2-cell-thick wall about 19 cells long goes across the
nest→food line, perpendicular and centred on the midpoint, in front of the source that
supplied the most pick-ups in the previous 200 steps (decided from the food grid, so the rule
is the same for every colony). Pre-block rate = mean deliveries per step over steps 300–500.
Recovery time = steps after the block until the trailing 100-step delivery rate (averaged over
32 worlds) is back to ≥50% / ≥80% of the pre-block rate. The detour is longer, so 100% may be
unreachable. Switching to the other source counts as recovering.

**Compute and selection.** PPO used 32 worlds × 64 ants × 100-step rollouts (about 200k ant-steps
per update), 3 epochs, minibatch 8192, lr 3e-4, γ 0.99, λ 0.95, clip 0.2, entropy 0.01, and a
128-128 tanh MLP. ppo1 trained for 28 min of wall clock (964 updates, about 190M ant-steps), ppo1_seed1 for
30 min, and ppo2 is ppo1 + 30 min.
ES used a population of 48 (antithetic), 2 worlds × 600 steps per member, σ 0.03, Adam lr 0.02,
same network, 15 min (106 generations, about 390M ant-steps; the final mean parameters are evaluated, with no checkpoint selection). The no-compass run had 11
min. Every 5 episodes, PPO is scored on 16 *validation* worlds (seeds 900–915, disjoint from
train and test), and the reported checkpoint is the best one there (`model_best.safetensors`; ppo2 is the exception, at its final weights, see failure 1;
`ANTFARM_LAST=1` evaluates the last one instead). The hand-written rules were written once and
never tuned (the cross-review found that this matters; see the caveats). During development I did run them on seeds 0–31 to calibrate the world's food
amount (15 → 40 units per cell, so a good colony isn't capped by depletion). That calibration
touched the test seeds, but it changes the world for every colony equally. Compute is **not** matched between PPO and
ES (28 vs 15 min); see the caveats.

## Results

<!--TABLES-START-->

**Unseen worlds from the training family (seeds 0-31)**, 32 colonies × 64 ants × 1000 steps

| colony | food delivered (mean ± sd) | food / energy | ants alive at end | colonies ≥50% alive | delivered without pheromone |
|---|---|---|---|---|---|
| random walkers | 6 ± 4 | 0.07 | 0.00 | 0.00 | n/a |
| hand-written classic (no compass) | 620 ± 139 | 2.81 | 0.85 | 1.00 | 39 |
| hand-written + compass | 836 ± 173 | 3.57 | 1.00 | 1.00 | 315 |
| **PPO** (shared policy) | 1040 ± 0 | 5.18 | 0.38 | 0.19 | 0 |
| PPO, 2nd seed | 1029 ± 63 | 4.95 | 0.68 | 1.00 | 5 |
| PPO continued, 3x starvation penalty | 1034 ± 31 | 4.85 | 1.00 | 1.00 | 97 |
| PPO, no compass | 293 ± 281 | 1.77 | 0.54 | 0.53 | 2 |
| ES (same network) | 867 ± 182 | 4.09 | 0.98 | 1.00 | 134 |

**Held-out family: food 24-30 cells away, long walls (seeds 50000-50031)**, 32 colonies × 64 ants × 1000 steps

| colony | food delivered (mean ± sd) | food / energy | ants alive at end | colonies ≥50% alive | delivered without pheromone |
|---|---|---|---|---|---|
| random walkers | 0 ± 0 | 0.00 | 0.00 | 0.00 | n/a |
| hand-written classic (no compass) | 65 ± 35 | 0.52 | 0.08 | 0.00 | 7 |
| hand-written + compass | 415 ± 55 | 1.78 | 1.00 | 1.00 | 72 |
| **PPO** (shared policy) | 542 ± 276 | 3.15 | 0.53 | 0.62 | 0 |
| PPO, 2nd seed | 587 ± 249 | 3.13 | 0.69 | 0.94 | 0 |
| PPO continued, 3x starvation penalty | 390 ± 256 | 2.19 | 0.67 | 0.72 | 0 |
| PPO, no compass | 0 ± 0 | 0.00 | 0.00 | 0.00 | 0 |
| ES (same network) | 23 ± 65 | 0.22 | 0.06 | 0.06 | 0 |

**Route-block recovery** (train-family seeds 0-31, abundant food, wall at step 500)

| colony | pre-block rate (food/step) | rate in first 100 steps after (rel.) | rate in last 200 steps (rel.) | t to 50% | t to 80% | worlds back to 50% | median t50 in those worlds | ants alive at 1500 |
|---|---|---|---|---|---|---|---|---|
| hand-written classic (no compass) | 1.40 | 0.15 | 0.28 | never (in 1000 steps) | never (in 1000 steps) | 0.34 | 176 | 0.64 |
| hand-written + compass | 1.84 | 0.15 | 0.40 | never (in 1000 steps) | never (in 1000 steps) | 0.31 | 198 | 0.93 |
| **PPO** (shared policy) | 3.02 | 0.19 | 0.48 | 348 | never (in 1000 steps) | 0.62 | 240 | 0.74 |
| PPO, 2nd seed | 3.23 | 0.43 | 0.61 | 121 | never (in 1000 steps) | 0.88 | 120 | 0.95 |
| PPO continued, 3x starvation penalty | 3.16 | 0.39 | 0.58 | 140 | never (in 1000 steps) | 0.84 | 100 | 0.94 |
| PPO, no compass | 0.45 | 0.16 | 0.19 | never (in 1000 steps) | never (in 1000 steps) | 0.38 | 172 | 0.31 |
| ES (same network) | 1.13 | 0.44 | 0.17 | never (in 1000 steps) | never (in 1000 steps) | 0.56 | 100 | 0.25 |

<!--TABLES-END-->

**Added by the cross-review** (same 32 worlds, 64 ants, 1000 steps, same `evaluate.py`; from
`results/review_hw_tuned.json` and `results/review_ppo1_final.json`):

| colony | in-dist food | in-dist food/energy | in-dist alive | held-out food | held-out food/energy | held-out alive | recovery t to 50% | worlds back to 50% |
|---|---|---|---|---|---|---|---|---|
| hand-written + compass, **tuned** (`compass_tuned`) | 1006 ± 70 | 4.30 | 1.00 | 579 ± 73 | 2.48 | 1.00 | 107 | 0.91 |
| PPO ppo1, **final weights** (`ANTFARM_LAST=1`) | 1034 ± 34 | 4.78 | 1.00 | 442 ± 263 | 2.32 | 0.72 | 184 | 0.84 |

`compass_tuned` changes three weights of the compass rule (forward bias 1→2, compass weight
2.5→5, random-move ε 0.05→0.02). The reviewer picked them with a 60-config grid on the
validation seeds 900–915 (about 1 minute of CPU) and never looked at the test seeds. Pre-block
rate for tuned hand-written: 1.92 food/step (PPO: 3.0–3.2).

Charts: `media/efficiency.png`, `media/recovery.png`, `media/training.png`, `media/es_training.png`.

Videos:
* `media/side_by_side_ppo2.mp4` and `.gif`: hand-written vs. ppo2 on an unseen held-out world
  (seed 50001), with a wall dropped across the busiest trail at step 500. Food delivered by step
  1400: 336 vs. 1263.
* `media/side_by_side.mp4` and `.gif`: hand-written vs. ppo1 vs. ES on the same world and
  wall (336 / 932 / 0; the ES colony starves on held-out worlds).
* `media/ppo_after_harvest.gif`: PPO strips 4 worlds bare, then starves.
* `media/nocompass_collapse.gif`: the stay-home collapse.
* `media/es_heldout_failure.gif`: ES on held-out worlds.
* `media/demo.html`: interactive terrarium.

## What the learned colony does

* **Huddle, then an expanding ring.** For the first 100–250 steps PPO ants stay in a tight
  ball around the nest, covering it in pheromone. Then they push outward as a ring-shaped search
  front whose edge is marked with pheromone (the polygon outlines in the video). When the front
  touches food, a trail snaps into place and the ring dissolves. It is a systematic search, not
  a random walk. It pays off: once a trail exists, the PPO colony delivers about 3 food/step vs.
  about 1.8 for the hand-written colony. But its first delivery comes later (median step 62 vs. 17;
  ppo2 huddles less, with a median first delivery at step 25).
* **It depends on pheromone completely.** With deposits disabled at test time, PPO delivers 0
  and every ant dies. The hand-written + compass colony still manages 315, and ES 134. Its
  deposits are far less clear-cut than the textbook rule's. Measured over 8 worlds and 600 steps,
  searching PPO ants choose none/home/food 38/31/31% of the time and carrying ants 34/23/42%.
  The hand-written ants choose home 97% of the time while searching and food 100% while
  carrying. So PPO's two channels are only weakly specialised: carrying ants lean towards
  "food" (the green in the videos is mostly a rendering-brightness effect; by mass the field
  is 60% home, 40% food). With the compass zeroed at test time,
  PPO falls to 204 food and 6% survival, so it uses both senses.
* **Detours.** When the wall drops, PPO ants restart the ring search around the wall and
  re-form a bent trail around its end (`side_by_side` strip, step 1052). The hand-written ants
  *as written* follow the compass straight into the wall and pile up there. A lightly tuned
  version of the same memoryless rule (stronger forward bias, less noise) does get around:
  it recovers to 50% in 107 steps (cross-review). So the pile-up comes from the untuned
  parameters, not from a limit of local rules.

## Failures worth keeping

1. **Starving next to a full pantry (PPO).** In every in-distribution world ppo1 has harvested
   all 1040 units by about step 750, about twice as fast as the hand-written colony. After
   that, ants keep searching or rest in the field (55% "stay" actions by step 1000) and die with
   the nest store at about 930 energy. Of 271 deaths logged on 8 worlds, 247 happened after the
   food was gone, a median 13 cells from the nest (`media/ppo_after_harvest.gif`). With γ =
   0.99, the −1 starvation penalty is barely visible from the moment an ant decides to keep
   wandering, and empty worlds were rare during training.
   * **Cross-review: this failure belongs to one checkpoint, not to the recipe.** The
     reported ppo1 is `model_best`, the *first* validation checkpoint to deliver 1040 (episode
     60, 37.5% alive on validation). Selection is a strict `>` on deliveries alone, so later
     checkpoints with the same 1040 and 83–100% alive (episodes 80 and 95) never replaced it.
     ppo1's final weights (episode 97) get 1034 food, 4.78 food/energy and 100% survival on the
     test worlds (`results/review_ppo1_final.json`).
   * `ppo2` continues ppo1's *final* weights (plus Adam state and counters) for 30 more minutes
     with the starvation penalty raised from 1 to 3. It ends at 99.9% survival, 1034 food and
     4.85 food/energy, which is what its starting point already had. So the penalty shows **no
     measurable survival effect**. On held-out worlds ppo2 delivers 390 vs. 442 for its
     starting weights. Whether that drop comes from the penalty or from more training can't be
     told apart with one run.
   * ppo2 is evaluated at its **final weights**: it inherited ppo1's best validation score
     (1040), so no later checkpoint could beat it under the strict-`>` rule. ppo1 and
     ppo1_seed1 are their best-on-validation checkpoints. The selection rule differs across
     rows, and that matters (see above).
2. **Stay-home collapse without a compass (PPO, twice).** Without path integration, ants that
   leave the nest often die (−1 each). In the first run (resting cost 1/1500), the policy
   learned within 15 episodes that *nobody leaves the nest*: 0 deliveries, 100% survival,
   because a resting colony could outlast a 1000-step episode on its starting store. That was
   an environment flaw, so I raised basal metabolism to 1/600 (a resting colony now starves at
   about step 900) and re-ran everything; **all numbers in this file use 1/600**. The second
   run collapsed the same way: by episode 30 every ant sits in the nest until the whole colony
   starves at about step 900 (`media/nocompass_collapse.gif`). The death comes about 600 steps
   after the decision, far beyond a 0.99 discount. The no-compass row in the table is that
   run's best validation checkpoint (episode 5), before the collapse.
3. **ES learns a leash.** ES (fitness over 600-step episodes on the training family) evolved
   a colony that searches only within about 8–10 cells of the nest. That is enough when food is
   10–22 cells away (867 delivered, 98% survival), but on the held-out family, with food 24–30
   cells away, it finds almost nothing and starves (23 delivered, 6% alive;
   `media/es_heldout_failure.gif`, and the ES panel of `side_by_side`). PPO's ring search
   generalises to the farther food. ES's doesn't.
4. **Recovery numbers and deaths.** The recovery run lasts 1500 steps, and some colonies lose
   many ants along the way. For the hand-written + compass colony this doesn't matter: 93% of
   its ants are alive at step 1500, yet it never gets back to 50%, so the wall itself is the
   cause (its ants are trapped behind it; the tuned version escapes, see the cross-review row). For ES (25% alive) and PPO without a compass (31%),
   the slow recovery can't be separated from the colony dying. PPO (74%) and the classic
   colony (64%) fall in between.

## Honest caveats

* **Test-seed leak in the world design.** The food amount per cell (15 → 40) was set by
  running the hand-written colonies on the *test* seeds 0–31. The same world applies to every
  colony, so it doesn't favour one of them directly. But it is why 1040 is a hard ceiling that
  PPO hits. Its effect on the ranking was not measured. The cross-review judges it small: it
  is one world-level scalar, chosen for the hand-written colony rather than for PPO. A clean
  protocol would calibrate on the validation seeds.
* **The baseline was not tuned; PPO was selected.** The original hand-written colonies were
  written once. PPO got 28–58 min of training plus checkpoint selection on validation worlds.
  A 1-minute validation-seed tune of three hand-written weights closes most of the
  in-distribution gap and beats PPO on held-out deliveries and on recovery time (cross-review
  row above). Read comparisons with the "as written" hand-written rows with this in mind.
* **ppo1's 5.18 food/energy is partly a starvation artifact.** Dead ants spend no energy. The
  healthy PPO policies score 4.78–4.95.

* **Ceiling.** In-distribution worlds hold 1040 food units and PPO takes all of them, so the
  1040 is a ceiling, not a margin. Food per unit energy (healthy PPO policies 4.78–4.95 vs. 4.30 for
  the tuned hand-written colony; ppo1's 5.18 is starvation-inflated) and the per-100-step
  delivery curves (`delivery_curve_100` in `results/eval.json`) are the fairer comparison.
* **Compute isn't matched.** PPO trained for 28–30 min (ppo2: 58 min in total); ES for 15 min
  and on 600-step episodes.
  The no-compass PPO had 11 min. The hand-written colonies needed none.
* **Training is noisy.** PPO's validation score swings between about 250 and 1040 from one
  evaluation to the next (`media/training.png`). The reported checkpoint is selected on
  separate validation seeds, never on the test seeds.
* **The compass is a real sense,** but a strong one. Without it, neither learning method found
  a stable colony in the time available, and the classic hand-written colony (no compass) is the
  only colony here that survives without one.
* Rendering, demo.html and short diagnostics ran outside the GPU lock (each under 1 minute).
  Every training and evaluation run went through `../bin/gpu_run`.

## Next steps

* Fix checkpoint selection: break ties on validation deliveries by survival or efficiency
  (or select on a combined score), then re-evaluate ppo1 and ppo1_seed1. Any 3×-penalty study
  should start from scratch on several seeds against the same budget with penalty 1. Try a
  larger γ so starvation 600 steps out becomes visible (that is also what killed the
  no-compass runs).
* Tune the hand-written baselines properly (on validation seeds) before claiming the learned
  colony beats them.
* Recurrent memory (GRU) for wall-following and for remembering where the food was. It is not
  built: no time was left once the shared GPU queue was accounted for.
* Train on a mix of food distances, and see whether ES's leash goes away.
* The "go deeper" fork experiment: save a world (`World.save` already stores the RNG), change
  one condition, and continue both copies.

## Gaps and environment notes

* No packages were missing.
* `../bin/gpu_run` reads the slot list once at start-up. When the lab went from 4 to 8 slots at
  about 06:30, jobs already waiting could only ever get the old 4 slots, so I re-queued mine.
  Other builders' jobs queued before then have the same problem.
* My jobs spent about 2 h 20 min waiting for a slot.
