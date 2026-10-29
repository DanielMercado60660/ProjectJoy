# 06 · Traffic Control From Hell: results

All numbers come from real runs on 2026-10-22, on a busy machine (ten builders training at once).
Sources: `runs/eval_results.json` and `runs/eval_table.md` (held-out evaluation), `runs/train_log.jsonl`
(training curve), `checkpoints/ppo_best.json` (selected checkpoint).

## Setup in one paragraph
A 2×2 grid. Two heavy east–west avenues (540 veh/h per direction, ~1030 veh/h at the rush peak),
a medium north–south Elm St (310 veh/h), and **Quiet Lane** (54 veh/h, one car every ~67 s).
2-s steps, one-hour episodes, 4-s all-red per switch, 16-car internal links with spillback. Every
controller decides every 10 s and the simulator enforces non-conflicting phases. Evaluation: **20
held-out arrival seeds** per pattern (seed 10000+, never used in training or tuning), the same
arrivals for every controller:
- `steady`: base demand all hour.
- `rush`: avenues ×1.9 at 17:30 (Gaussian, σ = 10 min), Elm St ×1.3, Quiet Lane unchanged.
- `hell`: avenues ×2.1 at the peak and lopsided (eastbound +30%, westbound −30%). This is
  oversaturated: queues on the eastbound avenue grow through the peak no matter who is in charge.

The demand *shapes* for `rush` and `hell` lie inside the training distribution (random scale
0.75–1.15, peak 1.0–2.1, asymmetry ±30%); the arrival seeds are held out.

## Controllers and budgets
| controller | how it was set up | compute |
|---|---|---|
| fixed-time (tuned) | grid search over 2,304 plans (EW green 10–80 s, NS green 10–60 s, chosen separately for the Elm St column and the Quiet Lane column) on 8 training-distribution demands, picked by the same validation score as PPO (`mean wait + 0.1 × max wait`). Chosen: 40 s EW / 20 s NS at both columns. | ~1 min CPU |
| fixed-time (tuned for mean wait) | same grid, picked by mean wait only. Chosen: Elm column 60/30 s, Quiet column 70/20 s. | same run |
| longest-queue-first | serve the phase with more cars at the stop line | none |
| max-pressure | serve the phase with the larger (stop-line queue − downstream link load) | none |
| max-pressure + guard | max-pressure, but serve any approach whose front car waited more than X s; X ∈ {30, 45, 60, 90, 120, 180} tuned on the same training demands with the same score. Chosen: 30 s (the bottom of the grid). | seconds |
| *reviewer:* LQF + 20 s min green, max-pressure + 2-veh hysteresis | LQF/max-pressure above may switch at every 10-s decision (no minimum green), paying 4 s of all-red each time. The reviewer added a minimum green (10–80 s) and a switching hysteresis (0–12 vehicles), tuned on the same 8 training demands with the same score, never the test seeds (`review_check.py` → `runs/review_check.json`). Chosen: LQF min green 20 s; max-pressure hysteresis 2 veh (a min green did not help max-pressure or the guard on training demand). | seconds |
| **PPO (learned)** | one 2×64 MLP policy shared by all four intersections, PPO, 32 envs × 1 h per iteration, **307 iterations in 15.0 min** (`--minutes 15`), ~44 simulated days. Checkpoint chosen at iteration 290 by validation on training-distribution seeds. | 15 min, 1 slot |

## Headline: held-out results (mean ± std over 20 seeds)
Waits are per signal crossing, in seconds. "Worst" waits include cars still waiting when the hour ends.

### Rush hour (the main test)
| controller | veh/h | mean wait | p95 | worst wait | Quiet mean | **Quiet worst** | worst approach mean | Jain | spillback steps |
|---|---|---|---|---|---|---|---|---|---|
| fixed-time (tuned) | 3772 | 13.6 ± 0.8 | 44.9 | 96 ± 15 | 13.2 | **46 ± 1** | 30.2 | 0.756 | 0 |
| fixed-time (mean-wait tuned) | 3771 | 12.3 ± 0.6 | 53.8 | 94 ± 9 | 21.5 | **76 ± 0** | 35.9 | 0.658 | 4 |
| longest-queue-first | 3777 | 19.7 ± 4.3 | 70.8 | 353 ± 190 | 31.8 | **320 ± 222** | 76.6 | 0.629 | 167 |
| max-pressure | 3773 | 23.0 ± 3.8 | 72.3 | 211 ± 55 | 12.0 | **102 ± 87** | 61.9 | 0.699 | 58 |
| max-pressure + 30 s guard | 3772 | 25.0 ± 4.5 | 81.8 | 144 ± 39 | 16.0 | **109 ± 51** | 48.0 | 0.806 | 181 |
| *LQF + 20 s min green (reviewer)* | 3772 | 14.8 | 39.0 | 123 | — | **117** | 23.2 | — | — |
| *max-pressure + 2-veh hysteresis (reviewer)* | 3772 | 19.6 | 57.8 | 171 | — | **67** | 44.4 | — | — |
| **PPO (learned)** | 3779 | **10.3 ± 0.5** | **35.0** | **90 ± 6** | 16.5 | **76 ± 11** | **23.1** | 0.781 | 15 |

### Steady
| controller | veh/h | mean wait | p95 | worst wait | Quiet mean | Quiet worst | Jain |
|---|---|---|---|---|---|---|---|
| fixed-time (tuned) | 2850 | 10.2 | 43.2 | 74 | 13.2 | 46 | 0.734 |
| fixed-time (mean-wait tuned) | 2850 | 10.1 | 50.9 | 80 | 22.3 | 76 | 0.594 |
| longest-queue-first | 2856 | 7.9 | 21.2 | 56 | 11.9 | 46 | 0.878 |
| max-pressure | 2852 | 10.6 | 26.4 | 54 | 7.6 | 28 | 0.917 |
| max-pressure + 30 s guard | 2852 | 10.6 | 26.6 | 44 | 7.6 | 28 | 0.915 |
| **PPO (learned)** | 2858 | **6.8** | 23.1 | 51 | 11.9 | 46 | 0.756 |

### Hell hour (oversaturated, lopsided)
| controller | veh/h | mean wait | p95 | worst wait | Quiet mean | Quiet worst | worst approach mean | Jain | left in network at 18:00 |
|---|---|---|---|---|---|---|---|---|---|
| fixed-time (tuned) | 3888 | 51.0 | 306 | 429 | 13.2 | **46** | 188.5 | 0.295 | 91 |
| fixed-time (mean-wait tuned) | 3930 | 41.7 | 254 | 374 | 21.9 | 76 | 158.7 | 0.338 | 48 |
| longest-queue-first | 3822 | 59.5 | 274 | 653 | 14.8 | 69 | 305.4 | 0.314 | 156 |
| max-pressure | 3800 | 64.8 | 289 | 687 | 12.6 | 69 | 319.3 | 0.308 | 179 |
| max-pressure + 30 s guard | 3754 | 77.2 | 398 | 521 | 12.7 | 47 | 225.9 | 0.381 | 224 |
| *LQF + 20 s min green (reviewer)* | 3949 | 36.6 | 145 | 380 | — | 106 | 128.1 | — | 30 |
| *max-pressure + 2-veh hysteresis (reviewer)* | 3856 | 57.7 | 253 | 629 | — | 85 | 287.1 | — | 122 |
| **PPO (learned)** | **3958** | **24.9** | **119** | **234** | 23.3 | **96 ± 9** | **68.1** | **0.594** | **21** |

Full tables with std for every column: `runs/eval_table.md`. Chart: `media/summary.png`. Reviewer rows
(italic, means only; "—" = not computed there): `runs/review_check.json`, same 20 held-out seeds and the
same arrivals for every controller (checked: identical `entered` counts per episode).

**How unfinished trips count.** `mean wait` and `p95` average over completed signal crossings only;
cars still queued at 18:00 enter the *worst* waits (front car's wait so far) but not the mean. The
reviewer recomputed the mean with every queued car counted at its wait so far
(`mean_wait_incl_unfinished` in `runs/review_check.json`): hell hour PPO 24.9 → 24.9 s, fixed-time
51.0 → 51.5, mean-wait fixed-time 41.7 → 41.8, LQF 59.5 → 60.8, max-pressure 64.8 → 66.6, guard
77.2 → 80.0. Leaving cars out flattered the rules slightly, so PPO's hell-hour win is not a
censoring artefact.
Fraction of time an internal link was full (`link_full_frac`) stayed at or below 1% for every
controller; spillback shows up mostly as blocked discharge steps (column above). No edge
arrivals were ever rejected.

## What the numbers say
- **The learned controller reduces waiting a lot.** Mean wait per crossing against the best rule in
  each pattern: −16% in rush hour (10.3 vs 12.3 s for mean-wait-tuned fixed-time; 14.8 s for LQF
  with a 20-s minimum green), −32% in the hell hour (24.9 vs 36.6 s for LQF with a 20-s minimum
  green; 41.7 s for the best fixed-time plan), and −14% vs longest-queue-first in steady traffic
  (6.8 vs 7.9 s). Worst wait, p95 and the worst approach's mean wait also drop under rush and hell.
  In the hell hour it leaves the fewest cars on the map (21 at 18:00; LQF with a minimum green
  leaves 30, the free-switching rules 156–224) and has the highest throughput, but only by 9 veh/h
  over min-green LQF (3958 vs 3949), well inside the ±58 seed-to-seed spread: call throughput a tie.
- **Queue-greedy rules starve the quiet street, as expected.** In rush hour longest-queue-first lets
  a Quiet Lane car wait **320 s on average over seeds, 558 s in the demo episode**. Max-pressure:
  102 s (up to ~4 min on bad seeds). In both, the Quiet Lane queue has to outgrow the avenue
  queues before it gets a green.
- **PPO does not starve Quiet Lane, but it does make it wait longer than the best fixed-time
  plan.** Quiet Lane worst wait under PPO: 46 s (steady), 76 s (rush), 96 s (hell). The
  score-tuned fixed-time plan holds it at 46 s in every pattern because its cycle is 60 s. So the
  first-version goal is met against free-switching LQF (4× better in rush) and LQF with a 20-s
  minimum green (117 s, 1.5× worse than PPO), and matched against the mean-wait-tuned fixed plan
  (76 s). It is **not** met against the score-tuned fixed plan (46 s) or max-pressure with a
  2-vehicle hysteresis (67 s), although both have higher mean waits (13.6 and 19.6 vs 10.3 s). In the hell hour PPO's Quiet Lane worst wait (96 s) is the highest of the builder's six
  controllers (the reviewer's min-green LQF reaches 106 s): it buys its average-wait lead partly by holding the quiet side street for
  1.5 minutes at the peak. That is the honest trade-off, and the reward weights decide it.
- **The max-wait guard does not work well; a minimum green does better.** Max-pressure plus a
  max-wait guard (tuned to 30 s) cut the rush-hour worst wait from 211 to 144 s, but the Quiet Lane worst wait still
  reached 109 s, and mean wait got worse. Under heavy load every approach's front car is over
  30 s, so the guard just serves whichever is oldest and thrashes (181 blocked-discharge steps,
  the most of any controller). The reviewer found the rules' bigger handicap is switching freely:
  LQF and max-pressure switch 178–276 times per intersection per hour, losing 4 s of all-red each
  time. A 20-s minimum green, tuned on training demand, takes LQF from 19.7 to 14.8 s (rush) and
  59.5 to 36.6 s (hell); a 2-vehicle hysteresis takes max-pressure from 23.0 to 19.6 s (rush).
  PPO still beats both.
- **Fairness is a trade-off, not a ranking.** Jain's index over the 16 approaches' mean waits
  favours max-pressure in steady traffic (0.92: it serves everyone short and often, ~276
  switches per intersection per hour), while PPO is best in the hell hour (0.59 vs ≤ 0.38).

## What worked
- A batched numpy queue model is plenty: 32 one-hour episodes simulate in ~0.6 s, so PPO saw
  ~44 simulated days of traffic in 15 minutes on a shared machine.
- PPO did not fall into the starvation trick the greedy rules use, without any hard max-red
  constraint. In the one-seed ablation below, a PPO trained on queue length alone (β = 0) also
  kept Quiet Lane's rush-hour worst wait at 76 s, so the squared front-wait penalty may not be
  what prevents starvation; one seed per setting cannot settle it.
- Sharing one policy across intersections with a one-hot id: one network, four times the data.
- Enforcing non-conflicting phases in the simulator: the policy never has to learn safety.

## What didn't / caveats
- PPO switches less than the greedy rules (157 vs 178–240 switches per intersection per hour in
  rush) but still about every 23 s. The 4-s all-red is the only switching cost.
- The validation score (`mean + 0.1 × max`) and the reward both weight the average more than the
  quiet street's worst case. A controller tuned to Quiet Lane's worst wait first would look
  different. Not tried: a hard max-red constraint in the simulator.
- Straight-through traffic only (no turns), one lane per approach, no pedestrians, a delay-line
  model of driving (no positions). With no turns there are no route cycles, so the grid cannot
  gridlock. Throughput differences are small in steady and rush because demand is below
  capacity there.
- Fixed-time has no offsets (no green wave); tuning offsets could help it on the avenues.
- One training seed per setting. Seed-to-seed variance of PPO: **not measured**.
- Speed numbers are under load (builder-reported load average 13–14 during training and eval; not
  logged to a file. Reviewer's `uptime` at 19:23: 4.8 / 9.1 / 10.7). Training: 307
  PPO iterations in 902 s, ~2.9 s per iteration. Evaluation incl. all tuning: 94 s.

## Surprises
- The tuning objective matters more than the plan family. Of 2,304 fixed-time plans, the best by
  mean wait gives Quiet Lane a 76-s worst case. The plan picked with a small worst-wait term holds
  it to 46 s at almost the same mean (10.2 vs 10.1 s in steady traffic). The best fixed plan is a
  decent baseline and loses mainly when demand moves away from what it was tuned for.
- In the hell hour the victim of the greedy rules is **Elm St, not Quiet Lane**. Under
  longest-queue-first and max-pressure, the Elm St approaches (I0 southbound, I2 northbound)
  average 260–290 s of waiting per crossing, because the oversaturated avenues always have the
  longer queue. Quiet Lane gets greens there because the lopsided demand keeps the westbound
  queue at I1/I3 short. Fixed-time instead lets the eastbound avenue (I0/I2 eastbound) back up
  (~165 s mean). PPO's worst approach averages 57 s. Per-approach means:
  `lane_mean_wait` in `runs/eval_results.json`.
- PPO in the hell hour leaves 21 cars on the map at 18:00, the fewest of any controller. The
  builder's rule-based controllers end the hour with a backlog of 48–224 cars, but LQF with a
  20-s minimum green (reviewer) leaves only 30, so most of that gap was the free-switching
  handicap. Why PPO does better here is not analysed beyond these numbers.

## Ablation: no fairness term (β = 0), run after the write-up
Same code and seed, `--beta 0 --minutes 10` (342 iterations in 600 s, more than the main run's
307, because the machine was less loaded). Checkpoint chosen by the same validation score, at
iteration 80. Evaluated on the same 20 held-out seeds. Source: `runs/ablation_beta0/eval_table.md`.

| pattern | metric | PPO β = 0.5 (main) | PPO β = 0 (queue only) |
|---|---|---|---|
| steady | mean wait / Quiet worst | 6.8 / 46 s | 6.9 / 49 s |
| rush | mean wait / worst / Quiet worst | 10.3 / 90 / 76 s | 10.8 / 95 / 76 s |
| hell | mean wait / worst / Quiet worst | 24.9 / 234 / 96 s | 26.6 / 284 / 114 s |

**The surprise, in one seed per setting: the queue-only policy also avoided starvation.** It
keeps Quiet Lane far from the 320-s worst case of longest-queue-first. The β = 0.5 run looks
better in the hell hour (Quiet worst 114 vs 96 s, overall worst 284 vs 234 s), but with one
training seed each and the β = 0 checkpoint picked at iteration 80 of 342 (main run: 290 of 307),
this does not show whether the fairness term helps or not. Likely reasons, not tested separately: (1) minimising total queued vehicles over a
~50-decision horizon already counts a waiting Quiet Lane car every step, and the policy sees how
long it has waited; (2) checkpoint selection uses `mean + 0.1 × max wait`, which rewards low worst
waits for both runs. This is a single seed per setting, so small differences (steady, rush) are
within noise.

## Demo
- `media/demo.html`: offline, two controllers side by side on the first held-out rush-hour
  episode (selectable), with queues, signal states, and a live Quiet Lane wait counter. Add
  `#t=1486` to the URL to jump to the moment longest-queue-first has held a Quiet Lane car for
  558 s, while PPO's worst so far is 80 s.
- `media/quiet_lane_starved_t1486.png`: that frame.
- `media/summary.png`: mean wait and Quiet Lane worst wait for every controller and pattern.

## Next steps
- A hard max-red constraint (say 90 s) in the simulator, and a reward that weights Quiet Lane's
  worst wait directly. Then see whether PPO can match fixed-time's 46 s without giving back the
  average.
- Turning movements and a demand adversary under a fixed budget (the "Go deeper" stage).
- More training seeds. Separate the two candidate reasons PPO avoids starvation: select the
  β = 0 checkpoint on mean wait alone, and drop front-wait from the observation.

## Follow-up 1 (round 3, 20:04–20:25): training-seed variance
> **Supersedes the one-seed caveats above** ("one training seed", "the β = 0 ablation is one seed"): read this section for seed variance; it was cross-reviewed.

Why: after review, PPO's lead over the best tuned rule was modest and rested on one training seed,
and the β = 0 ablation was one seed too. Five more β = 0.5 seeds and three more β = 0 seeds were
trained with the same code and config as the original runs (`train.py --seed s`), each checkpoint
picked by the same `validate()` score on the same validation seeds (5000/5001), then every one
evaluated on the same 20 held-out arrival seeds per pattern against the rules. Nothing was re-tuned:
the fixed-time plans, the 20-s LQF minimum green and the 2-vehicle max-pressure hysteresis are taken
from `runs/eval_results.json` / `runs/review_check.json` (the rule numbers reproduce those files to the
decimal). Sources: `runs/seeds/eval_seeds.md` (all tables), `runs/seeds/eval_seeds.json` (per-episode
values), `runs/seeds/train_*.jsonl` (curves), `checkpoints/seeds/*_best.{npz,json}`.

**Budget.** Iterations were capped at the original runs' counts (β = 0.5: 307, β = 0: 342) with an
11-min wall cap; seeds were trained four at a time, single-threaded, inside one `gpu_run` job each.
Job 1 (β = 0.5 seeds 1–3, β = 0 seed 1) ran on a quiet machine and reached the caps (307 / 342 iters
in 3.5–4.7 min). Job 2 (β = 0.5 seeds 4–5, β = 0 seeds 2–3) ran under load (uptime 13–14) and hit the
wall cap at 298–301 iterations: about the main run's count for β = 0.5, ~13% short for β = 0.
The original curve plateaus early (val score 14.2 by iteration 120, best 13.65 at 290), so this
does not look like the limiting factor. Selected iterations: β = 0.5 seeds 0–5 → 290, 180, 260, 230,
190, 270; β = 0 seeds 0–3 → 80, 160, 90, 280.

### Across seeds (mean ± sd across training seeds [min–max]; each seed is a mean over 20 held-out episodes)
| pattern | controller | mean wait s | worst wait s | Quiet worst s | veh/h | cars left |
|---|---|---|---|---|---|---|
| steady | best rule: LQF (no min green) | 7.9 | 56 | 46 | 2856 | 20 |
| steady | **PPO β=0.5, 6 seeds** | **6.9 ± 0.1** [6.8–7.0] | 52 ± 1 [51–55] | 46 ± 1 [46–47] | 2857 | 19 |
| steady | PPO β=0, 4 seeds | 6.9 ± 0.1 [6.8–7.0] | 56 ± 1 [55–57] | 50 ± 2 [47–52] | 2857 | 19 |
| rush | best rule: fixed-time (mean-wait tuned) | 12.3 | 94 | 76 | 3771 | 27 |
| rush | LQF + 20 s min green | 14.8 | 122 | 117 | 3772 | 26 |
| rush | max-pressure + 2-veh hysteresis | 19.6 | 171 | 67 | 3772 | 26 |
| rush | fixed-time (score tuned) | 13.6 | 96 | **46** | 3772 | 26 |
| rush | **PPO β=0.5, 6 seeds** | **10.5 ± 0.2** [10.3–10.8] | 90 ± 3 [86–94] | 73 ± 3 [68–76] | 3779 | 19 |
| rush | PPO β=0, 4 seeds | 10.6 ± 0.4 [10.3–10.9] | 102 ± 6 [95–107] | 82 ± 10 [73–95] | 3779 | 19 |
| hell | best rule: LQF + 20 s min green | 36.6 | 380 | 106 | 3949 | 30 |
| hell | fixed-time (mean-wait tuned) | 41.7 | 374 | 76 | 3930 | 48 |
| hell | fixed-time (score tuned) | 51.0 | 429 | **46** | 3888 | 91 |
| hell | **PPO β=0.5, 6 seeds** | **25.1 ± 1.1** [23.6–26.3] | 255 ± 20 [234–287] | 105 ± 8 [93–115] | 3958 ± 0 | 21 ± 0 |
| hell | PPO β=0, 4 seeds | 27.0 ± 5.4 [23.0–34.7] | 353 ± 61 [284–419] | 175 ± 61 [114–259] | 3942 ± 19 | 36 ± 18 |

Paired per held-out episode against the best rule on mean wait in that pattern (`runs/seeds/eval_seeds.md`):
every β = 0.5 seed has the lower mean wait in 20/20 episodes in steady (Δ −0.9 to −1.1 s) and rush
(Δ −1.5 to −2.0 s), and in 19–20/20 in the hell hour (Δ −10.4 to −13.1 s). Reviewer check: the same
counts and Δs (within 0.1 s) hold for `mean_wait_incl_unfinished`, computed identically for every controller.

### What the seeds say
- **"PPO has the lowest mean wait in every pattern" survives across seeds.** All six β = 0.5 seeds
  beat the best rule in every pattern: the *worst* seed is 7.0 vs 7.9 s (steady), 10.8 vs 12.3 s (rush),
  26.3 vs 36.6 s (hell). The seed spread (sd 0.1 / 0.2 / 1.1 s) is far smaller than the margin. The
  original seed 0 was a good draw (rank 1 of 6 in rush, 3 of 6 in hell), but it doesn't matter: the worst
  seed still beats the best rule everywhere.
  The rush margin is still modest: about −15% (10.5 vs 12.3 s).
- **Worst wait:** PPO β = 0.5 is at or below the best rule in rush on every seed (86–94 s vs 94 s for mean-wait fixed-time) and
  in the hell hour (234–287 s vs 374–380 s). In steady, LQF + min green (51 s) and the 30-s guard (44 s)
  match or beat it.
- **The Quiet Lane hell-hour weakness is consistent, and a bit worse than seed 0 showed.** Quiet Lane
  worst wait in the hell hour is 93–115 s across the six seeds (mean 105 s; seed 0's 96 s was the
  second best). The score-tuned fixed-time plan holds it at 46 s, mean-wait fixed-time at 76 s, and
  LQF + min green at 106 s, i.e. PPO now ties the min-green rule there instead of beating it. In rush,
  PPO's Quiet worst is 68–76 s on every seed: below LQF + min green (117 s), about equal to
  max-pressure + hysteresis (67 s) and mean-wait fixed-time (76 s), above score-tuned fixed-time (46 s).
- **Throughput and cars left:** 3957–3958 veh/h and 21 cars left in the hell hour on every β = 0.5 seed,
  vs 3949 veh/h / 30 cars for LQF + min green. The cars-left gap is consistent across seeds. The
  9-veh/h throughput gap (about 0.2%) is the same 9 vehicles seen from the other side (common arrivals:
  completed + still queued = arrived), so it is consistent but not a separate finding or a throughput win.

### β = 0 ablation, now 4 seeds: the fairness term looks like it matters in the hell hour
The one-seed ablation could not tell whether the squared front-wait term helps. With four β = 0 seeds
against six β = 0.5 seeds: steady and rush are close (mean wait equal; rush Quiet worst 82 ± 10 vs
73 ± 3 s, worst 102 vs 90 s). The hell hour separates them: **Quiet Lane worst 175 ± 61 s (114–259)
vs 105 ± 8 s**, worst wait 353 vs 255 s, and two of four β = 0 seeds have collapse episodes. β = 0
seed 3, episode 0: 673 cars left at 18:00, a 1688-s (28-min) worst wait, Quiet Lane car held 400 s,
throughput 3270 veh/h, yet its completed-trips mean wait is only 32 s (120 s counting the queued cars).
β = 0 seed 2: episodes 0 and 19 end with 195 and 288 cars left. No β = 0.5 seed has a hell-hour episode
above 34 cars left. So the earlier "the queue-only policy also avoided starvation" holds for steady and
rush, but not for the hell hour. Caveats: 4 vs 6 seeds; **the two collapsing β = 0 seeds (2 and 3) are exactly the two runs that
stopped ~13% short of 342 iterations** (298/299, job 2 under load), so "β = 0 collapses" and "β = 0 was
trained less" are confounded (seed 2's pick is from iteration 90, so a longer run could only have helped by
finding a better-validating later checkpoint). The full-length β = 0 seed 1 does not collapse but still has
a 173-s Quiet worst, and seed 0 (114 s) matches β = 0.5, so the Quiet-Lane gap rests on 3 of 4 seeds; selection by `mean + 0.1 × max` on training-distribution validation never sees a
demand as extreme as the hell hour, so it cannot screen these policies out.

### Surprise: the completed-trips mean hides the collapse
For β = 0 seed 3 the held-out mean wait (23.0 s) is the *lowest* of any controller in the hell hour,
while one of its episodes strands 673 cars. Mean wait over completed crossings rewards a policy that
stops serving an approach; cars-left, worst wait and `mean_wait_incl_unfinished` catch it
(27.4 ± 21.6 s). For β = 0.5 the two means agree within 0.1 s on every seed.

### Not measured
- Seeds were not checked for checkpoint/resume (runs ≤ 5 min, not needed). No new demo media.
- No re-tuning of the rules (and no joint min-green × hysteresis search, fixed-time offsets): the
  reviewer's remaining baseline gaps are unchanged.
- Why β = 0 collapses in the hell hour (which approach is starved, and whether it is spillback lock-up
  or plain starvation) was not analysed beyond the numbers above.
- Speed numbers: none new (training ran 4-at-a-time; job 2 under load average 13–14).
