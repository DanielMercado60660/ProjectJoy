# Results: Roomba Rocket League (first satisfying version)

## Headline (all numbers are on held-out starts, 500 matches per cell)
| policy | vs goalie (trained) heldout / corner / kickoff | vs keeper (unseen) heldout | vs chaser (unseen) heldout goal / conceded+own | file |
|---|---|---|---|---|
| scripted attacker (baseline) | 2.0% / 2.4% / 0.6% | **49.4%** | **33.0% / 5.0%** | `results/eval.json` |
| PPO run 2, main run, final (18 min) | 6.8% / 4.8% / 9.0% | 0.0% | 13.2% / 67.6% | `results/eval.json` |
| PPO run 2 snapshot (10 min) | 27.8% / 19.6% / 31.6% | 0.6% | 12.2% / 74.0% | `results/eval_run2_snapshot_10min.json` |
| **PPO run 4, fixed, final (10 min)** | **72.8% / 54.4% / 88.8%** | 0.2% | 9.8% / 62.2% | `results/eval_run4_final.json` |
| reviewer's off-centre-aim attacker (aim 0.65, tuned on seed 77) | 4.2% / 1.4% / 2.6% | 39.2% | 36.2% / 3.4% | `results/review_checks.json` |

The short version:
- The learned robot destroys the scripted attacker against the goalie it trained on. Run 4 scores 73% vs 2% held-out, and 54% vs 2.4% in the corner region that no training reset ever used.
- It is useless against a defender it never saw. It scores 0.2% against the line keeper, which the scripted attacker beats 49% of the time.
- It also concedes most matches against the striker, because it never learned to defend.
- The main run (run 2) collapsed late. Two bugs were found: unbounded action means and a CPU/GPU log-prob mismatch. Both were fixed together in run 4 and their effects were not separated (reviewer: the log-prob mismatch accounts for only ~10–15% of run 2's logged KL, see Diagnosis). Run 4 is the same setup with those two fixes, plus run 3's entropy 0 and lr anneal, on a 10-min budget. It was run after the write-up as an ablation, so it is clearly labelled as a separate run.
- Single seed for everything (training seed 0, one run per config).
- The scripted attacker is weak against the goalie: it drives straight through the goal centre, where the goalie stands, and never reverses. The reviewer's cheap fix (aim at the open side of the mouth, tuned on non-eval starts) only raises it to 4.2% held-out, so the PPO gap shrinks from 36× to ~17× but stays large. A stronger heuristic (feints, dribbling around the goalie) was not tried.
- Mild selection across runs: all four runs were evaluated on the same held-out seeds (1001–1003), and run 4's config was chosen after seeing runs 2–3 evaluated. Run 4 itself was run once, and the headline is its final checkpoint, not the best-by-training one.

## Setup in one paragraph
One PPO learner (MLX, 64×64 tanh actor and critic, Gaussian actions, GAE λ=0.95, γ=0.99) plays 1-v-1 against
a scripted **goalie** on a 2 m × 1.4 m pitch. It runs 256 batched numpy envs with 128-step rollouts,
4 epochs of minibatch 4096, lr 3e-4 and entropy coefficient 0.002 (runs 1–2; runs 3–4 change this, see below). Training stops on an 18-minute
wall-clock budget. Curriculum: ball starts ramp linearly from "near the opponent goal" to "anywhere"
over the first half of the budget, and 20% of resets are always "anywhere". The **baseline** is a
scripted attacker (get behind the ball on the goal→ball line, drive through). It sees the same full
state and drives the same robot at the same top speed, so information and embodiment are matched;
compute is not (it is a hand-written controller). The scripted opponents are 0.7× slower than the
learner. No `mx.compile` is used anywhere, so the orchestrator's compile-bakes-weights warning doesn't apply.

## Definitions (used in every table)
- **goal**: the ball crosses the +x goal line inside the mouth.
- **conc.**: the ball enters the learner's goal with the opponent as last toucher.
- **own goal**: the ball enters the learner's goal with the learner as last toucher.
- **timeout**: 15 s with no goal.
- **1st touch**: fraction of matches where the learner touched the ball before the opponent did (the interception race).
- **touch share**: learner touches / all touches.
- **closer to ball**: fraction of steps where the learner is nearer the ball than the opponent. This is a weak possession proxy, because a goalie that stays home loses it by design.
- **regains/match**: number of times last-touch passes from the opponent to the learner (interceptions / tackles won).
- **pin frac**: fraction of steps where the learner touches the ball while the ball is within 2 cm of a wall.
- **pin ≥2s**: fraction of matches with at least 40 such steps.
- Start sets, all on fixed seeds never used in training:
  - `heldout`: the training distribution at full spread, from an unseen seed.
  - `corner`: ball in the four corners (|x|>0.5, |y|>0.42). This region is **excluded from every training reset**.
  - `kickoff`: ball at centre, learner in its own half.
- Each cell is 500 matches with a deterministic policy (mean action).

## Run 1 (killed at 5.1 min): a reward bug that taught corner-pinning
Run 1 paid **+0.05 for every step the learner touched the ball**. A ball wedged into a corner is a
touch every step: 300 steps × 0.05 = **15 per match**, more than the +10 for a goal, which would also
end the match. Discounted with the timeout bootstrap, pinning is a risk-free farm. By 3 minutes the
eval showed the learner pinning the ball in 87% of matches. I killed the run at update 520 (5.1 min,
17 M steps), kept its update-500 checkpoint (`checkpoints/run1_touchbonus.npz`), and relaunched with the
touch bonus paid **once, on first touch**. Nothing else changed.

The exploit is a nice one. The learner pushes the ball along the end wall into the corner **beside the
opponent's goal mouth**, which is the highest-potential place on the pitch outside the goal itself
(Φ = −2·|ball−goal| is large there). It then sits on the ball and twitches to keep contact. The goalie
just watches, because its rush zone (0.45 m from goal centre) doesn't reach the corner.
Clip: `media/run1_touchbonus/ppo_pin_exploit.gif` (selected: the 4 matches with most pin time out of 200).
Unselected matches: `media/run1_touchbonus/ppo_vs_goalie.gif`.

Run 1 eval (`results/eval_run1_touchbonus.json`, 500 matches per cell):

| policy | opponent | starts | goal | conc. | own goal | timeout | 1st touch | touch share | closer to ball | regains/match | t→goal s | pin frac | pin ≥2s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo | goalie | heldout | 0.092 | 0.000 | 0.004 | 0.904 | 0.922 | 0.979 | 0.966 | 1.72 | 5.47 | 0.635 | 0.862 |
| ppo | goalie | corner | 0.082 | 0.000 | 0.002 | 0.916 | 0.998 | 0.980 | 0.969 | 1.70 | 6.20 | 0.660 | 0.880 |
| ppo | goalie | kickoff | 0.124 | 0.000 | 0.002 | 0.874 | 1.000 | 0.980 | 0.975 | 1.55 | 4.58 | 0.638 | 0.848 |
| ppo | keeper | heldout | 0.156 | 0.000 | 0.004 | 0.840 | 0.994 | 0.785 | 0.855 | 2.33 | 5.57 | 0.451 | 0.686 |
| ppo | keeper | corner | 0.262 | 0.000 | 0.008 | 0.730 | 0.998 | 0.770 | 0.815 | 1.98 | 4.29 | 0.425 | 0.560 |
| ppo | keeper | kickoff | 0.178 | 0.000 | 0.002 | 0.820 | 1.000 | 0.755 | 0.860 | 2.51 | 5.41 | 0.412 | 0.642 |
| ppo | chaser | heldout | 0.272 | 0.362 | 0.114 | 0.252 | 0.696 | 0.679 | 0.691 | 1.90 | 4.32 | 0.139 | 0.172 |
| ppo | chaser | corner | 0.350 | 0.234 | 0.046 | 0.370 | 0.950 | 0.801 | 0.821 | 2.03 | 3.98 | 0.262 | 0.282 |
| ppo | chaser | kickoff | 0.162 | 0.552 | 0.132 | 0.154 | 0.518 | 0.548 | 0.603 | 1.70 | 4.68 | 0.085 | 0.098 |
| scripted_attacker | goalie | heldout | 0.020 | 0.000 | 0.012 | 0.968 | 0.916 | 0.831 | 0.918 | 4.22 | 4.57 | 0.026 | 0.020 |
| scripted_attacker | goalie | corner | 0.024 | 0.000 | 0.010 | 0.966 | 1.000 | 0.835 | 0.927 | 3.94 | 4.82 | 0.029 | 0.026 |
| scripted_attacker | goalie | kickoff | 0.006 | 0.000 | 0.008 | 0.986 | 1.000 | 0.826 | 0.930 | 4.56 | 11.28 | 0.027 | 0.012 |
| scripted_attacker | keeper | heldout | 0.494 | 0.000 | 0.000 | 0.506 | 0.992 | 0.650 | 0.733 | 5.90 | 8.18 | 0.026 | 0.006 |
| scripted_attacker | keeper | corner | 0.536 | 0.000 | 0.000 | 0.464 | 1.000 | 0.659 | 0.769 | 4.67 | 6.96 | 0.038 | 0.012 |
| scripted_attacker | keeper | kickoff | 0.472 | 0.000 | 0.002 | 0.526 | 1.000 | 0.624 | 0.725 | 6.49 | 8.02 | 0.027 | 0.010 |
| scripted_attacker | chaser | heldout | 0.330 | 0.020 | 0.030 | 0.620 | 0.754 | 0.602 | 0.696 | 7.37 | 5.39 | 0.112 | 0.322 |
| scripted_attacker | chaser | corner | 0.476 | 0.004 | 0.014 | 0.506 | 0.954 | 0.644 | 0.720 | 7.03 | 5.72 | 0.112 | 0.320 |
| scripted_attacker | chaser | kickoff | 0.408 | 0.028 | 0.070 | 0.494 | 0.930 | 0.624 | 0.742 | 6.32 | 4.66 | 0.096 | 0.252 |

Even while farming, run 1 out-scores the scripted attacker against the goalie: 9.2% vs 2.0% held-out,
8.2% vs 2.4% in the held-out corners, 12.4% vs 0.6% from kickoff. Against the unseen line keeper it is
much worse (15.6% vs 49.4%), because it spends the match pinning instead of shooting.

## Run 2 (main run): first-touch bonus only
Same config as run 1 except the touch bonus: +0.5 once, on the learner's first touch. The run ran
18.0 min (19:01–19:19, after waiting ~5 min for a GPU slot), 3,381 updates and 110.8 M env steps.
Throughput was ~103 k steps/s **under load** (uptime load average 11.6 at start and 7.7 at end, with ten
builders running). Logs: `logs/train.jsonl`; curve: `media/training_curve.png`.

**What happened in training.** The goal rate over training episodes rose to **91%** at 4 min, but on
the easy curriculum (ball near the goal, curriculum 0.46). It peaked at **36.5%** once starts were fully
spread (9 min), then **slowly collapsed to 5.5%** by 18 min. At the same time the action std, after
dropping to 0.11, **rose back to 0.83**, and the approximate KL per update sat at 0.02–0.03, high for PPO.
My first reading, that the entropy bonus plus a constant lr let the policy drift, turned out to be **wrong**. Run 3 refuted it and the diagnosis below gives the real cause. I didn't keep
best-so-far checkpoints (the trainer overwrote `ppo.npz` every 25 updates), so the final checkpoint
is the collapsed policy. One mid-run snapshot exists: I copied `ppo.npz` at 19:12 to check progress
(update 2450, 10 min, 80 M steps). That copy wasn't picked by looking at eval scores, and it is now
`checkpoints/run2_snapshot_10min.npz`. Both are reported below. Clips from the snapshot are in
`media/run2_snapshot_10min/` and clips from the final checkpoint are in `media/run2_final/`.

### Run 2 vs the baseline, against the training goalie (500 held-out matches per cell)
| starts | scripted attacker goal | PPO snapshot (10 min) goal | PPO final (18 min) goal | random goal |
|---|---|---|---|---|
| heldout | 2.0% | **27.8%** | 6.8% | 0.0% |
| corner (excluded from training) | 2.4% | **19.6%** | 4.8% | 0.0% |
| kickoff | 0.6% | **31.6%** | 9.0% | 0.0% |

Own goals against the goalie: snapshot 1.2% / 1.2% / 4.4% and final 0.2% / 0.6% / 0.4%, against 1.2% / 1.0% / 0.8% for
the scripted attacker. The goalie never attacks, so "conceded" is always 0 against it.

### Cross-play: defender styles never seen in training (held-out starts)
| opponent | metric | scripted attacker | PPO snapshot | PPO final |
|---|---|---|---|---|
| keeper (line keeper) | goal | **49.4%** | 0.6% | 0.0% |
| keeper | 1st touch | 99.2% | 55.8% | 42.4% |
| chaser (striker) | goal | **33.0%** | 12.2% | 13.2% |
| chaser | conceded + own goal | 5.0% | 74.0% | 67.6% |
| chaser | 1st touch | 75.4% | 35.0% | 39.0% |

The learned robot **overfits its one opponent hard**. Against the goalie it beats the hand-written
attacker by 10–50× in goal rate. Against a line keeper, which the scripted attacker beats half the time,
it almost never scores, and in about half the matches it doesn't even reach the ball first. Watch
`media/run2_snapshot_10min/ppo_vs_keeper.gif`: it hovers and noodles around the ball instead of lining up.
My hypothesis (not tested): the policy reads the defender's position as a cue. Against the goalie, a
defender sitting deep means "the ball is far from goal, go and get it". The keeper sits deep all the
time, which is out of distribution. Against the chaser it concedes most matches because it never had
to defend (the goalie never shoots), and it scores own goals in 9–15% of matches when it collides with
the chaser near its own goal.

Interception and possession against the goalie (snapshot vs scripted, held-out): 1st touch 80% vs
92%, touch share 0.69 vs 0.83, and regains per match 1.99 vs 4.22. The scripted attacker touches the
ball more and wins it back more often. It just can't get it past the goalie. The learner touches less
and scores more, so it seems to have learned *where to put the ball*, not just how to stay on it.

**Pinning after the fix:** with the per-step touch bonus gone, pin time dropped from 63% of steps (run 1) to
0.9% (snapshot) and 0.2% (final). The potential term alone did not bring the pinning back.

Full tables (500 matches per cell):

Run 2 snapshot, update 2450 (`results/eval_run2_snapshot_10min.json`):

| policy | opponent | starts | goal | conc. | own goal | timeout | 1st touch | touch share | closer to ball | regains/match | t→goal s | pin frac | pin ≥2s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo | goalie | heldout | 0.278 | 0.000 | 0.012 | 0.710 | 0.804 | 0.690 | 0.746 | 1.99 | 8.78 | 0.009 | 0.004 |
| ppo | goalie | corner | 0.196 | 0.000 | 0.012 | 0.792 | 0.614 | 0.519 | 0.739 | 1.19 | 8.31 | 0.005 | 0.000 |
| ppo | goalie | kickoff | 0.316 | 0.000 | 0.044 | 0.640 | 0.972 | 0.783 | 0.801 | 2.19 | 9.21 | 0.009 | 0.006 |
| ppo | keeper | heldout | 0.006 | 0.000 | 0.006 | 0.988 | 0.558 | 0.427 | 0.692 | 1.90 | 4.93 | 0.005 | 0.006 |
| ppo | keeper | corner | 0.016 | 0.000 | 0.002 | 0.982 | 0.608 | 0.500 | 0.667 | 4.15 | 6.03 | 0.022 | 0.028 |
| ppo | keeper | kickoff | 0.000 | 0.000 | 0.002 | 0.998 | 0.610 | 0.416 | 0.736 | 1.14 | – | 0.002 | 0.000 |
| ppo | chaser | heldout | 0.122 | 0.646 | 0.094 | 0.138 | 0.350 | 0.289 | 0.394 | 1.38 | 5.46 | 0.010 | 0.008 |
| ppo | chaser | corner | 0.208 | 0.490 | 0.122 | 0.180 | 0.584 | 0.402 | 0.472 | 1.79 | 5.45 | 0.023 | 0.016 |
| ppo | chaser | kickoff | 0.160 | 0.574 | 0.112 | 0.154 | 0.312 | 0.423 | 0.478 | 2.14 | 5.86 | 0.025 | 0.024 |

Run 2 final, update 3381, with baselines (`results/eval.json`):

| policy | opponent | starts | goal | conc. | own goal | timeout | 1st touch | touch share | closer to ball | regains/match | t→goal s | pin frac | pin ≥2s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo | goalie | heldout | 0.068 | 0.000 | 0.002 | 0.930 | 0.526 | 0.436 | 0.777 | 0.92 | 8.57 | 0.002 | 0.000 |
| ppo | goalie | corner | 0.048 | 0.000 | 0.006 | 0.946 | 0.450 | 0.358 | 0.741 | 0.64 | 7.91 | 0.002 | 0.000 |
| ppo | goalie | kickoff | 0.090 | 0.000 | 0.004 | 0.906 | 0.796 | 0.608 | 0.800 | 1.12 | 9.42 | 0.003 | 0.002 |
| ppo | keeper | heldout | 0.000 | 0.000 | 0.006 | 0.994 | 0.424 | 0.345 | 0.747 | 0.62 | – | 0.002 | 0.002 |
| ppo | keeper | corner | 0.002 | 0.000 | 0.000 | 0.998 | 0.500 | 0.413 | 0.663 | 1.86 | 11.05 | 0.020 | 0.032 |
| ppo | keeper | kickoff | 0.002 | 0.000 | 0.000 | 0.998 | 0.404 | 0.302 | 0.828 | 0.40 | 5.95 | 0.001 | 0.000 |
| ppo | chaser | heldout | 0.132 | 0.550 | 0.126 | 0.192 | 0.390 | 0.340 | 0.413 | 1.68 | 5.86 | 0.016 | 0.018 |
| ppo | chaser | corner | 0.178 | 0.426 | 0.148 | 0.248 | 0.600 | 0.407 | 0.474 | 2.44 | 5.77 | 0.022 | 0.012 |
| ppo | chaser | kickoff | 0.118 | 0.490 | 0.154 | 0.238 | 0.378 | 0.415 | 0.458 | 2.40 | 6.62 | 0.018 | 0.014 |
| scripted_attacker | goalie | heldout | 0.020 | 0.000 | 0.012 | 0.968 | 0.916 | 0.831 | 0.918 | 4.22 | 4.57 | 0.026 | 0.020 |
| scripted_attacker | goalie | corner | 0.024 | 0.000 | 0.010 | 0.966 | 1.000 | 0.835 | 0.927 | 3.94 | 4.82 | 0.029 | 0.026 |
| scripted_attacker | goalie | kickoff | 0.006 | 0.000 | 0.008 | 0.986 | 1.000 | 0.826 | 0.930 | 4.56 | 11.28 | 0.027 | 0.012 |
| scripted_attacker | keeper | heldout | 0.494 | 0.000 | 0.000 | 0.506 | 0.992 | 0.650 | 0.733 | 5.90 | 8.18 | 0.026 | 0.006 |
| scripted_attacker | keeper | corner | 0.536 | 0.000 | 0.000 | 0.464 | 1.000 | 0.659 | 0.769 | 4.67 | 6.96 | 0.038 | 0.012 |
| scripted_attacker | keeper | kickoff | 0.472 | 0.000 | 0.002 | 0.526 | 1.000 | 0.624 | 0.725 | 6.49 | 8.02 | 0.027 | 0.010 |
| scripted_attacker | chaser | heldout | 0.330 | 0.020 | 0.030 | 0.620 | 0.754 | 0.602 | 0.696 | 7.37 | 5.39 | 0.112 | 0.322 |
| scripted_attacker | chaser | corner | 0.476 | 0.004 | 0.014 | 0.506 | 0.954 | 0.644 | 0.720 | 7.03 | 5.72 | 0.112 | 0.320 |
| scripted_attacker | chaser | kickoff | 0.408 | 0.028 | 0.070 | 0.494 | 0.930 | 0.624 | 0.742 | 6.32 | 4.66 | 0.096 | 0.252 |

## Runs 3 and 4 (after the write-up): finding and fixing the collapse
**Run 3** (12 min) tested the obvious guess: set entropy to 0 and anneal the lr linearly to 0. It was
**worse**. The training goal rate peaked at 19% after 47 s (curriculum 0.13), then fell to ~0.5%, and
std again fell to 0.10 and climbed back to 0.51. Held-out goal rate against the goalie: 1.4% (best-by-training
checkpoint) and 0.0% (final) (`results/eval_run3_best.json`, `results/eval_run3_final.json`,
`media/run3_training_curve.png`). So the entropy bonus wasn't the cause.

**Diagnosis** (from the checkpoints, `ppo.py` unchanged otherwise):
1. **The action means blew up.** The Gaussian mean was an unbounded linear output, and the env clips actions to [-1, 1].
   On 256 mid-match states the mean |μ| was **22** (run 2 snapshot), **50** (run 2 final) and **105** (run 3 final),
   with 98–99% of means outside the box. Once the mean sits far outside the box, every sample clips to the same
   wheel command, so the gradient that would bring it back is gone and the mean random-walks outward.
2. **The rollout and the update disagreed about log-probs.** Rollouts ran on CPU (fp32) and the PPO update ran on GPU,
   where matmuls are TF32 by default (~1e-3 relative error). With |μ|≈100 that is ~0.1–0.4 of absolute error in μ
   (measured CPU-vs-GPU max diff: 0.05, 0.13 and 0.41 for the three checkpoints). Divided by σ≈0.1–0.4, that is
   enough to make the importance ratio noisy before any update has happened. *(Reviewer correction.)* The builder wrote
   that the logged "KL" of 0.02–0.06 was mostly this artefact and fell to ~1e-5 when recomputed on-device; no file
   records that measurement. The reviewer measured the artefact alone (same weights, CPU vs GPU, same k3 estimator as
   the training log, 256 mid-match states, `review_checks.py` → `results/review_checks.json`): approx-KL **0.004**
   (run 2 snapshot) and **0.002** (run 2 final), against a logged 0.022–0.033 late in run 2, so the mismatch is only
   ~10–15% of run 2's KL. For run 3 it is **0.066** against a logged 0.058–0.067, so there it explains essentially
   all of it. On run 4's final weights (tanh mean, std 0.008/0.033) the artefact is still 0.009 with 12% of ratios
   outside the clip, so the on-device recompute matters even with a bounded mean. Tanh alone and relp alone were
   never run: **both fixes were applied together and their effects are not separated.**

**Run 4** (10 min) changes only those two things: the mean is tanh-squashed (`--tanh-mean 1`) and the
old log-probs are recomputed on the training device before each update (`--relp 1`). It keeps ent 0 + lr
anneal + keep-best from run 3. Training goal rate: 82% at curriculum 0.55, then **86% at full spread**
(5 min), and it ended at **74–80%** (last five logged updates) with a steady std of 0.02 and KL ~0.002 (`logs/run4_tanh_relp.jsonl`,
`media/run4_training_curve.png`). Throughput was ~304 k steps/s (182.6 M steps / 600 s) under **light** load (uptime load average 2.0 at the end, `logs/run4_uptime_end.txt`; no start reading; much lighter than run 2's 11.6 / 7.7). The demo clips in `media/` are from run 4's final checkpoint
(`checkpoints/run4_tanh_relp.npz`). The best-by-training-goal-rate checkpoint (`run4_tanh_relp_best.npz`,
update 2805) scores 79.4% / 58.2% / 87.4% against the goalie (`results/eval_run4_best.json`).

**What run 4 learned:** it drives **backwards into the ball**. Over the first 150 steps of 500
held-out goalie matches, 47% of steps have negative forward speed, and in 74% of near-ball steps the
ball is *behind* the robot (`probe_reverse.py`, `results/probe_reverse.txt`; the run 2 snapshot shows the
same at 77%). The robot is a symmetric disc and reverse speed is the same as forward speed, so this is
a legal and apparently convenient way to dribble. It shows up in every clip (the white heading tick
points away from the ball).

**Cross-play is still bad after the fix.** Against the keeper it scores 0.2%, even though its first-touch
rate rose to 83%: it gets to the ball and then can't finish against a keeper that never leaves its line.
Against the chaser it concedes 31.6% and scores own goals in 30.6% of matches, because it plays no
defence and bundles the ball into its own net in scrambles. The overfitting finding from run 2
survives the fix, so it is about the training opponent, not the bug.

Run 4 final, full table (`results/eval_run4_final.json`):

| policy | opponent | starts | goal | conc. | own goal | timeout | 1st touch | touch share | closer to ball | regains/match | t→goal s | pin frac | pin ≥2s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo | goalie | heldout | 0.728 | 0.000 | 0.002 | 0.270 | 0.894 | 0.815 | 0.880 | 2.88 | 8.34 | 0.023 | 0.010 |
| ppo | goalie | corner | 0.544 | 0.000 | 0.010 | 0.446 | 0.834 | 0.716 | 0.894 | 2.39 | 8.80 | 0.019 | 0.004 |
| ppo | goalie | kickoff | 0.888 | 0.000 | 0.000 | 0.112 | 1.000 | 0.840 | 0.902 | 3.14 | 8.09 | 0.021 | 0.004 |
| ppo | keeper | heldout | 0.002 | 0.000 | 0.000 | 0.998 | 0.834 | 0.695 | 0.849 | 2.00 | 5.05 | 0.015 | 0.026 |
| ppo | keeper | corner | 0.014 | 0.000 | 0.006 | 0.980 | 0.888 | 0.760 | 0.854 | 2.10 | 7.17 | 0.023 | 0.040 |
| ppo | keeper | kickoff | 0.010 | 0.000 | 0.000 | 0.990 | 0.942 | 0.754 | 0.871 | 2.55 | 8.44 | 0.019 | 0.036 |
| ppo | chaser | heldout | 0.098 | 0.316 | 0.306 | 0.280 | 0.548 | 0.567 | 0.540 | 2.48 | 5.49 | 0.087 | 0.168 |
| ppo | chaser | corner | 0.122 | 0.162 | 0.256 | 0.460 | 0.798 | 0.675 | 0.639 | 2.88 | 5.66 | 0.123 | 0.198 |
| ppo | chaser | kickoff | 0.006 | 0.554 | 0.270 | 0.170 | 0.150 | 0.385 | 0.418 | 1.74 | 7.72 | 0.088 | 0.174 |
| scripted_attacker | goalie | heldout | 0.020 | 0.000 | 0.012 | 0.968 | 0.916 | 0.831 | 0.918 | 4.22 | 4.57 | 0.026 | 0.020 |
| scripted_attacker | goalie | corner | 0.024 | 0.000 | 0.010 | 0.966 | 1.000 | 0.835 | 0.927 | 3.94 | 4.82 | 0.029 | 0.026 |
| scripted_attacker | goalie | kickoff | 0.006 | 0.000 | 0.008 | 0.986 | 1.000 | 0.826 | 0.930 | 4.56 | 11.28 | 0.027 | 0.012 |
| scripted_attacker | keeper | heldout | 0.494 | 0.000 | 0.000 | 0.506 | 0.992 | 0.650 | 0.733 | 5.90 | 8.18 | 0.026 | 0.006 |
| scripted_attacker | keeper | corner | 0.536 | 0.000 | 0.000 | 0.464 | 1.000 | 0.659 | 0.769 | 4.67 | 6.96 | 0.038 | 0.012 |
| scripted_attacker | keeper | kickoff | 0.472 | 0.000 | 0.002 | 0.526 | 1.000 | 0.624 | 0.725 | 6.49 | 8.02 | 0.027 | 0.010 |
| scripted_attacker | chaser | heldout | 0.330 | 0.020 | 0.030 | 0.620 | 0.754 | 0.602 | 0.696 | 7.37 | 5.39 | 0.112 | 0.322 |
| scripted_attacker | chaser | corner | 0.476 | 0.004 | 0.014 | 0.506 | 0.954 | 0.644 | 0.720 | 7.03 | 5.72 | 0.112 | 0.320 |
| scripted_attacker | chaser | kickoff | 0.408 | 0.028 | 0.070 | 0.494 | 0.930 | 0.624 | 0.742 | 6.32 | 4.66 | 0.096 | 0.252 |

## Measure items: status
| item | status |
|---|---|
| goals | measured (tables above) |
| possession / interception | measured: 1st touch (the interception race), touch share, regains/match, and the closer-to-ball proxy |
| own goals | measured |
| cross-play with unfamiliar opponents | measured: keeper and chaser, never seen in training |
| cross-play with unfamiliar *teammates* | **not measured**, because this is 1-v-1 (teammates belong to the 2-v-2 "go deeper" stage) |
| held-out start positions | measured: unseen-seed starts, plus the corner region excluded from all training resets |
| idle-machine throughput | not measured (only the under-load figure above) |
| seeds | **not measured**: a single training seed (0), so the gaps above carry unknown seed variance |

## What worked
- The batched numpy arena is fast enough that compute was never the bottleneck: 100–300 k steps/s under load, with MLX policy inference on CPU during rollouts. CPU rollout inference was 4× faster than GPU at 64 envs in the smoke test (0.054 s vs 0.22 s per 128-step rollout; reviewer: this comparison is not in any log file).
- The curriculum (near goal → anywhere) got first goals within a minute in every run.
- After the fix (run 4), the learned striker beats the scripted striker against its training defender by 36× held-out and 23× in the never-trained corner region, with 0–1% own goals. Against the reviewer's off-centre-aim attacker (4.2% / 1.4%) the margins are ~17× and ~39×.
- Instrumenting pin time from the first version made the run 1 exploit visible at minute 3 rather than after the run.

## What didn't
- The main run (run 2) collapsed late, from 36% to 5.5% training goal rate. Unbounded action means plus the CPU-fp32/GPU-TF32 log-prob mismatch are the two fixes that made run 4 work; which one mattered more is not measured (the mismatch is only ~10–15% of run 2's logged KL). The main-run headline is the weaker one, and the strong number comes from a later 10-minute fix run (run 4).
- Generalisation to other opponents is poor in every run. Against the unseen keeper it scores ≤3.6% (all PPO checkpoints from runs 2–4 and all start sets; run 1's pinning policy scores 15.6–26.2%) vs 47–54% for the scripted attacker. Against the chaser it concedes or own-goals in 42–82% of matches (run 4 final), vs 2–10% for the scripted attacker.
- No best-checkpoint saving in run 2. That was added in run 3.
- Run 1's reward bug cost 5 minutes of slot time (a good failure, documented above).

## Surprises
- Run 1 farmed the touch bonus by pinning the ball in the corner **next to** the goal mouth, the highest-potential spot that the goalie's rush zone doesn't reach.
- The TF32 note from the brief bit here in a new way. It isn't a parity-test curiosity: with large action means, CPU-rollout vs GPU-update log-probs disagree enough to corrupt PPO's importance ratios.
- The learner dribbles **in reverse**: in 74% of near-ball steps the ball is behind it.
- Against the keeper, run 4 reaches the ball first 83% of the time and still almost never scores.

## Next steps
1. Train against a mix of goalie, keeper and chaser (an opponent population), and keep one style held out for cross-play.
2. Add defence: a conceded-goal signal from a striker opponent, or self-play.
3. Run 3 seeds of the run 4 config before trusting the size of any gap (everything here is seed 0).
4. Re-measure throughput on an idle machine.
5. "Go deeper" (2-v-2, imperfect sensors, communication) not started.

## Follow-up 1 (round 3, 2026-10-22 20:05–20:45): training against a defender pool
**Bottom line (read first): this run is not a test of opponent pools.** It was compute-starved (41.0 M env steps vs
run 4's 182.6 M, machine load 6.5 → 29.4) and ppo.py ties the curriculum and lr anneal to wall clock, so the schedule
was compressed too. Under that budget the pool policy collapsed on the goalie (72.8% → 1.4%) and barely moved on the
unseen keeper (0.2% → 1.8%). It does not show that pools fail here; a step-matched rerun is needed.

**Question.** Run 4 overfits its one opponent: 72.8% vs the goalie it trained with, 0.2% vs the unseen line keeper,
and it concedes 62% vs the unseen chaser. Does training against a mixed pool of defender styles fix that?

**Setup.** `pool.py` + `ppo_pool.py` (ppo.py with the opponent swapped; run 4's flags: tanh mean, on-device
log-probs, ent 0, lr anneal, keep-best, 256 envs, minibatch 4096, **10-min wall budget, seed 0, one run**). Each
training episode draws one defender uniformly from:
- `goalie`, the fixed training goalie of runs 1–4 (kept, so the cost on the goalie compares like with like),
- `goalie_rand`, a randomized goalie family drawn per episode: guard standoff U[0.08, 0.40] m, rush radius
  U[0.25, 0.70] m, speed ×U[0.7, 1.0],
- `chaser`, the striker (unseen for run 4, **seen** here), using its own start positions as in eval.
**Held out entirely: `keeper`** (line keeper, never comes out). Evaluation is unchanged: `eval.py`, seeds
1001–1003, 500 matches per cell. Run 4 and scripted-attacker numbers are cited from `results/eval_run4_final.json`;
the scripted attacker in the pool eval json reproduces them exactly (2.0 / 49.4 / 33.0%). The off-centre aim attacker
comes from `results/review_checks.json`. No new held-out defender style was built (not measured).

**Big confound (read this before the table).** Load on the machine went from 6.5 at the start to 29.4 at the end
(`logs/pool_uptime_{start,end}.txt`). The run therefore got only **41.0 M env steps (1,251 updates) in its 10 minutes,
22% of run 4's 182.6 M (5,573 updates)**. The curriculum and lr anneal in ppo.py are tied to *wall clock*, so the pool
run also hit full curriculum after 23 M steps (run 4 took 90 M), and its lr reached 0 at 41 M. For reference, run 4
at the same 41 M steps (update 1,250) was already scoring **76.5%** of its training episodes vs the goalie (curriculum
0.46, `logs/run4_tanh_relp.jsonl`). Over its last 10 logged updates the pool run scored **0–4.5%** of training episodes vs the goalie,
4.6–10.7% vs goalie_rand and 15.8–18.8% vs the chaser (`logs/pool_mixed.jsonl`). So at matched steps the pool run is far behind,
not just shorter. But the two runs also differ in curriculum pace per step, and I could not separate "the pool is harder"
from "this run was starved and rushed". This is a negative result for *this budget*, not proof that pools don't work here.

**Results (held-out starts, 500 matches per cell; goal % / conceded+own-goal %).** Pool = `results/pool/eval_pool_final.json`
(final checkpoint, the headline); the best-by-training checkpoint (update 1,176) is within 1.6 points on goal % and 3.4 points on conceded %
everywhere, no change in conclusions (`results/pool/eval_pool_best.json`). Table from `tables_pool.py`.

vs **goalie** (seen by both):

| policy | heldout | corner | kickoff |
|---|---|---|---|
| run 4 PPO (goalie only) | **72.8 / 0.2** | **54.4 / 1.0** | **88.8 / 0.0** |
| pool PPO (final) | 1.4 / 0.0 | 0.4 / 0.0 | 10.6 / 0.0 |
| scripted attacker | 2.0 / 1.2 | 2.4 / 1.0 | 0.6 / 0.8 |
| aim attacker (0.65) | 4.2 / 1.0 | 1.4 / 0.8 | 2.6 / 0.4 |

vs **keeper** (UNSEEN by both):

| policy | heldout | corner | kickoff |
|---|---|---|---|
| run 4 PPO (goalie only) | 0.2 / 0.0 | 1.4 / 0.6 | 1.0 / 0.0 |
| pool PPO (final) | 1.8 / 0.0 | 5.4 / 0.0 | 1.8 / 0.0 |
| scripted attacker | **49.4 / 0.0** | **53.6 / 0.0** | **47.2 / 0.2** |
| aim attacker (0.65) | 39.2 / 0.0 | 44.2 / 0.0 | 44.6 / 0.0 |

vs **chaser** (seen by the pool run only):

| policy | heldout | corner | kickoff |
|---|---|---|---|
| run 4 PPO (goalie only) | 9.8 / 62.2 | 12.2 / 41.8 | 0.6 / 82.4 |
| pool PPO (final) | 20.8 / 53.0 | 13.0 / 49.8 | 4.4 / 80.0 |
| scripted attacker | 33.0 / 5.0 | 47.6 / 1.8 | 40.8 / 9.8 |
| aim attacker (0.65) | **36.2 / 3.4** | **43.8 / 1.0** | **40.6 / 11.2** |

**Answer.** Generalization to the unseen keeper did **not** meaningfully improve: 0.2% → 1.8% held-out (1 → 9 goals of 500; 5.4% corner).
That is still ~27× below the scripted attacker's 49.4%. The cost on the goalie was nearly everything: 72.8% → 1.4%.
Against the chaser, which it now trained on, it scores more (9.8 → 20.8%) and concedes a little less (62.2 → 53.0%),
but it is still much worse than both scripted attackers (≤5% conceded). On the heldout and corner sets the pool policy is below both scripted attackers against every defender; the one
exception is goalie/kickoff (10.6% vs at most 2.6%). Under this budget the pool made things worse overall.

**The good failure: "win the race, then park".** The pool policy reaches the ball first more often than run 4 (first
touch 92% vs the goalie, 99% vs the keeper; run 4 had 89% / 83%), and it gets there faster (1.07 s vs 1.71 s). Then it
almost never shoots: 98.6% of matches vs the goalie and 98.2% vs the keeper end in **timeouts with zero conceded**.
The new part is vs the goalie (run 4: 27.0% timeouts); run 4 already timed out 99.8% vs the keeper.
In the clips (`media/pool/pool_vs_goalie.gif`, `media/pool/pool_vs_keeper.gif`, first 4 held-out starts, unselected,
all 8 are timeouts) it nudges the ball toward the defender's corner and loiters beside it. **Suspected, untested** reason (no chaser-free pool was run, so this is a hypothesis): about a third of
its training episodes were against the chaser, which turns any loose ball into a −10, and the goalie/keeper rarely
concede a goal to a weak shot. So "keep the ball away from our goal and stall" beats "shoot" for a policy that hasn't
learned to beat a goalie yet. Mean training reward per step was negative in 1,210 of 1,251 updates (run 4: positive in 5,522 of 5,573); not like for
like, since the pool's chaser episodes carry −10 concessions that run 4 never saw. Vs the chaser
(`media/pool/pool_vs_chaser.gif`): timeout, conceded, timeout, goal. Training curves per pool opponent, with run 4 for
reference: `media/pool/pool_training_curve.png`.

**What's not measured.** Seeds beyond 0. A step-matched run (same 182.6 M steps, step-based curriculum). A pool without
the chaser (goalie + goalie_rand only), which would show whether the chaser's −10 pressure causes the stalling. Any
new held-out defender style. Speed numbers: none are claimed (busy machine, load 6.5 → 29.4).

**Next steps.** (1) Make the curriculum and lr anneal step-based instead of wall-clock-based, so a loaded machine can't
compress the schedule, then rerun at matched steps. (2) Start from run 4's weights and fine-tune on the pool, instead of
training from scratch. (3) Ablate the chaser out of the pool. (4) Add an opponent-ID / recent-history input or a
recurrent policy, so it can adapt to the defender style.
