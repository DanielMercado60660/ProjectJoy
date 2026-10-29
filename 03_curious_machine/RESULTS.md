# RESULTS: The Machine That Gets Bored

**Short version.** In a reward-free nine-room house, plain lifelong-novelty explorers (PPO + RND,
PPO + hashed counts) beat a random walk only modestly per episode. They also cycle: they rush to
whatever is novel right now and dither there, and one RND agent spends its episodes flipping a
lamp lever. Adding **episodic novelty** (how often this observation has been seen *in the current
episode*, scaled by an RND multiplier, after NGU) changes the picture. That agent reaches
**7.3 of 9 rooms and 4.95 of 8 mechanisms per 512-step episode, against 2.6 and 1.4 for the random
walk** (3 seeds each), and it unlocks the locked door in 97% of episodes. The random walk unlocked
it once in 35,000 training episodes.
Then the **noisy TV** wrecks it. In a house with a flickering screen, the same agent stares at the
TV for **509 of 512 steps** and triggers nothing (all 3 seeds). Two fixes that favour learnable
structure both ignore the TV: ensemble disagreement and NGU-style episodic memory in an
inverse-dynamics embedding. Neither recovers the strong explorer: they reach 2.7 and 1.0
interactions per episode. TV-robustness here costs most of the exploration gains.

All numbers come from real runs. Per-run logs and eval JSONs were written to `runs/` (git-ignored);
the committed copies are in `results/` (every eval JSON, `summary.json` including the whole-budget
tables, `tv_diagnosis.json`, per-run `log.jsonl` + `config.json`). `plots.py`, `table.py` and
`diagnose_tv.py` read `runs/`, so they only regenerate where `runs/` exists.

## Setup (what was compared, under what budget)

* **World** (README has the map): 9 rooms, 233 walkable cells, 8 mechanisms, 7×7 egocentric view
  (1622 binary features: cell type, room floor colour, TV pixel, plus "holding key" and facing).
  There is no extrinsic reward anywhere.
* **Same action budget per episode:** every agent, including the random walk, gets 512 actions
  per episode from the same start cell.
* **Same training budget:** every learned agent trains for 6.0 M env steps (366 PPO updates of
  128 envs × 128 steps) with seeds 0, 1, 2. The random walk's "training" log is also 6.0 M steps
  per seed, so whole-budget statistics are matched.
* **Evaluation:** 64 fresh 512-step episodes per seed, eval seed 10000+seed (never used in
  training), with the stochastic policy (that is how it explores). Tables give the mean over 3 seeds
  of per-seed means, ± the population std across seeds (ddof = 0; the sample std with n = 3 is
  about 22% larger). Note that the per-episode evaluation measures a policy
  that has *already consumed* 6 M steps of experience, while the random walk needs none. The
  whole-budget tables compare the two at equal total experience.
* **Agents**
  * `random`: uniform random actions.
  * `count`: PPO + 1/√N(hash(obs)), lifelong counts.
  * `rnd`: PPO + Random Network Distillation (Burda et al. 2018). Observation normalisation,
    intrinsic reward divided by the running std of discounted intrinsic returns, non-episodic
    intrinsic value head (γ 0.99), and a separate extrinsic head.
  * `rnd_ep`: PPO + **RND × episodic count**. The episodic term is 1/√n, with n = visits to this
    observation hash in this episode. The RND term is a multiplier clipped to [1, 5]. This is the
    NGU recipe with a hash count instead of kNN.
  * `ep`: ablation, the episodic term alone.
  * `ngu`: noisy-TV fix, **NGU-style**. Like `rnd_ep`, but the episodic term is a kNN similarity
    (NGU kernel, k = 10) in a 32-d embedding trained only through an inverse-dynamics model.
  * `disagree`: noisy-TV fix, **ensemble disagreement** (Pathak et al. 2019). Variance across 5
    forward models that predict a fixed random embedding of the next observation.

## Metric definitions (fixed before the main runs)

* **Rooms reached**: distinct rooms (of 9) entered during one 512-action episode.
* **Useful interactions**: distinct mechanisms (of 8) triggered during one episode: lever A,
  button B, push box, pick up key, unlock door, button E, lever C, lever D. Each counts once per
  episode. Novelty-seeking and useful discovery are kept apart: none of these counts as useful
  just because it looked new.
* **Coverage**: distinct cells (of 233) visited during one episode.
* **Unproductive step**: a step that neither enters a cell not yet visited this episode nor triggers a
  mechanism not yet triggered this episode.
* **Repeat interactions**: re-toggling a lever or re-pressing a button already used this episode.
  An agent that triggers nothing scores 0 here trivially.
  Lever toggles light and unlight a lamp, so they create visual novelty without new discovery.
  This is the base world's measure of "time on unproductive novelty".
* **TV steps** (noisy-TV world only): steps with any of the 3 screen tiles inside the agent's 7×7
  view. This is the noisy-TV world's measure of "time on unproductive novelty". A random walk has
  the TV in view about 24% of the time because of where it is (room R10, next to the start room and
  on the way to the locked door). Views are not occluded by walls, so the TV can also be watched
  from room R11 next door.

## 1. Base world: does curiosity beat the random walk? (first satisfying version)

| agent (base) | seeds | rooms /9 | interactions /8 | coverage /233 | unproductive steps /512 | repeat interactions | TV steps /512 |
|---|---|---|---|---|---|---|---|
| random walk | 3 | 2.59 ± 0.10 | 1.39 ± 0.18 | 49.6 ± 1.9 | 462.4 ± 1.9 | 2.44 ± 0.27 | 0.0 ± 0.0 |
| PPO + hashed counts | 3 | 3.69 ± 0.66 | 2.40 ± 0.58 | 47.5 ± 13.2 | 464.6 ± 13.5 | 0.49 ± 0.63 | 0.0 ± 0.0 |
| PPO + RND | 3 | 3.03 ± 0.03 | 2.23 ± 0.13 | 38.1 ± 2.1 | 473.6 ± 2.3 | 12.96 ± 5.10 | 0.0 ± 0.0 |
| PPO + RND × episodic count | 3 | 7.34 ± 0.14 | 4.95 ± 0.03 | 146.0 ± 3.8 | 364.0 ± 3.8 | 0.10 ± 0.13 | 0.0 ± 0.0 |
| PPO + episodic count only (ablation) | 3 | 4.86 ± 1.30 | 2.32 ± 1.86 | 106.9 ± 24.0 | 405.1 ± 25.3 | 0.03 ± 0.04 | 0.0 ± 0.0 |
| PPO + NGU-style (inverse-dynamics kNN) | 3 | 3.98 ± 0.01 | 1.00 ± 0.00 | 92.2 ± 0.8 | 420.8 ± 0.8 | 0.00 ± 0.00 | 0.0 ± 0.0 |
| PPO + ensemble disagreement | 3 | 2.95 ± 0.90 | 2.35 ± 0.64 | 46.2 ± 11.9 | 465.4 ± 12.0 | 2.07 ± 0.47 | 0.0 ± 0.0 |

| % of eval episodes (base) | lever A (lamp) | button B (near door) | push box | pick up key | unlock door | button E (far door) | lever C | lever D |
|---|---|---|---|---|---|---|---|---|
| random walk | 65 | 30 | 28 | 3 | 0 | 0 | 13 | 0 |
| PPO + hashed counts | 19 | 51 | 95 | 48 | 4 | 0 | 22 | 0 |
| PPO + RND | 0 | 29 | 99 | 0 | 0 | 0 | 95 | 0 |
| PPO + RND × episodic count | 0 | 99 | 100 | 99 | 97 | 0 | 99 | 0 |
| PPO + episodic count only (ablation) | 0 | 33 | 100 | 33 | 33 | 0 | 32 | 0 |
| PPO + NGU-style (inverse-dynamics kNN) | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 0 |
| PPO + ensemble disagreement | 76 | 43 | 74 | 23 | 0 | 0 | 20 | 0 |

| whole training budget (base, 6 M steps/seed) | episodes | key picked up | door unlocked | button E | max rooms in any episode |
|---|---|---|---|---|---|
| random walk (3 seeds) | 34944 | 1891 | 1 | 0 | 6 |
| PPO + hashed counts (3 seeds) | 34944 | 17039 | 30 | 0 | 6 |
| PPO + RND (3 seeds) | 34944 | 3559 | 0 | 0 | 6 |
| PPO + RND × episodic count (3 seeds) | 34944 | 25415 | 18944 | 10 | 8 |
| PPO + episodic count only (ablation) (3 seeds) | 34944 | 8192 | 4951 | 0 | 8 |
| PPO + NGU-style (inverse-dynamics kNN) (3 seeds) | 34944 | 545 | 0 | 0 | 6 |
| PPO + ensemble disagreement (3 seeds) | 34944 | 7060 | 1 | 0 | 6 |

Figures: `media/interaction_timeline_base.png` shows when each mechanism is first triggered within
an episode, plus per-mechanism hit rates. `media/training_curves_base.png` shows per-episode metrics
during training. `media/lifetime_discovery_base.png` shows whole-budget reliability.
**Demo:** `media/explorer_vs_random.gif` shows the random walk next to PPO + RND × episodic count
(seed 0, a median eval episode), with accumulating visitation heatmaps. The last frame is
`media/explorer_vs_random_final.png`.

**What the numbers say**

* **First satisfying version: yes.** On the 3-seed mean, PPO + RND, PPO + counts, the episodic
  agents, and disagreement all reach more rooms and trigger more mechanisms than the random walk
  under the same 512-action budget. Per seed, only RND × episodic count, counts and RND beat the
  random walk's best seed on both rooms and interactions in every seed. The episodic-only ablation's
  seeds 0 and 1 trigger *fewer* mechanisms (1.00 vs 1.39), and disagreement seed 0 reaches fewer
  rooms (1.69 vs 2.59) with interactions inside the random walk's range (1.45). The NGU-style agent
  reaches more rooms but triggers fewer mechanisms (1.00 vs 1.39) in all seeds. RND × episodic count wins by a wide margin: 2.8× the
  rooms, 3.6× the interactions, and 2.9× the coverage of the random walk, with little spread
  across seeds. It completes the whole "button B → key → back across the house → unlock" chain in
  97% of episodes.
* **Plain lifelong novelty (RND, counts) gets bored and settles.** The policy is memoryless and the
  bonus is non-episodic, so the agent learns one route to whatever is novel *right now* and then
  dithers there. Its per-episode coverage is no better than the random walk's, and RND's is worse
  (38 vs 50 cells). The training curves oscillate: the agent chases a frontier, the frontier stops
  being novel, and behaviour collapses for a while. Around 3.5 M steps, RND seed 0 spent whole
  episodes inside the start room (1 room and 1 interaction per episode).
* **The RND agent made its own small noisy TV out of lever C** (a preserved failure). Toggling a
  lever lights or darkens a lamp and changes the observation, so RND keeps paying for it. RND
  agents re-toggle levers 13 times per episode on average (18 for seed 2; random walk 2.4).
  `media/failure_lever_spam_rnd.gif` shows the agent heading straight for the dead-end room behind
  the box and staying there, flipping lever C.
* **The learned agents ignore the start room's lever.** Lever A is in the start room, the most
  familiar place in the house because every episode begins there. The random walk flips it in 65%
  of episodes. RND, RND × episodic, and the episodic-only agent never do (0%): they leave
  immediately. The machine really does get bored. Disagreement is the exception (76%): a forward
  model keeps finding the lever→lamp effect worth testing.
* **Episodic novelty is the big gain, and lifelong novelty makes it reliable.** The episodic-only
  ablation sometimes matches RND × episodic (seed 2: 6.7 rooms, 4.95 interactions). Its other two
  seeds get stuck sweeping the first four rooms (4 rooms, 1 interaction). Multiplying by the RND
  term, which points at what is new over the agent's whole life, got all 3 seeds to the key and
  the lock.
* **No agent solves the hardest chain within 6 M steps.** Button E, in the bottom-left corner of
  R20, opens the door into R22, the only way to reach lever D. RND × episodic pressed it in 10 of
  about 35k training episodes and never in evaluation, and no agent ever reached the ninth room.
  (§3 covers a 30 M-step run.)
* **Whole-budget view.** Over the same 6 M steps, the random walk unlocked the door once in 34,944
  episodes and RND × episodic unlocked it in 18,944. On the question "did it *ever* happen",
  curiosity buys less here than per-episode numbers suggest: with 128 parallel envs, every agent
  triggers the easy mechanisms within the first few hundred episodes. The difference is how
  *reliably* each agent triggers them, and whether it gets past the locked door at all.

## 2. Noisy TV (go deeper)

Three floor tiles in room R10 (east wall) show a fresh random colour (1 of 8) every step, so
there are 8³ = 512 possible screens. Nothing about them is learnable. Everything else is identical
to the base world. Same budgets and seeds.

| agent (tv) | seeds | rooms /9 | interactions /8 | coverage /233 | unproductive steps /512 | repeat interactions | TV steps /512 |
|---|---|---|---|---|---|---|---|
| random walk | 3 | 2.62 ± 0.09 | 1.41 ± 0.19 | 49.0 ± 2.0 | 462.9 ± 2.0 | 2.50 ± 0.31 | 122.9 ± 17.1 |
| PPO + hashed counts | 3 | 3.00 ± 0.00 | 1.29 ± 0.25 | 20.8 ± 1.8 | 491.9 ± 2.1 | 2.27 ± 2.14 | 492.3 ± 5.4 |
| PPO + RND | 3 | 3.01 ± 0.01 | 1.43 ± 0.31 | 26.7 ± 4.6 | 485.9 ± 4.8 | 1.97 ± 0.64 | 439.5 ± 51.3 |
| PPO + RND × episodic count | 3 | 2.00 ± 0.00 | 0.00 ± 0.00 | 21.4 ± 0.1 | 491.6 ± 0.1 | 0.00 ± 0.00 | 509.2 ± 0.3 |
| PPO + episodic count only (ablation) | 3 | 2.00 ± 0.00 | 0.00 ± 0.00 | 21.0 ± 0.1 | 492.0 ± 0.1 | 0.00 ± 0.00 | 509.6 ± 0.1 |
| PPO + NGU-style (inverse-dynamics kNN) | 3 | 3.89 ± 0.11 | 1.00 ± 0.00 | 86.0 ± 4.6 | 427.0 ± 4.6 | 0.00 ± 0.00 | 123.2 ± 7.0 |
| PPO + ensemble disagreement | 3 | 3.61 ± 0.33 | 2.71 ± 0.19 | 53.1 ± 4.5 | 458.4 ± 4.5 | 2.32 ± 0.21 | 52.5 ± 20.9 |

| % of eval episodes (tv) | lever A (lamp) | button B (near door) | push box | pick up key | unlock door | button E (far door) | lever C | lever D |
|---|---|---|---|---|---|---|---|---|
| random walk | 65 | 31 | 29 | 3 | 0 | 0 | 13 | 0 |
| PPO + hashed counts | 0 | 0 | 100 | 0 | 0 | 0 | 29 | 0 |
| PPO + RND | 1 | 2 | 99 | 0 | 0 | 0 | 41 | 0 |
| PPO + RND × episodic count | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| PPO + episodic count only (ablation) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| PPO + NGU-style (inverse-dynamics kNN) | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 0 |
| PPO + ensemble disagreement | 45 | 68 | 98 | 22 | 0 | 0 | 38 | 0 |

| whole training budget (tv, 6 M steps/seed) | episodes | key picked up | door unlocked | button E | max rooms in any episode |
|---|---|---|---|---|---|
| random walk (3 seeds) | 34944 | 1902 | 2 | 0 | 6 |
| PPO + hashed counts (3 seeds) | 34944 | 50 | 0 | 0 | 5 |
| PPO + RND (3 seeds) | 34944 | 879 | 0 | 0 | 6 |
| PPO + RND × episodic count (3 seeds) | 34944 | 65 | 0 | 0 | 6 |
| PPO + episodic count only (ablation) (3 seeds) | 34944 | 46 | 0 | 0 | 5 |
| PPO + NGU-style (inverse-dynamics kNN) (3 seeds) | 34944 | 177 | 0 | 0 | 6 |
| PPO + ensemble disagreement (3 seeds) | 34944 | 7047 | 1 | 0 | 6 |

How interesting each trained bonus finds the TV (`diagnose_tv.py`, `media/tv_bonus_ratio.png`).
Every agent's trained reward module is fed the same 64 random-walk episodes, and the table compares
its raw bonus on TV-in-view steps with its bonus elsewhere. For the episodic agents this is the
episodic term alone.

| bonus | TV / elsewhere (seeds 0, 1, 2) |
|---|---|
| hashed counts (lifelong) | 2.04, 2.00, 1.90 |
| RND | 1.90, 3.14, 2.89 |
| episodic hash count (`rnd_ep`, `ep`) | about 1.5 (the hash bonus has no learned parameters, so it is the same for every seed) |
| NGU-style kNN in an inverse-dynamics embedding | 1.04, 1.04, 1.00 |
| ensemble disagreement | 1.30, 0.79, 1.25 |

**Demo:** `media/noisy_tv.gif` shows four panels with the same 512-action budget: random walk, RND ×
episodic count (glued to the TV), NGU-style, and ensemble disagreement. The last frame is
`media/noisy_tv_final.png`.

**What the numbers say**

* **Every novelty-seeker that keys on raw observations is captured.** The episodic hash agents
  (RND × episodic, episodic-only) sit in front of the screen for 509 of 512 steps, visit 21 cells,
  and trigger nothing, in all 6 runs. Their 0.00 repeat interactions mean nothing here: you cannot
  re-toggle something you never toggled. An unpredictable frame is a new observation every step, so
  the episodic count is always 1. Lifelong counts (492 TV steps) and RND (440) are captured too:
  RND's predictor cannot fit a fresh random screen, so its error stays high. In one rendered
  episode, RND seed 0 parks in room R11 and watches the TV *through the wall*, because views are
  not occluded. TV time beyond
  what the random walk gets by accident: +369 (counts), +317 (RND), and +386 (RND × episodic).
* **Ensemble disagreement ignores the TV.** It spends *less* time with the TV in view than the
  random walk (52 vs 123 steps), and its scores match its base-world scores (2.71 vs 2.35
  interactions, 3.6 vs 3.0 rooms; the small gain is within seed noise). This is the expected
  mechanism: every forward model learns the same conditional mean for a random screen, so the
  models agree there. The measured bonus ratio is close to 1 (1.11 averaged over seeds).
* **NGU-style episodic memory is also TV-proof, but explores less.** Its TV time is exactly the
  random walk's (123), and its bonus ratio is 1.00–1.04. The inverse-dynamics embedding learned to
  drop the screen. It has the second-best coverage in the TV world (86 cells), but it pushes the
  box and nothing else (1.00 interactions), and it does the same in the base world (3.98 rooms,
  1.00 interactions). The embedding only keeps what predicts the agent's own action, which is
  mostly its position, so a toggled lamp or an opened door barely registers as episodic novelty. It
  ends up in the same four-room sweep where two of the episodic-only seeds got stuck.
* **Honest bottom line on the fix:** both "learnable-structure" bonuses solved the distraction
  (TV time at or below random), and neither recovered the 4.95-interaction explorer. With
  disagreement the agent does not get worse when the TV is added, but it was never a strong
  explorer. The best exploration came from the most distractible bonus.

## 3. Longer training (one extra run, not part of the matched table)

RND × episodic count, seed 0, trained for 30 M steps (5× the budget) with the same config
(`runs/rnd_eplong_base_s0`, label `rnd_eplong`).

| | rooms /9 | interactions /8 | coverage | unlock door | button E | lever D |
|---|---|---|---|---|---|---|
| 6 M steps (seed 0) | 7.48 | 4.95 | 148.6 | 98% | 0% | 0% |
| 30 M steps (seed 0) | 6.75 | 4.25 | 137.3 | 67% | 0% | 0% |

The fair comparison is peak against peak. During the 30 M run the per-episode training average
peaked at 7.78 rooms (about 10 M steps), above the 6 M run's final eval (7.48). Caveat (from the
cross-review): 7.78 is the maximum over ~114 bins of 512 training episodes, compared with a single
fixed-seed eval, so it is biased upward; treat "slightly above" as "about the same". The final 30 M
snapshot is worse only because the policy drifts. More training did **not** crack the last chain. Over 58,496 training episodes the agent pressed
button E 19 times and never reached the ninth room. The per-episode curve plateaus at about
7.5 rooms and 4.9 interactions by 6–8 M steps, then drifts down (7.7 rooms at 10 M steps, 6.6 at 30 M), and the final
snapshot evaluates *worse* than the 6 M-step one. This looks like the same frontier-chasing cycle
as plain RND, on a slower timescale. The bottleneck is probably not the amount of training: a
memoryless policy cannot tell "I already pressed button E this episode" from "I haven't". Its
effect, an open door three rooms away, is out of view, so the episodic bonus never rewards the
long walk back.

## 4. What worked, what didn't, surprises

* **Worked:** episodic × lifelong novelty (NGU recipe) with a plain hash count. It is cheap and
  stable across seeds, and it is the only bonus that reliably took the agent through the
  key → lock chain. Numba-vectorised environments plus MLX PPO ran at 20–60k env steps/s per
  process, so a 6 M-step run took 2–9 minutes.
* **Didn't work:** lifelong novelty alone, with a memoryless policy and a per-episode metric. The
  noisy-TV fixes kept their TV immunity but lost the gains above. The NGU-style kNN variant was
  also slower (10–14k steps/s, because of the numpy kNN).
* **Surprises**
  * The weight on the extrinsic head was silently harmful. The extrinsic reward is identically 0,
    but with Burda's coefficient of 2 the extrinsic advantage (pure value-head noise, std about
    0.005) was half the size of the intrinsic advantage (std about 0.01) and was mixed into every
    policy update. I set its weight to 0 and kept the head. The 3 runs made before the change are
    archived in `runs/v1_extcoef2/`: RND 3.0/3.8 rooms and 2.5 interactions, counts 3.0 rooms and
    1.9 interactions. They are not used in any table.
  * Learned agents never touch the lever in the start room ("boredom" in the literal sense), while
    disagreement plays with it in 76% of episodes.
  * The RND agent's lever spam is a homemade noisy TV, built from a mechanism the designer meant
    as a "useful interaction".
  * Ensemble disagreement *avoided* the TV room (52 TV steps vs 123 for the random walk) instead of
    being neutral about it.

## 5. Judgment calls (made without the user; all reversible)

* **World size:** 3×3 rooms with 5×5 interiors (a 19×19 grid), chosen so the hardest chain is
  hard but a 6 M-step budget is enough to see differences. Views are not occluded (simpler and
  faster). The noisy TV is passive: it flickers every step with no remote, so "watching" is simply
  having it in view.
* **Matched budgets:** 512 actions per episode, 6 M training steps for every learned agent, and
  3 seeds. The GPU slot queue was the main bottleneck (one wait was 90 minutes), which set the
  6 M-step budget.
* **Extrinsic head kept but given weight 0** (see Surprises). The RND predictor trains on 25% of
  samples (Burda et al.'s proportion), so novelty does not vanish within a few updates.
* **Added agents beyond the brief.** `rnd_ep`, `ep`, and `ngu` were added after plain RND and
  counts plateaued (the doc's "memory" tinker item). `ngu` simplifies NGU: no recurrent policy,
  no mixture of β/γ, the bounded reward 1/√(ΣK+1), and RND as the lifelong multiplier.
* **The policy samples stochastically in evaluation**, as it does in training.
* **Up to 3 runs share one GPU slot** (`worker.sh`, 1 numba/OMP thread each), which stays within
  the 4-thread-per-job cap. For about 25 minutes I held two slots (worker2 and worker3). By then the orchestrator had grown
  the shared pool from 4 slots to 8 (`../.locks/gpu0`…`gpu7`).

## 6. Gaps and next steps

* Not measured: occluded views, and a TV with a remote (action-triggered).
* No agent reached lever D or the ninth room in 6 M steps. Try longer runs (§3), a recurrent policy
  (so the agent can remember that it pressed button E), or a curriculum.
* A natural next fix: episodic novelty on an embedding trained with inverse dynamics *plus*
  something that keeps object state (for example a forward-prediction or disagreement term), to
  get NGU's TV robustness without losing the mechanisms.
* Seed variance for the count and episodic-only agents is large (see ± columns). Three seeds is
  the minimum.

## Files

* Checkpoints (reloadable by `render.py`, `evaluate.py`, `diagnose_tv.py`):
  `checkpoints/rnd_ep_base_s0.pkl` (main demo), `checkpoints/rnd_base_s2_leverspam.pkl`
  (failure clip), `checkpoints/{rnd_ep,ngu,disagree}_tv_s0.pkl` (noisy-TV demo). Every run's last
  checkpoint is `runs/<agent>_<world>_s<seed>/ckpt.pkl`. Each holds the weights, optimizer state,
  normalisers, count tables or episodic memories, env state, RNG state, and config. Exact resume is
  tested in `tests/test_pipeline.py::test_resume_matches_uninterrupted` for 4 agent types.
* Media: `explorer_vs_random.gif`, `failure_lever_spam_rnd.gif`, `noisy_tv.gif` (each with a
  `_final.png`), `interaction_timeline_{base,tv}.png`, `training_curves_{base,tv}.png`,
  `lifetime_discovery_{base,tv}.png`, `tv_bonus_ratio.png`.
* Smoke test: `./smoke.sh` (35–75 s depending on load) runs the 13 pytest tests, then a tiny train, kill, resume,
  eval, and render. Exact resume (the same weights as an uninterrupted run) is tested for `rnd`,
  `rnd_ep`, `ngu`, and `count`.
* Reproduction check: `evaluate.py --ckpt checkpoints/rnd_ep_base_s0.pkl --seed 10000`, run from a
  fresh shell, reproduces the table's seed-0 row exactly (7.484 rooms, 4.953 interactions,
  148.58 cells).
* `runs/` is git-ignored. `results/` holds the committed copies of every eval JSON, summary, and
  per-run training log. The whole-budget tables (from `episodes.jsonl`) regenerate only where
  `runs/` exists.

## Round 5: Does curiosity collect useful experience?

**Historical pre-run note; superseded by the measured section below.** At the Turn-1 handoff,
experiment and runner code were written; neural training, exploration comparisons,
surprise-task success, runtime and peak memory are **not measured**. Astra cannot run MLX in
this sandbox. Do not combine historical results above with the new protocol as a matched table.

Setup and hypotheses are in [DESIGN_R5.md](DESIGN_R5.md). The new progress method is a lagged-predictor ablation, not
a reproduction of LPM's learned error model. The six methods receive matched environment-step
budgets, and every task planner receives the same form and length of fresh exploration replay.
The frontier baseline has explicit object affordances; its tuning uses validation discovery only.

Local verification: 14 NumPy-only tests passed, covering map inference, executable replay plans,
seed separation, matched noise streams, failure penalties, classical exact resume, wall guards,
receipt preservation and headless GIF generation. MLX smoke, neural resume and main experiment
results remain unverified until Claude runs them. These checks establish pipeline properties,
not learned performance. No new MLX checkpoint or trained-agent demo has been produced yet.

Risks to inspect: sparse or noisy signed progress; undertraining within the battery budget;
conservative replay graphs that cannot compose unseen routes; and failures on the long key/door
chain. Fresh seeds vary stochasticity in the same fixed layout, not held-out houses. Preserve
failed task attempts and TV capture in the fixed-first-trial demos. Next: pass `smoke_r5.sh`,
freeze validation choices, run the bounded schedule, and trace every result to its saved receipt
before replacing this pending-results note with measured findings.

## Round 5: Curiosity that becomes useful later

**TL;DR (reviewed by Claude; every number below links to a receipt).** The question was whether the
things a curious agent stumbles on while exploring with no goal help it later, when it is suddenly given
a task. In the **plain house, yes**: after 512 reward-free steps, the replay of the RND × episodic-novelty
explorer lets a shared planner solve **64.0%** of seven surprise tasks, against **16.1%** for a random walk
and **7.1%** with no replay at all. In the **noisy-TV house that benefit disappears**: the same explorer
stares at the TV for 484 of 512 steps and its task success drops to **15.8%**, the random-walk level. The
new TV-robust learning-progress explorer (LPM) does ignore the TV (about 4 TV steps), but only because it
hardly moves: 1.7–1.8 rooms, no mechanisms, and task success equal to the empty-replay floor (7.1%) in
both houses. Its policy collapsed to a near-deterministic loop by PPO update 11–15 of 366 (one TV seed: 45),
a few updates after its reward switched on at update 7, and the reviewer traced why (below): the reward penalises any transition off that loop.
A classical frontier/BFS explorer with an explicit map and object priors finds all 9 rooms and 8
mechanisms and solves **98.2% (plain) / 100% (TV)** of the tasks. Turn 1's 28.6% task ceiling was a
planner limitation, fixed in Turn 2. Fresh evaluation randomness, but the same house layout throughout.

### Turn 2 (headline)

**Setup.** Six methods × two houses × three training seeds (610–612), each charged exactly 5,996,544
pretraining environment steps (Round 1's budget; 128 parallel environments). Evaluation: 16 fresh
episodes per cell (seeds 85400–85415, policy RNG 83519). In each episode the frozen explorer takes 512
reward-free actions; then seven tasks are revealed (lamp on in rooms 0/4/8; carry the key to room 0/3;
reach room 6/8). For each task the house is reset and the **same** compositional planner receives only
that episode's 512-step observation/action replay, a 128-action budget and 100,000 search expansions;
unplanned steps use the same seeded random fallback; a failure costs 128 steps. No policy weights or
training replay transfer. Means ± population SD over training seeds; `frontier` and `random` do not fit
anything, so their three rows are identical repeats, not independent replicates. All 36 cells finished
in one 25.9-minute wave (two workers, one GPU slot; [wave receipt](results/r5/t2_ac/wave1.json),
[manifest with code hashes](results/r5/t2_ac/manifest.json), which match the files on disk at review time).
[Generated report](results/r5/t2_ac/summary.md).

| world | method | rooms /9 | mechanisms /8 | TV steps /512 | task success | solved /112 per seed | plan coverage | capped task steps | source seeds |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| plain | lpm | 1.667 ± 0.471 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.071 ± 0.000 | 8 / 8 / 8 | 0.000 ± 0.000 | 122.661 ± 0.000 | [610](results/r5/t2_ac/lpm_plain_s610.json) / [611](results/r5/t2_ac/lpm_plain_s611.json) / [612](results/r5/t2_ac/lpm_plain_s612.json) |
| plain | rnd_ep | 7.312 ± 0.368 | 4.979 ± 0.029 | 0.000 ± 0.000 | 0.640 ± 0.004 | 71 / 72 / 72 | 0.568 ± 0.004 | 68.705 ± 0.467 | [610](results/r5/t2_ac/rnd_ep_plain_s610.json) / [611](results/r5/t2_ac/rnd_ep_plain_s611.json) / [612](results/r5/t2_ac/rnd_ep_plain_s612.json) |
| plain | disagree | 3.292 ± 0.230 | 2.104 ± 0.106 | 0.000 ± 0.000 | 0.202 ± 0.017 | 24 / 24 / 20 | 0.131 ± 0.017 | 108.821 ± 1.858 | [610](results/r5/t2_ac/disagree_plain_s610.json) / [611](results/r5/t2_ac/disagree_plain_s611.json) / [612](results/r5/t2_ac/disagree_plain_s612.json) |
| plain | ngu | 3.938 ± 0.088 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.071 ± 0.000 | 8 / 8 / 8 | 0.000 ± 0.000 | 122.661 ± 0.000 | [610](results/r5/t2_ac/ngu_plain_s610.json) / [611](results/r5/t2_ac/ngu_plain_s611.json) / [612](results/r5/t2_ac/ngu_plain_s612.json) |
| plain | frontier | 9.000 ± 0.000 | 8.000 ± 0.000 | 0.000 ± 0.000 | 0.982 ± 0.000 | 110 / 110 / 110 | 1.000 ± 0.000 | 46.366 ± 0.000 | [610](results/r5/t2_ac/frontier_plain_s610.json) / [611](results/r5/t2_ac/frontier_plain_s611.json) / [612](results/r5/t2_ac/frontier_plain_s612.json) |
| plain | random | 2.312 ± 0.000 | 1.562 ± 0.000 | 0.000 ± 0.000 | 0.161 ± 0.000 | 18 / 18 / 18 | 0.125 ± 0.000 | 110.259 ± 0.000 | [610](results/r5/t2_ac/random_plain_s610.json) / [611](results/r5/t2_ac/random_plain_s611.json) / [612](results/r5/t2_ac/random_plain_s612.json) |
| tv | lpm | 1.771 ± 0.281 | 0.000 ± 0.000 | 4.292 ± 3.312 | 0.071 ± 0.000 | 8 / 8 / 8 | 0.000 ± 0.000 | 122.661 ± 0.000 | [610](results/r5/t2_ac/lpm_tv_s610.json) / [611](results/r5/t2_ac/lpm_tv_s611.json) / [612](results/r5/t2_ac/lpm_tv_s612.json) |
| tv | rnd_ep | 2.667 ± 0.471 | 1.292 ± 0.913 | 483.521 ± 20.740 | 0.158 ± 0.061 | 22 / 23 / 8 | 0.086 ± 0.061 | 113.080 ± 6.786 | [610](results/r5/t2_ac/rnd_ep_tv_s610.json) / [611](results/r5/t2_ac/rnd_ep_tv_s611.json) / [612](results/r5/t2_ac/rnd_ep_tv_s612.json) |
| tv | disagree | 3.229 ± 1.064 | 1.750 ± 0.184 | 30.229 ± 22.684 | 0.241 ± 0.098 | 23 / 16 / 42 | 0.193 ± 0.078 | 104.423 ± 7.255 | [610](results/r5/t2_ac/disagree_tv_s610.json) / [611](results/r5/t2_ac/disagree_tv_s611.json) / [612](results/r5/t2_ac/disagree_tv_s612.json) |
| tv | ngu | 3.979 ± 0.029 | 1.000 ± 0.000 | 148.812 ± 26.363 | 0.071 ± 0.000 | 8 / 8 / 8 | 0.000 ± 0.000 | 122.661 ± 0.000 | [610](results/r5/t2_ac/ngu_tv_s610.json) / [611](results/r5/t2_ac/ngu_tv_s611.json) / [612](results/r5/t2_ac/ngu_tv_s612.json) |
| tv | frontier | 9.000 ± 0.000 | 8.000 ± 0.000 | 66.812 ± 0.000 | 1.000 ± 0.000 | 112 / 112 / 112 | 1.000 ± 0.000 | 45.714 ± 0.000 | [610](results/r5/t2_ac/frontier_tv_s610.json) / [611](results/r5/t2_ac/frontier_tv_s611.json) / [612](results/r5/t2_ac/frontier_tv_s612.json) |
| tv | random | 2.250 ± 0.000 | 1.562 ± 0.000 | 115.312 ± 0.000 | 0.161 ± 0.000 | 18 / 18 / 18 | 0.125 ± 0.000 | 110.259 ± 0.000 | [610](results/r5/t2_ac/random_tv_s610.json) / [611](results/r5/t2_ac/random_tv_s611.json) / [612](results/r5/t2_ac/random_tv_s612.json) |

Empty-replay control (same task seeds, same fallback, no replay): **8/112 = 0.071 in every cell**, all of
it from task `lamp_0` (8/16 by chance), in the `empty_replay` block of each receipt above.

**What worked.**
- *Curiosity's discoveries are useful later in the plain house.* RND × episodic novelty reaches 7.31
  rooms / 4.98 mechanisms (Round 1 reported 7.34 / 4.95 at the same budget) and 0.640 ± 0.004 task
  success (71/72/72 of 112), four times the random walk (18/112) and nine times empty replay. It solves
  `key_room_0`, `key_room_3` and `reach_6` in every episode and `lamp_4` in 98%, but never `lamp_8` or
  `reach_8` (room 8 is behind the longest mechanism chain); per-task rates are in each receipt's `per_task`.
- *More discovery means more tasks solved.* Across the 384 learned-agent test episodes, mechanisms found
  correlate with tasks solved at Spearman ρ = 0.71 (rooms: ρ = 0.48)
  ([receipt](results/r5/review_discovery_vs_tasks.json)).
- *The repaired task stage works.* With complete knowledge the frontier explorer's replay is planned for
  every task (plan coverage 1.000); it fails only 2/112 plain trials (`lamp_8` and `reach_8`, one episode each).
- *The secondary old split agrees.* On the four old test seeds 72500–72503 (`*_old.json` next to each
  receipt): RND × episodic plain 0.643, frontier 1.000 in both houses, random 0.107, disagreement TV 0.274,
  LPM and NGU 0.071.

**What didn't.**
- *The noisy TV still destroys the benefit.* RND × episodic in the TV house: 483.5 TV steps, 2.67 rooms,
  0.158 ± 0.061 task success (22/23/8 of 112; seed 612 sits exactly on the empty-replay floor).
- *The TV-robust methods are weak explorers.* Disagreement keeps TV time to 30 steps but finds 3.2–3.3
  rooms; its task success (plain 0.202, TV 0.241 ± 0.098, per seed 23/16/42) is not clearly above
  random's 0.161 with three seeds. NGU (inverse-dynamics episodic memory) reaches four rooms and one
  mechanism, and no plan ever succeeds (plan coverage 0; success = empty replay).
- *LPM fails to explore* (diagnosis below). It never discovers a mechanism in 96 test episodes and its
  replay never yields a plan.

**LPM diagnosis (reviewer).** Not an implementation bug, not a narrow tuning miss, and not evidence that
the published LPM method fails in general: this adaptation fails for identifiable reasons.
- *Code matches the design.* Reward = expected log error − current log error, scored before fitting; the
  error model regresses detached pre-update labels from a queue of the last `window` rollouts; reward is
  zeroed until update window+2; all values are finite; smoke kill/resume is exact
  ([smoke log](logs/r5/claude_t2_smoke.log)); the reload of `lpm_tv_s610` reproduces `episodes`, `tasks`,
  `transfer`, `empty_replay` exactly ([reload](results/r5/t2_ac/lpm_tv_s610_reload.json)).
- *Collapse is fast and LPM-specific.* In all six cells the reward switches on at update 7; policy entropy
  falls from about 1.5 to below 0.05 by update 11–15 (update 45 for TV seed 610). Over updates 7–32 the
  raw reward is 66–92% positive with mean 0.36–0.89, divided by an adaptive scale that starts at its 0.1
  floor, so normalised rewards hit the ±5 clip. Training-episode rooms fall from 2.7–2.9 (updates 1–6) to
  1.0–2.0 in the last 110 updates, and cell coverage from about 50 to 3–10. Every other learned method keeps
  phase-mean entropy at or above 0.57 throughout training ([phase summary](results/r5/review_lpm_training.json), from `logs/r5/t2_ac/*/updates.jsonl`).
- *After collapse, leaving the loop is punished.* The predictor's target (a fixed random embedding of a
  deterministic binary observation) is fully learnable, so on the agent's own loop log error reaches
  −13.81, the log(1e-6) = −13.82 floor, and reward there is ≈ 0. A NumPy forward pass of each final
  checkpoint on transitions it never makes (the saved first-episode replays of frontier and RND ×
  episodic) gives log errors around −6 (against −13.8 on its own loop), which the error model under-predicts, because it was fitted
  only to on-loop errors. The raw reward for transitions in rooms the agent did not visit is negative in
  11 of 12 comparisons (mean −0.49 to −2.78; 61–100% of transitions negative). Off-loop transitions in
  rooms it does visit are also negative (−1.0 to −3.5 on the frontier replay). Divided by the 0.1–0.6 scale, that is a strongly negative
  normalised reward for any deviation, with no compensating positive reward anywhere
  ([probe](results/r5/review_lpm_probe.json); script `review_r5_lpm.py`, <1 s, no MLX).
- *Its low TV time is not robustness.* During training, TV seed 610 averaged more than 400 TV steps per
  512-step training episode in 164 updates (between updates 24 and 195), and seed 612 in 68 updates (20–115).
  While captured, TV reward averaged about 0 and was about 50% positive, so nothing pushed the agent away. It
  moved off only later, as TV reward turned negative (−0.59 and −0.97 over updates 257–366). The
  evaluated policy shows about 4 TV steps because it barely leaves the start area.
- *Tuning was reasonable but could not rescue it.* Validation searched window {1,4} × entropy coefficient
  {0.001, 0.01} at 1,048,576 steps on seeds 64600–64603, selecting by discovery score only. Every
  candidate's entropy fell to 0.00–0.37 during updates 9–32 (partial recovery later in two pilots), and
  validation discovery stayed at 1.0–2.75 rooms ( scores 0.15–0.34 against
  frontier's 1.94–1.98) and the least-bad (window 4, entropy 0.001) was taken
  ([selection](results/r5/t2_tune/selection.json), `logs/r5/t2_tune_*/`). The grid was narrow. Untested
  fixes that target the mechanism above: a much longer or reservoir error queue, a pessimistic (high)
  error prediction for unfamiliar inputs, a larger log epsilon or a non-saturating target, and a novelty
  floor or entropy far above 0.01. Turn 1's `lp` failed for a different reason (vanishing reward scale);
  keep the two failures separate.

**Fairness (reviewer).**
- *Same observation and same deployment budget, different priors.* Frontier receives the same 7×7
  egocentric observation and the same 512 exploration actions, and infers odometry from observations
  only. It also integrates them into a map and uses BFS plus hand-coded generic affordances (keys, levers,
  buttons, box pushes, doors). The neural explorers are reactive (current observation only) and must
  learn what objects do. So this is "learned curiosity vs a classical explorer with an explicit map and
  object priors", not a matched-information contest. Frontier's tuning grid is flat (five of six settings reach
  9/8, the sixth 8.75/7.5; scores 1.937–1.984): the baseline is at the ceiling, not weak.
- *Pretraining budget.* Frontier and random are charged the same 5,996,544 simulator steps
  (`logs/r5/t2_ac/{frontier,random}_*/outcome.json`: all `complete`), but they fit nothing and reset their
  maps every episode, so those steps are unused by design. The receipts' `pretrain_steps: 0` and
  `charged_pretrain_steps: 5996544` record exactly this.
- *Transfer stage is identical across agents by construction.* There is one `evaluate()` path with the
  same 512-step single-episode replay, the same `Knowledge.from_replay` and `CompositionalPlanner`, 128
  actions, 100,000 expansions, the same task seeds and the same fallback RNG. One subtlety: the planner
  infers a button/lever link only when a unique observed activation explains a later change, so clean,
  one-at-a-time interaction (which frontier does by design) is easier to exploit than scattered
  interaction. The planner was validated on frontier tours, on validation seeds 64600–64603 only.
- *Floors.* Empty replay (0.071) is the right paired control for "did this replay contribute any plan":
  LPM and NGU sit exactly on it with plan coverage 0. The natural "no curiosity" baseline is the random
  walk (0.161), which also benefits from its own replay (plan coverage 0.125). Report both.

**Surprises.** Training was far faster than planned: 6M steps took 57–248 s per learned cell (73–83 s for
LPM; `train_seconds` in [review_lpm_training.json](results/r5/review_lpm_training.json)), so all 36 cells fit one wave; the planned waves 2 and 3 were never needed. Peak memory was not measured.

**Next steps.** (1) LPM fixes aimed at the diagnosed mechanism: a long or reservoir error queue,
pessimistic error predictions off-distribution, a non-saturating target, plus a broader entropy sweep.
(2) A TV-robust explorer that keeps discovery: for example RND × episodic with a learning-progress or
disagreement gate on the TV region. (3) Held-out house layouts; all evaluation here varies only
randomness within one fixed layout. (4) A recurrent or map-augmented learned explorer, to separate
"learned vs classical" from "memoryless vs mapped".

**Demo.** [LPM in the TV house](media/r5_t2_lpm_tv.gif) (stays near the start) and
[frontier in the TV house](media/r5_t2_frontier_tv.gif) (full tour, then a planned task), both from the
fixed first test episode, rendered from `results/r5/t2_ac/{lpm,frontier}_tv_s610.npz`. Reloadable
checkpoints are in `checkpoints/r5_t2_ac/`.

### Turn 1 (preliminary; superseded by Turn 2)

All 36 cells completed at 3,145,728 pretraining actions per cell (six methods × two worlds × three seeds; tags `r5_tune`/`r5_ac`, checkpoints in `checkpoints/r5_r5_ac/`). LP failed exploration, and the old transfer stage imposed a recorded-tour bottleneck even on complete knowledge, so these task scores are not a clean ranking of useful knowledge. **Reviewer note on the planner ceiling:** frontier's exploration is identical in both turns (9 rooms, 8 mechanisms), so its jump from 0.286 to 0.982/1.000 task success is entirely the planner repair. RND × episodic's 0.220 → 0.640 mixes the planner repair with the doubled training budget (3.1M → 6.0M steps) and is not a clean planner-only effect.

Setup: train seeds 510–512, 16 fresh episodes per seed (72500–72515), 512 reward-free actions followed by seven tasks, each reset and given 128 actions. The old replay graph had a 4096-node search cap. Means ± population SD across training seeds follow; classical repetitions are identical, not independent fitted-seed evidence. Each row links its three raw receipts. [Full report and source index](results/r5/r5_ac/summary.md); [independent NumPy recomputation](results/r5/t2_diagnosis.json).

| world | method | rooms /9 | mechanisms /8 | TV steps /512 | task success | plan coverage | source seeds |
|---|---|---:|---:|---:|---:|---:|---|
| plain | lp | 1.333 ± 0.471 | 0.333 ± 0.471 | 0.000 ± 0.000 | 0.083 ± 0.042 | 0.048 ± 0.067 | [510](results/r5/r5_ac/lp_plain_s510.json) / [511](results/r5/r5_ac/lp_plain_s511.json) / [512](results/r5/r5_ac/lp_plain_s512.json) |
| plain | rnd_ep | 6.354 ± 1.670 | 3.312 ± 1.692 | 0.000 ± 0.000 | 0.220 ± 0.120 | 0.167 ± 0.120 | [510](results/r5/r5_ac/rnd_ep_plain_s510.json) / [511](results/r5/r5_ac/rnd_ep_plain_s511.json) / [512](results/r5/r5_ac/rnd_ep_plain_s512.json) |
| plain | disagree | 4.021 ± 0.059 | 2.646 ± 0.290 | 0.000 ± 0.000 | 0.134 ± 0.015 | 0.086 ± 0.008 | [510](results/r5/r5_ac/disagree_plain_s510.json) / [511](results/r5/r5_ac/disagree_plain_s511.json) / [512](results/r5/r5_ac/disagree_plain_s512.json) |
| plain | ngu | 3.938 ± 0.088 | 0.979 ± 0.029 | 0.000 ± 0.000 | 0.054 ± 0.000 | 0.000 ± 0.000 | [510](results/r5/r5_ac/ngu_plain_s510.json) / [511](results/r5/r5_ac/ngu_plain_s511.json) / [512](results/r5/r5_ac/ngu_plain_s512.json) |
| plain | frontier | 9.000 ± 0.000 | 8.000 ± 0.000 | 0.000 ± 0.000 | 0.286 ± 0.000 | 0.286 ± 0.000 | [510](results/r5/r5_ac/frontier_plain_s510.json) / [511](results/r5/r5_ac/frontier_plain_s511.json) / [512](results/r5/r5_ac/frontier_plain_s512.json) |
| plain | random | 2.312 ± 0.000 | 1.375 ± 0.000 | 0.000 ± 0.000 | 0.152 ± 0.000 | 0.134 ± 0.000 | [510](results/r5/r5_ac/random_plain_s510.json) / [511](results/r5/r5_ac/random_plain_s511.json) / [512](results/r5/r5_ac/random_plain_s512.json) |
| tv | lp | 1.667 ± 0.471 | 0.229 ± 0.324 | 288.625 ± 209.674 | 0.071 ± 0.025 | 0.033 ± 0.046 | [510](results/r5/r5_ac/lp_tv_s510.json) / [511](results/r5/r5_ac/lp_tv_s511.json) / [512](results/r5/r5_ac/lp_tv_s512.json) |
| tv | rnd_ep | 2.979 ± 0.029 | 1.896 ± 0.078 | 488.333 ± 4.993 | 0.182 ± 0.004 | 0.128 ± 0.004 | [510](results/r5/r5_ac/rnd_ep_tv_s510.json) / [511](results/r5/r5_ac/rnd_ep_tv_s511.json) / [512](results/r5/r5_ac/rnd_ep_tv_s512.json) |
| tv | disagree | 3.354 ± 0.511 | 2.458 ± 0.118 | 58.021 ± 14.195 | 0.149 ± 0.015 | 0.107 ± 0.032 | [510](results/r5/r5_ac/disagree_tv_s510.json) / [511](results/r5/r5_ac/disagree_tv_s511.json) / [512](results/r5/r5_ac/disagree_tv_s512.json) |
| tv | ngu | 4.000 ± 0.000 | 1.000 ± 0.000 | 147.521 ± 24.681 | 0.054 ± 0.000 | 0.000 ± 0.000 | [510](results/r5/r5_ac/ngu_tv_s510.json) / [511](results/r5/r5_ac/ngu_tv_s511.json) / [512](results/r5/r5_ac/ngu_tv_s512.json) |
| tv | frontier | 9.000 ± 0.000 | 8.000 ± 0.000 | 65.188 ± 0.000 | 0.286 ± 0.000 | 0.286 ± 0.000 | [510](results/r5/r5_ac/frontier_tv_s510.json) / [511](results/r5/r5_ac/frontier_tv_s511.json) / [512](results/r5/r5_ac/frontier_tv_s512.json) |
| tv | random | 2.375 ± 0.000 | 1.375 ± 0.000 | 170.188 ± 0.000 | 0.152 ± 0.000 | 0.134 ± 0.000 | [510](results/r5/r5_ac/random_tv_s510.json) / [511](results/r5/r5_ac/random_tv_s511.json) / [512](results/r5/r5_ac/random_tv_s512.json) |

**What worked and failed.** Frontier discovers all nine rooms and eight mechanisms in both worlds. RND × episodic novelty is the strongest learned plain-house explorer (6.35 rooms) but still watches the TV for 488 steps. Disagreement reduces TV time to 58 steps while discovering comparatively few mechanisms. NGU stays around four rooms and one mechanism. LP is worse than random in both worlds. Round 1’s RND × episodic 7.34-room result used 5,996,544 actions; the smaller Turn-1 budget is a confound, not evidence of a regression at equal training. The paired empty-replay control succeeds on 6/112 trials (0.053571) per world, in every method receipt.

**The transfer bottleneck is a finding.** Frontier plans and solves only lamp 0 and lamp 4. In the first saved TV trajectory, the other task paths require 150, 162, 185, 290 and 295 recorded actions. Allowing 512 actions recovers all seven plans. Only 141 nodes were expanded at the 128-action cutoff, so the 4096-node cap was not binding. The graph’s complete remembered-map signature changes as knowledge grows; it mostly preserves the original acquisition tour and cannot compose a new route through known mechanisms. This was a planner limitation, not an impossible key/door task. Source: [replay-budget witnesses](results/r5/t2_diagnosis.json), using [frontier TV receipt](results/r5/r5_ac/frontier_tv_s510.json) and its `.npz`.

The 28.6% figure is **not a universal mathematical ceiling**: [RND × episodic, plain seed 511](results/r5/r5_ac/rnd_ep_plain_s511.json) reaches 37/112 = 0.330357, although every method’s three-seed mean is ≤0.285714. Different recorded tours can put more tasks inside the action limit.

**LP diagnosis.** The signal is not always zero: late positive fractions range roughly 0.41–0.70 across runs, and positive progress coexists with TV fixation. For TV seed 510, average reward SD falls from 1.36e-4 in updates 2–65 to 1.55e-6 in the final 64 updates, while the final cumulative-return divisor is 0.04608. Dividing that late raw SD by the final scale gives only 3.36e-5. This makes reward small relative to critic error; per-minibatch advantage normalization means scale alone does not prove causality. [Computed diagnostics](results/r5/t2_diagnosis.json) trace these to `logs/r5/r5_ac/lp_*_s*/updates.jsonl` and checkpoints.

One saved plain-house seed-510 episode uses the same action on 510/512 steps (99.6%); collapse is directly visible there. Other saved seeds have much broader action frequencies. These are **action-marginal**, not conditional policy entropy measurements. Conditional entropy was not logged and remains unmeasured by Astra; Claude’s MLX probe was planned for that. [Action-frequency receipt](results/r5/t2_action_diagnosis.json). **Reviewer addition:** Claude's MLX probe has since measured conditional policy entropy on the Turn-1 LP checkpoints: 0.019 nats for plain seed 510 (collapsed, matching the 510/512 single-action episode) and 0.88–1.37 for the other five ([entropy probe](results/r5/t2_entropy_probe.json); maximum log 5 = 1.61).

**Turn-2 repair checks (planner validation, run before the Turn-2 headline).** The new common planner builds a semantic map and conservative causal links from the same replay, then composes navigation, pickup, key-consuming unlock, switches and box pushes. With the true initial map/state supplied as an explicitly privileged oracle, the same planner solves 14/14 world/task pairs in 5–94 physical actions. Complete recorded knowledge passes the same witness checks. The action budget stays 128; the new search budget counts both navigation and macro expansions and is 100,000 for every method. [Oracle/witness receipt](results/r5/t2_diagnosis.json).

Normal frontier validation on four episodes per world reaches 9 rooms/8 mechanisms and succeeds on 28/28 tasks in each world, with mean task length 45.714 actions: [plain](results/r5/t2_planner_validation_plain.json), [TV](results/r5/t2_planner_validation_tv.json). Thus good observed knowledge can now distinguish an explorer from empty replay. This is validation of the task stage, not a new held-out neural comparison.

**Reproducibility and custody.** The existing smoke passed in [Claude’s log](logs/r5/claude_smoke1.log). The corrected old checkpoint path is `checkpoints/r5_r5_ac/`; [reload receipt](results/r5/r5_ac/lp_tv_s510_reload.json) exactly matches `episodes`, `tasks`, `transfer` and `empty_replay`, independently rechecked in the diagnostic receipt. Existing GIFs remain [LP](media/r5_ac_lp_tv.gif), [RND × episodic](media/r5_ac_rnd_ep_tv.gif), and [frontier](media/r5_ac_frontier_tv.gif). No old checkpoint, result, log or media file was overwritten.

**Budget.** Recorded Turn-1 training-only totals were 12.1, 26.4 and 27.8 minutes per seed, excluding evaluation (`training_seconds` in [t2_diagnosis.json](results/r5/t2_diagnosis.json)). Turn 2's throughput worry did not materialise; see Turn 2 surprises.

Local Turn-2 checks: **24 NumPy tests passed in 9.09 s**, 14 Python files parsed and `bash -n smoke_r5.sh` passed. The report regenerated from all 36 Turn-1 cells matches the original summary JSON exactly. [Verification log](logs/r5/astra_turn2_checks.txt); [rechecked report](results/r5/t2_turn1_recheck/summary.md). Claude's Turn-2 run then repeated these: 24 tests passed and all five `SMOKE_METHOD_OK` lines (including `lpm` exact kill/resume) appeared ([log](logs/r5/claude_t2_smoke.log)). Peak memory remains unmeasured.
