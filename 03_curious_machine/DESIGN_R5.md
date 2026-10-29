# Round 5: Does curiosity collect useful experience?

Historical Turn-1 design. See [DESIGN_R5_T2.md](DESIGN_R5_T2.md) for the task-planner repair,
new progress method and current validation/budget plan after Claude's first runs.

**Hypotheses.** (H1) Lagged prediction improvement spends less time viewing the TV than
RND × episodic novelty while retaining more mechanism discovery than the existing TV fixes.
(H2) Better discovery produces higher surprise-task success from an identical replay format.
A strong frontier baseline may beat all neural explorers; that is an informative outcome.

**Methods.** Train fresh PPO policies for `lp`, `rnd_ep`, `disagree`, and `ngu`; compare with
a systematic frontier/BFS explorer and uniform random actions, in both existing houses.
Reuse the existing PPO/network implementations with common settings: 32 environments,
128-step rollouts, 512-step episodes, hidden width 256, four epochs/four minibatches,
Adam 2.5e-4, entropy .001. No old checkpoints or extra normalisation rollouts. R5 masks
reset transitions and clips auxiliary gradients for all methods; historical results are context,
not rows in this new comparison.

`lp` predicts a frozen 64-dimensional random embedding of the next raw observation from
(current observation, action). Before fitting each rollout, reward is the **signed** difference
between a lagged predictor's squared error and current error on that same fresh transition,
clipped to [-1,1]. Refresh the reference every L updates, after scoring and before fitting.
Fixed inputs/targets avoid normalisation-induced progress; no positive-part clipping creates
a stationary-noise reward floor. The predictor still sees TV pixels. This is a small
lagged-predictor ablation inspired by [Learning Progress Monitoring](https://arxiv.org/abs/2509.25438),
**not** its learned error-model method and not a guarantee of noise immunity. Loss, signed
reward spread, positive/negative fractions and old/current errors are logged.

**Transfer protocol.** Freeze each explorer, collect exactly 512 actions on a fresh seed, then
reveal seven tasks: lamp on in room 0/4/8; carry the key to room 0/3; reach room 6/8.
For each task reset the physical house, transfer only that episode's action/observation replay,
and allow 128 actions plus at most 4096 BFS node expansions. All methods use the same planner:
graph edges are observed transitions; nodes include observation-derived position, facing, key
and the complete remembered semantic map. No simulator map, mechanism links, policy weights
or training replay transfer. TV intensity is excluded from **every** planner's memory equally.
Odometry uses overlapping observed walls/room colours. The frontier policy has generic object
affordances but no house layout or button-door links. Unplanned actions use the same seeded
random fallback; an empty-replay control measures how much the replay helps. Failures cost 128
steps. Actual simulator state scores success, never chooses explorer or planner actions.

**Tuning and splits.** Frontier searches mechanism priority {0,6,12} × frontier priority {0,2};
LP searches lag {4,16}, each with 131072 training steps/world using pilot seed 519. Select one
setting across both worlds on seeds 61500–61503 using rooms/9 + mechanisms/8 − .25×TV fraction;
never select on surprise-task scores. Existing explorers retain their previously developed
settings. Final train seeds are 510–512, fresh test seeds 72500–72515 (32 supported), and
10000–10003 are secondary only. These are held-out randomness, **not new layouts**.

**Metrics and budgets.** Report per-seed rooms/9, mechanisms/8, coverage, unproductive/repeated
steps, TV/512; task success, success-only steps, failure-capped steps, plan coverage, per-task
results and empty-replay success. Aggregate means and population SD over training seeds,
retaining individual episodes and source files. The nonlearning baselines also execute the
same pretraining step count (logged/checkpointed); this experience does not fit weights or
carry a map into evaluation. Their repeated results across training seeds are not independent
replicates. Environment budgets match; neural optimisation compute is separately disclosed.

**Minimum viable version / battery first.** Smoke ≤2 min; tuning capped at 3 min; one main
run capped at 25 min; ≤35 min total including any bounded continuation/reporting. One training
seed, all six methods/both worlds, 1048576 steps each, 16 test episodes: descriptive pilot only.
On AC, extend to three fresh seeds at 3145728 steps each, up to three 42-minute run invocations
(≤135 min including preparation). Every process checkpoints/resumes and has a wall guard;
run under `gpu_run`, ≤4 threads, estimated working memory comfortably below 16 GB (unmeasured).
If the guard stops a cell, preserve it and resume within the remaining allowance; never include
unequal or incomplete cells in the matched headline. A small run may not train a good explorer.
