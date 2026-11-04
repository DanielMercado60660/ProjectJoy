# Round 6: weight-space programming (WSP). Can an LLM merge where weight arithmetic failed?

Run 2026-11-04, 15:16–16:25 CST, on AC power. Runner, evaluator and reviewer: Claude. Weight writer: Astra
(OpenAI gpt-6-astra via `codex exec`, reasoning xhigh, workspace-write sandbox).

## Verdict

**Yes, in a narrow sense, and the mechanism matters more than the headline.** Astra wrote networks of
the *same* 19→128→128→6 architecture that keep **82–84% (one-shot, no rollouts)**, **83–89%
(10 rollout calls)** and **90% (50 calls)** of the specialists' return on held-out resets. Every
weight-arithmetic merge keeps at most 46%, and no merge beats the trivial 55% (fwd specialist for both
labels). All 6 independent Astra threads settled on the same construction: a task-routing skeleton
(shared layer 1, layer 2 split into a fwd bank and a bwd bank, the inactive bank switched off by
saturated tanh, the output bias cancelled per task), filled in by **closed-form ridge regression onto
the specialists' outputs on synthetic Gaussian inputs**. That is data-free distillation done with
linear algebra, not a sparse edit of the existing weights: 93–100% of weights change. The controls show
that the regression is what does the work:

* the routing skeleton with the specialists' original weights and no fitting keeps 24%;
* plain gradient-descent distillation on standard-normal synthetic inputs (0 env steps, 4000 Adam
  steps, ~8 s per seed) keeps **83%**, the same as Astra's one-shot.

With environment access, gradient methods still win at a matched budget. DAgger with the same env steps
keeps 96–97.5% (Astra: 89–90% with 300k steps over 4 seeds, ~75k per seed). PPO fine-tuning from the average at the same budget keeps
only 26–48%. So the LLM found and executed the right idea (route on the label, compress each specialist
to half width, fit the compression in closed form) with zero rollouts and no gradient descent. It did
not beat a dumb gradient baseline with the same information. "Merging" here works by rebuilding the
network around the label, not by editing a few weights.

The construction does not depend on shared pretraining. I ran the two one-shot builders unchanged on the
`init` condition (θ0 = random init), changing only the data path. There every weight merge keeps ~0%
and the average keeps −5%; the transferred builds keep **86% / 87%** held-out (DAgger 200k: 99%). This
run is mine, not an Astra attempt; see `init` transfer below.

The bar written down before any result (`Pre-registered expectation` below) was 56% / 68%. Every Astra
attempt cleared it on the mean.

## Pre-registered expectation (written 15:25, before any attempt finished)

Both specialists use all 128 hidden units in both layers. An exact label-gated union of two 128-128
networks needs 256 units per layer. In the same architecture, **an exact routing construction is
impossible without compressing each specialist to about half its width** (or finding shared structure).
Saturating tanh units with large one-hot weights can switch units on or off per task, and saturated
units can act as task-dependent biases for the next layer. The open question is how much each
specialist loses when pruned or compressed to ~64 units without data. The realistic bar is beating 56%
(fwd specialist for both labels) and 68% (tuned CPG gait), not reaching DAgger's 99%.

## Setup

* **Target:** project 25, condition `pre` (θ0 = 0.5M multitask PPO steps), training seeds 0–3,
  HalfCheetah-v5 `fwd` (+v − 0.1|a|²) and `bwd` (−v − 0.1|a|²) specialists at 2M PPO steps.
  Actor: tanh MLP 19→128→128→6, input [normalized obs (17), task one-hot (2)], deterministic mean
  action, 1000-step episodes. These are the same specialists and protocol as the earlier rounds.
* **What Astra got:** `wsp/data/s{k}/{theta0,fwd,bwd}.npz` (actor weights + frozen normalizer,
  exported from `checkpoints/s{k}/{base,pre_fwd,pre_bwd}.npz`) and `wsp/data/TASK.md`. TASK.md holds
  the architecture, IO index map, obs/action layout, rewards, the earlier results table, the zero-gradient
  label-column fact, the rules and the deliverable. Every attempt got identical text.
* **Rules:** use only `wsp/data` weights (no `checkpoints/`, `results/`, other attempts). No gradient
  descent, no iterative loss minimisation, no random or evolutionary search over weights. Closed-form
  linear algebra and unlimited numpy forward passes on self-generated synthetic inputs are allowed but
  must be declared. Write only in `wsp/attempts/<tag>/`. No installs, network or git.
* **Variants** (prompts in `wsp/tools/launch_attempt.sh`):
  * **one-shot**: no environment access at all, a single turn, no feedback.
  * **budget-N** (N = 10, 50): at most N calls to `wsp/tools/eval_budget.py` in total over the 4 seeds.
    One call = one candidate × one seed × both tasks × 3 episodes on the fixed TRAINING reset bank
    60000–60002 = 6000 env steps. It returns scalars only (returns, x-velocity, retained vs the specialist
    on those resets) and appends to a tool-owned ledger. Call N+1 is refused. Every candidate had to be
    a reasoned edit, kept only if it scored better.
* **Two independent attempts per variant** (fresh Codex threads), 6 in total. All are reported. The
  network scored is the one each attempt declared final. There was no re-selection by me.
* **Turn 0 smoke** (thread `01a1a870-cf99-7cc2-aaa3-97db1bedb974`): numpy loads the npz,
  `import mujoco, gymnasium` **works inside the Codex sandbox**, one evaluator call takes 0.25 s, and a
  numpy forward on 100k inputs takes 0.04 s. Since mujoco worked, the budget is enforced by the ledger
  plus an audit of every command in the Codex logs (below), not by the sandbox.
* **Held-out scoring, runner only** (`wsp/heldout.py`): after the attempts finished, every attempt and
  every baseline was scored on a fresh reset bank, **31000–31019 (20 episodes per task per seed)**.
  That bank existed only in my scratchpad until the attempts were done. The canonical 10000–10009 bank is
  a secondary column: it is readable in `common.py` but was never used by the attempts' evaluator.
  The canonical column reproduces the earlier rounds' numbers exactly (oracle 5320 / 8026, average
  429 / 3358), which validates the scorer.

### Attempts

| tag | variant | Codex thread | calls (per seed) | env steps | wall | Codex tokens: input (cached) / output (reasoning) | commands |
|---|---|---|---|---|---|---|---|
| `r6wsp_os_a` | one-shot | `01a1a871-e602-7f81-8da2-b6dbdc8a0df0` | 0 | 0 | 8.0 min | 390k (372k) / 17.6k (7.8k) | 13 |
| `r6wsp_os_b` | one-shot | `01a1a871-e602-7722-bf80-c0779681c6d3` | 0 | 0 | 7.2 min | 397k (380k) / 17.9k (8.0k) | 10 |
| `r6wsp_b10_a` | budget-10 | `01a1a885-7b2d-7af3-9aed-f5aa0268a0cc` | 10 (3/2/2/3) | 60,000 | 8.2 min | 921k (891k) / 19.4k (6.4k) | 22 |
| `r6wsp_b10_b` | budget-10 | `01a1a885-7b35-76b1-9ec1-8c8674027c80` | 10 (3/1/3/3) | 60,000 | 7.8 min | 636k (612k) / 18.6k (5.9k) | 25 |
| `r6wsp_b50_a` | budget-50 | `01a1a871-e602-7b03-add6-499634c38ffb` | 50 (15/4/16/15) | 300,000 | 12.5 min | 1.10M (1.06M) / 25.2k (10.7k) | 26 |
| `r6wsp_b50_b` | budget-50 | `01a1a871-e602-7952-ae41-dc77d543518d` | 50 (9/9/15/17) | 300,000 | 15.1 min | 1.81M (1.78M) / 31.3k (12.7k) | 38 |

Logs: `round5_logs/25_skill_transplantation_r6wsp_*.{jsonl,err,last.md}`. Final messages:
`wsp/audit/run_*.out`. Both budget-50 attempts and both budget-10 attempts used their full budget.
One-shot and budget-10 attempts ran concurrently in pairs; the budget-50 pair ran alongside the one-shots.

### Audit

* `wsp/tools/*`, `wsp/data/TASK.md` and all `wsp/data/s*/*.npz` hash-match before and after the
  attempts (`wsp/audit/sha_before*.txt`). The one mismatch is `launch_attempt.sh`, which I edited
  myself between the two batches to parameterise N.
* Every shell command in the six Codex logs and every `.py` file in the attempt folders was grepped
  for `checkpoints/`, `results/`, `logs/`, `media/`, other attempts' folders, the held-out bank,
  `gymnasium`, `mujoco`, `make_env` and `evaluate(`. There were **no hits** apart from attempts writing
  their own `STATUS.md` and a false positive (`62031000` contains `31000`). All env interaction went
  through the evaluator subprocess. Ledger counts are 50, 50, 10, 10.
* Every attempt's `build.py` rebuilds its four `merged_s*.npz` **byte-identically** from `wsp/data`
  alone (checked by me, SHA-256).
* No copying: every merged actor is ≥ 213 (L2) from every network in `checkpoints/s<k>/`
  (`wsp/results/mech.json`, `nearest_ckpt_dist`).

## Main result: held-out retained return (bank 31000–31019, 4 seeds × 20 episodes per task)

`retained` = return / the specialist's return on the same bank and seed. Mean ± std over the 4 training
seeds. *fails* = episodes below 50% of the specialist's mean (a fall or flip). Env steps are per
training seed. Full table, including the canonical bank: `wsp/results/table.md`.

| network | env steps / seed | fwd return | bwd return | **mean retained** | per seed | fails /160 |
|---|---|---|---|---|---|---|
| specialists switched by label (ceiling, 2× params) | (4.5M to make) | 5243 | 8035 | 100% | 100/100/100/100 | 1 |
| DAgger from θ0, 200k (earlier round's checkpoint) | 200k | 5235 | 7988 | 99.7% ± 1.2 | 102/98/99/99 | 1 |
| DAgger from θ0, 299k *(full b50 total per seed)* | 299k | 5198 | 8011 | 99.5% ± 0.8 | 100/100/98/99 | 0 |
| **DAgger from θ0, 74k** *(= b50 total / 4)* | 74k | 4937 | 7832 | **95.9% ± 2.6** | 99/94/92/98 | 5 |
| **DAgger from θ0, 61k** *(= b10 total per seed; above the 74k row only by 4-seed noise)* | 61k | 5059 | 7906 | **97.5% ± 1.8** | 99/99/95/96 | 0 |
| DAgger from θ0, 16k *(= b10 total / 4)* | 16k | 2491 | 6912 | 67.6% ± 23.3 | 90/92/46/42 | 46 |
| **Astra budget-50 A** | 75k avg | 4647 | 7238 | **89.6% ± 3.9** | 93/90/83/92 | 6 |
| **Astra budget-50 B** | 75k avg | 4768 | 7194 | **90.4% ± 4.1** | 96/86/87/92 | 5 |
| **Astra budget-10 A** | 15k avg | 3985 | 7228 | **83.0% ± 6.5** | 80/94/79/78 | 18 |
| **Astra budget-10 B** | 15k avg | 4581 | 7243 | **88.7% ± 2.0** | 86/89/89/91 | 6 |
| **Astra one-shot A** | 0 | 4038 | 6904 | **81.9% ± 11.6** | 94/93/70/71 | 18 |
| **Astra one-shot B** | 0 | 4560 | 6543 | **84.3% ± 7.6** | 92/92/76/78 | 15 |
| *control:* SGD distillation on synthetic N(0,1) inputs (runner) | **0** | 4097 | 7031 | **82.9% ± 8.8** | 88/90/86/68 | 14 |
| *control:* routing skeleton, original weights, no fitting (runner) | 0 | 808 | 2544 | 23.6% ± 15.9 | 25/8/13/49 | 120 |
| *control:* routing skeleton + closed-form output-layer refit (runner) | 0 | 2591 | 5431 | 59.4% ± 17.9 | 87/62/40/49 | 60 |
| PPO fine-tune from the average, 1M (earlier round's checkpoint) | 1M | 3425 | 7150 | 77.3% ± 4.1 | 79/83/74/73 | 8 |
| **PPO fine-tune from the average, 299k** | 299k | 1410 | 5600 | **48.3% ± 5.2** | 54/52/44/43 | 80 |
| **PPO fine-tune from the average, 74k** | 74k | 746 | 3736 | **30.1% ± 12.7** | 38/8/35/40 | 104 |
| PPO fine-tune from the average, 61k | 61k | 894 | 3600 | 30.7% ± 8.8 | 37/16/31/38 | 105 |
| PPO fine-tune from the average, 16k | 16k | 574 | 3310 | 25.9% ± 8.3 | 35/12/28/29 | 114 |
| fwd specialist under both labels | 0 | 5243 | 791 | 54.8% ± 6.5 | 49/48/62/61 | 81 |
| θ0 (shared start) | 0 | 1654 | 4808 | 45.8% ± 4.3 | 48/49/39/48 | 80 |
| naive weight average | 0 | 515 | 3230 | 24.9% ± 11.0 | 32/7/25/36 | 125 |
| task arithmetic / TIES / DARE, original λ grid | 0 (λ on val) | ≤1282 | ≤5091 | 38.0 / 39.3 / 29.0% | | 80–124 |
| task arithmetic / TIES / DARE, extended λ grid | 0 (λ on val) | ≤1653 | ≤4960 | 44.1 / 45.2 / 46.2% | | 80 |
| scripted CPG gait, wider box (earlier round, canonical bank only) | 5.76M CEM | 4075 | 4693 | 68% | | not measured |

Plot: `wsp/media/wsp_retained.png`.

**Matched budgets.** A budget-50 attempt spent 300,000 evaluation env steps in total, unevenly across
seeds (15/4/16/15 and 9/9/15/17 calls). The gradient arms (`wsp/baselines.py`, the same `ppo.train` call
as the earlier `pre_ft_average` arm and `pipeline.distill` for DAgger) ran with a uniform 73,728 steps
per seed (≈ 300k / 4). They also ran with 299,008 per seed, which gives each seed the attempt's whole
total and so is generous to the gradient side. The budget-10 analogues are 16,384 and 61,440. DAgger
uses no rewards but queries both specialists at every student step; those queries are free here, as they
are for Astra, who reads the specialists' weights. PPO steps are stochastic training steps; Astra's steps
are deterministic evaluation episodes.

**Selection overfit (budget attempts).** Each budget attempt picked per seed the candidate with the best
score on the same 3 training resets. On those resets the final picks score much higher than on held-out
resets. Unselected networks show no such gap.

| network | training resets 60000–2 (as Astra saw it) | held-out 31000–19 |
|---|---|---|
| b50_a / b50_b | 96.5% / 97.4% | 89.6% / 90.4% |
| b10_a / b10_b | 87.1% / 95.9% | 83.0% / 88.7% |
| one-shot a / b (never selected; scored on training resets by me afterwards) | 85.8% / 82.7% | 81.9% / 84.3% |
| SGD synthetic distillation (never selected) | 88.3% | 82.9% |

Selection over 15–20 candidates per seed on 3 resets inflates the budget attempts by about 7 points. The
real held-out gain from 50 calls over no calls is about +6 points (82–84% → 90%). Seed 1's training-reset
scores exceed 100% because its fwd specialist scores low on those three resets (3991 vs 5304 held-out).
(`wsp/results/train_vs_heldout.json`)

**Failures are bimodal.** Low means come from whole episodes in which the cheetah flips or stalls
(returns of ~100–1000 among episodes of ~5000), not from a uniformly slower gait. The one-shots fail 15–18
of 160 held-out episodes and budget-50 fails 5–6 (DAgger-74k: 5; ceiling: 1). The bottom row of the demo
is such a failure: one-shot A, seed 1, reset 31013, return 470. It flips onto its back like the naive
average does, while the same network runs normally on the other 19 resets.

## The construction Astra used (all 6 threads)

Indices are zero-based. Each attempt documents its exact variant in `wsp/attempts/<tag>/NOTES.md`.

1. **Layer 1: 127 shared observation features + 1 task unit.** Each pair of aligned rows (same neuron
   index, inherited from θ0) gets one shared 17-d observation row. That row is the plain average (os_a,
   b50_a seeds 1/3), a weighted fwd/bwd blend (b50_b: 50/50, 100/0, 75/25; b10_b: 62.5–75% fwd), a
   closed-form fit to both specialists' pre-activations (os_b), or a QR pick from the pooled 256 rows
   (b50_a seed 2). Each task's original first-layer bias goes into **that task's one-hot column**
   (bias + W[:,17] for fwd, bias + W[:,18] for bwd). This is the one place the label "naturally" enters,
   and Astra used it as a per-task bias. One row with the lowest output saliency is dropped, and unit 127
   becomes `tanh(K·x17 − K·x18)` = ±1 (K = 3–20).
2. **Layer 2: a fwd bank and a bwd bank (≈64 + 63 units) + 1 task unit.** Each bank keeps the
   specialist's most useful layer-2 units, chosen by output-weighted pivoted QR on synthetic activations.
   Their rows are **refit by one ridge least-squares solve** from the 127 shared features to the
   specialist's own layer-2 pre-activations, on 16k–49k synthetic inputs. The other bank is switched off
   by a weight from the task unit plus a bias offset M = Σ|w| + |b| + 12 (or the same idea). Under the
   wrong label the pre-activation is ≥ 12 in magnitude **for any input in the [−1,1]¹²⁷ box**, so tanh
   saturates to exactly ±1 in float32.
3. **Output layer: two refit read-outs with exact cancellation.** Each bank's 6 logits are fit by one
   ridge solve to the specialist's logits. Some attempts use sample weights of the form
   (1 − tanh²)² + floor, computed from the teacher only and never reweighted. The saturated bank
   contributes a known constant. The output bias and the weight from the layer-2 task unit are set to
   `(d_f + d_b ∓ …)/2` so that the constant cancels and each label gets its own intercept.

Synthetic inputs: standard-normal normalized observations, with hand-chosen priors in most attempts:
x-velocity drawn around ±3 to ±8.5 m/s by task, near-upright pitch, a scale mixture. These priors came
from the return table in TASK.md and physical intuition, not from rollouts. The budget attempts used their
calls to choose among synthetic priors (upright/motion/speed profiles), the fwd/bwd capacity split
(48/79 … 80/47), the first-layer blend, and the derivative weighting. In b10_b, call 7 moved to an 80/47
split, collapsed seed 3's bwd gait (557 return), and was rejected. Call 8 moved to 72/55 and was kept.

Both budget-50 attempts noticed that "algebraically identical" recombinations of fitted blocks change
rollout returns. b50_a writes that small floating-point perturbations seem to produce divergent rollouts.
I confirmed this: the same network and reset give a different trajectory when the numpy forward runs on a
batch of 20 observations instead of 1 (BLAS rounding). Single-episode returns are therefore not
reproducible across batch layouts. Means over 20 resets are stable: two identical held-out runs matched
to the digit. The demo replays the exact batched rollouts for this reason.

## Mechanistic check (`wsp/mech.py` → `wsp/results/mech.json`)

On each specialist's own trajectories (5 held-out resets × 1000 steps per task and seed). Means over the
4 seeds.

| network | mean \|π(s,fwd) − π(s,bwd)\| | action MSE vs fwd spec (fwd states) | vs bwd spec (bwd states) | live L2 units fwd / bwd / both | L2 units saturated (>95% of states) fwd / bwd label | weights changed vs average | max \|w\| |
|---|---|---|---|---|---|---|---|
| naive average | 0.454 | 0.252 | 0.213 | 128 / 128 / 128 | 0 / 0 | 0% | 0.78 |
| θ0 | 0.355 | 0.317 | 0.267 | 128 / 128 / 128 | 0 / 0 | 100% | 0.63 |
| Astra one-shot A / B | 0.741 / 0.743 | 0.010 / 0.010 | 0.013 / 0.013 | 64 / 63 / **0** | 64 / 65 | 94% / 100% | 20 / 17 |
| Astra budget-10 A / B | 0.744 / 0.742 | 0.010 / 0.006 | 0.013 / 0.012 | 63–70 / 57–64 / **0** | 58–64 / 63–71 | 93% / 98% | 16 / 20 |
| Astra budget-50 A / B | 0.748 / 0.752 | 0.009 / 0.010 | 0.016 / 0.016 | 60–64 / 64–67 / **0** | 64–68 / 61–64 | 100% / 95% | 16 / 16 |
| routing skeleton, no fitting | 0.676 | 0.085 | 0.151 | 64 / 63 / 0 | 64 / 65 | 96% | 16 |
| routing skeleton + output refit | 0.726 | 0.024 | 0.036 | 64 / 63 / 0 | 64 / 65 | 96% | 16 |
| SGD synthetic distillation | 0.728 | 0.009 | 0.018 | 104 / 106 / 82 | 8 / 2 | 100% | 1.2 |
| DAgger 74k | 0.543 | 0.001 | 0.001 | 128 / 128 / 127 | 0 / 0 | 100% | 0.93 |

("live" = |pre-activation| < 2 on > 5% of states; a unit is counted per label.)

* **The routing claim holds on real states.** Under either label, the live layer-2 units of every Astra
  network form a disjoint set (overlap 0). The other bank is saturated on > 95% of visited states.
  The analytic ≥ 12 margin makes the saturation hold on every state, not just these.
* **Fidelity explains the score.** Action MSE to the right specialist falls from 0.21–0.25 (average) to
  0.006–0.016 (Astra), close to SGD-on-synthetic (0.009–0.019) and about 10× above DAgger-74k (0.001).
  DAgger fits on-policy states; the synthetic fits never see one. That gap matches the gap in return
  (90% vs 96%) and the extra flips.
* **Not a sparse edit.** 93–100% of actor weights differ from the average, and the gate constants reach
  |w| ≈ 16–20 (specialists ≤ 0.8). The blog's sparse "remove one obstacle" edits (3.5% and 0.016% of
  weights) have no counterpart here. This is a re-synthesis of the network around the label.
* **A correction to TASK.md (my error, disclosed).** TASK.md told Astra that "the averaged network
  effectively ignores its task input", which repeated the brief. That holds in the `init` condition
  (earlier RESULTS.md: the average there runs backward under both labels). In `pre`, which is what Astra
  merged, the average's actions do depend on the label: mean |Δa| = 0.45, more than θ0's 0.36. Its
  x-velocity under the fwd label is +0.4…+1.6 m/s vs −0.7…−5.3 m/s under bwd. So in `pre` the average is a
  wrecked fwd gait rather than a label-blind one. The statement may have nudged every thread toward
  "routing" framing. The construction would be the natural one either way, but this is a confound for
  "6/6 threads converged". Also, all six threads are the same model on the same prompt, so their
  convergence is weak evidence that the construction is canonical.

## What this does and does not show

* **Shown:** with nothing but the weights, an architecture description and a text prompt, the LLM
  produced, in about 8 minutes and with no rollouts, same-architecture merges at 82–84% held-out. That is
  where task arithmetic, TIES and DARE sit at ≤ 46% and the averaged-then-PPO route needs ~1M env steps
  to reach 77%. The weights it wrote do what its notes say (disjoint routed banks, exact gates).
* **Not shown: that weight *reading* is the source of the gain.** The skeleton with the specialists'
  own weights scores 24% (59% with the output layer refit). The remaining ~25 points come from refitting
  layer 2 and the output to the specialists on synthetic inputs, which SGD does equally well (83%).
  Astra's contribution is choosing that program and its constants (and, with a budget, the priors and the
  capacity split). The weights are produced by least squares.
* **Not shown: better than gradient descent at matched information.** With 0 env steps it ties SGD
  distillation on synthetic inputs. With 15–75k env steps per seed it trails the matched DAgger runs by
  5.5–14.5 points and fails more episodes. It beats matched-budget PPO from the average by a lot (90% vs 30%), but that arm
  is a weak use of the budget.
* **Consistent with the blog's limits:** the nets fit in context (about 20k actor parameters per network,
  read with numpy rather than as text), and there is a nameable mechanism (the label as a per-task bias,
  saturation gating). "Merging" worked here because nothing new had to be learned: both gaits already
  existed and only had to be packed into half the width each.

## `init` transfer (runner-run, not an Astra attempt; `wsp/init_transfer.py` → `wsp/results/init_transfer.json`)

The two one-shot `build.py`s were copied to `wsp/init_transfer/r6wsp_os_{a,b}/`. The only change is
`DATA = … / "data_init"`, which points at `checkpoints/s*/{init,init_fwd,init_bwd}.npz` exported in the
same format. They were scored on the same held-out bank. The budget builders were not transferred:
their per-seed configs were selected on `pre` training resets, so carrying them over would be meaningless.

| `init` condition, held-out 31000–31019 | fwd | bwd | mean retained | per seed |
|---|---|---|---|---|
| specialists switched by label | 5507 | 7948 | 100% | |
| DAgger from θ0, 200k (earlier round's checkpoint) | 5374 | 7924 | 98.7% ± 1.3 | 99/97/100/99 |
| **Astra one-shot A, transferred** | 4439 | 7113 | **85.6% ± 11.7** | 82/68/97/95 |
| **Astra one-shot B, transferred** | 4529 | 7243 | **87.1% ± 9.1** | 85/73/95/96 |
| fwd specialist for both labels | 5507 | −4884 | 19.1% ± 7.1 | |
| θ0 (random init) | 0 | 0 | 0% | |
| naive average | −2109 | 2258 | −4.8% ± 4.1 | |

In `init` the averaged network really is label-blind: it runs backward under both labels, and every
earlier merge sits at about 0%. A seed-agnostic, data-free closed-form construction written for `pre`
recovers 86–87% here without modification. The shared first layer (aligned rows, averaged) survives
because both specialists still inherit their row identities from the same random init.

## Unfinished / not measured

* The compatible pair (fwd vs cruise): not run.
* Budget-25: not run (10 and 50 bracket it).
* CPG on the held-out bank: not measured (68% is on the canonical bank).
* Wall-clock or FLOP matching of Astra's API compute against gradient compute: not attempted. The token
  counts are above.

## Files

| path | what |
|---|---|
| `wsp/data/TASK.md`, `wsp/data/s*/` | exactly what Astra saw |
| `wsp/tools/eval_budget.py`, `budgets.json`, `ref_train.json`, `launch_attempt.sh` | budgeted training-reset evaluator, budgets, specialist references, attempt prompts |
| `wsp/attempts/r6wsp_*/` | each attempt's `merged_s0..3.npz` (final), `build.py`, `NOTES.md`, `ledger.jsonl`, rejected candidates |
| `wsp/baselines.py` | matched-budget PPO-from-average and DAgger arms → `wsp/baselines/{ppo_ft_avg,dagger}_<steps>_s*.npz` |
| `wsp/synth_distill.py` | control: SGD distillation on synthetic inputs → `wsp/baselines/synth_distill_s*.npz` |
| `wsp/ablations.py` | control: routing skeleton without / with output refit → `wsp/baselines/ablate_*_s*.npz` |
| `wsp/heldout.py` | held-out + canonical scoring of everything → `wsp/results/heldout.json` |
| `wsp/report_wsp.py` | → `wsp/results/table.{md,json}` |
| `wsp/mech.py` | mechanistic check → `wsp/results/mech.json` |
| `wsp/init_transfer.py`, `wsp/data_init/`, `wsp/init_transfer/` | runner-run transfer of the one-shot builders to the `init` condition |
| `wsp/demo_wsp.py` | → `wsp/media/wsp_demo.{mp4,gif}`, `wsp_demo_final.png`, `wsp_retained.png` |
| `wsp/audit/` | hashes before the attempts, attempts' final messages |
| `wsp/logs/` | job logs |

Rerun (from `25_skill_transplantation/`, every step through the GPU lock, each one under 2 min):

```bash
../.venv/bin/python wsp/attempts/r6wsp_b50_b/build.py           # any attempt: rebuilds its merged_s*.npz byte-identically
../bin/gpu_run ../.venv/bin/python wsp/baselines.py --steps 73728  # also 299008, 61440, 16384
../bin/gpu_run ../.venv/bin/python wsp/synth_distill.py
../.venv/bin/python wsp/ablations.py
../bin/gpu_run ../.venv/bin/python wsp/heldout.py                # ~1.5 min, pool of 4
../.venv/bin/python wsp/report_wsp.py
../bin/gpu_run ../.venv/bin/python wsp/mech.py
../bin/gpu_run ../.venv/bin/python wsp/demo_wsp.py               # ~20 s
../bin/gpu_run ../.venv/bin/python wsp/init_transfer.py          # init-condition transfer, ~1 min
```

The demo `wsp/media/wsp_demo.mp4` shows training seed 0, held-out reset 31000 (bottom row: seed 1,
reset 31013), labels fwd | bwd. Rows: specialists switched by label (4964 | 7700), naive average
(−26 | 4955, flips onto its back under fwd), Astra one-shot B (4545 | 7286), Astra budget-50 B
(4792 | 7538), and a one-shot failure (one-shot A, 470 | 7276). The clips replay the exact rollouts
behind the held-out scores.
