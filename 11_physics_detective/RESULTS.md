# RESULTS: The Physics Detective

All numbers come from real runs on 2026-10-22 (M5 Max, shared with 12 other jobs). Seeds are
fixed (`run_active.py --seed 2026`, SBI training seed 0, fresh SBI test set seed 999,
world 2 seed 11, world 3 seed 3). Raw outputs are in `results/`; `results/summary.json` holds
every number below except the reviewer-added rerun numbers.

## Setup (what the detective faces)

* **Hidden world (v1):** a 1 kg block with unknown kinetic friction μ_k ~ U[0.05, 0.8] and static
  friction μ_s = 1.2·μ_k. A push is a commanded force F ∈ {1, 2, …, 15} N held for
  T ∈ {0.1, 0.2, …, 1.0} s, so the menu has 150 pushes. The camera samples 20 frames
  (0.1–2.0 s) with σ = 8 cm noise and only sees x ≤ 1 m. Past that, a frame is reported as
  "out of view" (censored).
* **Why the choice of push matters:** with this noise level one push can give a posterior sd
  anywhere from 0.003 (a push that slides and stops inside the view) to 0.1 or more (a push that
  doesn't budge, or one that shoots out of view). The best push depends on the unknown μ_k.
* **Evaluation:** 500 hidden worlds drawn from the prior. Each strategy gets 15 pushes per world.
  For every world, 30 held-out "new interventions" (F ~ U[1, 15], T ~ U[0.1, 1]) are used to
  score prediction.
* **Strategies (matched budget: same menu, same number of pushes, same likelihood):**
  * **Active:** maximise expected information gain (EIG). Nested Monte Carlo with 64 posterior
    quantile particles and 2 noise draws each, scored over all 150 pushes.
  * **Random:** a uniform draw from the same 150-push menu.
  * **Fixed protocol:** "8 N for 0.3 s", repeated. This is a sensible-looking textbook push.
* **Estimators:** the exact Bayesian grid posterior (1500 points, censored/Tobit likelihood), and
  the **least-squares baseline**: minimise squared error over the visible frames (out-of-view
  frames dropped, as a practitioner typically would), with a Gauss–Newton/Laplace standard error.

## Measure 1: how well is the friction known (RMSE of the μ_k estimate over 500 worlds)

| pushes | active · Bayes | active · LSQ | random · Bayes | random · LSQ | fixed · Bayes | fixed · LSQ |
|---|---|---|---|---|---|---|
| 1  | 0.0198 | 0.0257 | 0.0958 | 0.178 | 0.0198 | 0.0312 |
| 2  | **0.0040** | 0.0068 | 0.0572 | 0.106 | 0.0174 | 0.0295 |
| 4  | **0.0021** | 0.0040 | 0.0255 | 0.0371 | 0.0156 | 0.0281 |
| 8  | **0.0013** | 0.0036 | 0.0120 | 0.0192 | 0.0147 | 0.0273 |
| 15 | **0.0010** | 0.0034 | 0.0046 | 0.0065 | 0.0142 | 0.0270 |

Active selection with Bayes reaches in 2 pushes an accuracy that random probing hasn't reached
after 15 pushes. See `media/active_vs_random.png`, left.

## Measure 2: prediction error on new interventions

Trajectory RMSE in cm on 30 held-out random pushes per world (20 frames each, positions clipped
at 2 m because past that the block has left the table). Bayes uses the posterior-predictive mean;
LSQ plugs in its point estimate.

| pushes | active · Bayes | active · LSQ | random · Bayes | random · LSQ | fixed · Bayes | fixed · LSQ |
|---|---|---|---|---|---|---|
| 1  | 4.00 | 4.58 | 13.95 | 21.5 | 3.84 | 5.10 |
| 2  | **0.92** | 1.35 | 7.81 | 11.45 | 3.31 | 4.71 |
| 4  | **0.49** | 0.92 | 3.63 | 4.46 | 2.69 | 3.95 |
| 8  | **0.32** | 0.78 | 1.78 | 2.38 | 2.28 | 3.48 |
| 15 | **0.24** | 0.76 | 1.02 | 1.38 | 2.00 | 3.35 |

## Measure 3: experiments needed to reach a target accuracy

The target is |μ̂_k − μ_k| ≤ 0.01, reached and held for every later push.

| strategy | Bayes: median pushes | Bayes: 90th pct | Bayes: share of worlds reached in 15 | LSQ: median | LSQ: reached |
|---|---|---|---|---|---|
| **active** | **1** | **2** | **100%** (worst case 4) | 2 | 99.0% |
| random | 4 | 11 | 96.6% | 5 | 87.6% |
| fixed (8 N, 0.3 s) | 3 | never | 81.6% | 4 | 70.6% |

*Reviewer note:* active's median of 1 is on a knife-edge. Exactly 50.2% of worlds reach and hold
±0.01 after the first push (the median absolute error after push 1 is 0.00996), and the mean is
1.53 pushes. In the independent seed-7 rerun (400 worlds) 47.5% reach it at push 1, so the median is
**2** there (mean 1.54). Read it as "1–2 pushes" rather than a hard 1. Random's median of 4 reproduced.
(Random: mean 4.8 among worlds that reach it.)

There's also a practical stopping rule that doesn't need the truth: stop when the detective's own
90% interval is narrower than ±0.01. Under that rule, active stops after a median of **2** pushes
(100% of worlds stop within 15), random after 8 (81%), fixed after 10 (64%). At the moment of
stopping, the 90th-percentile true error is 0.0062 (active), 0.0079 (random) and 0.0097 (fixed),
so the self-reported stop is honest.

**Cost of being smart:** choosing a push by EIG took a median of **0.19 s** (numpy, 150
candidates, under a machine load average of about 60). Random and fixed take microseconds.

**What the active detective learned to do** (`media/active_choices.png`): its first push is always
8–9 N for 0.3–0.4 s, a generic medium push. After that it pushes **just above the break-free force
μ_s·m·g** (median 0.65 N above it, 80% of pushes within 0.1–1.2 N) for a **long** time (91% of
later pushes have T ≥ 0.8 s): a slow slide that stops inside the camera view, so
the whole deceleration is observed. Nobody told it about the static threshold or the camera edge.
It found both through the EIG.

## Uncertainty calibration (coverage of central credible intervals)

Pooled over 500 worlds × 15 pushes = 7500 posteriors per strategy (`media/calibration.png`):

| strategy | Bayes 50% | Bayes 90% | LSQ ± z·SE 50% | LSQ ± z·SE 90% |
|---|---|---|---|---|
| active | 0.481 | 0.882 | **0.280** | **0.560** |
| random | 0.491 | 0.899 | 0.426 | 0.807 |
| fixed | 0.484 | 0.901 | 0.516* | 0.838* |

* The exact posterior is calibrated for every design policy, as theory says it should be (Bayes
  under the true prior doesn't care how the designs were chosen). Active's 0.882 at 90% was
  rechecked on a 20× finer grid (30,001 points): 0.496 / 0.886. The binomial SE for 500 worlds
  is about 0.013 per push count, and the rows are correlated within a world.
* *Reviewer update (cross-review):* "sampling noise" is not fully established. An
  independent rerun (seed 7, 400 new worlds) reproduced active-Bayes 90% coverage of **0.882**
  (random: 0.912). On the 30,001-point grid with mid-cell CDF the two runs give 0.887 and 0.889
  (50%: 0.494 / 0.501). Pooled over 900 worlds that is about 1.9 SE below 0.90 (SE from per-world
  means). So active-design 90% intervals are **slightly narrow, by about 1 point**: probably noise,
  but not proven. About 0.005 of the production 0.882 is a grid artefact. `GridPosterior.quantile`
  interpolates the right-edge cumulative sum, which shifts intervals left by half a cell, and that
  matters when active posteriors are only about 2 cells (sd ≈ 0.001, spacing 0.0005) wide.
* *Best case, by construction:* the inference model is **well-specified**. The likelihood uses
  exactly the simulator's physics, noise σ, field-of-view censoring rule and prior. Every
  calibration claim for the exact posterior is therefore a best-case result. Misspecification was
  not tested (see Next steps).
* **The least-squares intervals get worse the better the experiments are.** On the active
  detective's data its 90% coverage falls from 0.82 after 1 push to **0.39 after 15**. The
  cause is **truncation bias**: active pushes stop just inside the 1 m view, so noisy readings
  sometimes land past the edge and get dropped. The kept readings are biased low, so friction is
  biased high (mean error +0.0023 after 15 pushes vs a Bayes median abs error of 0.0006). Its SE
  shrinks anyway. The censored likelihood handles this exactly.
  (Reviewer note: censoring-aware least squares / Tobit regression is the standard textbook fix, so
  this is a failure of *ignoring censoring*, not of least squares as such. The independent seed-7
  rerun reproduced it: 0.85 → 0.45.)
* *Fixed LSQ coverage is inflated at low levels because in high-friction worlds (μ_k > 0.68)
  8 N never moves the block. SSE is then flat, JᵀJ = 0, and we report an infinite SE (the whole
  prior range), which always "covers". Its point estimate in those worlds is arbitrary (RMSE 0.030).

## ML element: amortized posterior network (MLX) vs the exact posterior

`sbi.py`: a Deep-Sets network (per-experiment MLP 42→192→192→128, sum-pool plus log(n), head
129→192→192→24) that outputs an 8-component Gaussian mixture over logit-scaled μ_k. It has 137k
parameters and was trained by maximum likelihood on fresh simulations only: 60,000 steps, batch
512, AdamW with cosine LR decay, 1–15 pushes per set. Training designs were half from the menu,
half continuous, and 30% of sets repeat only 1–3 distinct pushes (to look like an active
learner). Training took about 18 min wall time. **Checkpoint/resume** was verified: stopped at step 4000
(window NLL 0.017) and resumed (step 4500: −0.028, step 5000: −0.073), so the loss continued
smoothly. Final NLL (z-space) was −0.946.

**Accuracy and calibration** (`results/sbi_eval.json`). "gap/sd" is the median
|mean_net − mean_exact| / sd_exact, "sd ratio" is the median sd_net/sd_exact, and "W1/sd" is the
median Wasserstein-1 distance between the two posteriors in units of the exact sd.

| test set | n | RMSE net | RMSE exact | gap/sd | sd ratio | W1/sd | cov 50/90 net | cov 50/90 exact |
|---|---|---|---|---|---|---|---|---|
| fresh random-design sets, 1–15 pushes | 3000 | 0.0437 | 0.0436 | 0.20 | 1.08 | 0.23 | 0.505 / 0.901 | 0.501 / 0.909 |
| random detective, 4 pushes | 500 | 0.0263 | 0.0255 | 0.21 | 1.07 | 0.24 | 0.474 / 0.894 | 0.452 / 0.906 |
| random detective, 15 pushes | 500 | 0.0055 | 0.0046 | 0.33 | 1.14 | 0.36 | 0.522 / 0.904 | 0.514 / 0.902 |
| fixed protocol, 15 pushes | 500 | 0.0146 | 0.0142 | 0.24 | 1.12 | 0.26 | 0.460 / 0.878 | 0.462 / 0.890 |
| active detective, 2 pushes | 500 | 0.0054 | 0.0040 | 0.43 | 1.26 | 0.48 | 0.506 / 0.902 | 0.506 / 0.906 |
| active detective, 8 pushes | 500 | 0.0017 | 0.0013 | 0.58 | 1.41 | 0.63 | 0.510 / 0.896 | 0.500 / 0.876 |
| active detective, 15 pushes | 500 | 0.0019 | 0.0010 | 1.04 | **2.04** | 1.29 | 0.434 / 0.862 | 0.464 / 0.862 |

**Speed:** the exact grid posterior costs 12.4 ms per dataset (mean 7.8 pushes, numpy). The
network takes 1.3 ms for a single dataset and 0.0015 ms per dataset in a 3000-set batch
(forward only), or 0.46 ms including conversion to a grid density. That's a speed-up of roughly
10× single and roughly 8000× batched.
*Reviewer caveat:* the 8000× figure compares the net's batched forward pass (mixture parameters
only) with an exact posterior computed one dataset and one push at a time in a Python loop. Like
for like (both producing a grid density), the gap is about **27×** (0.46 ms vs 12.4 ms), and a
vectorised/batched numpy grid posterior would narrow it further. Treat 8000× as an upper bound
driven partly by loop overhead, not as the algorithmic speed-up.

**Verdict:** on random-style data the amortized net is essentially as good as exact Bayes, and
it is calibrated. It degrades gracefully on the **active detective's data**, which is both a
design-distribution shift (EIG repeats the same near-threshold push) and extremely sharp
(posterior sd about 0.001, or 0.005 in logit space). By 15 active pushes the net is about 2× too wide
and its mean is off by about 1 exact-sd. Its 90% coverage holds at 0.86 (the same as exact),
but 50% coverage dips to 0.43 because it is both too wide and off-centre. That is the known NPE weakness for very informative data. I preserved it rather
than chasing it (it's visible as the dashed purple curve in `media/detective.gif`).

## Go deeper 1: two hidden frictions (world 2)

Both μ_k ~ U[0.05, 0.8] and μ_s = r·μ_k with r ~ U[1, 1.8] are hidden. Sliding only reveals μ_k.
The only evidence about μ_s is whether the block breaks free at all, which is one bit per push.
The menu has finer forces here (0.25 N steps from 0.5 to 15 N, 5 durations, 295 pushes). The
posterior is a 1500 × 200 grid. EIG uses 96 resampled joint particles. 200 worlds × 12 pushes,
active vs random (`media/world2_two_frictions.png`):

| | active | random |
|---|---|---|
| RMSE μ_k after 3 / 6 / 12 pushes | **0.0047 / 0.0025 / 0.0015** | 0.055 / 0.027 / 0.0082 |
| RMSE μ_s after 3 / 6 / 12 pushes | **0.032 / 0.0076 / 0.0074** | 0.120 / 0.087 / 0.060 |
| median pushes until μ_s error ≤ 0.02 (and stays) | **4** | not reached in 12 |
| median final 90% CI width: μ_k / μ_s | 0.004 / 0.023 | 0.017 / 0.115 |
| coverage 50% / 90%, μ_k (pooled over pushes) | 0.524 / 0.892 | 0.558 / 0.929 |
| coverage 50% / 90%, μ_s | 0.455 / 0.885 | 0.471 / 0.885 |
| pushes within ±10% of the break-free force | **61%** | 8.5% |
| pushes that didn't move the block | 18% | 38% |

* **The active detective separates the two explanations by itself.** It spends most pushes
  right at the break-free force: a bisection on μ_s using the "did it move?" bit, while those
  same near-threshold pushes also give long, slow, in-view slides for μ_k.
* **Honest limit of identifiability:** μ_s can only be bracketed between two adjacent menu
  forces (0.25 N apart, so 0.0255 in μ_s). The posterior is flat inside that bracket, and the
  RMSE settles at exactly 0.0255/√12 = **0.0074** after about 6 pushes, the sd of a uniform
  bracket. The detective doesn't pretend to know μ_s better than the experiment resolution
  allows. The reported 90% CI width (0.023 ≈ 0.9 × 0.0255) says the same thing.
* **A bug I caught, preserved:** the first world-2 run used a 220-point μ_k grid (spacing 0.0034).
  The active posteriors got narrower than one grid cell, and 90% coverage for μ_k dropped to
  **0.77** (50%: 0.39). Random, with wider posteriors, looked fine (0.92). The coverage check
  flagged it, and a 1500-point grid restored 0.89. That run is kept as
  `results/world2_coarsegrid.npz`. Lesson: an "exact" grid posterior is only exact if the grid
  resolves the posterior, and active learning is exactly what makes posteriors sharp.

## Go deeper 2: a non-identifiable world, handled honestly (world 3)

*This is a single hidden world (seed 3), a demonstration rather than a statistical study.*

Hidden: the block mass m ∈ [0.5, 3] kg and the pusher's actuator gain k ∈ [0.5, 2] (log-uniform
priors). Friction is known (μ_k = 0.3, μ_s = 0.36). Truth: k = 1.3, m = 1.8, so k/m = 0.722.
Every push trajectory depends on k and m only through k/m: acceleration is (k/m)·F − μ_k·g, and
the break-free test is (k/m)·F > μ_s·g. A unit test checks that (k, m) and (2k, 2m) give
identical trajectories.

From `deeper.py world3` (seed 3), after 6 random pushes:

| quantity | posterior mean | 90% credible interval | comment |
|---|---|---|---|
| **k/m (identifiable)** | **0.719** | **[0.706, 0.731]** | contains the truth (0.722) |
| m | 1.51 | [0.74, 2.62] | prior 90% interval was [0.54, 2.74]: almost no information |
| k | 1.08 | [0.53, 1.87] | same: the data don't say |

* **Exact posterior:** a thin ridge along k = 0.719·m (`media/world3_ridge.png`, left). The
  marginal of m stays flat. It is cut only where the ridge leaves the k prior box. We report the
  ridge and the identifiable combination k/m, not a point estimate for m.
* **Conventional least squares on (k, m), 20 random starts:** every fit has the same SSE (0.7078)
  and the same k/m (0.7189), but the fits land anywhere on the ridge (m from 0.85 to 1.85
  depending on the start). cond(JᵀJ) is between 6e15 and 9e16, so the "standard errors" are
  meaningless. A naive fitter returns a confident-looking but arbitrary m.
* **Breaking the tie:** one spring-scale weighing (reading = m + N(0, 0.05²)) collapses the
  ridge: m = 1.79 [1.71, 1.87], k = 1.29 [1.23, 1.35], both covering the truth.
* **An EIG detective that may push or weigh** picks the weighing as its 2nd experiment
  (EIG 2.30 nats for weighing vs 1.07 for the best push). After 6 experiments (4 pushes, 2
  weighings) it has m = 1.83 [1.77, 1.89] and k = 1.31 [1.27, 1.35].

## What worked, what didn't, surprises

* **Worked:** EIG selection is a big win here: 1 push vs 4 to reach ±0.01 (median), and 4–7×
  lower prediction error at equal budget (4 to 15 pushes). The exact posterior is calibrated under every policy (active 90%
  runs about 1 point low, see the calibration section; well-specified setting).
  The amortized net matches it on in-distribution data at a small fraction of the cost.
* **Surprise 1 (the good kind of failure, baseline side):** the conventional least-squares
  fitter gets *worse calibrated the better the experiments are*. Dropping out-of-view frames
  biases it, and the active learner's favourite pushes are exactly the ones that graze the edge
  of the view. It can't be fixed by more data. It needs the right (censored) likelihood.
* **Surprise 2:** the "textbook" fixed push looks great after 1 push (RMSE 0.020, the same as
  active's first push) but never improves past 0.014. In 18% of worlds it never reaches ±0.01,
  because for μ_k > 0.68 it never moves the block, and for small μ_k the block leaves the view.
  Repeating a good-on-average experiment is not a strategy.
* **Surprise 3:** the active detective rediscovered the break-free threshold. Its pushes trace
  the line F ≈ μ_s·m·g + 0.65 N (`media/active_choices.png`).
* **Didn't work as well:** the amortized net under the active design shift (2× too wide at 15
  pushes, see above). Also, EIG cost (0.19 s per decision) dominates the loop. That's fine here
  but would matter with bigger menus.

## Next steps

* Use the amortized net *inside* the EIG loop (simulate, then amortized posterior) to make
  design selection fast for multi-parameter worlds. Fine-tune it on active-style design sets.
* Ball world (hidden restitution plus quadratic drag, drop/launch experiments). Not built.
* Misspecification test: give the true world a little unmodelled rolling resistance and see whether
  the posterior honestly widens or confidently lies.

## Things a reviewer should double-check

* The least-squares baseline deliberately drops out-of-view frames. That is the "conventional"
  choice and the source of its bias. A censoring-aware least squares would do better, but at that
  point it is basically the likelihood.
* The prediction metric clips positions at 2 m, so pushes that fly off the table don't dominate
  the RMSE. Unclipped numbers were not measured.
* Active-Bayes 90% coverage is 0.882 (0.886 on a 30k-point grid). I judge that to be sampling
  noise (500 worlds), not miscalibration, but it sits on the low side.
* World 3 is a single hidden world (seed 3), a demonstration rather than a statistical study.

## Gaps / environment notes

* Nothing was missing from the venv. MuJoCo was not needed: the analytic simulator is exact and
  transparent.
* The machine was heavily loaded (load average about 60), so wall times are pessimistic.

## Round 5: Probe to accomplish something

**TL;DR (reviewer-checked; every number below traces to a file named next to it).**

- **Turn 1, right physics:** probing turns a 6% blind guess into near-certain
  placement. EIG and task-directed VOI probing succeed in 95.5% / 94.5% of 512
  fresh worlds after two probes and 99.6% / 99.4% after four. Tuned least squares
  (LSQ) and Tobit fitters reach 88.5% / 90.4% after four. In these matched
  worlds task-VOI is not measurably better than EIG; every paired interval
  includes zero. Its only clear edge is +5.5 points with gain error at four probes.
- **Turn 1 surprise, hidden ~10% pusher-gain error:** the Bayesian detectives
  became *confidently wrong*. Their friction posteriors shrank onto a biased value:
  90% coverage was **0%**, and over half the worlds were "≥90% sure of success"
  but missed. Four-probe placement fell to 44% (EIG) / 49% (VOI), while the
  simpler tuned LSQ / Tobit controllers placed **59% / 57%**. Least squares did
  better at 2 and 4 probes, though not at 1. Same-history replays show this is
  mostly about which push is made at the end (force transfer under the wrong
  model), not about Bayes estimating worse. Bayesian *control* on LSQ's probe
  histories scored 79% (exploratory, reused test worlds).
- **Turn 2 fix:** add the unknown gain to the model (a joint friction × gain
  posterior on EIG's unchanged probes). On 128 new test worlds with gain error,
  four-probe placement is **97.7%**, against 46.1% for the same probes under the
  old model. Friction 90% coverage is **93.0%** (old model: 0%). The joint Tobit
  MLE with the same model reaches 92.2%. In worlds without gain error, the joint
  model costs 7.8 points at two probes (89.1% vs 96.9%) and ties at four (100%).
- **Caveat on the fix:** this fixes a *known* mismatch, not an unknown one. The
  gain-error family (|k−1| from 0.06 to 0.12) lies entirely inside the selected
  gain prior [0.85, 1.15], and gain-shift worlds were in the turn-2 validation
  set. It shows that modelling the right nuisance works. It says nothing about
  robustness to a mismatch the model doesn't contain.

Sources: `results/r5/main_summary.json`, `results/r5/turn2_diagnostics.json`,
`results/r5/turn2_main_summary.json`.

### Setup and verification

Claude completed the original round-5 runs on AC. Validation used 128 matched
worlds, seed 6202611, and 45 candidates per classical fitter. It selected LSQ
`adaptive_10_65` and Tobit `adaptive_10_90`, both with zero terminal aim offset:
push at 10 N, choose a menu duration to aim for 0.65 or 0.90 m using the current
fit. Validation took 23.16 s. Source: `results/r5/tuning_main.json`.

The headline uses 512 fresh worlds per family, seed 6202612, paired across
methods/families by friction, target and camera noise. The target is a 5 cm band
centred uniformly between 0.25 and 0.85 m. Exactly 0/1/2/4 preliminary probes cost
0/1/2/4 resets, followed by one scored push; no abstention. Scoring uses the true
eventual stop. All methods share the probe menu and final-action menu. Matched
worlds have unit gain; shifted worlds have an unknown, fixed gain
1±U[0.06,0.12]. Non-oracles assume unit gain. There is no neural component in this
round. The main run took 481.39 s. Sources: config and `elapsed_seconds` in
`results/r5/main_summary.json`; raw records in `checkpoints/r5_main.json`.

`diagnose_r5_turn2.py` reconstructed every headline summary group and paired
comparison exactly from that checkpoint. Its additional controller swaps use
only the first 128 saved worlds at four probes, explicitly **exploratory reused
test data**, not a fresh experiment. That diagnostic took 6.21 s. Source:
`results/r5/turn2_diagnostics.json` (`main_summary_recomputed_exactly`,
`swap_worlds`, `elapsed_seconds`).

### Placement success and its cost

All entries are percentages from `results/r5/main_summary.json`, `groups`.
One/two/four probes mean **two/three/five total attempts**, respectively.

| Method | Matched: 1 probe | 2 probes | 4 probes | Gain shift: 1 probe | 2 probes | 4 probes |
|---|---:|---:|---:|---:|---:|---:|
| EIG → Bayes | 57.23 | 95.51 | 99.61 | 41.80 | 42.58 | 43.95 |
| Task-VOI → Bayes | 60.35 | 94.53 | 99.41 | 38.87 | 44.34 | 49.41 |
| Tuned LSQ | 38.87 | 70.12 | 88.48 | 30.27 | 51.56 | 59.18 |
| Tuned Tobit | 42.97 | 74.61 | 90.43 | 28.13 | 51.76 | 57.23 |

Acting from the prior succeeds in 6.45% of matched and 1.76% of shifted worlds,
at one total attempt. The μ-only oracle succeeds in 100% and **0%**, respectively;
the full-world oracle succeeds in 100% of both. Source:
`results/r5/main_summary.json`, `blind`, `mu_oracle`, `world_oracle` groups.

More probes have a real cost. Matched EIG's success per attempt is 0.318 at two
probes and 0.199 at four; shifted EIG's is 0.142 and 0.088. Additional attempts
improve reliability in the correct model, but barely improve EIG placement
under gain error. These ratios are reported efficiencies, not an assumed utility
function or a claim about optimal stopping. Source: the same file,
`mean_attempts` and `success_per_attempt`.

Task-VOI minus EIG on matched worlds is +3.125 percentage points at one probe,
with paired approximate 95% interval **[−0.292, +6.542]**. At two/four probes the
differences are −0.977/−0.195 points; both intervals include zero. Thus this run
does not establish a matched-world advantage for task-directed probing. Under
gain shift at four probes, task-VOI beats EIG by +5.469 points [2.539,8.399], but
trails LSQ by −9.766 points [−15.591,−3.940] and Tobit by −7.813 points
[−13.584,−2.041]. Source: `results/r5/main_summary.json`,
`paired_success_differences`. Worlds are the sampling unit; budgets are not
pooled as independent observations.

### Uncertainty: narrower, confidently wrong intervals

Four-probe metrics below all come from `results/r5/main_summary.json`. Width is
the mean 90% friction interval width. “Confident miss” is the joint fraction of
worlds with predicted placement success ≥0.9 and an actual miss, not a fraction
conditioned on making a confident prediction.

| Method | Matched μ RMSE | Matched cov90 | Shifted μ RMSE | Shifted cov90 | Shifted width90 | Shifted confident miss |
|---|---:|---:|---:|---:|---:|---:|
| EIG | 0.00187 | 90.04% | 0.05028 | 0.00% | 0.00531 | 55.86% |
| Task-VOI | 0.00237 | 89.84% | 0.05145 | 0.00% | 0.00652 | 50.39% |
| LSQ | 0.00462 | 90.23% | 0.05855 | 0.00% | 0.01139 | 14.65% |
| Tobit | 0.00356 | 89.06% | 0.05933 | 0.00% | 0.01057 | 21.29% |

LSQ/Tobit intervals in this table belong to the **diagnostic Bayesian posterior
on their chosen probes**, not confidence intervals for their fitted estimate.
All four probe policies create confidently biased one-parameter posteriors under
the omitted gain. Smaller true-μ error does not guarantee better task performance:
EIG estimates physical friction more accurately than LSQ under mismatch but
places less successfully. Source: the table's `mu_rmse` and success fields in
`results/r5/main_summary.json`.

Correction to the initial runner interpretation: underresolution is zero in the
matched family, **not in the shifted family**. At four probes it affects 5.66%
of EIG, 4.30% of task-VOI, 4.69% of LSQ and 6.84% of Tobit diagnostic posteriors.
All flagged cases lie within 0.005 of a friction-prior edge. Sources:
`results/r5/main_summary.json`, `posterior_underresolved_fraction`, and
`results/r5/turn2_diagnostics.json`, `boundary_diagnostics`. This numerical
limitation must be retained; it does not explain the general calibration collapse,
which persists in the finer-grid run described below.

### Why the conventional controllers win under gain error

The proposed “effective μ” explanation is partly right, but its Bayesian-prior/
censoring explanation is not supported. A wrong unit-gain model can match a
particular moving push's stopping distance by fitting

\[
D(F,T;\mu,k)=\frac{T^2}{2}\left[\frac{(kF)^2}{\mu g}-kF\right],
\qquad
\mu_{\mathrm{eff}}(F)=\frac{\mu}{k^2+(1-k)\mu g/F}.
\]

These expressions apply to the moving branches; crossing a static threshold
adds another mismatch. Crucially, the effective coefficient **depends on force**.
A coefficient inferred from one push need not transfer to a different scored
push. This follows directly from `physics.rest_distance` and the model's static
threshold, not from a fitted explanation of the success counts.

The replay crosses estimators and controllers while holding each saved probe
history fixed. Shifted-world success on the **same 128-world subset**:

| Probe history | Original pipeline | Bayesian terminal action on that history | Posterior-mean point controller |
|---|---:|---:|---:|
| EIG | 50.78% | 50.78% | **72.66%** |
| LSQ | 54.69% | **78.91%** | 57.03% |

Source: `results/r5/turn2_diagnostics.json`, shifted `swaps` rows and
`controllers.saved/bayes/point_eig`. These are counterfactual actions scored in
the saved true worlds; their outcomes were not used by any original policy.
They show that neither “LSQ is intrinsically better” nor “Bayes cannot absorb
gain bias” explains the original ranking.

On those shifted histories, posterior means differ from censored MLE estimates
by only 0.00014–0.00023 on average. Bayesian inference is already concentrating
around a biased effective parameter, much like the fitter. Even excluding
histories with ≥1% posterior mass within 0.005 of a prior edge, Bayesian control
on LSQ histories succeeds in 85.47%, versus the original LSQ's 58.97% on that
same interior subset. Source: `results/r5/turn2_diagnostics.json`,
`mean_abs_post_mean_minus_mle`, `interior_n`, `interior_controllers`.

Probe and action distributions explain the transfer problem more directly.
In this subset, EIG's last probe averages 5.38 N for 0.888 s; LSQ's averages
10 N for 0.313 s. EIG's scored Bayesian action averages 11.13 N. Replacing only
that action with posterior-mean centring lowers its average force to 7.81 N,
and lowers mean distance from the force-specific effective coefficient from
0.01227 to 0.00778. Conversely, Bayesian control on the LSQ histories averages
11.96 N, near its 10 N probes, and performs better. Sources:
`results/r5/turn2_diagnostics.json` and `results/r5/turn2_controller_detail.json`.
Together with the controlled swaps, this supports acquisition/control transfer
under misspecified physics as a substantial cause. It is not proof that one
particular coefficient formula explains every censored/noisy history.

### Why task VOI is close to EIG, and the μ-oracle fails

In this assumed one-parameter world, learning μ determines the entire stopping
curve, and the common action menu can hit targets throughout the tested range.
Both criteria therefore resolve much the same decision-relevant uncertainty;
the 5 cm reward then saturates. They are not choosing identical probes: last-probe
agreement in matched worlds is 40.43% at one probe and 25.78% at four, while mean
predicted final success at four is 0.9963 for EIG and 0.9957 for task-VOI. Source:
`results/r5/turn2_diagnostics.json`, `policy_overlap`. This is an explanation
consistent with this experiment, not a theorem equating the two objectives.

The μ-oracle knows friction but still commands force through the wrong actuator
model. Stopping distance contains a squared-force term: fractional force errors
are amplified, and near-threshold actions can also fail to move. Its smallest
absolute shifted placement error is 4.25 cm, already beyond the 2.5 cm half-band;
its maximum nominal centring error is only 0.514 cm. Source:
`results/r5/turn2_diagnostics.json`, `oracle_stats`. The full-world oracle's
100% success confirms that these are reachable targets, not impossible commands
(`results/r5/main_summary.json`). Knowing μ alone was deliberately never a
ceiling under gain mismatch.

### Secondary checks and limits

The old-friction secondary split has 128 worlds, with new tasks/noise and the
original seed-2026 friction values. At four probes both Bayesian active methods
score 100% matched, but shifted success is EIG 42.19%, task-VOI 46.88%, LSQ
60.16%, Tobit 64.06%. Shifted cov90 remains zero for all four diagnostic
posteriors. Source: `results/r5/legacy_summary.json`. This supports the qualitative
failure pattern; these are reused friction values, not a fresh structural test.

The AC check doubled inference resolution and planning particles/fantasies.
Compare it with the **same first 128 main worlds**, not the full 512-world
aggregate. At four probes, shifted success changes EIG 50.78→49.22%, task-VOI
53.13→48.44%, LSQ 54.69→52.34%, Tobit 52.34→50.78%; both active methods remain
100% matched on that prefix. Sources: `results/r5/ac_resolution_summary.json`
and `results/r5/turn2_diagnostics.json`, `ac_same_prefix_differences`.
The check changes planning noise as well as grid size, so it is not a pure
inference-grid convergence test. Small ranking margins should not be overstated.
Shifted cov90 still equals zero, and residual four-probe underresolution is
0.78–1.56%, depending on method (`results/r5/ac_resolution_summary.json`).

The original failure example and all raw evidence remain in
`media/r5_main_demo.html` and `checkpoints/r5_main.json`. No old result, log,
checkpoint or demo was replaced during analysis.

### Turn 2: infer the pusher gain (complete, 2026-10-24)

**What changed.** The fix adds one unknown constant actuator gain k to the
inference model: a joint posterior over 2,401 friction × 641 gain cells with a
uniform gain prior. Probe histories are **not** re-planned. `gain_bayes`
re-reads the saved probe history of a chosen "owner" policy and picks the final
push that maximizes success integrated over (μ, k). `gain_mle` fits (μ, k) by
censored maximum likelihood on the same history and the same grid, then aims the
deterministic point controller at the target centre. `same_probes_unit_bayes`
keeps the old unit-gain model on the identical history. All original methods,
the blind action and both oracles are rerun on every new world.

**Selection (validation only).** There were 32 new paired worlds per family
(seed 8202611), and each candidate was scored on 192 cases (2 families ×
budgets 1/2/4). Candidates were probe owner {EIG, LSQ} × gain half-width
{0.15, 0.25}, the same four for `gain_bayes` and `gain_mle`. Both selected **EIG
owner, half-width 0.15**. On validation, gain-Bayes scored 146/192 and gain-MLE
114/192. The run hit its 10-minute guard at 22/32 worlds and was resumed from the
checkpoint to 32/32 (910 s total). Original LSQ/Tobit protocols stayed frozen from
`tuning_main.json`. Source: `results/r5/turn2_tuning.json`;
`logs/r5/turn2_tune.log`, `turn2_tune_resume_01.log`.

**Test.** 128 new worlds per family (seed 8202612, disjoint from every earlier
R5 seed), 7,936 records, 1,104 s. Source: `results/r5/turn2_main_summary.json`
(config, `elapsed_seconds`).

Placement success (%), n = 128 per cell. Source:
`results/r5/turn2_main_summary.json`, `groups`.

| Method | Matched: 1 probe | 2 probes | 4 probes | Gain shift: 1 probe | 2 probes | 4 probes |
|---|---:|---:|---:|---:|---:|---:|
| **Gain-Bayes** (joint μ,k; EIG probes) | 60.94 | 89.06 | 100.00 | 59.38 | **88.28** | **97.66** |
| Gain-MLE (joint μ,k; EIG probes) | 31.25 | 81.25 | 86.72 | 34.38 | 72.66 | 92.19 |
| Same probes, unit-gain Bayes (= EIG) | 64.84 | 96.88 | 100.00 | 55.47 | 53.12 | 46.09 |
| Task-VOI → Bayes (unit gain) | 67.19 | 97.66 | 100.00 | 52.34 | 57.81 | 53.91 |
| Tuned LSQ (frozen) | 46.88 | 73.44 | 85.94 | 32.03 | 53.91 | 59.38 |
| Tuned Tobit (frozen) | 46.09 | 80.47 | 89.06 | 33.59 | 56.25 | 63.28 |

Because the selected owner is EIG, `same_probes_unit_bayes` makes exactly EIG's
own decisions. Its rows equal the EIG rows in every metric, so it is the EIG
baseline rather than an additional independent control. It still isolates the
model change, because gain-Bayes sees the same observations. Prior-only
action: 5.47% matched / 1.56% shifted (gain-Bayes prior-only 3.12% / 8.59%).
The μ-oracle is 100% / 0%, and the full-world oracle is 100% / 100%.

Paired differences (percentage points, approximate 95% interval, discordant
worlds a-only/b-only). Source: `turn2_main_summary.json`,
`gain_fix_paired_differences`.

| Comparison | Family | 1 probe | 2 probes | 4 probes |
|---|---|---|---|---|
| Gain-Bayes − same-probes unit Bayes | shift | +3.9 [−2.0, 9.8] | **+35.2 [26.3, 44.0]** (47/2) | **+51.6 [42.6, 60.5]** (67/1) |
| Gain-Bayes − same-probes unit Bayes | matched | −3.9 [−11.6, 3.8] | **−7.8 [−13.0, −2.7]** (1/11) | 0.0 (0/0) |
| Gain-Bayes − gain-MLE | shift | +25.0 [16.6, 33.4] | +15.6 [7.4, 23.9] | +5.5 [1.5, 9.4] (7/0) |
| Gain-Bayes − gain-MLE | matched | +29.7 [19.2, 40.2] | +7.8 [1.8, 13.8] | +13.3 [7.4, 19.2] (17/0) |
| Gain-Bayes − tuned Tobit | shift | +25.8 [14.9, 36.7] | +32.0 [22.3, 41.7] | +34.4 [25.8, 42.9] |

The four-probe gain-Bayes vs gain-MLE margin under shift rests on only 7
discordant worlds. Gain-Bayes's own matched→shift drop at four probes is −2.3
points [−5.0, +0.3], computed by the reviewer from `checkpoints/r5_turn2_main.json`.

Calibration and errors at **four probes**. Source: `turn2_main_summary.json`,
`groups`. As in turn 1, the gain-MLE and LSQ/Tobit interval columns belong to a
diagnostic Bayesian posterior on their probes. Gain-MLE shares gain-Bayes's
posterior, so those columns are identical.

| Method | Family | μ RMSE | μ cov90 | μ width90 | gain RMSE | gain cov90 | confident miss | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gain-Bayes | shift | 0.0359 | **92.97%** | 0.0857 | 0.0500 | 94.53% | **0.78%** | 0.017 |
| Gain-MLE | shift | 0.0502 | (92.97%) | (0.0857) | 0.0676 | (94.53%) | 0.78% | 0.054 |
| Unit-gain EIG | shift | 0.0498 | 0.00% | 0.0055 | — | — | 53.12% | 0.537 |
| Tuned LSQ | shift | 0.0591 | 0.00% | 0.0119 | — | — | 14.06% | 0.253 |
| Gain-Bayes | matched | 0.0251 | 92.97% | 0.0949 | 0.0427 | 94.53% | 0.00% | 0.002 |
| Unit-gain EIG | matched | 0.0020 | 88.28% | 0.0060 | — | — | 0.00% | 0.0001 |

**What worked.** Modelling the gain removes the confident-wrong failure. Under
gain shift, coverage returns from 0% to roughly nominal (93% μ, 95% gain), and
confident misses fall from 53% to under 1%. Placement rises from 46% to 98% on
the identical observations. The turn-1 failure also replicated on these new
worlds: unit-gain EIG/VOI 46.1% / 53.9% vs LSQ/Tobit 59.4% / 63.3% at four
shifted probes. The reviewer computed paired EIG − LSQ as −13.3 [−24.9, −1.7]
from the checkpoint.

**What it costs.** With k free, friction and gain trade off along a ridge.
Even in matched worlds, the friction estimate is about 13× worse (RMSE 0.025 vs
0.002) and its 90% interval is about 16× wider than the unit-gain model's.
This is honest uncertainty, not error: coverage stays near nominal. Placement
does not need the separate values, only the stopping curve near the forces used,
so four-probe placement is unaffected. At two probes the extra uncertainty costs
7.8 points in matched worlds (11 worlds lost, 1 gained). At one probe, neither
family shows a difference distinguishable from zero.

**Why gain-MLE trails.** It uses the same likelihood and grid, but a plug-in point
on a ridge is poorly determined with 1–2 probes (μ RMSE 0.098 vs 0.058 at one
shifted probe). The point controller then picks whichever of the 2,715 menu
actions lands nearest the target, without considering how uncertain that
prediction is. In a reviewer check on reused test worlds (first 40 per family;
exploratory, not a headline), one change helped: restricting gain-MLE's final
push to forces already probed. Matched four-probe success rose from 85% to 100%,
equal to gain-Bayes, but shifted two-probe success rose only from 72.5% to
77.5% (gain-Bayes 92.5%). Part of gain-Bayes's margin over gain-MLE is therefore
controller choice, not inference. The margin at two probes under shift
survived this check. Sources: `review_r5_mle_controller.py`,
`results/r5/review_mle_controller_{matched,gain_shift}.txt`.

**Grid resolution (AC check).** The first 32 test worlds were replayed with
the joint grid doubled on both axes (4,801 × 1,281). Everything else was
unchanged, including the probe histories, which are identical. Gain-Bayes changed
**zero** outcomes at any budget or family, and its coverage did not change.
Gain-MLE flipped 1–3 worlds in 4 of its 6 probe-budget cells (budgets 1/2/4; 0 at one probe). Marginal underresolution is 0.00 for μ and
gain at both resolutions, conditional (ridge) underresolved mass is ≤0.0002, and
the largest joint cell holds ≤0.0015 of the mass. Doubling the grid changed nothing
material. Sources: `results/r5/turn2_resolution_summary.json` (850 s), compared
by the reviewer against the same 32 worlds in `checkpoints/r5_turn2_main.json`.

**Caveats.**
- **This fixes a known mismatch, not an unknown one.** The turn-2 design was
  written after the turn-1 test results and diagnostics. The gain family
  (|k−1| ∈ [0.06, 0.12]) lies entirely inside the selected prior [0.85, 1.15].
  The narrower half-width was selected on validation worlds that *include*
  the gain-shift family, whereas turn 1 held that family out of tuning. The
  summary's inherited caveat "gain_shift is held out of tuning" applies only to
  the frozen LSQ/Tobit protocols. A mismatch the model does not contain (rolling
  resistance, velocity-dependent friction) is untested and could fail exactly as
  turn 1 did.
- Gain-MLE received the same four validation candidates as gain-Bayes but no
  aim-offset sweep. Offset 0 won for both turn-1 fitters, so this probably
  matters little. Its controller is a plain plug-in, as above.
- Probe planning still assumes unit gain. A gain-aware EIG/VOI planner was not built.
- n = 128 per family; shift-family differences under ~10 points between
  original methods have intervals that include zero at this n.
- The main and resolution runs used `--max-minutes 40`, not the design's 20
  (AC power). Both completed without hitting the guard.

Demo: `media/r5_turn2_main_demo.html`, an offline replay of recovered and
remaining-failure worlds from `checkpoints/r5_turn2_main.json` (summary copy:
`results/r5/turn2_main_report_summary.json`). Implementation checks: 30 CPU tests
passed with one MLX test deselected (`logs/r5/turn2_smoke_claude.log`). Tiny
pilots under `results/r5/turn2_astra_*` are pipeline checks only.

### Round 5 next steps

- Test a mismatch family the model does **not** contain, such as rolling
  resistance or velocity-dependent friction. This shows whether the joint model
  merely moves the confident-wrong failure elsewhere.
- Make the probe planner gain-aware. Joint EIG/VOI could choose probes at
  two forces to break the μ–k ridge, which may recover the matched two-probe loss.
- Give the classical baseline an uncertainty-aware action rule, for example
  pushing at a probed force, and tune it on validation, to separate inference
  from control cleanly.
