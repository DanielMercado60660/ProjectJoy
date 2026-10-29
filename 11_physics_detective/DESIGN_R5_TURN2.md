# Turn 2: one fix — infer the pusher gain

**Evidence motivating the fix.** The completed test demonstrates failure of the
unit-gain physics assumption. It does not demonstrate that Bayesian inference
cannot learn an effective friction: same-history replays improve substantially
when probe acquisition and terminal control are crossed. The original test has
now informed this redesign and will not be reused as its evaluation set.

**Hypothesis.** Adding the nuisance gain to the inference model improves placement
and friction coverage under gain error without sacrificing most matched-world
performance. Probe histories remain unchanged, so a same-history unit-gain
Bayesian control isolates the model change. This is a model fix, not a new probe
planner, residual-noise inflation rule, or neural learner.

**Model and controls.** Use a joint grid over friction (2,401 midpoint cells on
[0.05, 0.8]) and actuator gain (641 midpoint cells). Independent uniform gain
priors have half-width 0.15 or 0.25 around one. Reuse the existing censored camera
likelihood with effective force kF. Integrate final-placement probability over
both parameters using one friction CDF per gain cell; do not materialize an
actions × friction × gain array. The final menu, target band, 0/1/2/4 probes,
attempt costs, camera noise and world distributions remain the same.

`gain_bayes` maximizes integrated success. `gain_mle` fits both parameters by
censored maximum likelihood and aims deterministically at the target centre.
Both get the same four validation candidates: EIG or original LSQ probe owner,
crossed with the two prior/support widths. A probe owner continues to select
probes with its original unit-gain estimator; the joint posterior never changes
its observations. `same_probes_unit_bayes` uses the selected gain-Bayes owner's
identical history but keeps the old model. Rerun all original methods, blind
action and both oracles on every fresh test world. Original LSQ/Tobit protocols
stay frozen from `tuning_main.json`; the new nuisance-MLE is the equally tuned
classical comparison with the added physical model.

**Selection and fresh test.** Validation: 32 new paired worlds per family, seed
8202611. Select independently for gain-Bayes and gain-MLE by mean placement
success across both families and budgets 1/2/4; break ties by placement MAE, then
fixed enumeration. Each candidate sees 192 scored validation cases. Freeze before
128 new test worlds per family, seed 8202612. Reject all three original R5 seeds
and validation/test seed collisions. Tiny smoke/pilot seeds are separate.

**Measures and limits.** Retain the original success, cost, placement error,
friction RMSE, posterior coverage/width, confidence and paired-difference metrics;
add gain RMSE/coverage and both marginal and conditional grid-resolution checks.
All MLE coverage is explicitly from a diagnostic Bayesian posterior. Conditional
checks detect thin ridges that wide marginals hide. The gain-shift family is now
represented in the physics and validation; this is **not** generalization to an
unseen mismatch type. Its mixture of gains also differs from the uniform prior,
so nominal coverage is measured rather than assumed. Save recovered cases and
remaining failures; no abstention or extra probe is permitted.

**Budget.** NumPy/SciPy only, ≤4 threads. Validation guard 10 min, main guard 20 min,
smoke/report allowance 3 min: ≤33 min battery plan. Main expected 7–17 min, subject
to load and checkpoint I/O; full runs go through `gpu_run`. MVP: 32 validation
worlds and 64 fresh test worlds, clearly labelled. **If AC:** freeze the selected
configuration and replay the first 32 test worlds with 4,801 × 1,281 cells, changing
only the gain-model grid. This is a numerical sensitivity check, not another
selection round or independent replication. Full follow-up is **pending runner**.
