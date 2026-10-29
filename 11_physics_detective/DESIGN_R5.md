# Round 5 design: probe to place

**Hypotheses.** H1: probes improve placement over acting from the prior. H2:
myopic task value of information (VOI) improves success over information gain
(EIG), especially at one probe; no advantage is a valid outcome. H3: unmodelled
actuator error can shrink intervals while reducing coverage; accurate placement
need not imply accurate parameter recovery.

**World and methods.** Reuse the existing analytic 1 kg Coulomb block, μ_k uniform
on [0.05, 0.8], μ_s=1.2μ_k, 20 noisy/censored camera frames (σ=8 cm, view=1 m).
Disclosed target centres are uniform on [0.25, 0.85] m. Budgets are exactly
0/1/2/4 probes, each costing a reset to rest at the origin, plus one scored push;
no abstention. All methods share 150 probe commands and 15 forces × 181 final
durations (0.1–1 s). Success means the true eventual stop lies within ±2.5 cm
of target; no scoring noise or clipping.

Compare EIG→Bayesian action, task-VOI→the same Bayesian action, visible-frame
least squares→deterministic controller, and censoring-aware Tobit maximum
likelihood→deterministic controller. Include a prior-only Bayesian action,
μ-only oracle, and full-world oracle, all with one attempt and zero probes.
The μ-only oracle assumes unit gain and is **not** a ceiling under mismatch.
The paired held-out family has fixed gain 1±U[0.06, 0.12], with random sign.
Exclude it from tuning. Non-oracles assume unit gain and never receive truth.

**Approximation and tuning.** Inference uses 6,001 midpoint grid cells and a
mid-cell CDF. EIG and task VOI use identical 64 quantile particles, 16 synthetic
observations and common noise over all probes; both scores are computed for
both planners. VOI is expected best final success after one observation minus
current best success, myopically even with four probes. No neural component.
Each fitter gets **45** validation candidates:
15 probe protocols × three terminal aim offsets (−0.5/0/+0.5 target half-widths).
Protocols: two fixed pushes, a force ladder, and adaptive pushes at forces
10/12/15 N aiming for 0.35/0.65/0.90 m or the actual target. Select separately
for LSQ/Tobit on 128 matched validation worlds (seed 6202611), maximizing mean
success over budgets 1/2/4; tie-break by placement MAE, then enumeration.
Freeze before testing; active hyperparameters are predeclared.

**Measurements and splits.** Headline: 512 fresh worlds, seed 6202612, paired
across methods and families with identical target/μ/camera-noise streams.
Report success with Wilson 95% intervals, attempts/resets, success per attempt,
placement MAE/p90, μ RMSE/bias, 50/90% posterior coverage and 90% width,
predicted-success Brier score, confident failures, resolution diagnostics and
probe-selection time. Pair method differences and family degradation by world;
never pool correlated budgets. LSQ/Tobit coverage is explicitly the diagnostic
Bayesian posterior on their probes, **not** a confidence interval for the fit.
Preserve every observation and the first task-VOI failure in an offline demo.
Secondary: 128 original seed-2026 friction values with new task/noise seed
6202613; this is an old-friction comparison, not a repeated old task metric.

**Budget / minimum viable version.** CPU only, ≤4 threads, modest memory; all
full runs use `gpu_run`. Battery plan: validation ≤3 min, main ≤20 min, secondary
≤4 min, smoke/report ≤3 min (≤30 total; no new battery job below 50%). Expected
main runtime is 5–12 min, extrapolated from the tiny pilot, not a guarantee.
Wall guards save resumable checkpoints; actual kill/reload equivalence is tested.
MVP is frozen tuning plus 128 fresh worlds/family at all budgets, both oracles,
raw records, summary and failure demo; label the smaller n. **If AC:** run an
additional 128-world quadrature check (12,001 cells, 128 particles, 32 fantasies)
on the first 128 headline worlds. This paired sensitivity check is not an
independent replication. Do not redesign from headline outcomes.
