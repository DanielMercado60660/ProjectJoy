# r6wsp_os_b — one-shot algebraic packing

The deliverables are four ordinary 19→128→128→6 tanh actors. Each combines
127 shared first-layer features with two separate, compressed second-layer
banks selected by the task label. All network arrays are float32 and have
exactly the seven required keys. The task's existing normalizer is assumed at
inference; no new normalizer is embedded in the files. `log_std` is copied from
that seed's forward specialist and is unused by deterministic inference.

**Environment interactions: zero. Evaluation-budget calls: zero. Gradient
steps: zero.** No simulator, environment, rollout, dataset, or external model
was used. There was no evolutionary search, random weight search, iterative
loss minimization, or tuning against environment returns. Construction reads
only the forward and backward NPZs of the seed being built. It does not need
theta0. Every fit below is one closed-form linear solve, with no repeated
optimization or refitting of that stage.

## Reproduce

From the project directory:

```sh
/Users/danielmercado/ProjectJoy/.venv/bin/python wsp/attempts/r6wsp_os_b/build.py
```

The script also works from another working directory because source and output
paths are resolved relative to `build.py`. It writes the four NPZs and
`diagnostics.json` beside itself. It uses NumPy and SciPy's pivoted QR, with
thread-count environment variables capped at four before numerical imports.
The measured first complete four-seed build, including synthetic checks, took
3.56 seconds. No run approached one minute.

`diagnostics.json` records source hashes, selected specialist neuron indices,
gate magnitudes, synthetic errors, and output hashes. `verification.json`
records a second, byte-identical rebuild and independent routing checks on
loaded final files. Neither report is an input to construction.

## Synthetic inputs, fixed in advance

For each seed and task, the fit uses 49,152 independently fabricated normalized
17-dimensional observations. A NumPy generator is initialized with
`SeedSequence([62031, seed, task, stream])`. Stream 0 is construction; stream 1
and stream 2 are separate checks. No states are propagated through time.

The input generator is explicitly a coverage heuristic, not an estimate of
the specialists' actual visitation distributions:

- Start with independent standard Gaussian coordinates multiplied by a shared
  per-row scale chosen from `(0.45, 0.85, 1.25, 1.9)` with probabilities
  `(0.12, 0.38, 0.35, 0.15)`.
- For 80% of rows, replace root pitch with raw `Normal(0, 0.48)` radians,
  transformed with the supplied frozen mean and standard deviation. The
  remaining pitch coordinates retain the normalized Gaussian values.
- Replace x-velocity with a task-dependent synthetic raw-speed mixture:
  18% `Normal(0, 1.2)`; 22% uniform between zero and 8 for forward or 11 for
  backward; 60% `Normal(5.5, 2.2)` for forward or `Normal(8, 2.2)` for backward.
  Negate the sampled speed for backward, then apply the supplied velocity
  normalization. These speeds are chosen constants, not measured outcomes.
- Clip all normalized coordinates to `[-10, 10]`.

The first 8,192 construction rows per task are used for first-layer fitting
and feature ranking. All 49,152 are used for second-layer and output fitting.
The independent matched check uses 16,384 rows per task from stream 1. The
broader stress check uses 16,384 stream-2 rows, multiplies normalized inputs by
1.5 before clipping, and replaces normalized x-velocity with uniform
`[-7.5, 7.5]`, regardless of task. No check results alter the construction.

## What each layer computes

Indices below are zero-based.

### First layer: 127 shared features and a task bit

The specialists descend from the same initialization, so neuron index `j` is
paired with the same index in the other specialist. For each of the 128 pairs,
compute the teacher's first-layer preactivation `z_j` on the appropriate
synthetic task inputs. Stack both tasks into a design matrix `A` with 17
observation columns and two task-indicator columns. Fit one common 19-vector:

```
d_j = (0.2 + 0.8 * (1 - tanh(z_j)^2))^2
beta_j = solve(A.T @ diag(d_j) @ A + 1e-8 * I,
               A.T @ diag(d_j) @ z_j)
```

The weights are fixed functions of the teacher, not updated from fit errors.
The slope weighting emphasizes regions where a preactivation mismatch changes
the tanh response, while retaining weight in saturated regions. The final two
coefficients provide independent task-conditioned offsets. This uses the
architecture's label columns to compensate for different operating regions.

Drop one shared feature by column-pivoted QR on its centered synthetic
activation matrix. Each feature is weighted by the square root of the sum,
over both specialists and second-layer units, of
`(a2.weight[i,j] * norm(a3.weight[:,i]))^2`. Keep the first 127 QR pivots and sort
their original indices. This is a deterministic pruning/ranking heuristic.

Rows 0–126 of the merged first layer hold these fitted coefficients; their
explicit biases are zero. Row 127 has no observation weights and computes

```
s = tanh(12 * task_fwd - 12 * task_bwd).
```

Thus `s` is exactly +1 or -1 in the tested float32 NumPy implementation.

### Second layer: forward bank 0–63, backward bank 64–126, task bit 127

For each task independently, evaluate the 127 shared first-layer features and
append a constant column to obtain `B`. Regress all 128 specialist second-layer
preactivations `Z` onto `B`:

```
C = solve(B.T @ B + R, B.T @ Z + R @ P).
```

`P` is the corresponding specialist's original second-layer coefficients on
the 127 retained first-layer indices, plus its original bias. `R` is diagonal:
`49152 * 2e-5` for feature coefficients and `1e-8` for the intercept. This small,
fixed Tikhonov anchor stabilizes correlated features. No task-to-task or
seed-to-seed data are mixed in this fit.

Compute `tanh(B @ C)`, center its first 8,192 rows, weight each column by the
norm of that specialist's outgoing output-weight column, and use pivoted QR
to retain 64 forward features and 63 backward features. The one-unit asymmetry
reserves a second-layer task unit while retaining the prescribed total width.
The chosen specialist indices are listed in `diagnostics.json`.

For a retained row with coefficients `(w, b)`, define

```
M = sum(abs(w)) + abs(b) + 12
forward preactivation  = w @ shared_h1 + b + M * (s - 1) / 2
backward preactivation = w @ shared_h1 + b + M * (-s - 1) / 2.
```

The active bank computes its fitted feature. The inactive bank's
preactivation is at most -12 for **any** shared hidden vector in `[-1, 1]^127`,
so its tanh output is float32 -1, independent of observations. The large
offset is encoded in the ordinary weight from first-layer unit 127 and the
ordinary second-layer bias. Active-path cancellation has ordinary float32
roundoff; the output fit uses these actually compiled float32 activations.

Second-layer unit 127 computes `tanh(12*s)`, giving an output-layer task bit.
No runtime router, extra layer, extra parameters, or external controller is
required.

### Output layer: two fitted torque maps with exact constant cancellation

For each task and each of six outputs, fit the specialist's **pre-tanh output
logit** onto the compiled active bank plus an intercept. The per-sample weight
is `max(0.1, 1 - tanh(teacher_logit)^2)^2`. This is one closed-form weighted
ridge solve per output. Feature regularization is `49152 * 1e-6`, intercept
regularization is `1e-8`, and the anchor is the specialist's original outgoing
coefficients at the retained second-layer indices plus its output bias.

Call the resulting maps `(Cf, df)` and `(Cb, db)`, where the coefficient
matrices have shape 6×bank-width. Write the two coefficient matrices into
the corresponding output-weight columns. Define

```
offset_f = df + sum(Cb, axis=1)
offset_b = db + sum(Cf, axis=1)
output_bias = (offset_f + offset_b) / 2
task_bit_output_weight = (offset_f - offset_b) / 2.
```

Under the forward label this cancels the inactive backward bank's constant
contribution and leaves `Cf @ active_h2 + df`; under the backward label it
leaves `Cb @ active_h2 + db`. The actor's final tanh produces the six actions.

## Approximation checks and limitations

These errors compare final merged actions with specialist actions on
independently generated synthetic inputs. They are pooled across all six
action coordinates. They are **not** returns, success rates, or rollout
measurements.

| Seed | Task | Matched action RMSE | Matched absolute-error p95 | Broader action RMSE |
|---|---|---:|---:|---:|
| 0 | fwd | 0.077180 | 0.172938 | 0.104284 |
| 0 | bwd | 0.078462 | 0.176635 | 0.108157 |
| 1 | fwd | 0.077935 | 0.174150 | 0.108378 |
| 1 | bwd | 0.067873 | 0.154368 | 0.093854 |
| 2 | fwd | 0.072284 | 0.160208 | 0.104081 |
| 2 | bwd | 0.069556 | 0.158253 | 0.097775 |
| 3 | fwd | 0.074852 | 0.166245 | 0.099125 |
| 3 | bwd | 0.075414 | 0.170611 | 0.099157 |

Rare errors are appreciably larger: maximum absolute action error across the
matched checks was 1.175562, and across broader checks it was 1.296227. The
largest final output coefficient magnitude across seeds was about 1.264.

The inactive-bank deviation from -1 was exactly zero in every matched and
broader check. A separate check loaded the saved NPZs and tested both labels
on 2,048 uniform full-clip-box inputs, 2,048 random clip-box vertices, and the
zero vector. Task bits and inactive saturation were exact, and logits were
finite. Each saved key, dtype, shape, and value was checked after serialization.
A complete second build reproduced all four NPZ SHA-256 hashes byte for byte.

The approximation costs are shared first-layer directions, removal of one
first-layer feature, and reduction to roughly half the original nonlinear
second-layer features per task. Synthetic regression compensates for these
losses only over its chosen distribution. Independent coordinate samples omit
physical correlations and gait phase; errors on reachable trajectories can
therefore differ substantially. Feedback can amplify small action errors.
I expect task routing itself to be reliable, but do not infer retained return
or actual forward/backward locomotion from these checks. Both are **not
measured**. No numerical retained-score prediction is made.

Before the final construction, one seed-0 feasibility probe used 16,384
normalized standard-Gaussian inputs with coordinate 8 replaced by uniform
`[-6, 6]`, RNG seed 4701. It averaged the two observation-weight matrices,
kept specialist task offsets, and did one unweighted second-preactivation
least-squares solve with `0.01*I` regularization and all 128 units retained.
Its action RMSE was 0.026702 forward and 0.030449 backward on those same probe
inputs. This established that shared first-layer features were a plausible
construction ingredient; it was not a routed deliverable, an independent
generalization check, or an environment evaluation. No probe data or files
are needed to rebuild the final actors.

FINAL: merged_s0..3.npz
