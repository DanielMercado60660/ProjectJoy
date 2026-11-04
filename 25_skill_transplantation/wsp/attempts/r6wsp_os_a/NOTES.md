# r6wsp_os_a — ONE-SHOT conditional compression

Four networks were constructed from their own seed's supplied weights only.
There were **zero environment interactions, rollouts, evaluation-tool calls,
gradient steps, iterative loss-minimization procedures, or weight searches**.
The construction and constants were fixed before inspecting its synthetic
approximation errors. Those errors did not select or modify a candidate.

## Rebuild and artifacts

From `25_skill_transplantation/`:

```sh
/Users/danielmercado/ProjectJoy/.venv/bin/python wsp/attempts/r6wsp_os_a/build.py
```

Optional: append `--seed 0` (or 1, 2, 3) to build just one seed. The complete
four-seed build took approximately 7.2 seconds on this machine. Numerical
libraries are capped at four threads. No packages or external data are needed.
The script locates `wsp/data` relative to its own path and writes only beside
itself. Its only weight inputs are `sN/theta0.npz`, `sN/fwd.npz`, and
`sN/bwd.npz`. Theta0 supplies `log_std`; the specialists supply the actor
construction. Their frozen normalizers are verified identical within each seed.

Each `merged_sN.npz` contains exactly the seven requested float32 arrays:
`a1.weight`, `a1.bias`, `a2.weight`, `a2.bias`, `a3.weight`, `a3.bias`,
`log_std`. The architecture remains 19 -> 128 -> 128 -> 6, with tanh after
each layer. No sidecar, alternate actor, or runtime routing code is needed.
`diagnostics_sN.json` records source SHA-256 hashes, selected neuron indices,
the gating bounds, and synthetic-only diagnostics. It is not a model input.

## What the hidden units compute

All indices below are zero-based.

* First layer 0..126: a shared nonlinear observation dictionary. Corresponding
  specialist first-layer observation rows are averaged. One row is removed by
  the smallest combined squared-path saliency
  `sum_task sum_j ||W3[:,j]||^2 * W2[j,i]^2`.
  Removed original indices for seeds 0..3 are 64, 23, 100, and 80.
* Each shared first-layer unit receives a distinct effective bias under each
  task. For task t, this is the specialist's active task column plus its bias,
  plus `(W1_t - W1_shared) @ mean(synthetic_obs_t)`. Thus its preactivation
  matches the specialist at that task's synthetic input center. The ordinary
  first-layer bias is zero; task columns carry these effective biases.
* First layer 127: `t = tanh(20*x[17] - 20*x[18])`. This is exactly +1 for
  forward and -1 for backward in the float32 calculations checked here.
* Second layer 0..63: 64 compressed forward-specialist features.
* Second layer 64..126: 63 compressed backward-specialist features.
* Second layer 127: `tanh(20*t)`, preserving the task sign for independent
  output intercepts.

The 64/63 split is fixed for every seed; the extra feature goes to forward,
the weaker specialist in the task's published summary. Within each branch,
the features are selected from that specialist's second-layer neurons by
output-weighted pivoted QR. Specifically, form a square root of their centered
activation covariance using an eigendecomposition, scale column j by
`sqrt(sum_a W3[a,j]^2 * mean(0.1 + 0.9*(1-action_a^2)^2))`, and take the
first 64 or 63 QR pivots. This is a fixed linear-algebra pruning rule, with no
candidate-performance search. Exact source-neuron order is in each JSON file.

## Exact task routing and readout

Let a fitted branch unit's active preactivation be `w @ h + b`, with h the
127 shared tanh features. Set `G = (sum(abs(w)) + abs(b) + 12)/2`.
The forward unit computes `tanh(w @ h + b + G*(1-t))`; the backward unit
computes `tanh(w @ h + b + G*(1+t))`.

The active branch has its fitted preactivation. Every inactive preactivation
is at least 12 in exact arithmetic, for every possible observation, since
each shared feature is bounded by one. After storing float32 parameters,
the measured algebraic lower bounds exceed 11.999998. Inactive activations
were exactly +1 on all synthetic checks, including 1,024 random corners of
the full [-10,10]^17 clipping box under both labels for every seed.

Let the independently fitted branch readouts be `C_f*h_f + d_f` and
`C_b*h_b + d_b`, with `s_f = C_f*ones` and `s_b = C_b*ones`. The final
readout contains both C matrices, task-column coefficient
`(d_f-d_b+s_f-s_b)/2`, and shared bias `(d_f+d_b-s_f-s_b)/2`.
This cancels the inactive branch's constant contribution and restores the
appropriate branch intercept. The final tanh produces the action. Float32
rounding is the only approximation in this routing algebra.

## Exactly what was fitted, and on which inputs

Each specialist receives 32,768 independently generated synthetic normalized
observations. RNG seed is `62026000 + 100*training_seed + task_index`, with
task_index 0 for forward and 1 for backward. The three input groups are:

* 70% directional samples: independent standard-normal coordinates scaled
  per sample by a log-uniform factor in [0.65,1.55]; normalized height is
  further scaled by 0.60 and vertical/pitch angular velocities by 0.80.
  Physical pitch is independently N(0,0.25^2), then normalized. Physical
  x-velocity is N(6,2.5^2) for forward or N(-8.5,2.5^2) for backward;
  20% of these velocities instead use a uniform draw between zero and the
  corresponding directional mean. These are chosen priors, not observations
  of a running policy. The published return scale motivated the velocity
  constants, without any new measurements.
* 20% broad samples: independent N(0,1.5^2) normalized coordinates, replacing
  x-velocity with a normalized physical N(3,5^2) or N(-4.25,5^2) draw.
* Remaining approximately 10%: raw first-eight coordinates independently
  uniform in [-0.1,0.1], and raw nine velocity coordinates independently
  N(0,0.1^2), all transformed by the supplied normalizer. This is a synthetic
  near-zero family, not a reset sampled or reproduced from an environment.

All normalized coordinates are clipped to [-10,10]. No synthetic input
depends on a previous action; there are no trajectories or simulated dynamics.

Two successive feedforward stages use direct, closed-form linear solves:

1. For each selected second-layer neuron, fit its **teacher preactivation**
   as an affine function of the 127 shared first-layer activations. Its fixed
   sample weights are `0.1 + 0.9*(1-teacher_h2_j^2)^2`, emphasizing the tanh
   transition while keeping every sample in the solve.
2. For each of six actions, fit its **teacher output preactivation** as an
   affine function of the newly computed compressed second-layer activations.
   Its fixed sample weights are `0.1 + 0.9*(1-teacher_action_a^2)^2`.

Each fit centers predictors and targets with its fixed weights, solves the
normal equations in float64, and leaves the intercept unpenalized. The ridge
coefficient is `1e-6 * mean(diag(weighted_predictor_covariance))`, with a
1e-12 floor on the latter quantity. Weights come directly from teacher
forward passes and are never recomputed from fitted residuals. Each target
receives one solve; there is no alternating optimization, gradient update,
iteratively reweighted regression, or tuning loop. Final parameters are
converted to float32.

## Synthetic verification only

Checks use 8,192 new synthetic inputs per task, from the same specified
mixture, with RNG seed `62029000 + 100*training_seed + task_index`. These
samples were excluded from every fit and selection. The reported RMSE is
across all six bounded action coordinates against that task's specialist.

| Seed | Forward action RMSE | Backward action RMSE | Forward action MAE | Backward action MAE |
|---|---:|---:|---:|---:|
| 0 | 0.073947 | 0.075296 | 0.038857 | 0.038162 |
| 1 | 0.074368 | 0.066353 | 0.040338 | 0.033609 |
| 2 | 0.066032 | 0.064842 | 0.032112 | 0.031105 |
| 3 | 0.065099 | 0.075510 | 0.036106 | 0.038759 |

For context within these synthetic checks, directional-family RMSE ranges
from 0.05990 to 0.07446, broad-family RMSE from 0.08049 to 0.10434, and
near-raw-zero-family RMSE from 0.01191 to 0.02570. Overall 95th-percentile
absolute action error ranges from 0.14032 to 0.16880. The JSON files retain
per-action and per-family details rather than hiding the broader-input errors.

All four artifacts pass exact key/shape/dtype checks, finite-parameter and
bounded-output checks, save/reload equality, and task-routing assertions.
Seed 0 was rebuilt in an independent process: every parameter array was
bit-identical. The second run used the same fixed construction and constants.

## Approximations and expected costs

Sharing the first-layer observation matrix loses specialist-specific nonlinear
features; task-dependent centering and the first affine fits only approximate
their effects. Removing about half of each second layer loses nonlinear
capacity; QR and output fitting compensate on the synthetic distribution.
The broad-family errors demonstrate that compensation is imperfect. Ridge
regularization introduces a small additional approximation. Routing itself
does not leak the other branch's varying features.

The expected cost is action error near decision boundaries and on states
poorly covered by the chosen independent-input distribution. In closed-loop
locomotion, these errors could change gait stability or compound over time;
synthetic RMSE cannot predict how much return will be retained. Actual forward
return, backward return, velocity, stability, retained score, and held-out
reset-seed performance are **not measured**. No locomotion success is claimed.

FINAL: merged_s0..3.npz
