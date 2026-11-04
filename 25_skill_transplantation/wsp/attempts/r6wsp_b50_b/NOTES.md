# r6wsp_b50_b — BUDGET-50

Started 2026-11-04 15:23 CST. Scope: at most 50 calls total to the supplied
`wsp/tools/eval_budget.py`; no other environment interaction. Only the supplied
TASK, weights, evaluator, and permitted actor conventions in `common.py` were read,
along with this attempt's own artifacts. All writes are inside this attempt folder.
No gradient training, random weight search, or iterative loss minimization.

Completed all 50 evaluation calls. Every evaluation and retention
decision is recorded below. Candidate archives are evidence, not retained
final choices. Retention is based strictly on improved mean retained score on the
fixed training resets; synthetic approximation error alone never retains an edit.

## Final measured result

Training reset seeds are exactly the evaluator's fixed bank 60000, 60001, 60002.
There are three 1000-step episodes per task per call: 50 calls, 300,000 total
environment steps, all through the supplied evaluator. No extra evaluations were
performed. **Held-out performance is not measured.** This table is selected on
training resets and is not an unbiased generalization estimate.

| Seed | Selected call | Forward return | Backward return | Forward retained | Backward retained | Mean retained |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 46 | 4818.8 | 7396.5 | 95.89% | 93.43% | 94.66% |
| 1 | 14 | 5092.8 | 7452.8 | 127.62% | 93.61% | 110.61% |
| 2 | 42 | 5514.1 | 6925.4 | 98.95% | 81.11% | 90.03% |
| 3 | 45 | 5340.6 | 7064.8 | 99.92% | 89.03% | 94.47% |

Across seeds the selected mean retained is **97.4425%** (averaging the evaluator's
rounded scores), versus **55.23%** for the unmodified forward-specialist baselines
from calls 1–4 on the same task labels and reset bank. Mean raw returns across
seeds are 5191.575 forward and 7209.875 backward. Mean forward velocities are
5.226, 5.450, 5.842, 5.658 m/s; backward velocities are -7.794, -7.867, -7.332,
-7.426 m/s. Every final episode has positive task reward and the correct velocity
sign. Seed 1 exceeds its specialist on this training bank; this does not establish
that the merged network exceeds the specialist on held-out resets.

## Rebuild

From the project folder, run:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/danielmercado/ProjectJoy/.venv/bin/python wsp/attempts/r6wsp_b50_b/build.py
```

`build.py` has the four final configurations embedded as literals. It reads only
that seed's supplied fwd/bwd/necessary theta0 normalizer arrays in `wsp/data/`;
it does not read the ledger, best.json, candidate archives, earlier attempts, or
any trained weights elsewhere. It performs no environment calls. No installation,
network, Git, autograd, gradient descent, random weight search, or iterative loss
minimization is used. `scipy.linalg.qr` and fixed ridge normal-equation solves are
closed-form constructions. Extra modes in the builder reproduce rejected designs.
The final files contain only the seven requested float32 actor/log_std arrays;
the frozen supplied normalizer remains external as specified by TASK.md.

## Final architecture and task routing

All final networks retain exactly 19 -> 128 tanh -> 128 tanh -> 6 tanh.
There is no external policy switch, simulator dependence, recurrent state, or
additional runtime model. The label acts through a1 input columns 17 and 18.
Neuron indices in this description are zero-based.

**Layer 1 neurons 0–126** share observation features across tasks. Their 17 input
weights are corresponding specialist rows blended 50/50 for seeds 0 and 1,
100% forward for seed 2, and 75% forward/25% backward for seed 3. Their stored
bias is zero; the forward task column equals the forward specialist's bias plus
its forward task column, and the backward column analogously uses the backward
specialist. Thus each task preserves its own first-layer offset while sharing
observation directions. Original neuron 127 is sacrificed in every final model.

**Layer 1 neuron 127** has zero observation weights and task weights (+10, -10),
so g = tanh(10) for fwd and -tanh(10) for bwd; float32 represents these as +1/-1.
**Layer 2 neuron 127** computes tanh(10g), providing the output with the same
task sign. All other second-layer units belong exclusively to one task:

| Seed | Forward second-layer units | Backward second-layer units | First-layer blend | Synthetic distributions fwd/bwd | Sensitivity weighting fwd/bwd |
|---|---|---|---|---|---|
| 0 | 0–47 (48) | 48–126 (79) | 50/50 | normal / normal | none / both layers |
| 1 | 0–63 (64) | 64–126 (63) | 50/50 | motion / motion | none / none |
| 2 | 0–63 (64) | 64–126 (63) | forward | normal / motion | none / both layers |
| 3 | 0–63 (64) | 64–126 (63) | 75/25 | speed / normal | none / both layers |

For each task, output-weighted column-pivoted QR selects that many original
specialist second-layer neurons. Specifically, center the first 8192 synthetic
teacher h2 rows, multiply each column by the Euclidean norm of its six original
output weights, and select columns in QR pivot order. The exact original-neuron
index lists are saved in the corresponding `merged_s*.json` diagnostics.

For every selected second-layer neuron, a single affine least-squares fit maps
the 127 shared h1 values to that teacher neuron's pre-tanh activation. This is
not a fit to environment states. For an active-task preactivation w·h+b, define
M = sum(abs(w)) + abs(b) + 12. A forward unit stores gate coefficient -M/2 and
bias b+M/2; a backward unit stores +M/2 and bias b+M/2. Under its own label it
computes w·h+b. Under the other label it is at least +12 for any h in [-1,1]^127,
and therefore saturates to +1. The inactive branch is thus a known constant.

One affine fit then maps the active task's selected tanh features to the six
teacher output logits. Let C_f, d_f and C_b, d_b be these two fits; let S_f and
S_b be their row-wise sums. Store C_f and C_b in the corresponding a3 columns,
a3.bias = (d_f+d_b-S_f-S_b)/2, and gate output weights
(d_f-d_b+S_f-S_b)/2. The constant inactive branch cancels, giving C_t h2_t+d_t
under label t before the final tanh. log_std is copied from that seed's forward
specialist and is unused by deterministic scoring.

## Exact fitting inputs and objectives

Each task uses 24,000 generated normalized observation vectors. NumPy's
`default_rng(7000 + 100*seed + task_index)` draws independent standard normal
values for all 17 coordinates, with task_index 0 for fwd and 1 for bwd.
All normalized coordinates are finally clipped to [-10,10]. No observed
environment states, trajectories, hidden reset states, or other learned models
are used. The original specialist's own task column is used when generating its
synthetic targets.

- **normal:** the standard normal vectors above.
- **motion:** replace root z with a raw Normal(-0.03, 0.06) draw, pitch with raw
  Normal(0, 0.30), and x velocity with raw Uniform(0,11) for fwd or its negative
  for bwd. Convert these three coordinates using the seed's supplied frozen
  norm_mean and sqrt(norm_var+1e-8). Other coordinates retain standard normals.
- **speed:** draw the same auxiliary raw arrays in the same order as motion,
  but replace only x velocity; retain the broad Gaussian torso coordinates.
- **mixed** (rejected): apply the motion replacements only to the first half.

Unweighted regression solves (A^T A + ridge) c = A^T y with an appended intercept,
ridge coefficient 1e-7 times the number of samples on slope coefficients and no
intercept penalty. For final backward sensitivity-weighted fits on seeds 0, 2,
and 3, each target column y independently uses fixed weights
q = max(1 - tanh(y)^2, 0.05)^2, then solves (A^T diag(q) A + ridge) c =
A^T diag(q) y. Ridge is 1e-7*sum(q), again excluding the intercept. These q values
come only from the teacher targets and are never updated iteratively. The hidden
fit targets teacher z2; the output fit targets teacher logits. There is one solve
per fixed objective, with no alternating refits or optimization loop.

The 6000-input synthetic diagnostic uses a separate RNG stream (+10000 to the
seed). It reports action RMSE versus the original specialist; it is not a policy
evaluation, and neither its samples nor its errors decide retention. No final
construction uses the experimental input-gated assignment, union dictionary,
near-linear residual units, saliency gate replacement, or fully shared layer 2.

## Approximations, failures, and limits

Sharing first-layer directions, sacrificing one first-layer unit, pruning the
second layer, and least-squares preactivation/output approximations all change
the controller. There is no guarantee that synthetic Gaussian states match the
on-policy state distribution. The most visible remaining deficit is seed 2
backward running (81.11% retained on training resets). Some rejected candidates
had smaller synthetic action error but much worse task reward; error on these
synthetic states does not certify stable control.

Fully shared second layers failed badly (calls 9–12). Upright motion assumptions
often improved backward speed but damaged forward running (calls 13–16). The
union first-layer dictionary failed on seed 3 (call 27). Allocating more backward
units helped seed 0 but harmed other seeds. One-shot derivative weighting helped
some backward fits but was not uniformly beneficial. The final candidate for each
seed is its strict best observed mean retained score, not the last candidate.

Mathematically independent task blocks do not imply identical finite-precision
trajectories after recombination: output constants/gate coefficients change, and
small float32 differences can grow over 1000 steps. Calls 8/16/22 and 38/29/40
show different returns after algebraically combining blocks. Numerical sensitivity
is an inference from that construction and the observed scores, not a separately
isolated simulator experiment. The combined networks were evaluated as complete
candidates, and the final selected weights are preserved exactly. Three resets
and selection over 50 candidates can overfit; held-out evaluation is deferred to
the scorer and remains entirely unseen.

## Weight analysis

Across seeds, corresponding specialist first-layer observation weights remain
similar (whole a1 weight cosine .915–.929), while a2 cosines are .887–.894 and
a3 cosines .713–.747. This motivates shared first-layer features, followed by
task-routed separate second-layer features, and a fully shared alternative.

## Evaluation log

### Call 1: seed 0, baseline_fwd

Unchanged forward specialist establishes a strong single-network baseline; its existing task columns remain intact.

Configuration: `{"mode": "fwd"}`.

Returns fwd/bwd: 5025.5, -307.4; retained: 1.0, -0.0388; mean retained **0.4806**. **KEEP** (initial baseline).

### Call 2: seed 1, baseline_fwd

Unchanged forward specialist establishes a strong single-network baseline; its existing task columns remain intact.

Configuration: `{"mode": "fwd"}`.

Returns fwd/bwd: 3990.7, -370.6; retained: 1.0, -0.0465; mean retained **0.4768**. **KEEP** (initial baseline).

### Call 3: seed 2, baseline_fwd

Unchanged forward specialist establishes a strong single-network baseline; its existing task columns remain intact.

Configuration: `{"mode": "fwd"}`.

Returns fwd/bwd: 5572.8, 1714.7; retained: 1.0, 0.2008; mean retained **0.6004**. **KEEP** (initial baseline).

### Call 4: seed 3, baseline_fwd

Unchanged forward specialist establishes a strong single-network baseline; its existing task columns remain intact.

Configuration: `{"mode": "fwd"}`.

Returns fwd/bwd: 5345.0, 2403.7; retained: 1.0, 0.3029; mean retained **0.6514**. **KEEP** (initial baseline).

### Call 5: seed 0, shared_gate

Share the highly aligned first-layer observation weights, preserve task-specific biases, fit second-layer preactivations once, prune by output-weighted activation QR, and route 64 forward/63 backward units with exact saturation gates.

Configuration: `{"mode": "gated"}`.

Returns fwd/bwd: 4668.8, 7240.6; retained: 0.929, 0.9146; mean retained **0.9218**. **KEEP**; prior best 0.4806 from call 1.

### Call 6: seed 1, shared_gate

Share the highly aligned first-layer observation weights, preserve task-specific biases, fit second-layer preactivations once, prune by output-weighted activation QR, and route 64 forward/63 backward units with exact saturation gates.

Configuration: `{"mode": "gated"}`.

Returns fwd/bwd: 4940.5, 6925.7; retained: 1.238, 0.8699; mean retained **1.0539**. **KEEP**; prior best 0.4768 from call 2.

### Call 7: seed 2, shared_gate

Share the highly aligned first-layer observation weights, preserve task-specific biases, fit second-layer preactivations once, prune by output-weighted activation QR, and route 64 forward/63 backward units with exact saturation gates.

Configuration: `{"mode": "gated"}`.

Returns fwd/bwd: 4879.8, 6822.9; retained: 0.8756, 0.7991; mean retained **0.8374**. **KEEP**; prior best 0.6004 from call 3.

### Call 8: seed 3, shared_gate

Share the highly aligned first-layer observation weights, preserve task-specific biases, fit second-layer preactivations once, prune by output-weighted activation QR, and route 64 forward/63 backward units with exact saturation gates.

Configuration: `{"mode": "gated"}`.

Returns fwd/bwd: 4043.6, 5206.0; retained: 0.7565, 0.6561; mean retained **0.7063**. **KEEP**; prior best 0.6514 from call 4.

### Call 9: seed 0, full_shared

Test whether all 128 second-layer units can be shared: one stacked closed-form solve matches each task teacher preactivations, then one stacked output solve matches teacher logits. This avoids the gated model capacity split.

Configuration: `{"mode": "shared"}`.

Returns fwd/bwd: 1015.8, 2540.8; retained: 0.2021, 0.3209; mean retained **0.2615**. **REJECT**; prior best 0.9218 from call 5.

### Call 10: seed 1, full_shared

Test whether all 128 second-layer units can be shared: one stacked closed-form solve matches each task teacher preactivations, then one stacked output solve matches teacher logits. This avoids the gated model capacity split.

Configuration: `{"mode": "shared"}`.

Returns fwd/bwd: 604.6, 5516.2; retained: 0.1515, 0.6928; mean retained **0.4221**. **REJECT**; prior best 1.0539 from call 6.

### Call 11: seed 2, full_shared

Test whether all 128 second-layer units can be shared: one stacked closed-form solve matches each task teacher preactivations, then one stacked output solve matches teacher logits. This avoids the gated model capacity split.

Configuration: `{"mode": "shared"}`.

Returns fwd/bwd: 354.5, -137.8; retained: 0.0636, -0.0161; mean retained **0.0238**. **REJECT**; prior best 0.8374 from call 7.

### Call 12: seed 3, full_shared

Test whether all 128 second-layer units can be shared: one stacked closed-form solve matches each task teacher preactivations, then one stacked output solve matches teacher logits. This avoids the gated model capacity split.

Configuration: `{"mode": "shared"}`.

Returns fwd/bwd: 1709.2, 3408.5; retained: 0.3198, 0.4296; mean retained **0.3747**. **REJECT**; prior best 0.7063 from call 8.

### Call 13: seed 0, motion_gate

Standard-normal normalized velocity covers mostly low speeds. Use synthetic task-signed raw speeds uniformly 0 to 11 m/s, upright torso height/pitch, and Gaussian joints to fit the same gated construction over a more plausible running-state range.

Configuration: `{"distribution": "motion", "mode": "gated"}`.

Returns fwd/bwd: 2523.5, 7050.0; retained: 0.5021, 0.8905; mean retained **0.6963**. **REJECT**; prior best 0.9218 from call 5.

### Call 14: seed 1, motion_gate

Standard-normal normalized velocity covers mostly low speeds. Use synthetic task-signed raw speeds uniformly 0 to 11 m/s, upright torso height/pitch, and Gaussian joints to fit the same gated construction over a more plausible running-state range.

Configuration: `{"distribution": "motion", "mode": "gated"}`.

Returns fwd/bwd: 5092.8, 7452.8; retained: 1.2762, 0.9361; mean retained **1.1061**. **KEEP**; prior best 1.0539 from call 6.

### Call 15: seed 2, motion_gate

Standard-normal normalized velocity covers mostly low speeds. Use synthetic task-signed raw speeds uniformly 0 to 11 m/s, upright torso height/pitch, and Gaussian joints to fit the same gated construction over a more plausible running-state range.

Configuration: `{"distribution": "motion", "mode": "gated"}`.

Returns fwd/bwd: 3538.1, 7291.1; retained: 0.6349, 0.8539; mean retained **0.7444**. **REJECT**; prior best 0.8374 from call 7.

### Call 16: seed 3, motion_gate

Standard-normal normalized velocity covers mostly low speeds. Use synthetic task-signed raw speeds uniformly 0 to 11 m/s, upright torso height/pitch, and Gaussian joints to fit the same gated construction over a more plausible running-state range.

Configuration: `{"distribution": "motion", "mode": "gated"}`.

Returns fwd/bwd: 2779.4, 7421.4; retained: 0.52, 0.9353; mean retained **0.7277**. **KEEP**; prior best 0.7063 from call 8.

### Call 17: seed 0, bwd_capacity

Backward retention is below forward retention in the first gated networks. Reallocate the second-layer capacity from 64/63 to 48 forward and 79 backward units while keeping the successful standard-normal fit distribution.

Configuration: `{"allocation": 48, "mode": "gated"}`.

Returns fwd/bwd: 4720.9, 7410.2; retained: 0.9394, 0.936; mean retained **0.9377**. **KEEP**; prior best 0.9218 from call 5.

### Call 18: seed 1, bwd_capacity

Backward retention is below forward retention in the first gated networks. Reallocate the second-layer capacity from 64/63 to 48 forward and 79 backward units while keeping the successful standard-normal fit distribution.

Configuration: `{"allocation": 48, "mode": "gated"}`.

Returns fwd/bwd: 3946.3, 7534.4; retained: 0.9889, 0.9463; mean retained **0.9676**. **REJECT**; prior best 1.1061 from call 14.

### Call 19: seed 2, bwd_capacity

Backward retention is below forward retention in the first gated networks. Reallocate the second-layer capacity from 64/63 to 48 forward and 79 backward units while keeping the successful standard-normal fit distribution.

Configuration: `{"allocation": 48, "mode": "gated"}`.

Returns fwd/bwd: 2951.9, 7350.8; retained: 0.5297, 0.8609; mean retained **0.6953**. **REJECT**; prior best 0.8374 from call 7.

### Call 20: seed 3, bwd_capacity

Backward retention is below forward retention in the first gated networks. Reallocate the second-layer capacity from 64/63 to 48 forward and 79 backward units while keeping the successful standard-normal fit distribution.

Configuration: `{"allocation": 48, "mode": "gated"}`.

Returns fwd/bwd: 3723.7, 6187.9; retained: 0.6967, 0.7798; mean retained **0.7383**. **KEEP**; prior best 0.7277 from call 16.

### Call 21: seed 2, combined_distributions

Earlier evaluations show the normal-distribution forward block and motion-distribution backward block each retain more on these seeds. Because the gate makes blocks independent, combine those two constructions at the original 64/63 allocation and evaluate the complete network before keeping it.

Configuration: `{"distribution": ["normal", "motion"], "mode": "gated"}`.

Returns fwd/bwd: 4944.1, 7358.6; retained: 0.8872, 0.8618; mean retained **0.8745**. **KEEP**; prior best 0.8374 from call 7.

### Call 22: seed 3, combined_distributions

Earlier evaluations show the normal-distribution forward block and motion-distribution backward block each retain more on these seeds. Because the gate makes blocks independent, combine those two constructions at the original 64/63 allocation and evaluate the complete network before keeping it.

Configuration: `{"distribution": ["normal", "motion"], "mode": "gated"}`.

Returns fwd/bwd: 2425.4, 7516.9; retained: 0.4538, 0.9473; mean retained **0.7006**. **REJECT**; prior best 0.7383 from call 20.

### Call 23: seed 0, action_sensitive

Ordinary logit regression spends capacity matching saturated torques. A single weighted closed-form output solve uses squared teacher tanh derivatives, floored before squaring at 0.05, to prioritize errors that actually change actions. No iterative reweighting.

Configuration: `{"mode": "gated", "output_weight": 1.0}`.

Returns fwd/bwd: 4095.2, 7289.8; retained: 0.8149, 0.9208; mean retained **0.8679**. **REJECT**; prior best 0.9377 from call 17.

### Call 24: seed 1, action_sensitive

Ordinary logit regression spends capacity matching saturated torques. A single weighted closed-form output solve uses squared teacher tanh derivatives, floored before squaring at 0.05, to prioritize errors that actually change actions. No iterative reweighting.

Configuration: `{"mode": "gated", "output_weight": 1.0}`.

Returns fwd/bwd: 4105.5, 7084.2; retained: 1.0288, 0.8898; mean retained **0.9593**. **REJECT**; prior best 1.1061 from call 14.

### Call 25: seed 2, action_sensitive

Ordinary logit regression spends capacity matching saturated torques. A single weighted closed-form output solve uses squared teacher tanh derivatives, floored before squaring at 0.05, to prioritize errors that actually change actions. No iterative reweighting.

Configuration: `{"mode": "gated", "output_weight": 1.0}`.

Returns fwd/bwd: 4923.3, 5790.2; retained: 0.8835, 0.6781; mean retained **0.7808**. **REJECT**; prior best 0.8745 from call 21.

### Call 26: seed 3, action_sensitive

Ordinary logit regression spends capacity matching saturated torques. A single weighted closed-form output solve uses squared teacher tanh derivatives, floored before squaring at 0.05, to prioritize errors that actually change actions. No iterative reweighting.

Configuration: `{"mode": "gated", "output_weight": 1.0}`.

Returns fwd/bwd: 3169.5, 7237.6; retained: 0.593, 0.9121; mean retained **0.7526**. **KEEP**; prior best 0.7383 from call 20.

### Call 27: seed 3, union_features

The weakest seed may lose task-specific nonlinear directions when first-layer rows are averaged. Select 127 original observation-weight rows from the union of both specialists by centered activation QR on 8192 synthetic inputs, then perform the same separate closed-form second-layer/output fits.

Configuration: `{"feature": "union", "mode": "gated"}`.

Returns fwd/bwd: 2358.9, 658.9; retained: 0.4413, 0.083; mean retained **0.2621**. **REJECT**; prior best 0.7526 from call 26.

### Call 28: seed 3, input_gate_aligned

Test the complementary capacity allocation: split and gate the first layer, give each task all 128 second-layer units, and align backward neurons to forward neurons by an exact minimum-cost assignment of output-weight vectors with tanh sign symmetry. Closed-form fits map compressed first features to teacher preactivations and one shared output matrix to both teachers.

Configuration: `{"mode": "input_gated"}`.

Returns fwd/bwd: 4270.6, 4799.9; retained: 0.799, 0.6049; mean retained **0.702**. **REJECT**; prior best 0.7526 from call 26.

### Call 29: seed 3, hidden_sensitive

Extend the fixed tanh-derivative weighting to each second-layer preactivation fit: one weighted solve per neuron emphasizes unsaturated hidden responses. Combine with the output sensitivity weighting that improved this seed in call 26.

Configuration: `{"hidden_weight": 1.0, "mode": "gated", "output_weight": 1.0}`.

Returns fwd/bwd: 3705.5, 7272.2; retained: 0.6933, 0.9165; mean retained **0.8049**. **KEEP**; prior best 0.7526 from call 26.

### Call 30: seed 0, mixed_coverage

The running-state distribution helps backward speed but its narrower torso range can damage recovery. Fit on a fixed 50/50 mixture of broad normalized Gaussian states and upright task-speed synthetic states to cover both cruising and recovery, with the original balanced 64/63 split.

Configuration: `{"distribution": "mixed", "mode": "gated"}`.

Returns fwd/bwd: 2871.2, 6912.9; retained: 0.5713, 0.8732; mean retained **0.7223**. **REJECT**; prior best 0.9377 from call 17.

### Call 31: seed 1, mixed_coverage

The running-state distribution helps backward speed but its narrower torso range can damage recovery. Fit on a fixed 50/50 mixture of broad normalized Gaussian states and upright task-speed synthetic states to cover both cruising and recovery, with the original balanced 64/63 split.

Configuration: `{"distribution": "mixed", "mode": "gated"}`.

Returns fwd/bwd: 4665.5, 7567.8; retained: 1.1691, 0.9505; mean retained **1.0598**. **REJECT**; prior best 1.1061 from call 14.

### Call 32: seed 2, mixed_coverage

The running-state distribution helps backward speed but its narrower torso range can damage recovery. Fit on a fixed 50/50 mixture of broad normalized Gaussian states and upright task-speed synthetic states to cover both cruising and recovery, with the original balanced 64/63 split.

Configuration: `{"distribution": "mixed", "mode": "gated"}`.

Returns fwd/bwd: 4688.9, 6925.6; retained: 0.8414, 0.8111; mean retained **0.8263**. **REJECT**; prior best 0.8745 from call 21.

### Call 33: seed 3, mixed_coverage

The running-state distribution helps backward speed but its narrower torso range can damage recovery. Fit on a fixed 50/50 mixture of broad normalized Gaussian states and upright task-speed synthetic states to cover both cruising and recovery, with the original balanced 64/63 split.

Configuration: `{"distribution": "mixed", "mode": "gated"}`.

Returns fwd/bwd: 3371.6, 7253.8; retained: 0.6308, 0.9142; mean retained **0.7725**. **REJECT**; prior best 0.8049 from call 29.

### Call 34: seed 3, linear_residual

Compression may waste nonlinear units representing the approximately linear contribution of discarded neurons. Reserve six units per task as near-linear residual channels: one joint closed-form regression on selected h2 features plus all h1 features yields six linear residual directions; encode each as 20*tanh(residual/20), retaining the same 128-unit architecture.

Configuration: `{"linear_units": 6, "mode": "gated"}`.

Returns fwd/bwd: 2262.7, 7085.4; retained: 0.4233, 0.8929; mean retained **0.6581**. **REJECT**; prior best 0.8049 from call 29.

### Call 35: seed 0, speed_broad_torso

Isolate the useful speed coverage from the harmful upright-torso assumption: task-signed raw velocity is uniform 0 to 11 m/s, but every other normalized coordinate remains standard Gaussian. Keep the original balanced gated construction.

Configuration: `{"distribution": "speed", "mode": "gated"}`.

Returns fwd/bwd: 3438.6, 7190.4; retained: 0.6842, 0.9082; mean retained **0.7962**. **REJECT**; prior best 0.9377 from call 17.

### Call 36: seed 1, speed_broad_torso

Isolate the useful speed coverage from the harmful upright-torso assumption: task-signed raw velocity is uniform 0 to 11 m/s, but every other normalized coordinate remains standard Gaussian. Keep the original balanced gated construction.

Configuration: `{"distribution": "speed", "mode": "gated"}`.

Returns fwd/bwd: 3577.0, 5544.8; retained: 0.8963, 0.6964; mean retained **0.7964**. **REJECT**; prior best 1.1061 from call 14.

### Call 37: seed 2, speed_broad_torso

Isolate the useful speed coverage from the harmful upright-torso assumption: task-signed raw velocity is uniform 0 to 11 m/s, but every other normalized coordinate remains standard Gaussian. Keep the original balanced gated construction.

Configuration: `{"distribution": "speed", "mode": "gated"}`.

Returns fwd/bwd: 2318.5, 6856.5; retained: 0.416, 0.803; mean retained **0.6095**. **REJECT**; prior best 0.8745 from call 21.

### Call 38: seed 3, speed_broad_torso

Isolate the useful speed coverage from the harmful upright-torso assumption: task-signed raw velocity is uniform 0 to 11 m/s, but every other normalized coordinate remains standard Gaussian. Keep the original balanced gated construction.

Configuration: `{"distribution": "speed", "mode": "gated"}`.

Returns fwd/bwd: 5197.0, 2647.0; retained: 0.9723, 0.3336; mean retained **0.653**. **REJECT**; prior best 0.8049 from call 29.

### Call 39: seed 2, sensitive_hybrid

Apply the hidden/output derivative weighting that improved seed 3 to seed 2, preserving its retained task-specific synthetic distributions. This targets unsaturated hidden/output errors without sacrificing its established velocity coverage.

Configuration: `{"distribution": ["normal", "motion"], "hidden_weight": 1.0, "mode": "gated", "output_weight": 1.0}`.

Returns fwd/bwd: 4055.2, 7578.0; retained: 0.7277, 0.8875; mean retained **0.8076**. **REJECT**; prior best 0.8745 from call 21.

### Call 40: seed 3, best_task_components

Call 38 produced the strongest forward behavior for this seed while call 29 produced strong backward behavior. Combine the speed-distribution unweighted forward block with the normal-distribution derivative-weighted backward block; evaluate the complete combination because finite-precision differences can affect long trajectories.

Configuration: `{"distribution": ["speed", "normal"], "hidden_weight": [0.0, 1.0], "mode": "gated", "output_weight": [0.0, 1.0]}`.

Returns fwd/bwd: 4355.9, 7227.8; retained: 0.8149, 0.9109; mean retained **0.8629**. **KEEP**; prior best 0.8049 from call 29.

### Call 41: seed 2, best_task_components

Preserve the unweighted normal-distribution forward block from call 21 and use the stronger derivative-weighted motion-distribution backward block from call 39. The complete merged candidate must beat the retained score before it is kept.

Configuration: `{"distribution": ["normal", "motion"], "hidden_weight": [0.0, 1.0], "mode": "gated", "output_weight": [0.0, 1.0]}`.

Returns fwd/bwd: 4888.3, 7629.7; retained: 0.8772, 0.8936; mean retained **0.8854**. **KEEP**; prior best 0.8745 from call 21.

### Call 42: seed 2, preserve_forward_features

Forward performance is more fragile than backward performance. Preserve the forward specialist first-layer observation directions exactly (except the reserved gate neuron) instead of averaging, and let the closed-form backward projection absorb the direction differences.

Configuration: `{"distribution": ["normal", "motion"], "feature": "fwd", "hidden_weight": [0.0, 1.0], "mode": "gated", "output_weight": [0.0, 1.0]}`.

Returns fwd/bwd: 5514.1, 6925.4; retained: 0.9895, 0.8111; mean retained **0.9003**. **KEEP**; prior best 0.8854 from call 41.

### Call 43: seed 3, preserve_forward_features

Test whether preserving forward first-layer observation directions stabilizes the weakest forward gait, while retaining the selected speed/normal task distributions and backward derivative weighting from call 40.

Configuration: `{"distribution": ["speed", "normal"], "feature": "fwd", "hidden_weight": [0.0, 1.0], "mode": "gated", "output_weight": [0.0, 1.0]}`.

Returns fwd/bwd: 5079.5, 6762.0; retained: 0.9503, 0.8522; mean retained **0.9012**. **KEEP**; prior best 0.8629 from call 40.

### Call 44: seed 2, forward_biased_features

Call 42 improved forward return by preserving its input features but reduced backward return. Use a fixed 75% forward/25% backward observation-weight blend as a compromise between the measured full-forward and equal-average constructions, leaving all other choices fixed.

Configuration: `{"distribution": ["normal", "motion"], "feature": "forward75", "hidden_weight": [0.0, 1.0], "mode": "gated", "output_weight": [0.0, 1.0]}`.

Returns fwd/bwd: 5089.5, 6963.2; retained: 0.9133, 0.8155; mean retained **0.8644**. **REJECT**; prior best 0.9003 from call 42.

### Call 45: seed 3, forward_biased_features

Call 43 improved forward return and mean retention but reduced backward return, matching the tradeoff seen in seed 2. Test a fixed 75% forward feature blend to preserve much of the forward benefit while reducing the backward projection error.

Configuration: `{"distribution": ["speed", "normal"], "feature": "forward75", "hidden_weight": [0.0, 1.0], "mode": "gated", "output_weight": [0.0, 1.0]}`.

Returns fwd/bwd: 5340.6, 7064.8; retained: 0.9992, 0.8903; mean retained **0.9447**. **KEEP**; prior best 0.9012 from call 43.

### Call 46: seed 0, backward_sensitivity

Preserve seed 0 retained 48-unit forward block and 79-unit backward capacity, while applying the one-shot derivative weighting to the backward fits. This tests the successful backward approximation refinement without changing the forward construction.

Configuration: `{"allocation": 48, "hidden_weight": [0.0, 1.0], "mode": "gated", "output_weight": [0.0, 1.0]}`.

Returns fwd/bwd: 4818.8, 7396.5; retained: 0.9589, 0.9343; mean retained **0.9466**. **KEEP**; prior best 0.9377 from call 17.

### Call 47: seed 1, backward_sensitivity

Preserve seed 1 strong motion-distribution forward block, and refine only the backward preactivation/output fits with fixed tanh-derivative weights. The entire candidate is retained only if its measured mean improves.

Configuration: `{"distribution": "motion", "hidden_weight": [0.0, 1.0], "mode": "gated", "output_weight": [0.0, 1.0]}`.

Returns fwd/bwd: 4791.1, 7415.7; retained: 1.2006, 0.9314; mean retained **1.066**. **REJECT**; prior best 1.1061 from call 14.

### Call 48: seed 2, balanced_capacity

Preserving forward first-layer features nearly recovered specialist forward return, leaving backward compression as the larger deficit. Transfer eight second-layer units to backward (56/71 split), retaining the best feature choice and task-specific fitting rules.

Configuration: `{"allocation": 56, "distribution": ["normal", "motion"], "feature": "fwd", "hidden_weight": [0.0, 1.0], "mode": "gated", "output_weight": [0.0, 1.0]}`.

Returns fwd/bwd: 5051.9, 7272.1; retained: 0.9065, 0.8517; mean retained **0.8791**. **REJECT**; prior best 0.9003 from call 42.

### Call 49: seed 3, balanced_capacity

The retained seed 3 network recovers essentially all forward specialist return but less backward return. Shift eight second-layer units from forward to backward while preserving the successful 75% forward first-layer blend and task-specific fitting rules.

Configuration: `{"allocation": 56, "distribution": ["speed", "normal"], "feature": "forward75", "hidden_weight": [0.0, 1.0], "mode": "gated", "output_weight": [0.0, 1.0]}`.

Returns fwd/bwd: 5004.0, 7076.5; retained: 0.9362, 0.8918; mean retained **0.914**. **REJECT**; prior best 0.9447 from call 45.

### Call 50: seed 2, least_salient_gate

The gate previously replaced first-layer neuron 127 arbitrarily. Instead remove the neuron with the smallest summed squared a2 contribution weighted by squared a3 output-column norms across both specialists, then reserve the freed slot for the gate. This deterministic pruning heuristic tests whether preserving more important input features repairs the remaining backward deficit.

Configuration: `{"distribution": ["normal", "motion"], "feature": "fwd", "gate_selection": "saliency", "hidden_weight": [0.0, 1.0], "mode": "gated", "output_weight": [0.0, 1.0]}`.

Returns fwd/bwd: 5178.4, 6882.5; retained: 0.9292, 0.8061; mean retained **0.8677**. **REJECT**; prior best 0.9003 from call 42.

## Final verification

Completed 2026-11-04 15:38:19 CST. Rebuilt all four final networks from the embedded configurations and supplied data only. Each rebuilt NPZ is byte-identical (full SHA-256) to its evaluated selected candidate, and its hash matches the evaluator ledger receipt.

Verified exact seven-key schema, required shapes, float32 dtype, finite parameters, and bounded finite actions. For each seed and each label, checked 6000 independent Gaussian observations and 6000 synthetic corner observations with coordinates at -10 or +10. The inactive second-layer block is exactly +1 in float32 for all samples; the analytic worst-case inactive preactivation bound exceeds 11.999 for every unit. Labels yield distinct actions. These are synthetic forward checks, not additional environment evaluations. Full hashes and bounds are in `verification.json`.

The read-only ledger audit confirms calls 1 through 50 exactly once, 300000 environment steps, and 50 matching NOTES call entries. No environment calls remain in progress. Final selection calls: seed 0 = 46; seed 1 = 14; seed 2 = 42; seed 3 = 45.

FINAL: merged_s0..3.npz
