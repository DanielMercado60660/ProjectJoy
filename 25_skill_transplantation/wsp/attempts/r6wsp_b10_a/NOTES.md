# BUDGET-10: r6wsp_b10_a

## Delivered result and scope

One ordinary 19 -> 128 -> 128 -> 6 tanh network per seed. No ensemble,
external routing, extra runtime state, or architecture changes. The supplied
one-hot label controls internal routing. Files contain exactly the six actor
arrays plus `log_std`, all float32. Normalization remains the evaluator's
frozen seed normalizer. `log_std` is copied from that seed's theta0 and unused.

Only the allowed TASK.md, source weights in wsp/data, evaluator source, and
this attempt's own files were read. No other attempts, checkpoints, result
banks, project rollout code, network, installs, git, or environment APIs were
used. The sole environment interaction was **10 calls** to eval_budget.py,
60,000 evaluator steps total, each on the permitted fixed training resets.
The evaluator wrote ledger.jsonl; this attempt never modified it. No GPU or
long-running job was needed; all individual builds and checks finished in
seconds. Numerical-library thread caps are four.

There is **no gradient descent, autograd, iterative loss minimization,
evolutionary/random weight search, or random weight perturbation**. Random
numbers are used only to generate deterministic synthetic inputs. All edits
are explicit averaging, pruning/ranking, one-shot ridge solves, and analytic
routing. No repeated reweighting or optimization loop is used.

## What each part computes

Indices below are zero-based.

1. **First-layer units 0..126: shared nonlinear observation features.** Pair
   corresponding fwd/bwd neurons inherited from the same theta0. For each
   original column j, rank downstream energy by
   `sum_i(fwd.W2[i,j]^2 + bwd.W2[i,j]^2)`. Drop the weakest column, retaining
   all others in original order. Dropped original indices for seeds 0..3 are
   respectively **88, 78, 100, 75**. Each retained observation weight row is
   `(fwd.W1[j,:17] + bwd.W1[j,:17])/2`. Its bias is zero; label column 17 is
   `fwd.b1[j] + fwd.W1[j,17]`, and column 18 is
   `bwd.b1[j] + bwd.W1[j,18]`. Thus each task retains its own effective
   first-layer bias while sharing observation directions.
2. **First-layer unit 127: label marker.** Observation weights and bias are
   zero and task weights are `[3,-3]`. Output is `s*g`, where `s=+1` for fwd,
   `s=-1` for bwd, and `g=tanh(3)`.
3. **Second-layer units 0..62: forward features. Units 63..126: backward
   features.** For each task independently, project all 128 teacher
   second-layer preactivations onto the 127 shared first-layer activations
   with a single centered ridge solve. Take tanh of the projected values.
   Run column-pivoted QR on every third synthetic row after column centering,
   with each column multiplied by the L2 norm of its teacher output weights.
   The first 63 fwd / 64 bwd pivots select independent, output-important
   features. Save their fitted incoming coefficients and intercepts. Each
   diagnostics JSON lists the original second-layer neuron IDs in the exact
   packed order. These are compressed specialist features, not individually
   identified gait-phase neurons.
4. **Inactive branch saturation.** Let a fitted selected row be `v` and its
   intercept be `c`; let its task sign be `q`. Set
   `M=sum(abs(v))+abs(c)+12`. Write weights `v` on shared features, weight
   `-q*M/(2*g)` on the label marker, and bias `c+M/2`. Under the active label
   the shift cancels, giving `v*h+c`. Under the other label it becomes
   `v*h+c+M >= 12`, because every shared feature lies in [-1,1]. Thus the
   inactive branch is +1 to float32 precision even for extreme input states.
5. **Second-layer unit 127: output label marker.** Its only input weight is
   `3/g` from first-layer unit 127; it outputs `s*g` again.
6. **Output: independent compressed heads with constant cancellation.** Fit
   the teacher's six output preactivations from its selected active features
   and an intercept. If heads are `(Af,bf)` and `(Ab,bb)`, define
   `cf=bf-Ab*1` and `cb=bb-Af*1`. Store Af and Ab on their respective packed
   features, bias `(cf+cb)/2`, and marker weight `(cf-cb)/(2*g)`. The inactive
   branch's constant contribution is canceled, yielding the active fitted
   specialist head before the architecture's final tanh.

## Exactly what was fitted, and on which inputs

For each seed and each task, generate **24,000** independent synthetic input
vectors, with NumPy `default_rng(812000 + 13*seed + task)` (task 0=fwd, 1=bwd).
Start with 17 independent standard normal coordinates; multiply each row by
one sampled scale from `[0.5,1,1.5]` with probabilities `[0.2,0.6,0.2]`.
On the 80% of rows whose index modulo 5 is nonzero, overwrite these three
coordinates using explicit raw-coordinate priors and the supplied normalizer:

- Root z: Normal(mean=-0.06, standard deviation=0.12).
- Root pitch: Normal(mean=0, standard deviation=0.35).
- Root x velocity: Uniform(-1,11) for fwd, Uniform(-13,1) for bwd.

The remaining coordinates and the remaining 20% of rows keep the scaled
normalized Gaussians. Clip all normalized coordinates to [-10,10] and cast
to float32. These independent priors are assumptions about input coverage,
not recorded states, dynamics simulation, rollouts, or sampled environment
observations. They intentionally cover upright motion and broader states.

All fit targets come only from NumPy forward passes of that seed's supplied
specialist under its own label. The first fit targets the 128 teacher layer-2
preactivations. The second targets the six teacher output preactivations
using the selected approximate layer-2 activations. Both ordinary fits solve
`(Xc.T Xc / N + 1e-5 I) B = Xc.T Yc / N`, then set the unregularized intercept
from the means. Solves accumulate in float64 and coefficients become float32.
There is no iterative refinement or convergence-based stopping criterion.

**Final seed-specific output fits:** seeds 1 and 2 use the ordinary solve.
Seeds 0 and 3 use a forward-only weighted solve, independently for each of
the six outputs. Each sample's fixed weight is
`(1-tanh(teacher_output_preactivation)^2)^2 + 0.01`.
This is a one-shot local action-space quadratic weighting; the positive
floor retains information about saturated outputs. The intercept is
unpenalized and the other coefficients use the same 1e-5 ridge. Backward fits,
feature selection and all hidden weights are unchanged. This analytic edit
was tested and kept only after its training retained score increased.

Diagnostic validation uses 12,000 fresh synthetic rows per task with RNG seed
increased by 10,000. These are **synthetic validation inputs**, not held-out
reset seeds. No synthetic inputs are carried into an environment.

## Every evaluation call and retention decision

Each call evaluated one candidate on both tasks, three training resets each,
with exactly the evaluator's deterministic action and 1000-step episodes.
The incumbent is tracked separately for each seed. Baseline calls are
unchanged fwd specialists; an edit is retained only if mean retained strictly
exceeds that seed's incumbent. Every tested edit happened to improve it.
All original candidates and evaluator receipts are preserved in this folder.

| Call | Seed | Candidate | Fwd return | Bwd return | Mean retained | Decision |
|---:|---:|---|---:|---:|---:|---|
| 1 | 0 | `baseline_s0.npz` | 5025.5 | -307.4 | 48.06% | Baseline established |
| 2 | 0 | `shared_motion_s0.npz` | 2579.0 | 7156.2 | 70.86% | Kept; > 48.06% |
| 3 | 1 | `baseline_s1.npz` | 3990.7 | -370.6 | 47.68% | Baseline established |
| 4 | 1 | `shared_motion_s1.npz` | 4850.8 | 7497.5 | 107.86% | Kept; > 47.68% |
| 5 | 2 | `baseline_s2.npz` | 5572.8 | 1714.7 | 60.04% | Baseline established |
| 6 | 2 | `shared_motion_s2.npz` | 4815.6 | 7356.5 | 86.29% | Kept; > 60.04% |
| 7 | 3 | `baseline_s3.npz` | 5345.0 | 2403.7 | 65.14% | Baseline established |
| 8 | 3 | `shared_motion_s3.npz` | 3871.0 | 6992.9 | 80.27% | Kept; > 65.14% |
| 9 | 0 | `derivative_motion_s0.npz` | 2794.5 | 7171.8 | 73.10% | Kept; > 70.86% |
| 10 | 3 | `derivative_motion_s3.npz` | 3879.3 | 7119.2 | 81.15% | Kept; > 80.27% |

1. Unchanged fwd specialist: establish the seed 0 incumbent.
2. Shared 127-neuron first layer and gated 63/64 specialist second-layer branches; recover both behaviors without doubling network size.
3. Unchanged fwd specialist: establish the seed 1 incumbent.
4. Apply the same reasoned shared-basis compression to seed 1.
5. Unchanged fwd specialist: establish the seed 2 incumbent.
6. Apply the same reasoned shared-basis compression to seed 2.
7. Unchanged fwd specialist: establish the seed 3 incumbent.
8. Apply the same reasoned shared-basis compression to seed 3.
9. Refit seed 0 forward output only with fixed teacher-tanh-derivative weights; target unsaturated action errors. Synthetic forward RMSE 0.07833 -> 0.07595.
10. Apply that same forward-only weighted linear solve to seed 3. Synthetic forward RMSE 0.07172 -> 0.07031.

## Final measured training results

| Seed | Selected call | Fwd return | Bwd return | Fwd mean x velocity | Bwd mean x velocity | Mean retained |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9 | 2794.5 | 7171.8 | 3.184 | -7.575 | 73.10% |
| 1 | 4 | 4850.8 | 7497.5 | 5.205 | -7.906 | 107.86% |
| 2 | 6 | 4815.6 | 7356.5 | 5.150 | -7.783 | 86.29% |
| 3 | 10 | 3879.3 | 7119.2 | 4.170 | -7.488 | 81.15% |

Mean retained across four seeds: **87.100%**, versus **55.230%**
for the unchanged-fwd baselines on these same training resets. Mean returns
are **4085.05 fwd** and
**7286.25 bwd**.
Held-out reset returns and retained score: **not measured**.

Seed 1 exceeds 100% on these training resets because the supplied forward
specialist had one low-return training episode (1256.9), while the merged
candidate's three forward returns were 4908.5, 5006.2, and 4637.7. This does
not establish superiority on unseen resets.

## Approximation costs and preserved failure

Compression preserves two behaviors at the cost of imperfect action
imitation. The shared first layer changes specialist observation directions;
its projection has layer-2 preactivation RMSE about 0.059-0.065. Keeping only
63/64 layer-2 features introduces another approximation. Synthetic independent
action RMSE for final fwd/bwd heads is:

| Seed | Fwd RMSE | Bwd RMSE |
|---:|---:|---:|
| 0 | 0.07595 | 0.08047 |
| 1 | 0.07938 | 0.07175 |
| 2 | 0.07081 | 0.06598 |
| 3 | 0.07031 | 0.07771 |

These errors can compound over a 1000-step episode; synthetic action accuracy
alone is not a locomotion guarantee. In particular seed 0 still has forward
training returns **4700.4, 1374.7, 2308.2**, and seed 3 has **5094.9, 1374.0,
5169.2**. Backward motion is more consistent. No rollouts or videos were
available through the authorized interface, so the precise failure mechanism
was **not observed** and no visual failure clip is claimed.

Although the forward-only output edit leaves backward behavior algebraically
equivalent, it changes the shared output bias/marker cancellation in float32.
Backward episode returns changed (e.g. seed 3 mean 6992.9 -> 7119.2). Small
floating-point action differences can amplify through long dynamics; this
is a plausible explanation, not a diagnosed trajectory-level cause. Final
files are byte-identical to their actually evaluated candidates, and both
task returns, including these changes, enter every retention decision.

Task routing itself has an analytic lower bound of 12 on the inactive branch
preactivation. The approximate specialists and synthetic coverage, rather
than accidental mixing of the inactive branch, are the main limitations.
The next useful check is the externally run held-out reset evaluation; no
further evaluator call is permitted in this attempt.

## Rebuild and verification

From the project directory:

```sh
/Users/danielmercado/ProjectJoy/.venv/bin/python wsp/attempts/r6wsp_b10_a/build.py
/Users/danielmercado/ProjectJoy/.venv/bin/python wsp/attempts/r6wsp_b10_a/smoke_check.py
```

The default build hard-codes the accepted output-fit choice for each seed and
reconstructs all four final files from wsp/data alone. It does not read the
ledger, prior candidates, diagnostics, or evaluation results. It also writes
fresh synthetic diagnostics. No environment is imported or invoked.

The synthetic-only smoke check passed for all four seeds: exact keys/shapes,
float32, finite values, action bounds, full 19-input forward execution, task
routing on 10,000 new bounded synthetic inputs per label, and bitwise equality
both to a fresh reconstruction and to the selected evaluated candidate.
Each network has 19,852 parameters including log_std. Measured minimum
inactive preactivations in that stress check were all above 16.8. See
verification.json for exact hashes and results. The build and smoke check
complete in seconds; no budgeted evaluation is part of either command.

The preserved baselines, intermediate candidates, diagnostics, call receipts,
and ledger provide the audit trail. The four files named below are the only
submitted networks.

FINAL: merged_s0..3.npz
