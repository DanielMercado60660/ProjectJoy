# r6wsp_b10_b — BUDGET-10

Read `wsp/data/TASK.md` before inspecting weights. Only supplied seed-local weights and synthetic numpy inputs are used. No simulator imports or interactions except the authorized evaluation script; no gradients, iterative loss minimization, random weight search, other attempts, or external weights.

Initial construction: compare explicit two-branch compression using a shared averaged first-layer basis against independent pruned first-layer branches. All fits are one closed-form affine ridge least-squares solve. Task routing uses saturated tanh task neurons, with analytic inactive-branch correction. Full construction and each evaluation are documented below as completed.

Synthetic inputs use deterministic independent Gaussian draws, a fixed scale mixture, a directional invented root velocity distribution, and an invented mixture of upright and broad root pose distributions. These are not rollout states. Randomness generates only calibration inputs; it never perturbs candidate weights.

## Evaluation ledger

No evaluations yet. Budget limit: 10 calls total over all four seeds. An evaluated revision replaces a seed's incumbent only if its mean retained score improves on the same fixed training resets.

Call 1: seed 0, `shared_s0.npz`. Averaged aligned first-layer observation weights; task-specific teacher effective biases; 63 forward and 64 backward second-layer units selected by output-weighted activation QR, with closed-form layer/output refits. Forward return 3365.0 (0.6696 retained); backward 7145.4 (0.9026); mean retained 0.7861. Kept as seed 0 initial incumbent. Forward episode returns vary substantially (4211.0, 4521.4, 1362.6); this motivates preserving more forward capacity in a later reasoned revision. Independent split prototype has worse synthetic action RMSE on both tasks (0.0964/0.0999 versus shared 0.0809/0.0772), so it has not consumed an evaluation call.

Call 2: seed 1, `shared_s1.npz`, same construction as call 1. Returns 5049.8 forward / 7201.4 backward; retained 1.2654 / 0.9045, mean 1.0850. Kept as initial seed 1 incumbent. Above 100% is specific to these training resets and the specialist denominator, not a claim of held-out superiority.

Call 3: seed 2, `shared_s2.npz`, same construction. Returns 2939.6 / 7280.6; retained 0.5275 / 0.8527, mean 0.6901. Kept as initial seed 2 incumbent. Forward startup is inconsistent across resets.

Call 4: seed 3, `shared_s3.npz`, same construction. Returns 3415.9 / 7101.0; retained 0.6391 / 0.8949, mean 0.7670. Kept as initial seed 3 incumbent. Again forward startup is the limiting branch.

Next reasoned edit: favor the less robust forward branch by assigning it 80 of the 127 available second-layer feature units and moving the shared observation basis from a 50/50 weight average to a 75/25 forward/backward average. This sacrifices some backward feature precision to reduce forward distortion; it is a fixed mechanistic change, not a random perturbation. Test seed 2 first because its current retained score is lowest.

Call 5: seed 2, `forward_capacity_s2.npz`, 80/47 second-layer units and 75/25 forward/backward shared observation basis. Returns 5328.5 / 6959.3; retained 0.9562 / 0.8150, mean 0.8856. **Kept**, improving seed 2 from 0.6901. All three forward reset returns exceed 5200; the anticipated backward cost is present but smaller than the forward gain. Apply the same reasoned construction to seeds 0 and 3, which had the same forward inconsistency.

Call 6: seed 0, `forward_capacity_s0.npz`, same 80/47 and 75/25 construction. Returns 4744.0 / 6698.7; retained 0.9440 / 0.8461, mean 0.8950. **Kept**, improving from 0.7861.

Call 7: seed 3, `forward_capacity_s3.npz`, same change. Returns 5412.9 / 557.5; retained 1.0127 / 0.0703, mean 0.5415. **Rejected**, retaining seed 3's call-4 score of 0.7670. The backward branch fails despite good forward performance; do not hide this failure. No rollout data or clips are available from the authorized scalar-only evaluator.

With three calls left, investigate an independent closed-form refinement on synthetic inputs: derivative-weighted least squares. For a target preactivation z and tanh output h, set fixed sample weights to (1-h^2)^2 + 0.02. These weights emphasize unsaturated regions where preactivation error changes behavior, while preserving a floor for coverage. Each layer's fit is still one direct linear-system solve per target column; there are no repeated residual updates or iterative loss minimization. Compare synthetic validation before selecting any evaluated candidate.

Synthetic-only diagnostics on seed 3 (independent bank, action RMSE forward/backward): original 0.07511/0.08340; derivative-weighted 0.07297/0.07976; additionally correcting each first-layer task bias so that its synthetic mean preactivation equals its teacher's mean 0.07277/0.07923. The bias correction is exactly `(W_teacher - W_shared) @ mean(x_task)` and is not optimized iteratively.

For call 8 choose a moderated capacity tradeoff: 72 forward / 55 backward units, 62.5% forward first-layer basis, derivative-weighted fits, and analytic mean-bias correction. Synthetic RMSE is 0.05873/0.09475, substantially improving forward fidelity without the earlier backward error of 0.12259. This explicitly responds to call 7's overcompression failure. None of the intervening synthetic-only diagnostic networks have been evaluated or retained as final networks.

Call 8: seed 3, `balanced_s3.npz`, moderated tradeoff and fixed derivative weighting/mean correction described above. Returns 5150.6 / 7286.9; retained 0.9636 / 0.9183, mean 0.9409. **Kept**, improving from the retained call-4 incumbent's 0.7670. Both tasks now have consistent returns over all three training resets.

Calls 9 and 10 will apply this same construction to seeds 0 and 2. The reason is to recover backward capacity from their retained 80/47 networks while retaining more forward capacity and fidelity than the original 63/64 networks. Seed 1's strong original incumbent is left alone.

Call 9: seed 0, `balanced_s0.npz`. Returns 4549.4 / 7091.8; retained 0.9053 / 0.8958, mean 0.9005. **Kept**, improving from 0.8950. Backward improvement outweighs the small forward cost under the predeclared mean-retained criterion.

Call 10: seed 2, `balanced_s2.npz`. Returns 5196.5 / 7550.1; retained 0.9325 / 0.8842, mean 0.9083. **Kept**, improving from 0.8856. Budget exhausted: exactly 10 calls, 60000 total environment steps through the authorized evaluator only. No further evaluation calls will be made.

## Final construction, precisely

All four files implement the specified 19 → 128 → 128 → 6 tanh architecture. They contain exactly the seven requested float32 actor keys; the evaluator supplies the unchanged seed-local normalizer. `log_std` is copied from the seed's forward specialist and is unused for deterministic scoring. No seed borrows another seed's weights.

### First layer

Units 0–126 are a shared observation feature bank. Original neuron row identities are preserved because both specialists descend from the same initialization. Rank original first-layer rows by the sum of their forward and backward second-layer outgoing column norms and drop the single least influential row. For each remaining row, use observation weights `alpha * W_fwd + (1-alpha) * W_bwd`. Its two task columns encode the corresponding teacher's effective first-layer bias (`teacher bias + teacher active task column`). Actual first-layer biases are zero.

For seeds 0, 2, and 3, alpha is 0.625 and each task column additionally receives `(W_teacher - W_shared) @ synthetic_task_input_mean`. This preserves the teacher's mean preactivation under that task's synthetic calibration distribution. Seed 1 retains the evaluated original alpha 0.5, without mean correction.

Unit 127 is the task bit `r = tanh(20*x[17] - 20*x[18])`, which is exactly +1 for forward and -1 for backward in float32.

### Second layer

The two skill banks are disjoint:

| Seeds | Forward units | Backward units | Router |
|---|---|---|---|
| 0, 2, 3 | 0–71 (72) | 72–126 (55) | 127 |
| 1 | 0–62 (63) | 63–126 (64) | 127 |

For each task, compute the teacher's 128 second-layer preactivations on 24000 invented normalized inputs. Solve an affine least-squares system reconstructing those preactivations from the 127 shared first-layer activations. This gives 128 candidate tanh features. Select the stated number by column-pivoted QR of their mean-centered synthetic activations, using the first 8000 inputs and scaling each column by the square root of its original output-weight column norm. The selected rows stay in ascending original index order; exact row maps are in each `merged_sK.json`.

For each selected row, let `w, b` be its fitted observation-feature weights and bias. Define `K = 12 + sum(abs(w)) + abs(b)`. For the forward bank use preactivation `w·h + b + K*(1-r)/2`; for the backward bank use `w·h + b + K*(1+r)/2`. Under the active label this is precisely the fitted branch preactivation. Under the inactive label it is at least 12 for any h in [-1,1], so tanh saturates to +1. Unit 127 computes `tanh(20*r)` and carries the exact task bit to the output layer.

### Output layer

Independently fit each branch's affine output preactivation to the corresponding teacher's original pre-tanh output, using the branch's selected fitted tanh features on the same 24000 synthetic inputs. Call these fitted output matrices and biases `(V_f, c_f)` and `(V_b, c_b)`.

The inactive branch contributes a known constant. Let `d_f = c_f - V_b·1` and `d_b = c_b - V_f·1`. The merged output uses both feature banks, bias `(d_f+d_b)/2`, and task-router coefficient `(d_f-d_b)/2`. Thus the inactive constant cancels analytically, and the final tanh produces the corresponding compressed specialist's action. Routing is exact up to floating-point arithmetic; the approximation is in the shared features and compressed/fitted skills.

### Exactly which closed-form fits and inputs

Every fit solves `(Aᵀ D A + ridge) C = Aᵀ D Y` directly, with an unpenalized affine intercept. The ridge on feature coefficients is `1e-6 * sum(sample_weights)`. For seed 1, D is the identity. For seeds 0, 2, and 3, each target column y uses fixed diagonal weights `D_ii = (1-tanh(y_i)^2)^2 + 0.02`. Second-layer targets are teacher second-layer preactivations; output targets are teacher output preactivations. There is exactly one linear solve per target column, with no residual iteration, gradient step, optimizer, or loss-driven weight search.

Synthetic generation is fixed and task-specific:

- RNG seed: `30100 + 101*seed + 13*(task == bwd) + 7919*bank`.
- Draw independent standard-normal normalized observations and multiply each row by a scale from {0.6, 1.0, 1.5}, with probabilities {0.2, 0.5, 0.3}.
- Override root x-velocity with an invented raw velocity N(5, 2.5²) for forward or N(-7, 2.5²) for backward. For 20% of examples instead use N(0, 1²). Normalize with the provided theta0 mean/variance.
- On 75% of examples, override raw root height with N(0, 0.12²) and raw pitch with N(0, 0.45²), then normalize with the provided normalizer. Other entries retain the independent synthetic values.
- Clip all normalized observation coordinates to [-10,10]. No environment states, transitions, dynamics, actions executed in a simulator, or hidden reset seeds enter construction.
- Bank 0 contains 24000 fit inputs per task. Bank 1 contains 12000 independent diagnostic inputs per task. Input-generation randomness is fixed across recipes and is never used to perturb weights.

## Final measured results and limitations

| Seed | Final recipe | Selected call | Forward return | Backward return | Mean retained | Synthetic action RMSE fwd / bwd |
|---|---|---:|---:|---:|---:|---:|
| 0 | balanced | 9 | 4549.4 | 7091.8 | 90.05% | 0.06565 / 0.09190 |
| 1 | shared | 2 | 5049.8 | 7201.4 | 108.50% | 0.08012 / 0.07421 |
| 2 | balanced | 10 | 5196.5 | 7550.1 | 90.83% | 0.05737 / 0.07729 |
| 3 | balanced | 8 | 5150.6 | 7286.9 | 94.09% | 0.05873 / 0.09475 |

Mean retained over the four final networks: **95.8675% on training resets**. Mean return: 4986.575 forward, 7282.55 backward. The original four shared candidates averaged 83.205% retained on these same resets. The meaningful reference is the per-seed, per-task specialist return denominator reported by the authorized evaluator on the same three training resets. Its switched-specialist ceiling uses two networks; the submitted files each use one network of the required original size. No new environment baseline was run outside the ten calls.

Held-out reset performance: **not measured**. Synthetic action error does not guarantee trajectory stability: seed 3's rejected call 7 shows that a fairly small imitation error can destroy the backward gait. Its failure remains recorded in the evaluator ledger and this document; its candidate is preserved for audit but is not a final deliverable. The synthetic distribution assumes mostly upright poses and invents velocity ranges, so state-distribution mismatch is a material limitation. The strongest training result (seed 1) may reflect reset-specific behavior, and all candidate selection used training resets only.

## Rebuild and verification

From `/Users/danielmercado/ProjectJoy/25_skill_transplantation`:

```sh
/Users/danielmercado/ProjectJoy/.venv/bin/python wsp/attempts/r6wsp_b10_b/build.py
/Users/danielmercado/ProjectJoy/.venv/bin/python wsp/attempts/r6wsp_b10_b/verify.py
```

`build.py` defaults to the fixed selected recipe for each seed and reconstructs all four final files from `wsp/data/` alone. It does not read candidates, evaluation results, or the ledger. `--recipe shared`, `--recipe forward_capacity`, or `--recipe balanced` reproduces an evaluated construction, with `--seed K` to restrict the seed. Optional `split`, `derivative`, and `centered` recipes reproduce synthetic-only diagnostics.

`verify.py` is a synthetic-only artifact check. It reads the immutable evaluator ledger, verifies exactly ten calls, confirms every final tensor is bit-identical to the highest-scoring evaluated candidate for its seed, checks the exact key set/shapes/float32 dtype/finite values, and tests both task paths on independent synthetic inputs plus randomly selected corners of the entire [-10,10]^17 observation box. Every inactive branch saturated exactly to +1 in this check; all actions were finite and bounded. The check passed, and detailed hashes/results are in `verification.json`. No verification step interacts with an environment.

Only this attempt directory was written. The evaluator alone wrote `ledger.jsonl`; it was never edited manually. No packages, network access, git commands, other attempts, project result/checkpoint folders, or prohibited environment interfaces were used.

FINAL: merged_s0..3.npz
