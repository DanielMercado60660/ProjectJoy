# r6wsp_b50_a: BUDGET-50

Only the supplied per-seed weight files are used. No environment access except the budget evaluator, no gradient descent, no random weight search. Synthetic input sampling is deterministic and is not environment sampling. All selection decisions compare the same training-reset mean retained score. Rejected candidates are preserved as audit artifacts but do not replace the selected merged networks.

Construction and final validation will be recorded after the evaluation log below. Each call tests a stated structural edit; the tool-owned ledger is never modified.

## Evaluation log

### Call 1: seed 0, fwd_baseline

Establish the measured single-specialist baseline before accepting any compressed routed edit.

Config: `{"mode": "fwd"}`.

Training resets: fwd 5025.5, bwd -307.4; mean retained 0.4806. **KEPT**; initial measured baseline.

### Call 2: seed 1, fwd_baseline

Establish the measured single-specialist baseline before accepting any compressed routed edit.

Config: `{"mode": "fwd"}`.

Training resets: fwd 3990.7, bwd -370.6; mean retained 0.4768. **KEPT**; initial measured baseline.

### Call 3: seed 2, fwd_baseline

Establish the measured single-specialist baseline before accepting any compressed routed edit.

Config: `{"mode": "fwd"}`.

Training resets: fwd 5572.8, bwd 1714.7; mean retained 0.6004. **KEPT**; initial measured baseline.

### Call 4: seed 3, fwd_baseline

Establish the measured single-specialist baseline before accepting any compressed routed edit.

Config: `{"mode": "fwd"}`.

Training resets: fwd 5345.0, bwd 2403.7; mean retained 0.6514. **KEPT**; initial measured baseline.

### Call 5: seed 0, shared63

Average aligned first-layer observation weights, keep task-specific biases, reconstruct second-layer preactivations by closed-form regression, retain 63 output-weighted independent features per expert, and silence the other branch with a bounded saturation gate.

Config: `{"mode": "shared"}`.

Training resets: fwd 2793.6, bwd 7152.1; mean retained 0.7296. **KEPT**; prior best 0.4806.

### Call 6: seed 1, shared63

Average aligned first-layer observation weights, keep task-specific biases, reconstruct second-layer preactivations by closed-form regression, retain 63 output-weighted independent features per expert, and silence the other branch with a bounded saturation gate.

Config: `{"mode": "shared"}`.

Training resets: fwd 4985.3, bwd 7278.4; mean retained 1.0817. **KEPT**; prior best 0.4768.

### Call 7: seed 2, shared63

Average aligned first-layer observation weights, keep task-specific biases, reconstruct second-layer preactivations by closed-form regression, retain 63 output-weighted independent features per expert, and silence the other branch with a bounded saturation gate.

Config: `{"mode": "shared"}`.

Training resets: fwd 4833.8, bwd 6792.6; mean retained 0.8315. **KEPT**; prior best 0.6004.

### Call 8: seed 3, shared63

Average aligned first-layer observation weights, keep task-specific biases, reconstruct second-layer preactivations by closed-form regression, retain 63 output-weighted independent features per expert, and silence the other branch with a bounded saturation gate.

Config: `{"mode": "shared"}`.

Training resets: fwd 2930.4, bwd 477.9; mean retained 0.3042. **REJECTED**; prior best 0.6514.

### Call 9: seed 0, broad63

Broaden synthetic joint and velocity coverage with a fixed three-scale Gaussian mixture; the initial narrow synthetic fit may be inaccurate once a fast gait reaches larger normalized inputs.

Config: `{"mixture": true, "mode": "shared", "scale": 1.5}`.

Training resets: fwd 3551.7, bwd 7166.9; mean retained 0.8060. **KEPT**; prior best 0.7296.

### Call 10: seed 1, broad63

Broaden synthetic joint and velocity coverage with a fixed three-scale Gaussian mixture; the initial narrow synthetic fit may be inaccurate once a fast gait reaches larger normalized inputs.

Config: `{"mixture": true, "mode": "shared", "scale": 1.5}`.

Training resets: fwd 4830.7, bwd 7354.0; mean retained 1.0671. **REJECTED**; prior best 1.0817.

### Call 11: seed 2, broad63

Broaden synthetic joint and velocity coverage with a fixed three-scale Gaussian mixture; the initial narrow synthetic fit may be inaccurate once a fast gait reaches larger normalized inputs.

Config: `{"mixture": true, "mode": "shared", "scale": 1.5}`.

Training resets: fwd 5002.9, bwd 6656.1; mean retained 0.8386. **KEPT**; prior best 0.8315.

### Call 12: seed 3, broad63

Broaden synthetic joint and velocity coverage with a fixed three-scale Gaussian mixture; the initial narrow synthetic fit may be inaccurate once a fast gait reaches larger normalized inputs.

Config: `{"mixture": true, "mode": "shared", "scale": 1.5}`.

Training resets: fwd 4556.1, bwd 3383.0; mean retained 0.6394. **REJECTED**; prior best 0.6514.

### Call 13: seed 0, transition63

Include synthetic start-up velocities as well as both full-speed regimes using a uniform task-signed velocity range, while retaining broad joint coverage; narrow full-speed fitting omitted the near-zero start.

Config: `{"mode": "shared", "profile": "transition", "scale": 1.5}`.

Training resets: fwd 3493.5, bwd 7116.5; mean retained 0.7971. **REJECTED**; prior best 0.8060.

### Call 14: seed 2, transition63

Include synthetic start-up velocities as well as both full-speed regimes using a uniform task-signed velocity range, while retaining broad joint coverage; narrow full-speed fitting omitted the near-zero start.

Config: `{"mode": "shared", "profile": "transition", "scale": 1.5}`.

Training resets: fwd 3101.7, bwd 5742.5; mean retained 0.6145. **REJECTED**; prior best 0.8386.

### Call 15: seed 3, transition63

Include synthetic start-up velocities as well as both full-speed regimes using a uniform task-signed velocity range, while retaining broad joint coverage; narrow full-speed fitting omitted the near-zero start.

Config: `{"mode": "shared", "profile": "transition", "scale": 1.5}`.

Training resets: fwd 5227.4, bwd 830.3; mean retained 0.5413. **REJECTED**; prior best 0.6514.

### Call 16: seed 0, upright63

Concentrate synthetic root pitch around upright (raw angle zero) and narrow vertical-position variation while covering start-up through full-speed velocities; this allocates approximation accuracy to running postures.

Config: `{"mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 4624.7, bwd 7110.8; mean retained 0.9092. **KEPT**; prior best 0.8060.

### Call 17: seed 2, upright63

Concentrate synthetic root pitch around upright (raw angle zero) and narrow vertical-position variation while covering start-up through full-speed velocities; this allocates approximation accuracy to running postures.

Config: `{"mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 4690.6, bwd 7285.9; mean retained 0.8475. **KEPT**; prior best 0.8386.

### Call 18: seed 3, upright63

Concentrate synthetic root pitch around upright (raw angle zero) and narrow vertical-position variation while covering start-up through full-speed velocities; this allocates approximation accuracy to running postures.

Config: `{"mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 3936.6, bwd 7117.1; mean retained 0.8167. **KEPT**; prior best 0.6514.

### Call 19: seed 0, action63

Use one closed-form weighted output regression per torque, with weights 0.1 plus squared teacher tanh derivative; reduce error where logit error most affects the action, without iterative optimization.

Config: `{"mode": "shared", "profile": "upright", "scale": 1.5, "weighted3": true}`.

Training resets: fwd 4713.7, bwd 7030.6; mean retained 0.9130. **KEPT**; prior best 0.9092.

### Call 20: seed 2, action63

Use one closed-form weighted output regression per torque, with weights 0.1 plus squared teacher tanh derivative; reduce error where logit error most affects the action, without iterative optimization.

Config: `{"mode": "shared", "profile": "upright", "scale": 1.5, "weighted3": true}`.

Training resets: fwd 4763.7, bwd 7407.7; mean retained 0.8612. **KEPT**; prior best 0.8475.

### Call 21: seed 3, action63

Use one closed-form weighted output regression per torque, with weights 0.1 plus squared teacher tanh derivative; reduce error where logit error most affects the action, without iterative optimization.

Config: `{"mode": "shared", "profile": "upright", "scale": 1.5, "weighted3": true}`.

Training resets: fwd 2535.2, bwd 7125.1; mean retained 0.6861. **REJECTED**; prior best 0.8167.

### Call 22: seed 0, sensitive63

Fit each retained second-layer preactivation with a single teacher-derivative-weighted linear solve, emphasizing unsaturated hidden units whose approximation errors reach the output; retain ordinary output regression.

Config: `{"mode": "shared", "profile": "upright", "scale": 1.5, "weighted2": true}`.

Training resets: fwd 2673.3, bwd 7199.7; mean retained 0.7207. **REJECTED**; prior best 0.9130.

### Call 23: seed 2, sensitive63

Fit each retained second-layer preactivation with a single teacher-derivative-weighted linear solve, emphasizing unsaturated hidden units whose approximation errors reach the output; retain ordinary output regression.

Config: `{"mode": "shared", "profile": "upright", "scale": 1.5, "weighted2": true}`.

Training resets: fwd 4730.8, bwd 7308.8; mean retained 0.8524. **REJECTED**; prior best 0.8612.

### Call 24: seed 3, sensitive63

Fit each retained second-layer preactivation with a single teacher-derivative-weighted linear solve, emphasizing unsaturated hidden units whose approximation errors reach the output; retain ordinary output regression.

Config: `{"mode": "shared", "profile": "upright", "scale": 1.5, "weighted2": true}`.

Training resets: fwd 4765.6, bwd 4670.6; mean retained 0.7401. **REJECTED**; prior best 0.8167.

### Call 25: seed 0, splice_best

Splice the forward branch from action63 and backward branch from sensitive63; their first-layer weights are identical and saturation gates isolate their contributions exactly. Each branch has the best measured task return among evaluated compatible constructions.

Config: `{"branches": [{"mode": "shared", "profile": "upright", "scale": 1.5, "weighted3": true}, {"mode": "shared", "profile": "upright", "scale": 1.5, "weighted2": true}], "mode": "splice"}`.

Training resets: fwd 3177.9, bwd 6931.1; mean retained 0.7539. **REJECTED**; prior best 0.9130.

### Call 26: seed 1, splice_best

Splice the forward branch from shared63 and backward branch from broad63; their first-layer weights are identical and saturation gates isolate their contributions exactly. Each branch has the best measured task return among evaluated compatible constructions.

Config: `{"branches": [{"mode": "shared"}, {"mixture": true, "mode": "shared", "scale": 1.5}], "mode": "splice"}`.

Training resets: fwd 4995.6, bwd 7390.3; mean retained 1.0900. **KEPT**; prior best 1.0817.

### Call 27: seed 2, splice_best

Splice the forward branch from broad63 and backward branch from action63; their first-layer weights are identical and saturation gates isolate their contributions exactly. Each branch has the best measured task return among evaluated compatible constructions.

Config: `{"branches": [{"mixture": true, "mode": "shared", "scale": 1.5}, {"mode": "shared", "profile": "upright", "scale": 1.5, "weighted3": true}], "mode": "splice"}`.

Training resets: fwd 3979.5, bwd 7331.4; mean retained 0.7863. **REJECTED**; prior best 0.8612.

### Call 28: seed 3, splice_best

Splice the forward branch from transition63 and backward branch from action63; their first-layer weights are identical and saturation gates isolate their contributions exactly. Each branch has the best measured task return among evaluated compatible constructions.

Config: `{"branches": [{"mode": "shared", "profile": "transition", "scale": 1.5}, {"mode": "shared", "profile": "upright", "scale": 1.5, "weighted3": true}], "mode": "splice"}`.

Training resets: fwd 5149.7, bwd 7243.9; mean retained 0.9382. **KEPT**; prior best 0.8167.

### Call 29: seed 0, pool63

Replace averaged first-layer directions with 127 output-influence-weighted QR-selected directions from the union of both specialists. Preserve exact specialist directions where possible while ranking diverse synthetic activations across both labels.

Config: `{"basis": "pool", "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 3597.0, bwd 7383.2; mean retained 0.8242. **REJECTED**; prior best 0.9130.

### Call 30: seed 2, pool63

Replace averaged first-layer directions with 127 output-influence-weighted QR-selected directions from the union of both specialists. Preserve exact specialist directions where possible while ranking diverse synthetic activations across both labels.

Config: `{"basis": "pool", "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 4927.7, bwd 7503.2; mean retained 0.8815. **KEPT**; prior best 0.8612.

### Call 31: seed 3, pool63

Replace averaged first-layer directions with 127 output-influence-weighted QR-selected directions from the union of both specialists. Preserve exact specialist directions where possible while ranking diverse synthetic activations across both labels.

Config: `{"basis": "pool", "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 5288.1, bwd 3235.3; mean retained 0.6986. **REJECTED**; prior best 0.9382.

### Call 32: seed 0, linear63

Reserve 17 first-layer neurons as nearly linear coordinates tanh(0.1*x), keeping 110 influential averaged nonlinear features. The coordinates provide a direct basis for compensating small specialist first-layer direction changes with a closed-form second-layer fit.

Config: `{"basis": "linear", "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 2023.7, bwd 7254.2; mean retained 0.6595. **REJECTED**; prior best 0.9130.

### Call 33: seed 2, linear63

Reserve 17 first-layer neurons as nearly linear coordinates tanh(0.1*x), keeping 110 influential averaged nonlinear features. The coordinates provide a direct basis for compensating small specialist first-layer direction changes with a closed-form second-layer fit.

Config: `{"basis": "linear", "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 4790.1, bwd 7459.0; mean retained 0.8666. **REJECTED**; prior best 0.8815.

### Call 34: seed 3, linear63

Reserve 17 first-layer neurons as nearly linear coordinates tanh(0.1*x), keeping 110 influential averaged nonlinear features. The coordinates provide a direct basis for compensating small specialist first-layer direction changes with a closed-form second-layer fit.

Config: `{"basis": "linear", "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 4945.5, bwd 1408.3; mean retained 0.5514. **REJECTED**; prior best 0.9382.

### Call 35: seed 0, backward71

Allocate 71 nonlinear second-layer features to backward and 55 to forward, since backward has the larger synthetic logit approximation error in all three seeds; test whether asymmetric capacity reduces the harder branch error.

Config: `{"count_f": 55, "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 4506.6, bwd 7257.2; mean retained 0.9067. **REJECTED**; prior best 0.9130.

### Call 36: seed 2, backward71

Allocate 71 nonlinear second-layer features to backward and 55 to forward, since backward has the larger synthetic logit approximation error in all three seeds; test whether asymmetric capacity reduces the harder branch error.

Config: `{"count_f": 55, "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 3891.8, bwd 7299.7; mean retained 0.7767. **REJECTED**; prior best 0.8815.

### Call 37: seed 3, backward71

Allocate 71 nonlinear second-layer features to backward and 55 to forward, since backward has the larger synthetic logit approximation error in all three seeds; test whether asymmetric capacity reduces the harder branch error.

Config: `{"count_f": 55, "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 2804.8, bwd 7089.9; mean retained 0.7091. **REJECTED**; prior best 0.9382.

### Call 38: seed 0, upright_mixture

Combine the useful upright/start-up prior with the earlier multi-scale Gaussian mixture: include near-rest small joint excursions and large gait excursions in one fixed closed-form fit, rather than a single joint-amplitude scale.

Config: `{"mixture": true, "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 4561.6, bwd 7198.3; mean retained 0.9084. **REJECTED**; prior best 0.9130.

### Call 39: seed 2, upright_mixture

Combine the useful upright/start-up prior with the earlier multi-scale Gaussian mixture: include near-rest small joint excursions and large gait excursions in one fixed closed-form fit, rather than a single joint-amplitude scale.

Config: `{"mixture": true, "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 4772.4, bwd 7285.2; mean retained 0.8548. **REJECTED**; prior best 0.8815.

### Call 40: seed 3, upright_mixture

Combine the useful upright/start-up prior with the earlier multi-scale Gaussian mixture: include near-rest small joint excursions and large gait excursions in one fixed closed-form fit, rather than a single joint-amplitude scale.

Config: `{"mixture": true, "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 3347.0, bwd 7070.5; mean retained 0.7587. **REJECTED**; prior best 0.9382.

### Call 41: seed 0, exact_fwd_basis

Use the forward specialist first-layer observation directions exactly (except one pruned unit), so forward compression avoids first-layer interpolation error; fit the backward branch into that fixed feature basis analytically.

Config: `{"mix": 1.0, "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 4822.2, bwd 6929.6; mean retained 0.9174. **KEPT**; prior best 0.9130.

### Call 42: seed 2, exact_fwd_basis

Use the forward specialist first-layer observation directions exactly (except one pruned unit), so forward compression avoids first-layer interpolation error; fit the backward branch into that fixed feature basis analytically.

Config: `{"mix": 1.0, "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 4398.5, bwd 4157.8; mean retained 0.6381. **REJECTED**; prior best 0.8815.

### Call 43: seed 3, exact_fwd_basis

Use the forward specialist first-layer observation directions exactly (except one pruned unit), so forward compression avoids first-layer interpolation error; fit the backward branch into that fixed feature basis analytically.

Config: `{"mix": 1.0, "mode": "shared", "profile": "upright", "scale": 1.5}`.

Training resets: fwd 5215.1, bwd 5768.2; mean retained 0.8513. **REJECTED**; prior best 0.9382.

### Call 44: seed 0, magnitude_rank

Replace diversity-based pivoted QR pruning with direct activation-standard-deviation times output-weight-norm ranking, preserving the largest individual teacher contributions even when correlated. Keep the retained construction otherwise unchanged.

Config: `{"mix": 1.0, "mode": "shared", "profile": "upright", "ranking": "weight", "scale": 1.5}`.

Training resets: fwd 4106.8, bwd 6959.8; mean retained 0.8481. **REJECTED**; prior best 0.9174.

### Call 45: seed 2, magnitude_rank

Replace diversity-based pivoted QR pruning with direct activation-standard-deviation times output-weight-norm ranking, preserving the largest individual teacher contributions even when correlated. Keep the retained construction otherwise unchanged.

Config: `{"basis": "pool", "mode": "shared", "profile": "upright", "ranking": "weight", "scale": 1.5}`.

Training resets: fwd 5141.9, bwd 7219.9; mean retained 0.8841. **KEPT**; prior best 0.8815.

### Call 46: seed 3, magnitude_rank

Replace diversity-based pivoted QR pruning with direct activation-standard-deviation times output-weight-norm ranking, preserving the largest individual teacher contributions even when correlated. Keep the retained construction otherwise unchanged.

Config: `{"branches": [{"mode": "shared", "profile": "transition", "ranking": "weight", "scale": 1.5}, {"mode": "shared", "profile": "upright", "ranking": "weight", "scale": 1.5, "weighted3": true}], "mode": "splice"}`.

Training resets: fwd 2160.8, bwd 7184.6; mean retained 0.6548. **REJECTED**; prior best 0.9382.

### Call 47: seed 2, pool_splice

Combine seed 2 magnitude-ranked forward branch (5141.9 return) with its QR-ranked backward branch (7503.2); both use exactly the same pooled first-layer basis. Evaluate the combined float32 network because earlier splices showed numerically sensitive rollouts.

Config: `{"branches": [{"basis": "pool", "mode": "shared", "profile": "upright", "ranking": "weight", "scale": 1.5}, {"basis": "pool", "mode": "shared", "profile": "upright", "scale": 1.5}], "mode": "splice"}`.

Training resets: fwd 5280.9, bwd 7491.0; mean retained 0.9124. **KEPT**; prior best 0.8841.

### Call 48: seed 0, integration64k

Increase deterministic synthetic quadrature from 16384 to 65536 inputs per task, retaining the current architecture, priors, and pruning rule. More samples reduce finite-sample error in the single closed-form regressions; this is not an iterative optimizer.

Config: `{"mix": 1.0, "mode": "shared", "n": 65536, "profile": "upright", "scale": 1.5}`.

Training resets: fwd 4724.4, bwd 6958.7; mean retained 0.9096. **REJECTED**; prior best 0.9174.

### Call 49: seed 2, integration64k

Increase deterministic synthetic quadrature from 16384 to 65536 inputs per task, retaining the current architecture, priors, and pruning rule. More samples reduce finite-sample error in the single closed-form regressions; this is not an iterative optimizer.

Config: `{"branches": [{"basis": "pool", "mode": "shared", "n": 65536, "profile": "upright", "ranking": "weight", "scale": 1.5}, {"basis": "pool", "mode": "shared", "n": 65536, "profile": "upright", "scale": 1.5}], "mode": "splice"}`.

Training resets: fwd 5180.5, bwd 7553.5; mean retained 0.9071. **REJECTED**; prior best 0.9124.

### Call 50: seed 3, integration64k

Increase deterministic synthetic quadrature from 16384 to 65536 inputs per task, retaining the current architecture, priors, and pruning rule. More samples reduce finite-sample error in the single closed-form regressions; this is not an iterative optimizer.

Config: `{"branches": [{"mode": "shared", "n": 65536, "profile": "transition", "scale": 1.5}, {"mode": "shared", "n": 65536, "profile": "upright", "scale": 1.5, "weighted3": true}], "mode": "splice"}`.

Training resets: fwd 5036.6, bwd 4734.4; mean retained 0.7694. **REJECTED**; prior best 0.9382.

## Final construction

All indices below are zero-based. The delivered architecture is exactly 19 -> tanh(128) -> tanh(128) -> tanh(6), with the required seven float32 arrays. There is one ordinary network per seed; no external router, ensemble, additional runtime state, or normalizer modification. `log_std` is copied from the seed's forward specialist and is unused in evaluation.

First-layer units 0..126 are shared observation features. Unit 127 has zero observation weights and task weights [3,-3], producing g = +tanh(3) for fwd and -tanh(3) for bwd. The shared units have zero explicit bias; their task columns contain the corresponding specialist's active effective bias b1 + W1[:,17+task]. Thus task-specific thresholds remain available in the shared features.

The observation-feature bases differ by seed:

- Seed 0 uses 127 forward-specialist directions unchanged. Rank original rows by the sum, over the two specialists, of the column norms of W3 @ W2; discard the weakest row. The backward branch is approximated in this forward basis.
- Seeds 1 and 3 use the elementwise average of the aligned forward/backward observation rows, with the same importance ranking and one discarded row. Alignment uses their inherited shared-start neuron indices, without learning a permutation.
- Seed 2 takes 127 directions from the union of all 256 forward/backward rows. For both tasks, evaluate this pool on 2048 synthetic inputs; concatenate the activation matrices, center columns, scale by the same per-original-row influence, and use pivoted QR to select a diverse weighted feature basis. Each chosen row retains the effective bias of its original row index for each task.

Second-layer units 0..62 implement 63 forward nonlinear features; unit 63 supplies a forward intercept. Units 64..126 implement 63 backward nonlinear features; unit 127 supplies a backward intercept. Every nonlinear feature's preactivation is a closed-form approximation to a selected specialist second-layer preactivation. For an active branch the intended preactivation is A @ h_shared + b. Write d = 12 + abs(b) + sum(abs(A)). The forward gate coefficient is -d/(2*tanh(3)) and bias is b+d/2; backward uses the opposite gate sign. Inactive preactivation is therefore at least 12 for every shared activation in [-1,1], so tanh rounds to exactly +1 in float32. Intercept units use the same gate with A=0, b=0, d=12: intended activation is 0 when active and +1 when inactive.

Each active branch predicts specialist output logits as U @ h_selected + c. Its intercept-unit output weight is -c - sum_columns(U). The network output bias is c_fwd+c_bwd. An inactive branch then contributes -c, which its part of the global bias cancels; an active branch contributes its fitted specialist logits. The final tanh is unchanged. Routing cancellation is exact in real arithmetic; float32 introduces small rounding differences, discussed below.

## Exact closed-form fitting and synthetic inputs

No environment observations, trajectories, dynamics, gradient optimizers, or other attempts' artifacts were used. Every candidate is either a supplied specialist, a stated analytic compression, or a compatible branch splice. Fitting always targets the supplied specialist of the same training seed and label.

All retained branches use N=16384 inputs per task. The independent numpy RNG seed is 70231 + 17*seed + task. Start with 17 independent standard-normal coordinates. Multiply by the branch scale; for mixture branches also multiply each row by a randomly selected member of [0.5,1,1.5] before applying the scale. Default velocity is x[8] += +3 for forward or -3 for backward. Transition/upright profiles replace x[8] with task-sign times Uniform(-0.5,6). The upright profile additionally multiplies x[0] by 0.6 and sets x[1] = 0.25*x[1] - norm_mean[1]/sqrt(norm_var[1]), concentrating raw root pitch near zero using only the supplied normalizer. Finally clip all coordinates to [-10,10]. These are generated synthetic priors, not inferred state measurements.

For each task, compute the specialist first hidden activations, second preactivations Z, second hidden activations, and output logits L. Compute the shared first hidden activations H. Fit [H,1] C2 = Z by one ridge linear solve, with objective mean squared residual plus 1e-5 times squared non-intercept coefficients; leave the intercept unpenalized. Evaluate tanh([H,1] C2), then select 63 columns. Usually selection is pivoted QR on the first 4096 centered activation rows, scaled per column by the norm of that specialist's original output-weight column. Seed 2's forward branch instead ranks activation standard deviation times original output-weight norm; its backward branch still uses QR. Finally fit the specialist logits L from the selected 63 nonlinear features and an intercept using another closed-form ridge solve with the same regularization.

Seed 3's backward output fit uses one solve per output, with sample weights 0.1 + (1 - tanh(L)^2)^2, emphasizing logit accuracy where the final action is sensitive. No iterative reweighting is performed. The rejected hidden-sensitivity experiment likewise made a single teacher-weighted replacement solve after feature selection, not a gradient or iterative optimization loop.

Retained branch settings:

| Seed | Forward branch | Backward branch |
|---|---|---|
| 0 | Forward-only observation basis; upright, scale 1.5; QR | Same basis and prior; QR |
| 1 | Averaged basis; normal, scale 1; QR | Same basis; normal mixture, scale 1.5; QR |
| 2 | Pooled basis; upright, scale 1.5; magnitude ranking | Same basis and prior; QR |
| 3 | Averaged basis; transition, scale 1.5; QR | Same basis; upright, scale 1.5; QR; sensitivity-weighted output fit |

Splicing copies the chosen second-layer and output-weight block under an identical first layer and recombines the two fitted intercepts. All final settings are embedded as `FINAL_CONFIGS` in `build.py`; the default rebuild reads only `wsp/data/` weights and does not depend on selected.json, history.json, candidate files, or evaluator results.

## Measured training results and limitations

Exactly 50/50 evaluator calls were used: 300000 environment steps, all through the supplied evaluator, on its three fixed training reset seeds per task. There were no other environment interactions. Calls per seed are 15, 4, 16, 15. Every call's rationale, config, measured score, and keep/reject decision appears above; full per-reset outputs are in the evaluator-owned ledger and the preserved evaluation receipts. A candidate replaced its seed's incumbent only on strictly higher mean retained score. Rejected files are retained as audit evidence and are not deliverables.

| Seed | Final call | Fwd return | Bwd return | Fwd mean x velocity | Bwd mean x velocity | Mean retained |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 41 | 4822.2 | 6929.6 | 5.229 | -7.333 | 91.74% |
| 1 | 26 | 4995.6 | 7390.3 | 5.350 | -7.799 | 109.00% |
| 2 | 47 | 5280.9 | 7491.0 | 5.619 | -7.917 | 91.24% |
| 3 | 28 | 5149.7 | 7243.9 | 5.463 | -7.607 | 93.82% |

Mean over the four seeds: **96.45% retained on training resets**. The measured forward-specialist-used-for-both-labels starting baseline averaged 55.23% under exactly the same reset bank and deterministic action scoring. Final policies have positive forward velocity and negative backward velocity on those resets. Seed 1 exceeds 100% because retained return is a ratio, not a bounded percentage; that is a training-bank result, not a held-out claim.

The cost of the construction is approximation error from a shared first feature basis and roughly half-width task-specific second layers. Increasing the backward capacity to 71 units hurt the forward task enough to lose overall. Adding near-linear coordinates did not help. More synthetic samples did not improve any incumbent in the final three calls. Synthetic action MSE alone did not reliably predict locomotion return.

A notable failure: the first shared construction for seed 3 scored only 30.42% (call 8), below its 65.14% baseline, and was rejected. Upright synthetic posture coverage repaired much of its backward failure. Another important limitation is numerical sensitivity: for seeds 0 and 2, nominally isolated branch splices changed actions on 10000 synthetic samples by at most about 1e-6, yet substantially changed long-rollout scores. Those splices were evaluated and rejected. This is consistent with small floating-point perturbations producing divergent rollouts; the experiment does not establish which particular dynamics amplify them. No claim of numerically identical per-task trajectories is made.

Training-reset selection over 50 calls can overfit the reset bank. **Held-out reset performance: not measured.** No held-out resets or earlier attempts were inspected. Synthetic verification is not a substitute for held-out locomotion evaluation.

## Rebuild and verification

From the project root `/Users/danielmercado/ProjectJoy/25_skill_transplantation`:

```sh
/Users/danielmercado/ProjectJoy/.venv/bin/python wsp/attempts/r6wsp_b50_a/build.py
/Users/danielmercado/ProjectJoy/.venv/bin/python wsp/attempts/r6wsp_b50_a/verify.py
```

The builder fixes BLAS/thread environment limits to 4 before importing numpy. It is deterministic on this installation and uses only the original data weights plus constants embedded in its source. No evaluation calls are required for rebuilding or verification. `verify.py` reads this attempt's saved receipts for audit checks, rebuilds each array, and uses synthetic forward passes only. It runs in a few seconds.

Verification passed: exact required keys, float32 dtypes, shapes, finite weights, bounded finite actions, bit-for-bit array reconstruction, and final NPZ SHA-256 matching the exact evaluated candidate. The tool-owned ledger contains exactly calls 1..50. Each final candidate is the highest measured mean-retained candidate for its seed. On 32768 fresh synthetic samples per task (RNG 918273 + 2*seed + task, upright scale-1.5 transition distribution), every inactive second-layer activation equals +1 exactly. Analytic box bounds exceed 11.99 on off-task preactivations after float32 conversion. These fresh synthetic errors were measured after final selection and were not used for choosing candidates.

| Seed | Fresh synthetic fwd action MSE | Fresh synthetic bwd action MSE | Final NPZ SHA-256 |
|---|---:|---:|---|
| 0 | 0.00713189 | 0.01612691 | `ca42faadc054f35b288e3badb160445e981839c459e8b0ec4b52dba127e3ab3c` |
| 1 | 0.01008397 | 0.00872701 | `ca58d48dbf00c7fe50d2b4adafae1b0d4ad5b4516ccfbc83e6fc021b6f8bee59` |
| 2 | 0.01024553 | 0.00794345 | `b708608fd91f67bdb2363a1548aa71fb00828d6666bad3681ef1a62da0b1303c` |
| 3 | 0.00858930 | 0.00915198 | `78632a56262e8944688ff8e887f14a828ce1962bbb84ecdac1b097ae25beff37` |

Artifacts: `merged_s0.npz` through `merged_s3.npz` are the final scored networks; `build.py` is the self-contained constructor; `NOTES.md` is this rationale and complete call log. `verify.py` and `validation.json` contain synthetic-only verification; `experiment.py`, `selected.json`, `history.json`, named candidate NPZs and evaluation receipts preserve the experiment. `ledger.jsonl` was written only by the supplied evaluator and was never edited.

FINAL: merged_s0..3.npz
