# Low-grip weight-space programming

Started 2026-11-04 15:39 CST. Wall-time limit approximately 16:39 CST.

## Rules and method

All driving uses the supplied `../eval_train.py`, at most 25 calls, with
`NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4`. No simulator imports, reimplementation,
training, evolution, random search, or automated simulator optimisation. All
candidates start from this folder's `start.npz`. No other policy artifacts read.
Forward passes and analytic probes use NumPy only. Unit indices are zero-based.

## Initial mechanism hypothesis

At fixed path curvature, lateral acceleration scales with speed squared; halving
grip suggests reducing speed by sqrt(0.5). The existing speed input has substantial
connections to h1 units 25 (-1.0291), 31 (+0.9045), 28 (+0.8728), 29 (+0.8137),
19 (-0.7910), 2 (-0.6795), 18 (+0.6647), 1 (-0.6426), 17 (-0.6424), and 11
(-0.6016). These are mixed features, not pure speed detectors. The second layer
has strong negative pedal readout through h2 units 5, 7, 4, 8, 26, and 18.
Probes will determine how these paths respond to speed and approaching walls.

An initial targeted intervention is to multiply only W1[10,:] by sqrt(2),
making the original policy perceive a higher speed at the same physical speed.
It changes 32 of 1,570 scalars, preserving all ray and action-history weights.
This is a hypothesis, not an assumption of exact physical similarity: timestep,
acceleration, drag, and recurrent previous-action inputs complicate the scaling.

## Evaluation ledger

| Call | Candidate / edit from start | mu=0.5 finishes | Progress | Lap s | mu=1 finishes | Lap s | Eligible |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | `start.npz`, unchanged | 1/32 | 0.267298 | 28.3000 | 32/32 | 25.3875 | yes |
| 2 | `c02_speed_sqrt2.npz`, W1 speed row x sqrt(2) | 32/32 | 1.000000 | 32.9125 | 27/32 | 29.9037 | no |
| 3 | `c03_speed_sqrt2_steer_half.npz`, speed x sqrt(2), steering logits x 0.5 | 32/32 | 1.000000 | 33.0531 | 31/32 | 29.5742 | yes |
| 4 | `c04_speed_sqrt2_steer_quarter.npz`, speed x sqrt(2), steering logits x 0.25 | 22/32 | 0.866090 | 33.2136 | 20/32 | 30.2150 | no |
| 5 | `c05_speed135_steer_half.npz`, speed x 1.35, steering logits x 0.5 | 30/32 | 0.988991 | 33.0133 | 31/32 | 28.6806 | yes |
| 6 | `c06_side_half.npz`, call-3 transformation plus virtual side contrast gain 0.5 | 31/32 | 0.995798 | 33.4065 | 31/32 | 30.5355 | yes |
| 7 | `c07_speed_adaptive_steering.npz`, speed x sqrt(2), explicit speed-adaptive steering readout | 32/32 | 1.000000 | 33.0125 | 32/32 | 29.5594 | yes |
| 8 | `c08_speed135_adaptive_steering.npz`, speed x 1.35, same adaptive steering target | 32/32 | 1.000000 | 33.0469 | 30/32 | 28.3167 | yes |

Lap means include finishers only. These are development evaluation tracks, not
the unseen judging tracks. Every full response is retained as `eval_*.json`.

Call 1 traces show already nearly maximal braking and steering before the wall;
late additional output force cannot recover the lost stopping/cornering distance.
Call 2 solves low-grip completion but has dry-grip inside-wall crashes: the car
is slower yet still commands large steering. The higher dry-grip yaw limit then
permits a tighter turn. Next test halves the steering output logits to reduce
over-turning and feedback oscillation, retaining the physics-derived speed scale.

Call 3 meets the constraint, with only a full-grip inside-left-wall failure on
track 26. At observed speeds 12-15 m/s it commands roughly 0.25-0.67 steering.
The algebraic half-grip steering saturation threshold is only about 0.12-0.20.
Call 4 checked whether further attenuation addresses the surplus. It fails:
traces show outside-wall collisions even at walking speed, where weakened
steering cannot recover tight corners. Retain this failure evidence.

Call 5 restored half steering and tested speed factor 1.35. Quadratic drag
remains present at both grips, so the braking-force ratio exceeds 0.5; a smaller
speed correction might preserve completion and improve dry pace. It instead
lost two low-grip finishes (tracks 12, 13); restore sqrt(2) for the next mechanism.

Call 6 hypothesis: add side-clearance feedback via the already strong oblique-ray
contrast. In the call-3 dry failure, steering-logit derivatives for right/left
90-degree rays were +0.19 to +0.33 and -0.02 to -0.05 (unhelpful for centering),
whereas left/right 18-degree rays were about +1.3/-1.0. Set
`delta = 0.5*(left90-right90)`, and make the first layer see
`left18 += delta; right18 -= delta`. This is exactly a two-row weight edit:
`W1[9] += 0.5*(W1[6]-W1[3]); W1[0] -= 0.5*(W1[6]-W1[3])`.
It should reduce inside-wall over-turning and increase outside-wall recovery,
without weakening low-speed turning globally. All weights still derive from start.

Call 6 reduces full-grip near-wall time from 0.0571 to 0.0168, but loses the tight
right turn on track 13 at both grips. Altering the path also alters the required
low-speed steering. It does not beat call 3.

Call 7 instead restores steering authority specifically at low speeds. The target
is `steer_logit = original_rescaled_logit * (0.5 + 0.5*clip(1-v/15,0,1))`.
At 15 m/s and above this is half gain; at zero speed it is full gain. The network
does not acquire a wrapper: `fit_steering.py` solves one regularized linear least
squares system for the existing 32 steering output weights and bias, using only
1,155 handmade ray/speed/previous-action combinations. The features are the
unchanged second-layer activations after the speed-row edit. Regularization is
0.001 toward the half-gain readout. No fitting uses simulator traces or scores.
Another 360 handmade interpolated observations have steering-action RMSE 0.0520
against the explicitly specified function (fit RMSE 0.0619); it is an approximation.
This candidate finishes all 32 at both grips. It changes only 65 scalars.

Call 8 combined the previously tested 1.35 speed scale with this low-speed
steering restoration: call 5's failed traces were at low speed and large heading
error, where the new readout is intended to improve recovery. It refits the same
explicit function on the same handmade grid under the new speed scale. It restores
32/32 low-grip completion, but is 0.0344 s slower on average than call 7 and loses
two full-grip finishes (tracks 26, 30). Call 7 remains the evaluator-selected best.

## Weight-only probes

`probe_weights.py` uses 20 explicitly handmade observations: straight, left/right
gentle/tight ray patterns, speeds 9/15/21/27 m/s, previous commands zero. Exact
chain-rule logit sensitivities are in `probe_start.txt`. This is network analysis,
not driving or a simulator surrogate.

Mean contributions to d(pedal logit)/d(normalized speed) are h1[31] -1.4152,
h1[25] -1.1484, h1[19] -0.9743, h1[28] -0.9551, h1[17] -0.9087,
h1[29] -0.8716. Every one is negative over these probes. The dominant h2
speed-to-brake paths are h2[5] -1.4030, h2[7] -1.2607, h2[4] -0.9686,
h2[8] -0.7487, h2[18] -0.7031, and h2[26] -0.4885. Thus the input
rescaling reuses an existing distributed speed-versus-free-space computation.
Some paths also affect steering: h1[25] mean steer-logit sensitivity +0.5145,
h1[19] -0.3019, h1[31] -0.1892. Unit functions are mixed and input-dependent;
these local derivatives do not establish a globally isolated semantic neuron.

## Final result

Stopped after **8 of 25 calls**; **17 calls remain**. The evaluator's `best.npz`
is exactly equal, array by array, to `c07_speed_adaptive_steering.npz`.
SHA-256 of best.npz:
`e58d10ebf8d559c3b5e98164cc60e74d79983e34cb4cc964a02312aa7c17d4a1`.

**Best call 7:** low grip 32/32, mean progress 1.0, mean lap 33.0125 s;
full grip 32/32, mean progress 1.0, mean lap 29.559375 s. Baseline was 1/32,
mean progress 0.267298 at low grip, and 32/32, 25.3875 s at full grip.
The baseline low-grip lap mean (28.3 s) is only one finisher and is not a matched
speed comparison. Full-grip mean lap time increases by 4.171875 s (16.43%).

### Final mechanism and scope

The original policy sees adequate wall clearance at speeds that were safe on dry
grip and brakes too late for half grip. Once understeering toward the outside wall,
its pedal and steering already saturate; stronger last-moment commands cannot fix
this. Multiplying W1[10,:] by sqrt(2) reuses the existing mixed speed/clearance
features to trigger braking at lower physical speeds. However, dry-grip turning
at those lower speeds can become too tight. Halving steering logits addresses
much of this, but tight low-speed corners still need steering authority. The final
readout approximates a steering gain that rises from 0.5 at >=15 m/s to 1.0 at rest.
This is an ordinary 13->32->32->2 network, with no extra controller or grip input.

The final edit changes **65/1,570 scalars = 4.140127%**: 32 speed-input weights,
32 steering-output weights, and the steering-output bias. All other 1,505 scalars
are bitwise unchanged. It is a targeted input/readout edit, not a hidden-layer
rewrite. All h1/h2 weights and biases other than the speed row retain their learned
values. The largest adaptive steering-readout additions relative to the half-gain
readout use h2[19] (-0.448185), h2[29] (-0.408188), h2[3] (-0.381345),
h2[17] (+0.328208), h2[18] (-0.328143), h2[11] (-0.318327), h2[27]
(+0.302490), and h2[23] (-0.301851). These mixed features jointly approximate
the specified speed-dependent function; no claim is made that a single unit
implements it in isolation.

### Verification and limitations

`verification.json` records shapes, finite float32 values, exact changed-scalar
counts, best/call-7 equality, exact reconstruction from start, and evaluator budget.
The reconstruction initially differed by up to 1.19e-7 in the speed row due to
NumPy float64 versus Python-scalar float32 promotion. `fit_steering.py` now exposes
that precision explicitly and reproduces call 7 bit for bit. Evaluated candidates
and evaluator-owned files were not modified to resolve this. Reconstruction and
all probes run only network arithmetic, consuming no driving calls.

All evaluated candidates and failures are retained. The 32 tracks are development
evaluation tracks reused under the stated budget. Unseen judging-track performance
is **not measured**. No maximum-speed or optimal-lap-time claim is made. The
speed-dependent readout is approximate, and full-grip near-wall time (0.057184)
is higher than baseline (0.006170), despite full completion. The final solution
has no online grip detection and pays a dry-track speed cost. Additional simulator
fine-tuning was stopped after reaching 32/32 at both grips with a compact edit.

### Reproduction without additional driving calls

From this folder, rebuild the best weights under a fresh filename:

```sh
OMP_NUM_THREADS=4 /Users/danielmercado/ProjectJoy/.venv/bin/python fit_steering.py --output rebuilt_call07.npz
```

`probe_weights.py` reproduces the initial handmade-observation analysis. Its output
is retained in `probe_start.txt`. `probe_adaptive_steering.json` and
`probe_adaptive_135.json` retain fit diagnostics. To reproduce call 8's exact
arithmetic, supply `--speed-scale 1.35 --speed-scale-precision float32` and a fresh
output filename. Calls 2-6 are reproduced by `edit_weights.py`, with the speed,
steering, and side-ray gains in the evaluation ledger/narrative. This script always
loads `start.npz`; it never loads another policy or calls the simulator.

The only permitted driving command remains:

```sh
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 /Users/danielmercado/ProjectJoy/.venv/bin/python ../eval_train.py --attempt_dir . --weights c07_speed_adaptive_steering.npz --note "Explicit repeat of call 7; consumes one of the 17 remaining calls"
```

It was not run again during verification. `best.npz` is owned by the evaluator;
the final verification only reads it.
