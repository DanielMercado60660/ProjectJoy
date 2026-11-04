# Weight-space programming notes

Started 2026-11-04 at 15:25 America/Chicago. Deadline approximately 16:25.

## Constraints and method

All candidates start from `start.npz`. No training, gradients, evolution, random
search, simulator imports, simulator reimplementation, or driving outside the
budgeted evaluator. Numpy probes use hand-made observations or the evaluator's
reported crash observations. Only this directory is writable. The evaluator
alone manages its budget, archives, and `best.*` files. Maximum 25 calls.

## Initial mechanism hypothesis

The trained network has distributed features rather than an obvious isolated
wall-contact switch. Its pedal output has a substantial positive bias (+0.6815).
The physics caps yaw rate at 9.81 / speed, so excessive corner-entry speed can
make correct-direction steering ineffective. First distinguish excessive speed
from a steering defect using one baseline call, hidden-unit probes, and local
input sensitivities. Unit numbering is zero-based throughout.

## Evaluation ledger

| Call | File / explicit edit | Strict finishes | Strict mean progress | Mean lap (finishers) | Soft contacts/lap |
|---|---|---:|---:|---:|---:|
| 1 | `start.npz`, unchanged diagnostic baseline | 0/32 | 0.232425 | n/a | 19.34375 |
| 2 | `c02_range_speed.npz`: h1[23]/h2[28] speed controller, target `11 + 20*mean(ray4,ray5)` m/s; freeze internal prev_pedal=1 | 26/32 | 0.900338 | 27.71923 s | 0.34375 |
| 3 | `c03_sharp_brake.npz`: target `7 + 26*mean(ray4,ray5)`; input speed-error gain .20 instead of .15 | 27/32 | 0.918500 | 27.68519 s | 0.25 |
| 4 | `c04_wider_steer.npz`: call 3 plus `W1[2] += .5*W1[3]`, `W1[7] += .5*W1[6]`, steer bias -= .08 | 32/32 | 1.000000 | 28.003125 s | 0 |
| 5 | `c05_faster_envelope.npz`: target `9 + 27*mean(ray4,ray5)` on call 4's corrected steering | 30/32 | 0.972476 | 26.90000 s | 0.0625 |
| 6 | `c06_clear_road.npz`: call 4 lower envelope + h1[29] smooth clear-road bonus, stronger feedback | 32/32 | 1.000000 | 26.58125 s | 0 |
| 7 | `c07_firmer_feedback.npz`: pedal output gain 2 -> 3, same target speed | 32/32 | 1.000000 | 26.321875 s | 0 |
| 8 | `c08_medium_visibility.npz`: bonus midpoint at 24 m instead of 30 m | 32/32 | 1.000000 | 26.140625 s | 0 |
| 9 | `c09_symmetric_readout.npz`: closed-form steering-readout correction toward reflection symmetry | 32/32 | 1.000000 | 26.146875 s | 0 |
| 10 | `c10_broad_bend_speed.npz`: clear-road bonus coefficient .45 -> .60 | 32/32 | 1.000000 | 25.896875 s | 0 |
| 11 | `c11_drag_feedforward.npz`: readout adds an approximation of `.8*speed_obs-.25` to the pedal logit | 32/32 | 1.000000 | 25.803125 s | 0 |
| 12 | `c12_compact_readout.npz`: restore 60 original W1 entries; re-express the call-11 steering function and drag term in the readout | 32/32 | 1.000000 | **25.775000 s** | **0** |

Call 1: reported crash traces have pedal logits 3.938–4.924 (almost full
throttle). Mean positive pedal-logit contributions include h2[8] +0.595,
h2[10] +0.404, h2[23] +0.327 and h2[27] +0.291. h2[8] is strongly negative
(-0.970 mean) with a negative pedal output weight (-0.6133); h2[10] is
positive (+0.862) with positive weight (+0.4683). Together with many other
positive terms and b3[1]=+0.6815, these saturate the throttle. On a hand-made
mean observation, the pedal-logit speed sensitivity is positive (+0.2923),
not the negative feedback needed to brake. Steering generally points into
the corner, but the simulator's grip limit prevents the requested yaw rate.

Call 2 replaces h1[23], whose variation has the lowest RMS linearized
steering effect (0.0027) across the specified probes, and h2[28], whose steer
output weight is only +0.0087. Absorb their mean contributions into downstream
biases. Folding prev_pedal=1 into b1 prevents the new pedal outputs from
perturbing the original steering representation. This changes 206/1570 scalars;
steering differs by RMS 0.002883, max 0.01438, on 441 hand-made probes.
The six remaining failures include sharp right turns where the target speed
stays above 11 m/s even when the forward range collapses. At first the network
also under-commands steering while the forward-side ray becomes discontinuous.
Next: lower the short-range speed envelope and strengthen the proportional
speed response while preserving long-range speed.

Call 3 proves speed reduction alone is insufficient: the remaining crash traces
reach only 8.5 m/s at impact yet still understeer. In the right turn on track 12,
the right 35-degree ray opens to .371 while steer is only -.071. Original
steering-logit sensitivities to the +/-18-degree rays are about -1.80/+2.17,
but the +/-35-degree pair is about +.114/+.053 (wrong sign on the right and
little directional contrast). This is a second part of the wall-dependent
behavior: the policy relies on narrow-ray transitions and eventual wall snaps
instead of sufficiently following a tight bend.

Call 4 propagates half of the 35-degree rays through the learned 18-degree
input weights. This is an input-feature edit, retaining both learned layers.
A -.08 steering-logit bias correction reduces the hand-probed left bias
(straight steering was +.10 at 21 m/s). On the same track-12 observation,
steer becomes -.278. All 32 finish, zero soft-wall contacts too, and only
1.9734% near-wall time. Next: test a higher corner-speed envelope now that the
wide-ray steering defect is corrected.

Call 5's two failures (tracks 12 and 13) still ask for enough steering to hit
the grip limit; the higher short-range speed floor is the problem. It is not
useful to turn the wheel harder there. Call 6 restores call 4's low-range
envelope and reuses another nearly constant, low-steering-influence unit,
h1[29], for a smooth long-range speed bonus. Its old mean contribution is
absorbed into b2, just as for h1[23]. With `f=(ray4+ray5)/2`, its pedal path is:

```
h1[23] = tanh(.1*(7 + 26*f - 30*speed_obs))
h1[29] = tanh(6*f - 3)
h2[28] = tanh(3*h1[23] + .45*(1+h1[29]))
pedal = tanh(2*h2[28])
```

This preserves the short-range envelope while adding up to about 3.1 m/s
when the forward rays are long. It also increases proportional feedback from
approximately .4 to .6 per m/s near target. Call 7 changes the final 2 to 3:
maximum throttle rises from .964 to .995, while small-error feedback becomes
.9 per m/s (local braking-loop factor 12*.1*.9=1.08, below the scalar stability
boundary 2). This calculation is a local controller check, not a simulation.

Call 8 changes h1[29]'s bias from -3 to -2.4. At a 24 m forward range the
small-angle corridor estimate gives a substantially broader bend than in
the failing short-range cases. The bonus raises the target there from about
18.1 to 18.9 m/s, still conservative relative to that approximate geometry.
It leaves the very-short-range bonus small. Call 8 has zero contacts and
1.1685% near-wall time. Calls 6–8 change 309/1570 scalars (19.6815%); the
architecture and most learned steering weights remain intact.

Call 9 addresses the remaining reflection asymmetry. For a hand-made sensor
vector x, reflect the rays and reverse prev_steer to obtain M(x). Specify
the desired steering logit as `(z(x)-z(M(x)))/2`, using call 8 as the source
function. A closed-form regularized linear solve changes only the steering
readout, keeping h2[28] disconnected from steering. This uses 2,268 hand-made
vectors; a separate 560-vector hand-made grid reduces symmetry-target logit
RMSE from .09853 to .03798. This is a function-approximation probe, not a
held-out driving test. The call is 0.00625 s slower than call 8 and is not
selected as best, but near-wall time drops from 1.1685% to .6192%.

Call 10 increases the bonus on long forward rays using the symmetry-corrected
steering. At very short range the sigmoid bonus remains small; no high
minimum corner speed is reintroduced. All 32 finish at 25.896875 s.

Call 11 compensates for quadratic drag. The physics needs pedal `v^2/1600`
to balance drag. The linear logit surrogate `.8*speed_obs-.25` approximates
the required pedal logit around the main 15–25 m/s operating band. A
closed-form solve expresses it through the existing h2 features (excluding
the programmed speed unit), using 2,646 hand-made inputs and a separate
672-vector validation grid. Validation logit RMSE is .02352, max .08657.
This preserves the programmed negative speed feedback while reducing its
steady tracking error. All 32 finish; near-wall time falls to .3113%.

Call 12 compiles the wide-ray steering behavior into the output readout.
Restore the original W1 rows for inputs 2 and 7, except the two repurposed
units (60 scalars restored). Fit the readout to call 11's steering function
on the same hand-made grid, and re-express the same drag surrogate in the
restored features. Validation steering-logit RMSE versus call 11 is .01397,
maximum .04350. Pedal-logit RMSE versus call 11 is .00659. All 32 finish at
25.775 s, with .6686% near-wall time. This becomes the evaluator's best.

### Proposals inspected without driving

- An unregularized symmetry solve on the initial 441 probes had small fitting
  error but output coefficients as large as 5.82; it was not evaluated. The
  broader, regularized call-9 solve was used instead.
- A quadratic drag-logit target `.65*speed_obs^2` was probed before call 11.
  Its validation RMSE .07182 and maximum error .20686 were too large; it was
  not evaluated. The linear surrogate was substantially more accurate in the
  available feature basis. No simulator call was used for either proposal.

## Final mechanism and scope of change

The source's wall riding comes from nearly saturated positive throttle
through corners, combined with weak sharp-turn steering cues in the wider
rays. Soft walls supply the heading correction and retain speed, so these
defects are rewarded. In strict mode the car runs outside the track. The
baseline traces, pedal-logit decomposition, and failure of speed-only edits
support this explanation; no single hidden unit is claimed to be a unique
"wall-riding neuron."

The final policy is a targeted partial rewrite. h1[23] computes speed error,
h1[29] computes a smooth forward-visibility bonus, and h2[28] combines them
for negative-feedback throttle/braking. Their final exact programmed path is:

```
f = (obs[4] + obs[5]) / 2
h1[23] = tanh(.1*(7 + 26*f - 30*obs[10]))
h1[29] = tanh(6*f - 2.4)
h2[28] = tanh(3*h1[23] + .6*(1 + h1[29]))
pedal = tanh(3*h2[28] + q(other_h2))
```

`q` is the fitted drag-logit surrogate. The other first-layer units have
prev_pedal folded to 1 internally, preserving their familiar representation
while actual throttle/brake commands now vary. Original near-constant feature
contributions were absorbed into downstream biases. The steering readout
preserves the learned driving function with stronger wide-angle response and
reduced reflection bias. Both output readouts are revised; this is disclosed
in the scalar count rather than described as an isolated ablation.

| Array | Changed float32 scalars |
|---|---:|
| W1 | 56 |
| b1 | 32 |
| W2 | 94 |
| b2 | 32 |
| W3 | 64 |
| b3 | 2 |
| **Total** | **280 / 1,570 = 17.8343949%** |

1,290 scalars remain bitwise unchanged. Final candidate:
`c12_compact_readout.npz`, automatically selected as `best.npz` by call 12.
SHA-256 prefix from the evaluator: `42667c1662bd`.

## Final measured result and verification

- **12 of 25 calls used**, including the unchanged baseline; 13 calls remain.
- Strict, grip 1.0, same 32 evaluator tracks as the baseline: **32/32 finishes,
  1.0 mean progress, 25.775 s mean lap, zero crashes**. Baseline: 0/32 finishes,
  .232425 mean progress.
- Soft-wall diagnostic: **32/32 finishes, 25.775 s mean lap, zero wall-contact
  steps on every track**, versus baseline 19.34375 contact steps/lap.
- Final near-wall fraction: .0066864 (about .669%) in both modes.
- Unseen-track completion, lap time and wall margin: **not measured**. The
  approximate controller times stated in TASK.md are supplied references,
  not separately reproduced measurements.
- Every evaluated candidate is retained at its ledger filename, and every
  raw evaluator response is retained as `callNN.json`. Evaluator-owned files
  were written only by the permitted evaluator.
- `verify_artifacts.py` checks all 12 candidate hashes against their receipts,
  tensor shapes/dtypes/finiteness, the automatic best selection, 441 bounded
  hand-made forward probes, and the exact changed-weight count. It also
  confirms all 32 strict finishes and all 32 zero-contact soft results.
- `reproduce_best.py` rebuilds call 12 from `start.npz` using only the disclosed
  weight arithmetic and closed-form hand-made-input solves. Reproduction was
  **bit-for-bit equal in every float32 array**. `verification.json` records
  the checks, timestamps, full hashes and results. No extra driving was used.

Reproduce and check without spending an evaluation call, from this folder:

```sh
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /Users/danielmercado/ProjectJoy/.venv/bin/python reproduce_best.py
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /Users/danielmercado/ProjectJoy/.venv/bin/python verify_artifacts.py
```

The reproduction script writes `reproduced_best.npz`; it does not overwrite
the evaluated candidates or any evaluator-owned state. Work stopped after
successful verification at 12 calls, within the 60-minute allowance.
