# One-shot weight-space edit

## Scope and result

`final.npz` is the single submitted policy, derived from `start.npz`. It contains
exactly `W1`, `b1`, `W2`, `b2`, `W3`, and `b3`, with the required shapes and float32
dtype. **394 of 1,570 scalar parameters changed: 25.0955%.** The other 1,176 are
bitwise unchanged. This is a substantial circuit edit: four first-layer units
and two second-layer units are repurposed, the pedal readout is replaced, and
the steering readout is damped and recentered.

**No simulator was imported or run. No car was driven, no tracks were generated,
and no driving evaluation was performed.** I read the permitted `racing/sim.py`
and `racing/track.py` source to understand the observation convention, grip,
collision rules, and curvature limits. I did not read another policy, results,
logs, media, other workspace folders, or the project's Markdown write-ups.
There was no training, gradient descent, random search, or optimization loop.
All probes used literal, hand-made sensor vectors and independent combinations
of input values. They are not driving evidence.

## Diagnosis: distributed full throttle, with useful steering still present

The strongest evidence is an almost unconditional accelerator. For the literal
centered-straight vector in `inspect_weights.py`, at 20 m/s and previous pedal
+1, the original pedal logit is **+5.0193**, giving pedal **+0.99991**. On the
hand-made `sharp_left` vector at the same speed, it is **+4.5083**, giving
**+0.99976**. Even at 30 m/s or with the near-left-wall vector, it stays close to
full throttle. The output bias alone is +0.6815.

Several second-layer units add positively to this pedal logit despite different
activation signs. At the centered-straight probe:

| Original unit | Activation | Pedal weight | Pedal-logit contribution |
|---|---:|---:|---:|
| h2[8] | -0.9755 | -0.6133 | +0.5983 |
| h2[10] | +0.8900 | +0.4683 | +0.4168 |
| h2[27] | -0.8070 | -0.4865 | +0.3926 |
| h2[23] | -0.8004 | -0.3810 | +0.3049 |

The local derivative of pedal logit with respect to normalized speed is
**positive**, approximately +0.1859 on that straight probe and +0.3640 on the
sharp-left probe. Thus the original circuit does not supply the braking needed
to respect the yaw-rate limit on these inputs. Saturation also makes small
readout corrections ineffective.

The steering circuit still uses useful directional information. For example,
h1[27] strongly contrasts left/right forward clearance: its input weights are
+0.9945 on left18, +0.3389 on left6, -0.3247 on right18, and -0.5711 on right6.
h2[13], h2[31], and h2[12] have substantial steering readout weights of -0.6660,
-0.6895, and +0.5480. A near-left-wall straight input produces a strong right
command, so these probes do not support a simple wall-attraction explanation.

My mechanism hypothesis is **saturated acceleration combined with steering
that becomes grip-limited**, followed by the soft-wall heading correction
described in TASK.md. The useful steering features remain, but the accelerator
fails to slow enough for those features to control the car legally. This is an
inference from the weights, static probes, and task description; it is not a
newly observed trajectory or a claim that one unit explicitly detects contact.

## The programmed circuit

All unit indices below are zero-based. Distances `d` are in meters, and `v` is in
meters/second. Inputs are converted from their /60 and /30 normalizations by the
weights themselves.

### 1. Preserve a consistent context for the retained steering features

For every retained first-layer unit, fold its original previous-pedal input
weight into its bias and set that input weight to zero. This makes its computation
exactly equal to the original unit with `prev_pedal = +1`, independent of actual
braking. The old accelerator nearly always supplied that context. The new brake
controller therefore cannot perturb steering through an unfamiliar previous-
pedal input. The final policy ignores observation 12 completely.

### 2. Reuse four features with small steering impact on the static probes

I replace h1[23], h1[29], h1[14], and h1[6]. The old h1[23] is mostly positive
and h1[29] mostly negative in the full-throttle context: their previous-pedal
weights are +0.6069 and -0.7617, respectively. Their activations, and those of
the other two units, have small effects on steering in the chosen probes.

Before disconnecting their old outgoing edges, fold their mean old contributions
into b2. The means come from 240 explicit hand-made input combinations:
8 ray vectors, their mirrors, 5 speeds, and 3 previous-steer values. No fitting
or search is used; this is arithmetic mean compensation.

| Reused h1 unit | Old mean activation | New role |
|---|---:|---|
| 23 | +0.8613976104 | right6 clearance/speed comparator |
| 29 | -0.9293991361 | left6 clearance/speed comparator |
| 14 | +0.3222478418 | absolute speed comparator |
| 6 | +0.2665312747 | previous-steer feature |

On those 240 inputs, the retained steering logit **before deliberate damping and
recentering** differs from the original by RMSE **0.0078611**, maximum absolute
error **0.0193378**. This only measures feature-preservation error on these
synthetic inputs; it does not measure final controller quality.

### 3. Replace full throttle with a clearance-based governor

Let:

```
width_proxy = d_right90 + d_left90
T_R = d_right6 - 1.5 - 0.5 * (width_proxy - 12)
T_L = d_left6  - 1.5 - 0.5 * (width_proxy - 12)
T_C = 26

h1[23] = tanh(0.35 * (T_R - v))
h1[29] = tanh(0.35 * (T_L - v))
h1[14] = tanh(0.35 * (T_C - v))
h2[23] = tanh(h1[23] + h1[29] + h1[14] - 2)
pedal = tanh(2.5 * h2[23])
```

The old pedal readout is zeroed; only h2[23] drives the new pedal. Its old
steering weight was approximately -0.0004 and is zeroed as well.

This is a smooth AND of three speed margins. If speed reaches or exceeds **any**
of the three targets, that comparator is nonpositive and each other comparator
is at most +1. Therefore the sum minus 2 is nonpositive, so **the pedal cannot
be positive**. This property follows algebraically, not from a driving test.
When several limits are near-active, it brakes earlier. Drag makes the eventual
speed lower than the zero-pedal target.

The 1.5 m allowance accounts for the 1 m car radius plus a small clearance
allowance. The modest width adjustment prevents a wider corridor from increasing
the speed target quite as much at the same corner geometry. These targets are
heuristics, not estimates of exact track curvature or a formal collision barrier.
The 26 m/s cap limits straight-line speed before unseen bends. At rest with clear
forward rays, pedal is about +0.9566. Maximum negative pedal is about -0.9866.

### 4. Dampen steering without deleting its learned corner features

The old immediate ray response is very strong. Reducing speed changes how much
steering the grip limit suppresses, so retaining that entire instantaneous gain
could create oscillation. Program h1[6] and h2[16] as:

```
h1[6] = tanh(0.25 * previous_steer)
h2[16] = tanh(h1[6])
```

h2[16]'s original steering output weight was only about -0.0011. Its former
incoming column is replaced. The retained steering logit is recentered by
subtracting **0.1689742597**, its value on the symmetric straight input at 16 m/s
with previous steer zero. The new steering output is:

```
steer = tanh(0.25 * (retained_steering_logit - 0.1689742597)
             + 3.2 * h2[16])
```

The memory term is approximately `0.8 * previous_steer` for modest steering
values. Sustained turn commands can build up while abrupt corrections are
smaller. This is not an exact linear low-pass filter and it does alter large
steady steering commands. Added lag and possible corner-cutting remain risks.

## Static checks and concrete examples

`static_checks.json` records the hashes, parameter changes, and hand-made probes.
The saved file was reloaded and checked for exactly the required keys, shapes,
float32 dtype, finite entries, and identical array contents.

On 1,323 independent static input combinations, the network pedal agrees with
the programmed formula to maximum absolute error **9.54e-7**. The checks also
confirm nonpositive pedal above any target, monotonic pedal response to speed
and forward clearance on the grid, and complete independence from previous
pedal. There are no vehicle states or trajectory updates in these checks.

Examples below use previous steer 0 and previous pedal +1. Steering therefore
shows the first filtered response, not a sustained-turn equilibrium.

| Literal input | Speed | Original (steer, pedal) | Edited (steer, pedal) |
|---|---:|---:|---:|
| centered straight | 20 | (+0.1527, +0.9999) | (-0.0038, +0.9538) |
| centered straight | 30 | (+0.1106, +0.9999) | (-0.0157, -0.9439) |
| sharp left | 20 | (+0.7737, +0.9998) | (+0.2101, -0.9714) |
| very sharp left | 20 | (+0.5874, +0.9996) | (+0.1240, -0.9862) |
| straight, 1.2 m left ray | 12 | (-0.9643, +0.9997) | (-0.4951, -0.8760) |

Changed scalar counts:

| Array | Changed | Total |
|---|---:|---:|
| W1 | 80 | 416 |
| b1 | 32 | 32 |
| W2 | 184 | 1,024 |
| b2 | 32 | 32 |
| W3 | 64 | 64 |
| b3 | 2 | 2 |
| **Total** | **394** | **1,570** |

Source SHA-256:
`50f875b604d183c9b6d000a790bc7d1f5d0bd551a49d6ad627307ed68a941231`

Final SHA-256:
`9f3891391b8d989f826e85171358c1a3b8cf5115b93010d2b2eacee839610ae3`

## Prediction and unmeasured quantities

My low-confidence point prediction is **28/32 strict completions**. I expect a
large improvement over the supplied 0/32 baseline because the main unsafe
full-throttle behavior is removed and useful steering features are retained.
This prediction is judgment, not a measured result or a calibrated probability.
I cannot establish that the car never touches a wall without driving it.

Strict completion, lap time, wall-contact count, near-wall fraction, and actual
speed trajectories are all **not measured**. I expect slower laps than the
soft-wall policy; the cautious governor may also be slower than the strict-trained
reference described in TASK.md. Abrupt curvature changes, delayed steering,
unusual sensor configurations, and a braking-induced stop with no reverse are
remaining failure modes. Static checks cannot resolve them.

## Reproduction

From this folder, reconstruct exactly the submitted arrays and repeat only the
static checks with:

```
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /Users/danielmercado/ProjectJoy/.venv/bin/python edit_weights.py
```

`inspect_weights.py` contains the literal probes and original-network analysis;
`edit_weights.py` contains the deterministic edit and static artifact checks.
Neither imports or calls the simulator. No evaluation command is provided or run.
