# One-shot weight edit

`final.npz` is the single submitted policy, derived from `start.npz`. All six arrays
have the required shapes and float32 dtype. **No simulator, trajectory, driving,
track generation, or evaluation was run.** Completion, contacts, and lap time are
not measured. The numerical checks below concern isolated, invented observation
vectors and network algebra only.

## Mechanism inferred from the original weights

The clearest defect is a distributed, saturated throttle circuit, rather than a
single identifiable wall-contact trigger. On the authored straight observation at
25 m/s, previous steer 0 and previous pedal +1, the pedal logit is **5.0463** and
the pedal is **0.999917**. A tighter left-turn example still produces logit
**4.4202**, pedal **0.999710**. Steering points into the bend, but there is almost
no speed reduction. Under the documented grip limit, more steering cannot recover
the missing cornering acceleration. The soft-wall correction can then supply
heading alignment and confinement that this controller fails to achieve itself.
That interpretation is consistent with TASK.md; it is not a trajectory-derived
causal diagnosis.

Units are zero-indexed. Several h2 units add positive pedal logit even when their
activations are negative. For the straight example above:

| h2 unit | Activation | Pedal weight | Pedal-logit contribution |
|---|---:|---:|---:|
| 8 | -0.9751 | -0.6133 | +0.5980 |
| 10 | +0.8936 | +0.4683 | +0.4185 |
| 27 | -0.8301 | -0.4865 | +0.4039 |
| 23 | -0.8202 | -0.3810 | +0.3125 |
| 16 | -0.8480 | -0.2924 | +0.2479 |

The output bias adds another **+0.6815**. Other units reinforce these terms.
These are mixed features, not cleanly separable semantic detectors. In the tight
turn, units 8 and 10 still contribute about +0.5994 and +0.4128. The original
pedal-logit derivative with respect to normalized speed is even positive in the
straight and tight-turn probes (about +0.1395 and +0.3244). Prior throttle also
reinforces the pedal logit locally. A small output-bias reduction would leave the
missing negative speed feedback unresolved.

Useful steering structure is present. For example, h1[27] receives -0.5711 from
the right 6-degree ray, +0.3389 from the left 6-degree ray, -0.3247 from the right
18-degree ray, and +0.9945 from the left 18-degree ray. It forms an opening
contrast favoring the left side; its local influence on the steering logit is
positive in the inspected examples. h1[5] has the opposite forward-ray contrast
(+0.4947 right, -0.2991 left) and negative local steering influence. All first-layer
features and essentially the entire steering circuit are retained.

## Edit: two explicit visibility/speed margins

I repurposed **h2[16] and h2[23]**. Their original steering weights are only
**-0.00112950** and **-0.00043589**, so they are good places to put a new pedal
circuit without disturbing the learned directional response.

Let `d_right` and `d_left` be the two 6-degree ray distances in meters, and let
`v` be speed in m/s. The specified replacement functions are:

```text
m_right = 0.15 * (8 + 0.4*d_right - v)
m_left  = 0.15 * (8 + 0.4*d_left  - v)

h2[16] approximately tanh(m_right)
h2[23] approximately tanh(m_left)
pedal = tanh(2 * (h2[16] + h2[23] - 1))
```

I solved one linear least-squares system mapping the **unchanged h1 features**
to these two specified affine margins, then wrote its coefficients into the two
W2 columns and b2 entries. This is a direct function compilation, with no gradient
descent, evolutionary search, simulator objective, or learned driving labels.
The 65,536 input vectors are deterministic mixtures and perturbations of authored
ray tables, speed values, and independent previous controls. They are not track
observations. The two fits were solved together once.

The sum of the two tanh margins provides a smooth conservative combination:
both directions need sufficient visibility for strong throttle. One short ray
can cause braking even if the other ray remains long. Equal distances `d` give
an ideal zero-pedal speed of **4.338 + 0.4*d m/s**, before drag and approximation
error. Thus 57.4 m visibility corresponds to about 27.3 m/s, and 20 m to about
12.3 m/s. When one ray is very long, the shorter ray's zero-pedal speed tends
toward `8 + 0.4*d_short`. These constants were chosen analytically for a cautious
one-shot edit; they were not tuned against driving outcomes. Maximum throttle
is tanh(2), approximately 0.964; strong braking approaches -1.

All original W3 pedal weights are zeroed, except these two new weights of +2.
The pedal bias becomes -2. This removes the accumulated throttle bias instead of
trying to subtract one contributor from a saturated sum.

I also zeroed the two reused units' steering weights. For **any identical input**,
the remaining steering logit is exactly the original minus those two original
terms. Since tanh is 1-Lipschitz, the absolute steering-action change is globally
bounded by **0.001565394**. This bound does not claim that closed-loop steering
will be identical: speed and previous-pedal observations will change while driving.

## Exact change count

| Array | Changed / total |
|---|---:|
| W1 | 0 / 416 |
| b1 | 0 / 32 |
| W2 | 64 / 1,024 |
| b2 | 2 / 32 |
| W3 | 34 / 64 |
| b3 | 1 / 2 |
| **Total** | **101 / 1,570 = 6.433121%** |

This is a replacement of the pedal controller using the original representation,
not a full-network rewrite. **1,469 parameters remain bit-for-bit unchanged.**
There was one saved edited policy and no performance-based candidate selection.

## Static evidence and limits

Representative isolated inputs, with previous steer and previous pedal both zero:

| Authored input | Speed (m/s) | Original pedal | Edited pedal |
|---|---:|---:|---:|
| Straight | 0 | +0.9995 | +0.9640 |
| Straight | 20 | +0.9997 | +0.9377 |
| Straight | 28 | +0.9998 | -0.3438 |
| Left medium | 15 | +0.9990 | -0.5999 |
| Left tight | 20 | +0.9984 | -0.9998 |
| Right tight | 20 | +0.9981 | -0.9998 |

On the synthetic fitting inputs, median margin approximation errors expressed as
speed are **0.212 / 0.206 m/s** for right/left; 99th percentiles are
**1.134 / 1.065 m/s**; maxima are **2.889 / 3.225 m/s**. These are in-sample
function-approximation errors, not generalization or racing scores. Both margins
have negative speed derivatives at every one of those inputs. The two derivative
ranges with respect to normalized speed are [-4.7613, -3.6694] and
[-4.8213, -3.4303], versus the specified -4.5.

A separate 1,650-vector static sweep over the authored tables, six speeds, and
five values of each previous control confirms positive pedal at rest, negative
pedal at 40 m/s, and a maximum same-input steering difference of **0.0014043**.
It does not advance a car or feed actions back into observations. File checks
confirm shapes, float32 dtype, finite values, and the exact unchanged slices.

**Prediction, not measurement:** approximately **30/32 strict completions**, with
low confidence and a broad plausible range of 24-32. I expect substantially less
understeer and wall dependence, at the cost of slower laps than a tuned strict
policy. I cannot guarantee even that range. The preserved steering may oscillate,
cut corners differently at lower speed, or depend on wall correction in ways the
static probes cannot reveal. Visibility is only a proxy for curvature and braking
distance; sudden bends, asymmetric openings, and out-of-domain hidden-feature
approximation errors remain risks. **Strict completion, lap time, and actual
wall-contact counts are all not measured.**

## Reproduction and files

Run inside this folder with the shared Python environment. These commands only
perform static weight algebra and file checks:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /Users/danielmercado/ProjectJoy/.venv/bin/python design_governor.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /Users/danielmercado/ProjectJoy/.venv/bin/python check_artifact.py
```

- `inspect_weights.py`: original-weight inspection and authored input tables.
- `design_governor.py`: deterministic direct edit; writes `final.npz`.
- `check_artifact.py`: array integrity and static algebra checks.
- `static_design.txt`: captured design and isolated-input output.
- `static_checks.json`: machine-readable counts, static diagnostics, and hashes.

SHA-256 of submitted `final.npz`:
`d30264369f5688a475cc1ef81a0d0f5f6778bc1308ec97515af70b63483750b9`

SHA-256 of unchanged `start.npz`:
`50f875b604d183c9b6d000a790bc7d1f5d0bd551a49d6ad627307ed68a941231`
