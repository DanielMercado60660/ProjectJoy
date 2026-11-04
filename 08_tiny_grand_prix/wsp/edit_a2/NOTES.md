# Direct weight editing experiment

Start: 2026-11-04 15:25 CST. Deadline: approximately 16:25 CST.

## Constraints and method

All candidates start from this folder's `start.npz`. Only the mandated `../eval_train.py` command drives the car. Maximum 25 calls. No simulator imports, simulator reimplementation, gradient descent, evolution, random search, other policies, or project write-ups. Weight algebra and deterministic hand-made observation probes are used to reason about edits. Evaluator stdout is retained as `callNN.json`.

## Initial hypothesis

The original policy's failure may be excessive speed and throttle persistence rather than deliberately steering toward walls: under the documented grip cap, corrective steering cannot rescue an excessively fast corner entry. This is a hypothesis to distinguish with unit contributions, input sensitivities, and crash observations. The dense network does not justify labeling a single unit a wall-riding unit without evidence.

## Evaluation ledger

| Call | Candidate / edit | Strict finish | Mean progress | Strict lap, finishers | Soft wall steps/lap |
|---|---|---:|---:|---:|---:|
| 1 | `start.npz`, unchanged baseline | 0/32 | 0.232425 | n/a | 19.34375 |
| 2 | `candidate02.npz`, replace 33 pedal readout values with least-squares approximation to `0.25*(7+27*mean(ray[-6],ray[+6])-30*speed)` on deterministic handmade observations | 30/32 | 0.967913 | 28.10333 s | 0.0625 |
| 3 | `candidate03.npz`, same target; pin internal previous pedal to +1 by folding W1 row12 into b1; 97 changes | 27/32 | 0.918826 | 27.21852 s | 0.1875 |
| 4 | `candidate04.npz`, call2 plus steer-logit correction `0.8*(r+35-r-35)+0.4*(r+60-r-60)+(r+90-r-90)`; 66 changes | 32/32 | 1.0 | 28.203125 s | 0 |

## Mechanism evidence and decisions through call 4

- Original baseline track 0, steps 51–65: speed rises from 25.86 to 30.15 m/s; pedal stays 0.999–1.000. Steering is already positive (left), but yaw is grip-capped. A larger steering command cannot substitute for braking.
- On a hand-made centered straight at 25 m/s and previous pedal +1, the pedal logit is +5.0463 (action about 0.999917). The largest positive hidden contributions include h2[8] (+0.5980), h2[10] (+0.4185), h2[27] (+0.4039), h2[23] (+0.3125), plus b3[1]=+0.6815. This is a distributed saturated accelerator, not evidence of one discrete wall-riding unit.
- Speed derivative of the original pedal logit is positive, +0.1395 per normalized speed unit at that probe, rather than regulating speed downward. Previous-pedal derivative is also positive, +0.2541 (and much larger at previous pedal 0 or -1).
- Original steering-logit derivatives are concentrated at +/-6 and +/-18 degree rays: about -1.273,+1.292 and -1.612,+1.939. The +/-35 and +/-60 rays have much weaker and asymmetric influence. Call2 failures include low-speed corners where the useful opening has moved into these wider rays, and discontinuous front-ray openings can briefly reverse steering.
- Call3 demonstrated that freezing the previous-pedal feature does not help: more low-speed corner crashes. It is retained as an informative failure, not selected.
- Call4 adds a symmetric preference for the wider side with more room, preserving both hidden layers and the original steering output as an additive component. All 32 laps finish with identical strict/soft lap statistics and zero soft-wall contacts.
- Least-squares targets are prescribed functions of handmade static inputs, never simulator actions/rewards. `edit_readout.py` and `edit_steer.py` contain the complete deterministic construction; they do not launch evaluations. This is direct output-weight programming, not gradient training or a simulator search loop.


## Calls 5–8: failure diagnosis and explicit governor

| Call | Candidate / edit | Strict finish | Mean progress | Strict lap, finishers | Soft wall steps/lap |
|---|---|---:|---:|---:|---:|
| 5 | `candidate05.npz`: call4 steering, speed target raised from `7+27*f` to `9+27*f` | 30/32 | 0.983332 | 26.91000 s | 0.0625 |
| 6 | `candidate06.npz`: same prescribed functions, expand hand-made lattice with single peripheral-ray openings to 1.0 | 32/32 | 1.0 | 27.13125 s | 0 |
| 7 | `candidate07.npz`: exact linear speed-error circuit in h1[14] and h2[16], target `9+27*f`; preserve steering through a deterministic solve | 29/32 | 0.967045 | 27.17241 s | 0.09375 |
| 8 | `candidate08.npz`: exact curved target `32*tanh(1.8*f)`, remove 9 m/s floor, increase braking feedback; 220 values changed | 32/32 | 1.0 | 26.715625 s | 0 |

Here `f=(obs[4]+obs[5])/2` is normalized forward-ray mean; speed is in m/s.

Call5 failures included spurious acceleration as a +/-60-degree ray suddenly opened to 60 m. These observations were used to diagnose an uncovered input regime, not as regression training data. Call6 expands the deterministic handmade lattice to 35,721 observations and eliminates these failures, although the prescribed pedal function remains approximate.

For call7, h1 units were ranked on a handmade lattice by steering-logit change when clamped to their mean. h1[14] had the smallest RMS change, 0.00833 (max 0.04228). h2[16] originally has steering output weight about -0.0011. Its existing pedal coefficient is -0.2924. Repurposing these units isolates an exact speed-control circuit with minimal structural steering disturbance. Other h2 biases absorb the removed h1 mean; the final steering readout approximates original steering plus the same broad-ray correction, excluding h2[16]. The pedal law is exact to float32 precision, but its 9 m/s floor makes braking too weak late in tight corners. Three crashes remain even while steering strongly into the corner.

Call8 also repurposes h1[28] (original clamp RMS steering change 0.01719) as a near-linear speed channel. The actual implemented circuit is:

- h1[14] = tanh(1.8*f)
- h1[28] = tanh(0.1*obs[10])
- h2[16] = tanh(4.8*h1[14] - 45*h1[28])
- pedal = tanh(3*h2[16])

Equivalently, target speed is `32*tanh(1.8*f)` and measured speed is approximated by `300*tanh(v/300)`. This tends toward zero target speed at zero forward clearance and saturates on open straights. Feedback logit gain near zero error is 0.45 per m/s. Compared with call7, it has no fixed unsafe speed floor and stronger braking feedback. No environment or dynamics is implemented in these scripts.

Call8 changes 220/1570 values (14.0127%), a larger targeted circuit rewrite than the 66-value readout edits; it is not presented as a tiny ablation. It finishes all 32 tracks, with 0 soft wall contacts and near-wall time 0.272608%. Its maximum pedal discrepancy from the prescribed analytic circuit over the handmade lattice is 1.71e-7. Steering-logit approximation RMS is 0.07228, max 0.37973 on the expanded lattice.

The `pedal05.npz` and `pedal06.npz` files are intermediate readout-only construction artifacts, not additional evaluations. Candidate metadata JSON files record every construction parameter. The local probe inspection that initially tried to index an empty soft crash list raised an IndexError after printing the baseline; it did not drive the car or consume an evaluation.

## Final calls and result

| Call | Candidate / edit | Strict finish | Mean progress | Strict lap, finishers | Soft wall steps/lap |
|---|---|---:|---:|---:|---:|
| 9 | `candidate09.npz`: target `36*tanh(1.6*f)`; keep low-clearance slope `36*1.6=57.6` and identical steering | 32/32 | 1.0 | 26.284375 s | 0 |
| 10 | `candidate10.npz`: target `40*tanh(1.44*f)`; same low-clearance slope, physical top-speed ceiling | 32/32 | 1.0 | 26.112500 s | 0 |

**Stopped after 10/25 evaluation calls. 15 calls remain unused.** Each numbered receipt was checked against its reported call number. The evaluator-selected `best.npz` matches `candidate10.npz` exactly in all six arrays. All evaluated candidates remain present, including the unchanged `start.npz` baseline and the failed edits. `best.*`, the evaluator, budget config, evaluator log and archived candidates were not modified by our scripts.

The final strict result is 32/32 completions, mean progress 1.0, mean lap time 26.1125 s. Soft rules give the same lap time, 32/32 completions, and 0.0 wall-contact steps per lap. Near-wall time is 0.0612266% in both modes. These are results on the 32 permitted evaluation tracks only. Unseen-track performance is not measured; a global fastest legal policy is not established. The reference policies and their quoted lap times come from TASK.md, not from any inspection or run of another policy.

### Original mechanism: a distributed accelerator, with inadequate corner recovery

The original steering generally turns toward an opening, but its accelerator stays saturated while grip caps the available yaw. It uses wall contact to recover the path and heading without paying the braking cost. Strong positive previous-pedal paths help sustain the accelerator: for example, prev_pedal -> h1[29] -> h2[8] -> pedal has weight signs (-,+,-), with product +0.22947 before tanh derivatives. The corresponding paths through h1[20] and h1[23] have positive products +0.16939 and +0.08591. All tanh derivatives are nonnegative, so these specific paths reinforce the previous pedal. h2[8] is strongly negative at the straight probe and its negative output weight makes a large positive pedal contribution; h2[10], h2[27], and h2[23] also contribute strongly to the saturated throttle. The hidden units mix many inputs, so these are identified signal paths and measured probe contributions, not claims that one unit alone encodes intentional wall riding.

The baseline's weak peripheral steering then causes a second failure mode when slower, braking-aware driving puts the useful corner opening outside the central rays. The additive wider-angle and side-clearance correction handles that regime.

### Final circuit and extent of rewrite

The final pedal computation is explicitly programmed:

```
f = (obs[4] + obs[5]) / 2
h1[14] = tanh(1.44 * f)
h1[28] = tanh(0.1 * obs[10])
h2[16] = tanh(6.0 * h1[14] - 45.0 * h1[28])
pedal = tanh(3.0 * h2[16])
```

Thus the target speed is `40*tanh(1.44*f)` m/s, compared against `300*tanh(v/300)` m/s, an almost linear measurement of speed. At the maximum sensor reading the target is about 35.75 m/s. The target tends to zero when forward clearance vanishes, and pedal decreases monotonically with speed for any fixed ray input. Previous pedal has no path into this new pedal circuit.

The old outgoing connections of h1[14] and h1[28] are removed, with their average contribution folded into h2 biases. Their incoming weights are replaced with the clearance and speed functions. h2[16] becomes the speed-error unit and is excluded from steering. The other steering features remain derived from the starting policy. The steering readout is fitted by a single linear solve to the starting steering logit plus `0.8*(r+35-r-35)+0.4*(r+60-r-60)+(r+90-r-90)` on handmade static observations. This is a larger targeted circuit rewrite, not a claim that the original network is wholly preserved.

| Array | Values changed |
|---|---:|
| W1 | 26 |
| b1 | 2 |
| W2 | 94 |
| b2 | 32 |
| W3 | 64 |
| b3 | 2 |
| **Total** | **220 / 1570 = 14.01273885%** |

All values are finite float32. A static 35,721-observation check matches the specified final pedal equation to maximum absolute error 1.43e-7, and a fixed-ray speed sweep confirms strictly decreasing pedal. These are network checks, not additional simulator evaluations. `verification.json` records the receipt checks, array equality, parameter counts, and best-file SHA-256.

### Reproduction and retained files

Construct the winning weights from the untouched start using only CPU weight algebra:

```sh
OMP_NUM_THREADS=4 /Users/danielmercado/ProjectJoy/.venv/bin/python edit_curved_governor.py reproduced10.npz --top-speed 40 --front-slope 1.44
```

If an additional evaluation is desired, the only permitted driving command remains the following, which consumes one of the 15 remaining calls:

```sh
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 /Users/danielmercado/ProjectJoy/.venv/bin/python ../eval_train.py --attempt_dir . --weights candidate10.npz --note "Recheck final call10 candidate"
```

- `start.npz` / `start.txt`: supplied baseline, unchanged.
- `candidate02.npz` through `candidate10.npz`: every edited candidate evaluated.
- `call01.json` through `call10.json`: complete evaluator stdout, including failure observations/actions.
- `candidate*.npz.json`, `pedal*.npz.json`: construction parameters and static approximation errors.
- `probe.py`, `edit_readout.py`, `edit_steer.py`, `edit_governor.py`, `edit_curved_governor.py`: deterministic inspection/construction tools; no simulator imports or evaluation loops.
- `pedal05.npz`, `pedal06.npz`: unevaluated construction intermediates.
- `best.npz`: evaluator-managed selected artifact; never manually written.
- `verification.json`: final structural and receipt verification.
- `STATUS.md`: milestones with timestamps taken from receipt modification times.

No further simulator calls are planned. The clean 32/32 result, diminishing improvement from raising only the open-road ceiling, and preserved failure receipts provide a concrete stopping point without spending the remaining budget on parameter search.
