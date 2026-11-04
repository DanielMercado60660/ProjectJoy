# Low-grip weight-space programming

## Outcome

The evaluator automatically selected **call 4**, `c04_clearance_speed135.npz`, as `best.npz`. On the supplied 32 tracks it finished **32/32 at mu=0.5** (mean lap **33.19375 s**, mean progress **1.0**) and **32/32 at mu=1.0** (mean lap **29.0625 s**, mean progress **1.0**). Exactly **96/1,570 parameters (6.11465%)** differ from `start.npz`, all in three rows of W1. All biases and all later-layer weights are unchanged. Used **4 of 25** evaluation calls; stopped after achieving full completion at both grips and one modest speed-relaxation check. This is a targeted input-pathway edit, not a larger policy rewrite.

Results are measured on the supplied evaluation tracks only. Unseen-track performance is **not measured**. The task supplied the original policy's 1/32 low-grip completion, approximately 27% mean progress, and 32/32 full-grip completion with a 25.4 s mean lap; I did not spend a call reevaluating that baseline. Final full-grip laps are about 3.66 s slower than that supplied baseline.

## Method and constraints

All candidates were constructed directly from `start.npz` with explicit algebraic edits. I read `racing/sim.py`, `wsp_common.py`, and `eval_train.py` to understand physics, inference, and the evaluator's contract. I did not import simulator code, reproduce the simulator, train a model, run gradient descent, use evolution or random search, or automate a candidate-selection loop. Weight probes use specified observation vectors and analytical derivatives only. No other policy's artifacts were opened. No installs, network requests, or git commands were used. All writes were inside this attempt folder. Protected evaluator files were only written by the supplied evaluator.

Every driving call used this exact form, with the relevant candidate filename and an edit-specific note:

```sh
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 /Users/danielmercado/ProjectJoy/.venv/bin/python ../eval_train.py --attempt_dir . --weights c04_clearance_speed135.npz --note '<edit explanation>'
```

The original stdout of each call is preserved in `call01.json` through `call04.json`, including the evaluator's available crash traces. Each evaluated candidate is retained as its own named NPZ, in addition to the evaluator-managed archives.

## What appears to limit the original policy

At half grip, both available braking acceleration and lateral acceleration are halved. Approximately preserving a path requires speed scaled by sqrt(0.5). The learned policy has no grip input and carries its full-grip speed/braking calibration into low grip. The initial edit was therefore to multiply the speed-input weights by sqrt(2), exactly making the unchanged network perceive a higher speed.

The network's braking computation is distributed over mixed speed/visibility units. In the original first layer, several units oppose speed and forward visibility:

| h1 unit | Speed weight | Sum of -6 and +6 degree ray weights | Interpretation of this part of the preactivation |
|---|---:|---:|---|
| 31 | +0.90454 | -1.51300 | Higher speed and shorter forward visibility raise the unit |
| 28 | +0.87284 | -0.82589 | Same speed-versus-visibility direction |
| 25 | -1.02911 | +0.69955 | Complementary direction: more visibility and less speed raise it |
| 19 | -0.79097 | +0.64716 | Complementary speed/visibility contribution, also previous-pedal dependent |
| 17 | -0.64244 | +0.75716 | Complementary speed/visibility contribution |
| 29 | +0.81367 | -0.05091 | Speed and command-history mixed feature; previous-pedal weight -0.58082 |

These are descriptions of the indicated terms, not claims that a unit implements only one concept. All units also combine other rays and previous actions.

For hand-specified left-bend rays `[.10,.12,.16,.21,.27,.40,.50,.25,.14,.11]`, with both previous commands zero, the original pedal changes from **+0.7269 at 15 m/s** to **-0.8790 at 20 m/s**. Reversing those ten rays gives **+0.6939** and **-0.8940**. These are synthetic weight probes, not simulator observations.

At the left-bend 15 m/s probe, h2 units 5, 7, 4, 18, 8, and 26 contribute approximately -2.9843, -2.1271, -1.4504, -1.2584, -1.0961, and -0.8096 to the derivative of pedal preactivation with respect to normalized speed. Their respective pedal output weights are -1.11847, -0.90438, -0.81104, -0.54912, -0.66518, and -0.54924. Higher speed raises these h2 preactivations and therefore suppresses throttle. This supports preserving the existing braking computation and recalibrating its speed input.

Call 1 confirmed that this simple recalibration can fix low-grip completion, but exposed a second coupling: at the reduced speed, full grip permits more yaw/curvature before saturation. Its full-grip traces show the car steering toward the shrinking inside-wall clearance. Weakening all steering in call 2 caused poor heading recovery at low speeds. That failure is retained, not discarded from the record.

For call 1's full-grip crash observations, the derivative of steering preactivation with respect to `ray[6]` (+18 degrees, left) is approximately +2 to +2.8, while that for `ray[3]` (-18 degrees, right) is approximately -1.4 to -2.2. Direct side-ray responses are weak and locally have the wrong sign for clearance correction: a larger right-side distance can increase left steering even when the left wall is close. This suggested routing the side-distance difference through the existing middle-ray steering pathway. This is local diagnostic evidence; it is not a global monotonicity proof or a complete causal decomposition of the network.

## Every edit and result

All indices below are zero-based. Each candidate is rebuilt from `start.npz`.

1. **`c01_speed_sqrt2.npz`**: `W1[10,:] *= sqrt(2)`. Changes 32 parameters. Physical hypothesis: lower the speed at which the existing braking circuit reacts by approximately sqrt(0.5). Result: all low-grip tracks finish, but five full-grip tracks fail, violating the full-grip constraint.
2. **`c02_speed_steer_quarter.npz`**: keep the sqrt(2) speed edit; also multiply `W3[:,0]` and `b3[0]` by 0.25. Changes 65 parameters. Hypothesis: smaller steering requests prevent full-grip inner-wall cutting at reduced speed. The gain approximated half-grip feasible commands of about 0.11-0.20 versus trace requests around 0.5-0.7 at 12-16 m/s. Result: worse at both grips. Crash traces include large heading errors and nearly stopped cars with inadequate corrective steering; shrinking the output also changes the previous-steer feedback distribution. Rejected.
3. **`c03_speed_wall_feedback.npz`**: restore the original steering head, retain sqrt(2) speed scaling, and set `delta = 0.75*(W1[6,:]-W1[3,:]); W1[9,:] += delta; W1[0,:] -= delta`. Changes 96 parameters. Hypothesis: retain steering authority and give the middle-ray steering computation explicit wall-clearance feedback. Result: 32/32 at both grips. Low-grip mean near-wall fraction falls from 0.00453551 in call 1 to 0.00023855; full-grip fraction falls from 0.08595098 to 0.01139046. Automatically retained as best at that point.
4. **`c04_clearance_speed135.npz`**: same lateral correction as call 3; change perceived-speed gain from sqrt(2) to 1.35. Changes 96 parameters. Hypothesis: substantially reduced near-wall exposure may permit a modest speed relaxation (nominally 4.76% more speed, 9.74% greater lateral acceleration demand). Result: 32/32 at both grips. Low-grip time improves by only 0.015625 s; full-grip time improves by 0.91875 s. This is a measured local tradeoff, not evidence that the gain is optimal. Evaluator automatically selected it as final best.

Lap means below include finishers only; rejected candidates therefore have different finishing subsets and their mean times are not directly comparable as matched full-set lap performance.

| Call | Changed / 1570 | mu=.5 finished | mu=.5 mean progress | mu=.5 mean lap (s) | mu=1 finished | mu=1 mean progress | mu=1 mean lap (s) | Constraint met |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 32 | 32/32 | 1.00000000 | 32.912500 | 27/32 | 0.89687069 | 29.903704 | no |
| 2 | 65 | 22/32 | 0.86608970 | 33.213636 | 20/32 | 0.80710499 | 30.215000 | no |
| 3 | 96 | 32/32 | 1.00000000 | 33.209375 | 32/32 | 1.00000000 | 29.981250 | yes |
| 4 | 96 | 32/32 | 1.00000000 | 33.193750 | 32/32 | 1.00000000 | 29.062500 | yes |

## Final computation

Only the following three W1 rows change:

```python
W1[10, :] = 1.35 * start_W1[10, :]
delta = 0.75 * (start_W1[6, :] - start_W1[3, :])
W1[9, :] = start_W1[9, :] + delta
W1[0, :] = start_W1[0, :] - delta
```

This is algebraically equivalent (up to float32 rounding) to running the original network on these virtual inputs:

```text
virtual_speed = 1.35 * speed
clearance_difference = left_side_ray - right_side_ray
virtual_left_18deg_ray = left_18deg_ray + 0.75 * clearance_difference
virtual_right_18deg_ray = right_18deg_ray - 0.75 * clearance_difference
```

The remaining inputs are unchanged. When the left wall is closer, the clearance difference is negative: the existing steering pathway receives a smaller left ray and a larger right ray, encouraging steering away from that wall. The opposite correction applies near the right wall. Both grips receive the same policy; it does not detect grip, access hidden simulator state, or branch on a track identifier. These virtual features are linear network inputs and are not clipped ray measurements.

## Verification and artifacts

`verify_weights.py` reconstructs all four documented candidates in memory from `start.npz` and checks exact equality against each saved candidate, array shapes, float32 dtype, finite values, parameter-change counts, evaluator call accounting, and equality of the final candidate with evaluator-managed `best.npz`. It also verifies the virtual-input identity over 315 deterministic hand-made observations covering both turn directions, near walls, seven speeds, and previous-command extremes. This audit drives no cars and consumes no evaluation calls.

```sh
/Users/danielmercado/ProjectJoy/.venv/bin/python verify_weights.py
```

Audit passed. Maximum first-layer preactivation discrepancy from the equivalent virtual-input formulation: **7.40091e-8**. `verification.json` records the full audit, SHA-256 checksums, and final summary metrics. `best.npz` SHA-256 is `44105385cbf10e3050d199ae95ab1151f9984bfbaf057d757cc544e7c267dda9`.

The unsuccessful experiments and their traces remain in `c01_speed_sqrt2.npz`, `c02_speed_steer_quarter.npz`, `call01.json`, and `call02.json`; no failure clips were generated because no extra driving/rendering was authorized by the task. The full successful raw evaluations are `call03.json` and `call04.json`.

No additional simulator calls are planned. The smallest next external check is the orchestrator's unseen-track evaluation. The main remaining uncertainty is generalization: side-ray correction can interact with geometry and heading outside the observed/synthetic cases, and a 1.35 speed gain is slightly less conservative than the physical sqrt(2) starting hypothesis.
