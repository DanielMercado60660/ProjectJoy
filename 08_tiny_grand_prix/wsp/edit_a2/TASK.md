# Task: edit a wall-riding racing policy's weights so it drives legally (weight-space programming)

You will change the weights of a tiny neural-network driving policy **directly**, by reasoning
about what its units compute, the way a programmer edits code. This is a test of
"weight-space programming": no gradient descent, no evolution, no random search, no automated
optimisation loop over the simulator. You get a small budget of simulator evaluations to check
your edits.

## The car and the world
* Top-down 2D race car on a closed track bounded by two walls. Track half-width 5-7 m; the car
  is a circle of radius 1 m. Tracks are smooth random loops 400-800 m long, corners both ways.
* Kinematic bicycle model, time step 0.1 s, wheelbase 2.6 m, max steering angle 0.45 rad.
  Throttle up to 6 m/s^2, brakes up to 12 m/s^2, quadratic drag (top speed ~40 m/s), no reverse.
  Grip limit (grip mu = 1.0 here): longitudinal acceleration capped at mu*9.81 m/s^2 and yaw
  rate at mu*9.81/v, so entering a corner too fast makes the car run wide (understeer).
* The physics source is in `/Users/danielmercado/ProjectJoy/08_tiny_grand_prix/racing/sim.py`
  (you may READ `racing/*.py` to understand the physics and observations).

## The policy
13 inputs -> 32 tanh (h1) -> 32 tanh (h2) -> 2 tanh outputs. Weights: `start.npz` in this
folder with arrays `W1 (13,32), b1 (32,), W2 (32,32), b2 (32,), W3 (32,2), b3 (2,)` (float32).
`h1 = tanh(obs @ W1 + b1); h2 = tanh(h1 @ W2 + b2); action = tanh(h2 @ W3 + b3)`.
`start.txt` is the same weights as labelled text. `../wsp_common.py` has load/save helpers.

Inputs (obs), in order:
| index | name | meaning |
|---|---|---|
| 0-9 | rays at -90, -60, -35, -18, -6, +6, +18, +35, +60, +90 deg | distance to the nearest wall along a ray from the car's heading, divided by 60 m (1.0 = no wall within 60 m). **Negative angles point to the car's RIGHT, positive to its LEFT.** |
| 10 | speed | speed / 30 m/s |
| 11 | prev_steer | steering command at the previous step, [-1, 1] |
| 12 | prev_pedal | pedal command at the previous step, [-1, 1] |

Outputs: `action[0]` = steer in [-1, 1] (x 0.45 rad), **positive = turn LEFT**;
`action[1]` = pedal in [-1, 1], positive = throttle, negative = brake. Episodes start at rest
on the centerline pointing along the track.

## What is wrong with this policy
It was trained by evolution strategies with **soft walls**: touching a wall did not end the
episode; the car was pushed back inside, its heading snapped to the wall direction, and it kept
97% of its along-wall speed. It learned to **wall-ride**: drive hard into the outside wall of
corners and let the wall steer it round. On the 32 evaluation tracks:
* soft-wall rules: finishes 32/32, mean lap 22.0 s, 19.3 wall-contact steps per lap, 30% of the
  time within 20% of the room to a wall;
* **strict rules (any wall contact = crash): finishes 0/32**, crashing after 23% of a lap on average.

For reference (not available to you): a policy of the same size trained under strict rules laps
these tracks in about 25.4 s with 32/32 finishes, and a tuned hand-written ray-following
controller does about 25.8 s. 22 s is not reachable without touching walls.

## Goal
Edit the weights so the car **never touches a wall** (strict completion as high as possible,
ideally 32/32), and, among edits that finish, laps as fast as possible. Prefer targeted edits
that remove the wall-riding mechanism and keep what the policy already does well. A larger
rewrite is allowed, but say so and say how much you changed; we will report the fraction of
weights changed and compare with other policies.

## Evaluation budget (hard limit: 25 calls)
The ONLY way to drive the car is:

    /Users/danielmercado/ProjectJoy/.venv/bin/python ../eval_train.py --attempt_dir . --weights <your.npz> --note "<what you changed>"

Run it from this folder with `NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4` set. Each call drives your
candidate one lap from the start line on the 32 evaluation tracks under strict and soft-wall
rules (grip 1.0) and prints JSON: summaries, per-track results, and for up to 4 crashed tracks
the observations and actions for the last 15 steps before the crash. One call takes about 2.5 s.
The score is (strict completion, then strict mean progress, then lower strict lap time). The
best-scoring candidate so far is kept automatically in `best.npz`; that is what will be judged
(on other tracks you never see). After 25 calls it refuses.

You may compute anything you like on the weights themselves with numpy (it runs on the CPU here;
MLX does not work in this sandbox): e.g. feed hand-made observation vectors through the network,
probe units, or solve for new weights to match a function you specify on hand-made inputs.
A numpy forward pass of 1 million observations takes well under a second.

## Rules
* Write only inside this folder. No pip install, no downloads, no git, no network.
* Do not import `racing/` or any simulator code into your own scripts, do not re-implement the
  simulator, and do not drive the car any way except through `../eval_train.py`. Do not edit
  `../eval_train.py`, `budget_config.json`, `eval_log.jsonl`, `best.*` or `candidates/`.
* Do not open or use any other policy's weights or results (nothing under `checkpoints/`,
  `results/`, `logs/`, `media/`, or other folders of `wsp/`), and do not read the project's
  `*.md` write-ups. Your edit must start from `start.npz`.
* No gradient descent, no evolution strategies, no random/black-box search over the simulator.

## Deliverables (in this folder)
* `NOTES.md`: what you think the wall-riding mechanism is (which units, what they compute), every
  edit you tried (call number, what changed, result), and your final explanation.
* Keep each candidate you evaluated as an .npz (eval_train.py also archives them).
* Final message: the mechanism in a few sentences, your best call's numbers, calls used, and
  the fraction of the 1,570 weights you changed in the best candidate.

You have about 60 minutes of wall time for this turn. Stop earlier if you are done.
