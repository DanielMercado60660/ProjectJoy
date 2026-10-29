# Phase 1 tasks: what every spec shares

Specs: [go_and_return.md](go_and_return.md), [set_lamp.md](set_lamp.md), [find_ball.md](find_ball.md).
Code: `../tasks.py` (episodes, judges), `../interface.py` (frozen command interface), `../oracle.py`.

- **Instruction (structured in Phase 1):** `{"task": <id>, "target": <pad colour | "on"/"off" | null>}`.
  No free text; wording tests belong to Phase 2.
- **Episode:** fully determined by (config, task, k). `tasks.make_episode()` seeds everything from the
  config hash, so an episode always stays inside its config's split.
- **Actions (frozen, `results/m2/interface_frozen.json`):** at 10 Hz:
  - a twist: vx −0.3…0.4 m/s, vy 0, wz −0.6…0.6 rad/s
  - head-pose deltas: neck pitch, head pitch, head yaw, head roll, each within 0.8 × its cap
  - a `done` bit.

  Each command is held for 5 walking-policy steps. Skills (`../skills.py`) are the provided way to
  produce these commands, but any controller may emit them directly.
- **Observations:** what the controller may use:
  - the head camera (128×128 RGB, rendered at the tick)
  - body state: gyro, projected gravity, joint positions and velocities, plus the IMU's five 50 Hz
    samples during the tick (`gyro_substeps`, `gravity_substeps`) for yaw odometry
  - the last command, the tick and the sim time.

  Not observable: the true pose, the lamp flag, the press count, the button depth and the ball
  position. Only the judge reads those. This is a convention, not a sandbox: a controller is handed
  the `Interface` (and an `Observation` keeps a private `_iface`), both of which reach the `World`.
  Non-oracle controllers must not touch `iface.world` or `obs._iface`.
- **Trivial floors** (`../results/m2/review_checks.json`): saying `done` at once scores 0% on
  go-and-return and find-ball and ~half on set-the-lamp (the episodes that need no press).
- **End of episode:** the first tick with `done = True`, or the task's step budget. Success is judged at
  that moment. A controller that never says `done` fails.
- **Always recorded:** falls (trunk tilt > 45°), ticks touching a wall, partition or occluder, sim time
  and ticks.
- **Splits:** see `../splits/` and RESULTS.md M1. Every factor changes one thing relative to the
  shared training pool. Never tune on a `test` set.
- **Sessions:** a session boundary falls between ticks. Skill-internal state is not saved, so resume at
  a skill boundary.
- **Body:** the frozen M0 policy body. Its quirks (start threshold, ~0.44× forward speed, ~1.4× turns,
  leftward drift, deaf neck pitch) are in the freeze file's `body_notes`.
