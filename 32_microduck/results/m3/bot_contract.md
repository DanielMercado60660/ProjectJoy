# Reference bot contract (M3): what the bot may know

The reference bot (`refbot.py` and its helpers `bodymodel.py`, `odometry.py`, `perception.py`,
`botmap.py`) is a controller like any other: it gets the instruction and, every 10 Hz tick, an
`interface.Observation`, and returns a `Command`. It is the bar every learned mind must clear, so it
must not see anything a learner wouldn't.

## Allowed
- **The observation:**
  - the head-camera frame (128×128 RGB)
  - the gyro, projected gravity, joint positions and joint velocities (plus the tick's five 50 Hz
    gyro and gravity samples)
  - the last command, the tick and the sim time.
- **The instruction:** `{"task", "target"}`.
- **Its own body:** the robot's kinematic model (`robot/robot_allcollisions.xml`), the default pose and
  the head-camera mounting, loaded on the bot side as a separate model. Real robots know their own
  kinematics; leg odometry from joint encoders + IMU is "body state", so it's inside the brief's rule.
- **Fixed priors, identical across every config** (nothing drawn per room):
  - The room is a 3 × 3 m square with walls.
  - The duck starts on the home mat (go and come home, set the lamp). *M3 review:* not in find the
    ball, whose start is a task-specific viewpoint only guaranteed 0.20 m from obstacles, so the floor
    model's "guaranteed floor" ring 0.28–0.40 m around the start can include an obstacle there
    (24 of the 250 M3 evaluation starts have one within 0.40 m; the bot solved 22 of those 24).
  - The colour *categories*: red, green, blue and yellow pads; a magenta ball; a cyan button in a
    dark frame; a white home mat. Reserved shades still belong to their category.
  - Sizes: pad 0.30 m, mat 0.40 m, button plate 0.24 m, ball 0.07 m, lamp ~0.10 m box on a wall.
  - The lamp is warm white when on and dark when off.
  - A walk across the plate centre toggles the lamp, and the plate re-arms after 1 s off it.
  - The camera intrinsics: 45° field of view, square image, rendered upright by `head_image`.
- **Episode-seeded randomness** (deterministic), and memory it writes itself (a JSON-able dict).

## Not allowed
- `World`, `iface.world`, `obs._iface` or any config field: layout, appearance, seeds, positions.
- The judge, the lamp flag, the press count, the button depth, the ball position, the true pose.
- Anything from `tasks.py` beyond the instruction it is handed.
- Tuning on any `test` split. Training and dev only; after tuning, dev numbers count as development,
  not held-out.

## Enforcement
`tests/test_m3.py::test_bot_contract`. The bot's modules must not import `room`, `tasks`, `configs`,
`oracle` or `session`, and must not contain `.world`, `_iface`, `cfg[`, `layout` or `lamp_on`. The
runner hands the bot only the observation stream and the instruction.

The check is static: it would not catch truth reached by another name (e.g. `getattr`). The M3 review
read every bot module and found no such route. The bot holds the `Interface` only to build a skills
`Ctx` (no layout); `Ctx.obs` calls `iface.observe()`, which returns the same `Observation` fields as
`tick()`. `skills.py` imports `configs`, but only uses it when a layout is passed (the oracle).
