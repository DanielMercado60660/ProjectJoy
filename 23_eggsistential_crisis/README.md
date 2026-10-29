# 23 Eggsistential Crisis: don't break the egg

A simulated Shadow Dexterous Hand (gymnasium-robotics `HandManipulateEggRotate_ContinuousTouchSensors`
assets, MuJoCo 3.16) learns to **yaw an egg lying in its palm to a target orientation** while keeping
it in the hand and staying under a contact-force budget. The learned method is PPO in MLX with an
asymmetric critic (it sees privileged friction, mass, true contact force and time) and an actor that
reads the 92 touch sensors. It's compared with cheap baselines that were tuned on training seeds:
hold still, a CEM-tuned constant posture, tuned random jiggling, and a CEM-tuned closed-loop scripted
finger gait. Results, numbers and caveats are in [RESULTS.md](RESULTS.md).

## The gymnasium-robotics bug and the fix
`HandManipulateEgg*-v1` crashes in `make()`/`reset()` under mujoco 3.16 with an AssertionError in
`gymnasium_robotics/utils/mujoco_utils.py:get_joint_qpos`. The cause isn't a wrong joint type.
`model.jnt_type[j]` is a `numpy.int32`, and `np.int32(3) in (mjtJoint.mjJNT_HINGE, mjtJoint.mjJNT_SLIDE)`
is `False` under mujoco 3.16's enum type, even though `np.int32(3) == mjJNT_HINGE` is `True`. So every
**hinge** joint of the hand trips the assert. `gymfix.py` replaces the four joint get/set helpers
in memory with versions that cast the type to `int`. Site-packages isn't edited.
`test_parity.py` checks that the patched helpers equal direct qpos/qvel slicing for every joint, and
that our own env (`egg_env.py`) produces the **identical 153-d observation** as the patched gym env,
both on the same state and after a control step from the same state.

## Task definition (our narrow first version)
- The egg (ellipsoid 3×3×4 cm, 0.151 kg as in the asset) starts on its side in the palm with a
  uniform random yaw. 10 zero-action settle steps run, as in gym's reset.
- Target: the settled pose rotated about world z by δ, with |δ| ~ U[0.15, 0.4] rad and a random
  sign. 0.15 is above the threshold, so doing nothing doesn't count as success. Curriculum: θmax
  widens by 0.2 rad, up to π/2, once training success exceeds 60%.
- Error: the angle between the egg's long axis and the target's long axis, treated as lines. The
  ellipsoid is symmetric about that axis and end to end, so spin about the axis is physically
  invisible. The gym full-quaternion distance is reported as well. **Headline caveat:** under gym's own
  full-quaternion criterion PPO succeeds in 27.3% of main-range episodes vs 25.4% for the tuned gait
  (RESULTS.md, Review), so the big gap is in pointing the long axis, not in full orientation.
- Success: error < 0.1 rad (5.7°, gym's rotation threshold) at the final step (t = 100, 4 s), and
  not dropped. Drop: egg centre below z = 0.10 m (it rests at about 0.163 m).
- Force budget: **10 N** total normal contact force on the egg, fixed before training (the egg
  weighs 1.48 N; hold-still peaks are about 2-3 N, random actions 35-50 N). In evaluation the peak
  is measured every physics substep (2 ms) with `mj_contactForce`.
- Control: absolute joint position targets in [-1, 1] mapped onto the actuator ranges (gym
  semantics), 20 substeps of 2 ms = 40 ms per action, 100 actions per episode.
- Training randomization: friction scale U[0.7, 1.3] (applied to all collision geoms, because
  MuJoCo takes the max of the pair) and egg mass/inertia scale U[0.7, 1.3].

## How to run (from this folder)
```bash
./smoke.sh                                   # < 1 min: parity tests + tiny train/tune/eval/render into smoke_out/
../.venv/bin/python -m pytest -q test_parity.py

# baseline tuning (CEM / grid on training seeds) -> checkpoints/baselines.json
../bin/gpu_run ../.venv/bin/python baselines.py --n_eps 64
# PPO training (each run <= 45 min; chained by resuming from checkpoint). The actual sequence:
../bin/gpu_run ../.venv/bin/python train.py --run run1 --minutes 43 --lr 1.5e-4
../bin/gpu_run ../.venv/bin/python train.py --run run2 --resume checkpoints/run1/latest.npz --minutes 43 --lr 1.5e-4 \
    --force_w 0.2 --force_margin 7 --force_cap 2
../bin/gpu_run ../.venv/bin/python baselines.py --force_aware checkpoints/baselines.json \
    --out checkpoints/baselines_forceaware.json --n_eps 64 --margin 7 --log logs/baselines_forceaware.log
../bin/gpu_run ../.venv/bin/python train.py --run run3 --resume checkpoints/run2/latest.npz --minutes 40 --lr 1.5e-4 \
    --force_w 0.2 --force_margin 7 --force_cap 2
# held-out evaluation (seeds >= 10000) -> results/eval.json, ~6 min
../bin/gpu_run ../.venv/bin/python evaluate.py --ckpt checkpoints/run3/latest.npz --ckpt_before checkpoints/run1/latest.npz
../.venv/bin/python tables.py            # markdown tables used in RESULTS.md
../.venv/bin/python plots.py             # media/*.png
# demo clips: reload the checkpoint, offscreen render (~10 s each)
../.venv/bin/python render_demo.py --ckpt checkpoints/run3/latest.npz
../.venv/bin/python render_demo.py --ckpt checkpoints/run3/latest.npz --theta 0.8 1.5708 --tag _wide
../.venv/bin/python render_demo.py --ckpt checkpoints/run3/latest.npz --n 64 --mass 0.4 --tag _lightegg_mass0.4
```
The final policy is `checkpoints/run3/latest.npz` (run1 → run2 → run3 chained).

Review additions (cross-reviewer):
```bash
# gait retuned on PPO's full curriculum range (0.15 to pi/2) -> checkpoints/baselines_wide.json, ~4 min
../bin/gpu_run ../.venv/bin/python baselines.py --force_aware checkpoints/baselines_forceaware.json \
    --theta_max 1.5708 --out checkpoints/baselines_wide.json --log logs/baselines_wide.log --n_eps 64 --margin 7
# same held-out cells with per-episode trajectories: held-for-last-0.5 s success, gym full-quaternion
# success, settle time; wide-retune gait rows -> results/eval_review.json, ~4.5 min
../bin/gpu_run ../.venv/bin/python review_metrics.py
```
`.gitignore` keeps `smoke_out/`, `logs/*.pid`, `MUJOCO_LOG.TXT` and the intermediate checkpoint
`checkpoints/run2/` out of git. They stay on disk. run2 is only needed to re-run run3 from
scratch, not for the demo or the evaluation.

## Files
| file | what |
|---|---|
| `gymfix.py` | in-memory monkeypatch for the gymnasium-robotics joint-type assert |
| `egg_env.py` | threaded vectorized env on the gym XML (task, metrics, force measurement) |
| `test_parity.py` | patch correctness + observation/dynamics parity with the patched gym env |
| `baselines.py` | zero / posture / random / scripted-gait baselines and their tuning |
| `train.py` | PPO + asymmetric critic in MLX, checkpoint/resume, curriculum |
| `evaluate.py` | held-out evaluation of all methods, robustness grid |
| `render_demo.py` | reloads a checkpoint and renders side-by-side MP4/GIF clips |
| `plots.py` | figures in `media/` |
| `tables.py` | markdown tables from `results/eval.json` |
| `review_metrics.py` | reviewer re-eval: stability (last 0.5 s), gym full-quaternion success, wide-retune gait |
| `queue_after.sh` | the same as `launch_after.sh`, for any script (used for the gait retune and the eval) |
| `launch_after.sh` | waits for my previous job, checks AC power, then launches a run via `gpu_run` |
| `checkpoints/` | `baselines.json`, `runN/latest.npz` (+ `.json` meta: config, normalizer, RNG, curriculum) |
| `logs/` | training logs (`runN.jsonl`, one line per PPO iteration), tuning log |
| `results/eval.json` | all evaluation cells |
| `results/eval_review.json` | reviewer's re-evaluation (same seeds) |
| `media/` | demo clips and figures |
