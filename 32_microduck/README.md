# 32 Microduck — One Duck, Many Minds (Phase 1, Phase 2 steps 1–5)

A simulated Pollen Robotics Microduck (25 cm, 800 g biped, 14 servos, head camera) in plain CPU
MuJoCo, walking under Pollen's published `velstand` policy. Later milestones add a room, skills, a
hand-written reference bot and a learned recurrent policy on one command interface (see
[design/PROPOSAL.md](design/PROPOSAL.md)).

**Status (2026-10-27, all with the real policy body): Phase 1 complete.**
- M0 (locomotion gate) passed.
- M1 (room, repeatable episodes, sessions, split files) built.
- M2 (frozen 10 Hz command interface, seven skills, three task specs, oracle feasibility 330/330)
  built and reviewed (interface re-frozen with the tick's 50 Hz IMU samples, find_ball
  judge hardened, skills' observation plumbing fixed); find-ball setup v2 (two hiding places) added
  before M3.
- M3 (sensors-only reference bot: leg odometry, colour perception, camera-built map, memory, one state
  machine per task; tuned on train/dev only; evaluated against the oracle) built and reviewed
  (contract and numbers hold; the find-ball no-memory searcher fixed and re-run, overclaims
  corrected, set-lamp demo replaced).
- M3.1 "chicken head" gaze reflex (the owner's idea) measured: a steadier camera, but no gain for the
  bot, so not adopted (`gaze.py` stays as an option).
- M4 (episode format, 1,029 recorded bot demonstrations with perturbations, `media/demo.html` replay)
  built.
- M5 (learned recurrent policy in MLX: behaviour cloning on the bot's demos, two DAgger rounds,
  checkpoints chosen by a rule written before training) built: the pipeline works, the policy does
  not do the tasks (closed loop it stays at the set-lamp done-at-once floor).
- M6 (single look at the reserved test rooms, 120 per family per task) done: the bot holds its dev
  numbers on instances, arrangements and distractor (82–97%) and drops on new appearances (go 78%,
  set-lamp 62%), as predicted; the oracle scores 120/120 everywhere; the learned policy scores 0 on
  go-and-return and find-ball and only the no-press set-lamp floor. Failure clips in
  `media/m6_failures.html`. Recorder: DAgger "done" labels fixed, and an opt-in mode to follow the
  evaluator exactly (a renderer's first frame differs by 1–2 grey levels).

**Phase 2, step 1 (2026-10-27 morning, dev only): the set-lamp failure.** A lamp/match auxiliary
loss trained on the bot's own beliefs, with clean DAgger labels, ties the policy's "done" to the lamp
in part: dev lamp AUC 0.84, against 0.48 for M5 and 0.53 without the loss. Said done without pressing in
2/18 press-needed dev episodes (M5: 17/18). It misses the protocol's bar (AUC ≥ 0.9, at least one
press-needed success): no version succeeds when a press is needed. In closed loop the policy never
decides to walk (its forward command stays at its standing level), though on the bot's demos it copies
the bot's walking. A walk-or-stand head made a press (as one M5 DAgger checkpoint once did), then never
said done, and lost most no-press successes. Reviewed. Plain-ReLU encoders can die early in training (P2-A was
blind); the image encoder is now leaky.

**Phase 2, step 2 (2026-10-27 midday, dev only): a DAgger round with P2-A′ driving.** 300 set-lamp
episodes with the bot labelling in shadow, then P2-E (P2-A′ + the round). No gain: 0/18 press-needed,
no presses; no-press 11 vs 17/22 and lamp AUC 0.77 vs 0.84 against P2-A′, but step 4's runs of the same
recipe (7 and 12/22, AUC 0.58 and 0.73) show these differences are within seed spread. The policy turns too weakly and
often the wrong way, so the bot rarely reached its "walk" step and the round taught turning (1.9% walk
labels). P2-E still doesn't walk when driving itself. First check on go-and-return and find-ball: 0/40
each for M5, P2-A′, P2-D, P2-E and a second P2-A′ seed. Reviewed..

**Phase 2, step 3 (2026-10-27 afternoon): two checks before a recovery round.** When the bot takes over
from P2-A′ partway through, it succeeds as often as when it drives from the start (34/48 vs 33/49
press-needed). It says done with the lamp wrong 12.5% of the time, against 11.6% alone, which fails the
5% gate as written; the bar was set without a baseline. After takeover 46% of the bot's labels are
"walk" (round 3: 1.9%). Fed the bot's demos, P2-A′ turns the right way 98% of the time but strongly
enough to start a turn (|wz| ≥ 0.5) only 68% of the time. Reviewed..

**Phase 2, step 4 (2026-10-27 afternoon): the recovery round.** Four arms, two seeds each: control,
+ recovery episodes (the bot takes over from the policy), + discrete motion heads (walk/stand and a
left/none/right turn at the bot's rate), and both. Discrete heads are the clearest gain of Phase 2: the
policy walks in its own states and turns hard enough, no-press successes rise (17 and 19 of 22 vs 7 and
12; P2-A′'s own seeds reached 17 and 15), and presses become more frequent (up to 4 of 18 rooms per run). Recovery data adds nothing measurable. Press-
needed successes: at most 1 of 18 in any run, so the round's bar is not met. Reviewed..

**Phase 2, step 5 (2026-10-27 evening): Astra's diagnosis and the history pilot.** At the owner's request
Astra reviewed the whole record. A measured flaw: a training window from an
episode longer than 128 ticks shows the episode's final "done" tick 0.2% of the time. A pilot on top of the
discrete heads tried ending-weighted windows (E), whole episodes with memory carried across chunks (H),
and the same chunks without memory (Hr); confirmed on 80 dev rooms never used before. Seeing endings
raises no-press successes (72, 69, 68 of 80 over two seeds vs 52), mostly by saying done more often (E
says done as often when a press is needed), and the duck still almost never presses (at most 1 of 80 press-needed successes per arm), and carrying a detached
memory across chunks changed no success count (H ≈ Hr; this does not test learned long memory). Next: skill selection with a shared sensors-only map (Astra's option 2). Reviewed.

See [RESULTS.md](RESULTS.md) and [tasks/](tasks/).

## Run
```bash
cd 32_microduck
./smoke.sh                                   # 112 tests, about 45 s: M0-M6 + Phase 2 (incl. the split check)
../.venv/bin/python policy/parity.py         # numpy port vs onnxruntime on 1,000 obs -> results/m0/parity.json
../.venv/bin/python m0_tune.py               # fix sweep (25 body settings)            -> results/m0/tune.json
../.venv/bin/python m0_from_rest.py          # start-threshold test                     -> results/m0/from_rest.json
../.venv/bin/python m0_gate.py               # the gate, chosen body (about 10 s)       -> results/m0/gate.json
../.venv/bin/python m0_gate.py '{"scene": "allcollisions"}'   # same gate, plain body    -> results/m0/gate_allcollisions.json
../.venv/bin/python m0_render.py             # media/m0_pad_run.mp4, m0_push_walk.mp4, m0_head_camera.png
../.venv/bin/python review_m0_checks.py      # reviewer spot checks (about 6 s)        -> results/m0/review_checks.json
# M1
../.venv/bin/python m1_physics_checks.py     # room vs M0 scene, button plate sweep, cross-process session, speed -> results/m1/physics_checks.json
../.venv/bin/python make_splits.py           # writes splits/*.json + MANIFEST.json (about 15 s; deterministic)
../.venv/bin/python check_splits.py          # re-verifies the splits from the files     -> results/m1/split_check.json
../.venv/bin/python m1_render.py             # media/m1_families.png, m1_button_lamp.mp4, m1_lamp_pair.png
../.venv/bin/python review_m1_checks.py      # M1 review: save/restore break tests, siblings, button edge, clearance, colours -> results/m1/review_checks.json (~30 s)
# M2 (set OMP_NUM_THREADS=1; each uses 4 processes)
../.venv/bin/python freeze_interface.py      # interface spec + sha256 of interface.py  -> results/m2/interface_frozen.json
../.venv/bin/python m2_battery.py            # skill battery, oracle feasibility, task-3 setup coverage, split re-check -> results/m2/battery.json (~1 min)
../.venv/bin/python m2_render.py             # oracle demo clips media/m2_*.mp4
../.venv/bin/python review_m2_checks.py      # M2 review: trivial controllers vs judges, watch time, IMU yaw drift, setup grids -> results/m2/review_checks.json (~1 min)
# M3 (set OMP_NUM_THREADS=1; each uses 4 processes; train/dev only)
../.venv/bin/python m3_odometry.py           # odometry candidates and leg scale    -> results/m3/odometry.json
../.venv/bin/python m3_perception.py v3      # perception on 560 frames             -> results/m3/perception_v3.json
../.venv/bin/python m3_lamp.py v0            # lamp on/off detectors, facing the lamp -> results/m3/lamp_v0.json
../.venv/bin/python m3_floor.py              # floor model thresholds               -> results/m3/floor_sep.json
../.venv/bin/python m3_drift.py v0           # in-episode drift, landmark correction on/off -> results/m3/drift_v0.json
../bin/gpu_run ../.venv/bin/python m3_eval.py final train:60 instances-dev:50 arrangements-dev:50 appearances-dev:50 distractor-dev:40 --oracle   # -> results/m3/eval_final.json (~30 min)
../.venv/bin/python m3_baselines.py          # find-ball no-memory searchers        -> results/m3/find_ball_baselines.json (~40 s)
../.venv/bin/python m3_render.py             # bot demo clips media/m3_*.mp4 (~30 s)
../.venv/bin/python m3_debug.py <config-id> <task> <out.png>   # one bot episode: its map vs the true layout
# M3.1 chicken head (optional reflex, off in the bot)
../.venv/bin/python m3_gaze.py               # camera wobble off/on on train-0020..0025  -> results/m3/gaze.json
../bin/gpu_run ../.venv/bin/python m3_eval.py final_gaze train:60 instances-dev:50 arrangements-dev:50 appearances-dev:50 distractor-dev:40 '--params={"gaze": true}'   # -> results/m3/eval_final_gaze.json
# M4 (train split only; episodes/ is gitignored)
../bin/gpu_run ../.venv/bin/python m4_record.py   # 1,029 episodes -> episodes/<task>/*.npz, episodes/index.jsonl, results/m4/recording.json (~1 h, 4 processes)
../.venv/bin/python m4_replay.py episodes/go_and_return/train_train-0280_k1.npz episodes/set_lamp/train_train-0084_k1.npz \
  episodes/find_ball/train_train-0116_k1.npz episodes/set_lamp/train_train-0052_k1_missed_press.npz \
  episodes/find_ball/train_train-0073_k1_ball_moved.npz episodes/go_and_return/train_train-0026_k1.npz   # -> media/demo.html
# M5 (train data only; checkpoints chosen on 30 instances-dev episodes by results/m5/selection_rule.json)
# results/m5/bc_index.jsonl = the 888 rows of episodes/index.jsonl with success true (every train split row)
../bin/gpu_run ../.venv/bin/python m5_train.py runs/m5_bc2 --index results/m5/bc_index.jsonl --steps 16000 --ckpt-every 4000   # BC (~26 min)
../bin/gpu_run ../.venv/bin/python m5_select.py runs/m5_bc2                      # -> results/m5/selection_m5_bc2.json (4 processes)
../bin/gpu_run ../.venv/bin/python m5_dagger.py runs/m5_bc2/ckpt_008000/weights.safetensors --n 150 --beta 0.5     # DAgger round 1 recording (~27 min, 4 processes)
../bin/gpu_run ../.venv/bin/python m5_train.py runs/m5_dagger1 --index results/m5/bc_index.jsonl --index episodes/dagger_index.jsonl --init runs/m5_bc2/ckpt_008000/weights.safetensors --steps 12000 --ckpt-every 3000
../bin/gpu_run ../.venv/bin/python m5_dagger.py runs/m5_dagger1/ckpt_012000/weights.safetensors --n 150 --start 150 --beta 0 --round 2
../bin/gpu_run ../.venv/bin/python m5_train.py runs/m5_dagger2 --index results/m5/bc_index.jsonl --index episodes/dagger_index.jsonl --index episodes/dagger2_index.jsonl --init runs/m5_dagger1/ckpt_012000/weights.safetensors --steps 12000 --ckpt-every 3000
../bin/gpu_run ../.venv/bin/python m5_select.py runs/m5_dagger2                  # chosen weights copied to checkpoints/ by hand
# Note: since M6, record.py ends a DAgger episode when the shadow bot finishes (stale-label fix), so re-running
# the two m5_dagger.py lines no longer reproduces the stored round-1/2 episodes.
OMP_NUM_THREADS=1 ../.venv/bin/python -m pytest -q tests/test_m5.py              # input path, train step, reload, resume, 30-tick closed loop (~3 s)
# M6 (protocol and predictions in results/m6/protocol.json, written before any test episode). As run, the
# bot/oracle and BC grids did not go through gpu_run and BC used M6_PROCS=6, and the policies were loaded
# from runs/ (byte-identical to checkpoints/); the lines below re-run them with the lock (the bot/oracle line is
# still one 2.1 h run, over the 45-minute cap: split it per set with --sets=<set>:120 and different tags to comply).
../bin/gpu_run ../.venv/bin/python m6_eval.py bot_oracle bot,oracle,searcher_all,searcher_occluder   # 4,800 episodes (2.1 h, 4 processes)
M6_PROCS=4 ../bin/gpu_run ../.venv/bin/python m6_eval.py policy_bc bc --sets=train:40,instances-test:40,arrangements-test:40,appearances-test:40,distractor-test:40 --policy=bc:checkpoints/m5_bc.safetensors
for s in train instances-test arrangements-test appearances-test distractor-test; do   # one ~18 min run per set
  M6_PROCS=4 ../bin/gpu_run ../.venv/bin/python m6_eval.py policy_dagger_$s dagger --sets=$s:120 --policy=dagger:checkpoints/m5_dagger.safetensors; done
../.venv/bin/python m6_report.py                                                 # -> results/m6/report.json, report.md
../bin/gpu_run ../.venv/bin/python m6_clips.py record && ../.venv/bin/python m6_clips.py page   # 7 clips -> media/m6_failures.html
# Phase 2 step 1 (dev only; protocol + amendments in results/p2/protocol.json). Runs are gitignored (runs/p2_*).
../.venv/bin/python p2_lamp_probe.py         # can a small CNN read the lamp at 32/64/96 px? -> results/p2/lamp_probe.json
P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_devdemos.py   # 80 bot demos on dev configs -> episodes/p2_dev/
T="--index results/m5/bc_index.jsonl --index episodes/dagger_index.jsonl --index episodes/dagger2_index.jsonl --img 64 --steps 16000 --ckpt-every 4000 --max-minutes 45"
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_train.py runs/p2_a2 $T --aux-weight 1 --act leaky            # P2-A' (~24 min)
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_train.py runs/p2_b2 $T --aux-weight 1 --act leaky --done-from-aux   # P2-B'
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_train.py runs/p2_c2 $T --aux-weight 0 --act leaky            # P2-C' (control)
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_train.py runs/p2_d  $T --aux-weight 1 --act leaky --vx-head binary  # P2-D
# (P2-A, the dead-encoder run, is the P2-A' line without --act leaky, into runs/p2_a)
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_select.py runs/p2_a2   # 40 dev episodes per checkpoint + open-loop diag -> results/p2/select_p2_a2.json
../.venv/bin/python p2_diag.py runs/p2_a2/ckpt_016000/weights.safetensors        # lamp AUC, early done, counterfactual -> results/p2/diag_*.json
../.venv/bin/python p2_aux_split.py runs/p2_a2/ckpt_016000/weights.safetensors   # lamp head in view / remembered -> results/p2/aux_split_*.json
../.venv/bin/python p2_enc_probe.py runs/p2_a2/ckpt_016000/weights.safetensors   # linear lamp probe on frozen image features -> results/p2/enc_probe.jsonl
../.venv/bin/python p2_vx_check.py <weights>                                      # forward commands on the bot's dev demos -> results/p2/vx_check.jsonl
OMP_NUM_THREADS=1 ../bin/gpu_run ../.venv/bin/python p2_vx_closed.py <weights>    # forward commands in closed loop -> results/p2/vx_closed.jsonl
# Phase 2 step 2 (amendments 3-4)
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_dagger.py runs/p2_a2/ckpt_016000/weights.safetensors --n 300 --k 3 --beta 0 --tag dagger3   # ~16 min
../.venv/bin/python p2_dagger_check.py --tag dagger3      # -> results/p2/dagger3_check.json; episodes/dagger3_train_index.jsonl flags its 18 bad done ticks
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_train.py runs/p2_e $T --index episodes/dagger3_train_index.jsonl --aux-weight 1 --act leaky   # P2-E
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_train.py runs/p2_a2_s1 $T --aux-weight 1 --act leaky --seed 1                               # P2-A' seed 1
# (both hit the 45-min guard while sharing the machine; continue with: p2_train.py <run> --resume --steps 16000 --max-minutes 45)
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_other.py NAME=WEIGHTS ...      # go-and-return / find-ball dev check -> results/p2/other_tasks.jsonl
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_shadow_dev.py NAME=WEIGHTS ... # policy vs shadow-bot turn/walk on dev -> results/p2/shadow_dev.jsonl
# Phase 2 step 3 (amendment 5)
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_takeover.py --arms bot_alone,takeover_100,takeover_250
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_takeover.py --arms takeover_400,fresh_250
../.venv/bin/python p2_takeover.py --summary                                     # -> results/p2/takeover.json
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_wz_check.py NAME=WEIGHTS ...  # teacher-forced turn output -> results/p2/wz_check.jsonl
../.venv/bin/python p2_wz_check.py --yaw dagger3                                 # replay-only: do weak turn commands rotate the duck?
# Phase 2 step 4 (amendment 6): recovery episodes, index, eight runs, per-pick checks
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_recovery.py --start 100 --n 100   # and --start 200 --n 100 -> episodes/p2_recovery/
../.venv/bin/python p2_recovery_index.py      # filtered DAgger 1-2 indexes + episodes/recovery_index.jsonl
./p2_lane.sh c_s0 r3_s0 r2_s1                 # train to 16k (resuming after 45-min stops) + select; 3 lanes at once
./p2_checks.sh c_s0 c_s1 ...                  # p2_diag, p2_aux_split, p2_wz_check, p2_vx_closed, p2_shadow_dev, p2_other per pick
# Phase 2 step 5 (amendments 7, 7a): history pilot
./p2_lane.sh e_s0 h_s1                        # e: --p-end 0.2; h: --stream; hr: --stream --stream-reset (2,000-step window warm-up, 512-stream pool)
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_confirm.py NAME=WEIGHTS ...   # 80 never-used dev rooms -> results/p2/confirm.jsonl
../.venv/bin/python p2_confirm.py --stages select_p2_r3_s1   # stage counts from a selection file's pick
```
Everything in M0 is CPU-only and quick (the gate is 40 simulated minutes in about 10 s on 4 processes),
so nothing goes through `bin/gpu_run`.

## Using the room
```python
import json, session
from room import World
cfg = json.load(open("splits/train.json"))["configs"][0]   # any config from splits/
world = World(cfg)                       # room + frozen body; world.duck is the Duck
world.duck.set_command(vx=0.3)
for _ in range(100):
    world.step()                         # 50 Hz: duck step + button/lamp/flicker logic
print(world.lamp_on, world.press_count, world.ball_pos())
world.reset()                            # back to the config's start, bit for bit
session.save("life.json", world, memory={"seen": []})      # continuing life
world, memory = session.start("life.json")                  # resumes exactly
```

## Running the reference bot
```python
from refbot import controller
factory, bot = controller("find_ball", budget=600)      # sensors only: results/m3/bot_contract.md
result, world, iface = run_episode(cfg, make_episode(cfg, "find_ball", 0), factory)
print(result["success"], bot.log, bot.mem.to_dict())
```

## Running a task
```python
import json
from room import World
from tasks import make_episode, run_episode
from oracle import controller_for          # or your own controller: (iface, instruction) -> generator of Commands
cfg = json.load(open("splits/train.json"))["configs"][0]
ep = make_episode(cfg, "set_lamp", k=0)     # instruction + start conditions, deterministic
world = World(cfg)
result, world, iface = run_episode(cfg, ep, controller_for(world, "set_lamp"), world=world)
print(result["success"], result)
```

## Using the body on its own
```python
from duck import Duck, BodyConfig
from m0_gate import FROZEN_BODY
duck = Duck(BodyConfig(**FROZEN_BODY))
duck.set_command(vx=0.3, wz=0.0, head=(0, 0, 0.5, 0))   # held until changed
for _ in range(50):                                      # 50 Hz control steps
    duck.step()
print(duck.pos(), duck.yaw(), duck.planar_velocity(), duck.tilt())
```
Frozen command bounds: [results/m0/body_frozen.json](results/m0/body_frozen.json).

## Layout
```
duck.py               body: scene load, 61-dim observation, 50 Hz loop, delay buffers, push force, head camera fix
policy/velstand_np.py numpy port of velstand.onnx (reads the initialisers with onnx)
policy/parity.py      parity check against onnxruntime
m0_tune.py            M0 development sweep of the allowed fixes
m0_from_rest.py       start-threshold test (from standing vs already walking)
m0_gate.py            M0 gate: pad reach, 20 x 60 s runs, tracking, head pose, push recovery
m0_render.py          demo clips and a head-camera frame
robot/                MJCF + meshes copied from microduck_rl (see below)
results/m0/           every M0 number (json)
media/                demo clips (no meshes here)
logs/                 stdout of the runs that produced results/m0
tests/test_m0.py      pytest checks run by smoke.sh (M0)
tests/test_m1.py      M1 checks: replay, reset, appearance independence, sessions, button/lamp, splits
room.py               room built with MjSpec from a config dict; World (state, reset, button -> lamp, flicker)
configs.py            family rules per split factor, generators, validator, hand_config() for tests
session.py            session files for continuing life (save / new / start)
make_splits.py        writes splits/ (run once, before any training)
check_splits.py       re-verifies splits/ from the files alone
m1_physics_checks.py  room vs M0 scene, button plate sweep, cross-process session, speed
m1_render.py          M1 media
review_m1_checks.py   M1 reviewer checks
splits/               train.json + instances/arrangements/appearances/distractor.json + MANIFEST.json
interface.py          FROZEN 10 Hz command interface (Command, Observation, Interface.tick)
skills.py             stop, turn_to, go_to(+planned), look, scan, press_button, return_home; TruePose hook
tasks.py              episodes, task-3 setup (viewpoint + ball launch), ground-truth judges, run_episode
tasks/*.md            the three task specs + shared conventions
oracle.py             ORACLE controllers (skills + simulator truth) for feasibility
m2_battery.py         skill battery + oracle feasibility + task-3 setup coverage
m2_render.py          oracle demo clips
freeze_interface.py   writes the interface freeze file
refbot.py             M3 reference bot: sensing loop, memory, three task state machines (sensors only)
bodymodel.py          bot's own kinematic model (feet, head camera) from joint encoders
odometry.py           leg odometry + gyro heading (bot side)
perception.py         colour segmentation, floor projection, lamp on/off detectors (bot side)
botmap.py             per-room floor model, occupancy grid, A* (bot side)
m3_*.py               M3 measurements (odometry, perception, lamp, floor, drift, eval, render, debug)
results/m3/           bot contract, tuning log (tuning.jsonl), every M3 number
gaze.py               M3.1 chicken-head reflex (adaptive gaze stabilisation; off by default)
record.py             M4 episode recorder (format, perturbations, DAgger mixing for M5)
m4_record.py          records the demonstrations; m4_replay.py builds media/demo.html
episodes/             recorded episodes + index.jsonl (gitignored)
episodes/dagger3_*    DAgger round 3 (P2-A′ driving, set-lamp, train k=3) index + training index (gitignored)
episodes/p2_shadow_dev/  p2_shadow_dev.py's dev episodes (gitignored)
episodes/p2_takeover/ check A's 500 episodes (train configs 0-99, k=4), by arm (gitignored)
episodes/p2_recovery/ step 4's 200 recovery episodes (train configs 100-299, k=4) (gitignored)
episodes/p2_look/     4 scratch closed-loop episodes of P2-A′ (dev, bot shadowing), the basis of amendment 2 (gitignored)
learner.py            M5 policy (CNN + GRU, MLX), input preparation, PolicyController for run_episode
m5_train.py           BC / DAgger training (resumable); m5_select.py applies the selection rule
m5_dagger.py          DAgger recording (policy and shadow bot on train configs)
checkpoints/          m5_bc.safetensors, m5_dagger.safetensors (the rule's picks, 2.9 MB each) + meta
m6_eval.py            M6 evaluation (test + familiar sets, Wilson intervals); m6_report.py builds results/m6/report.*
m6_clips.py           M6 failure clips: picks by fixed rules, re-records (cold renderer), media/m6_failures.html
tests/test_m6.py      recorder follows the evaluator (cold renderer); DAgger "done" labelled once
p2_learner.py         Phase 2 policy: Policy2 (lamp/match aux heads, leaky encoder, optional walk-or-stand and left/none/right turn heads), P2Controller
p2_train.py           Phase 2 training: clean labels (cut at the bot's first done), aux labels from the bot's beliefs, resumable
p2_select.py          Phase 2 dev selection (40 set-lamp episodes per checkpoint); p2_diag.py open-loop lamp diagnostic
p2_aux_split.py       lamp-head accuracy in view / remembered; p2_enc_probe.py linear probe on image features
p2_vx_check.py        forward commands on the bot's demos; p2_vx_closed.py the same in closed loop
p2_dagger.py          DAgger round with a Phase 2 policy driving; p2_dagger_check.py its pre-training data check
p2_other.py           go-and-return / find-ball dev check; p2_shadow_dev.py policy vs shadow-bot commands on dev
p2_takeover.py        teacher-takeover check (record.py takeover_tick / takeover_fresh); p2_wz_check.py turn-output check
p2_recovery.py        recovery episodes (P2-A′ drives, shadow bot takes over at T); p2_recovery_index.py the training indexes
p2_lane.sh            one lane of runs: train to 16k with --resume, then p2_select; p2_checks.sh the per-pick checks
p2_confirm.py         set-lamp confirmation on 80 never-used dev rooms + stage counts
p2_stream_debug.py    amendment 7a's sampler experiments (why stream training needs a window warm-up)
p2_lamp_probe.py      lamp readability at 32/64/96 px; p2_devdemos.py records the 80 dev bot demos
p2_labels.py          per-frame in-view labels + observable-only masks (built and tested, not used yet)
p2_pretrain.py        encoder pretraining on the bot's detections (built and tested, not used yet)
results/p2/           protocol + amendments, every Phase 2 number
tests/test_p2*.py     Phase 2 checks: labels, truncation, train step, reload, resume, controllers, walk/turn heads, takeover, recovery sampling, ending windows, streams
review_m2_checks.py   M2 reviewer checks
tests/test_m2.py      M2 checks: freeze, clipping, skills, episodes, launch session, oracle, review regressions
review_m0_checks.py   independent review's spot checks -> results/m0/review_checks.json
```
The checkpoint for M0 is Pollen's policy file itself (`../data/microduck_policies/velstand.onnx`,
sha256 in `body_frozen.json`); nothing is trained in M0.

## Sources and licences
- `robot/`: copied from `github.com/pollen-robotics/microduck_rl` at commit
  `b02fdea4902c2b93b7d34a83a6f05da4ececb7d3` (`src/mjlab_microduck/robot/microduck/`): the
  `robot_allcollisions`, `robot_groundcontact` and `robot_walk` MJCFs, their scenes, and the 38 STL
  meshes they use. Code is Apache-2.0 (`robot/LICENSE`); **the meshes are CC BY-NC-SA 4.0**
  (`robot/LICENSE-MODELS.md`), Copyright Pollen Robotics: shared here unmodified for non-commercial
  use with attribution, and not covered by this repository's MIT license. They stay out of `media/`.
- `../data/microduck_policies/`: Hugging Face `pollen-robotics/microduck-policies`
  (Apache-2.0), revision `665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448`.
- `duck.py` rebuilds the observation and loop of `scripts/infer_policy.py` (read, not imported) and
  the head-camera fix of `src/mjlab_microduck/sim/camera.py`.
