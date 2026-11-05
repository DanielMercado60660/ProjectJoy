# Round 8 staged plan: walk, steady eyes, then bridge to pretrained models (draft, 2026-11-05; nothing started)

The owner's frame: (1) walking, (2) walking like a chicken so the head stays stable, (3) take that stable
vision and bring in small VLAs / world models, bridging the disconnect between them and the duck. This plan
keeps that order and gives each stage the cheapest test that decides whether it is worth building.
`R8_PROPOSAL.md` (servo as teacher) is folded in: the servo becomes the demo source for stage 3 and a baseline
row everywhere.

**Bottom line.**
- **Three cheap, independent checks first, in parallel (§5):**
  - A: port the walker's training motor model (BAM) and re-measure tracking.
  - B: measure how steady the camera can get: oracle, trunk camera, software warp, phase-locked frames.
  - C: checkpoint checks: LeWM checkpoint vs scratch, NoMaD/ViNT zero-shot, encoder probes.
- **Stage 1** delivers a walker that does what it's told, plus a waypoint (or footstep) layer. That layer is the
  action interface every navigation model speaks.
- **Stage 2.** A physical chicken head fights this robot: the head is 38% of the mass and swings for balance.
  Software stabilization (rotation warp, phase-locked frames) gets the steady view without touching the gait.
  We found no study of whether a steady view helps a world model, so we measure it.
- **Stage 3.** Always a checkpoint plus our own data, judged against the same model trained from scratch. The
  first bets:
  - LeWM's released checkpoints answer the open question directly.
  - NoMaD/ViNT are tiny and MIT-licensed, take inputs close to our resolution, output waypoints, and steer by
    goal images. A goal image taken from on the plate solves the blind plate entry by construction.

## 1. What we actually tested, and what never got a fair test

| approach | what ran (from the records) | why it was not a fair test | a fair test |
|---|---|---|---|
| LeWorldModel (R6) | trained **from scratch** on 64 px head-camera frames, ViT-Tiny p8, 20k steps (time box, ~2.4 h). The published LeWM checkpoints were judged unhelpful (`LEWM_FEASIBILITY.md` §2: toy envs, other action spaces); no encoder-init ablation ran | the lamp is ~4 px and off camera 78% of the time; gait ego-motion dominates next-frame prediction; R6's own suggested fixes (96 px, oversampling presses, re-gating) never ran | a pretrained encoder, a steadier view, native resolution, the same gates |
| DreamZero-style world-action model (`WAM_S2_OPTION.md`) | **designed, never run**: gated on LeWM S1, which failed | the idea itself was never tested | the WAM on a representation that passes the gates, trained on a reactive teacher |
| chicken head (M3.1, R7 4A) | a head reflex on top of the frozen walker: camera wobble −35% yaw, −26% pitch. Judged by (a) whether the **bot** improves (it doesn't need it) and (b) **test-time swaps** into pixel policies trained on wobbly frames (a distribution shift). The "gaze-on training arm" was explicitly not run | nothing was ever trained on steadier frames; the question that matters (does a world model learn better from a steady view?) was never asked | stabilized frames into the representation gates (below) |
| walking layer (M0) | Pollen's velstand v8 run **without its BAM motor model** (the `bam` package isn't installed): forward ≈ 0.44× command, turns ≈ 1.39×, curves left 0.22 rad/s, vx < 0.2 and \|wz\| < 0.5 don't start, lateral unusable. Frozen since M0; RESULTS M0: "A BAM port is the obvious way to close it, out of scope for Phase 1" | every learned policy has had to learn around a body that doesn't do what it's told; BC averages near decision points fall into the dead zone and the duck stands still (the stall) | port BAM, re-run the M0 tracking battery, add a waypoint layer |
| teacher (all rounds) | every learned policy copied the reference bot (map + hidden program stage) | a pixel student can't see the bot's map; earlier DAgger labels were stale | the reactive 64 px servo (31/40 vs the bot's 21/40) as teacher |

**The test every stage has to respect.** The pixel-free belief policy (R7 1B) was given near-perfect supplied
perception (plate bearing/distance, lamp, memory) and still got 4/40 and 0/40. So "better eyes + the same
behaviour cloning *from the bot*" is already bounded. A reactive teacher (the servo) is not covered by that
bound. Stages 1 and 2 are enablers. Stage 3 earns its keep only if it changes
*how actions are produced* (latent planning, joint future + action prediction, a reactive teacher), not just
the encoder.

## 2. Stage 1: walking that does what it's told

**Facts.**
- The walker is Pollen's `velstand.onnx` v8 (`0045i2si` @ 2749, 2026-10-12), a 61→512→256→128→14 MLP at 50 Hz.
- Command block: twist (vx, vy, wz), head pose (4 joint deltas), body pose (6).
- Pollen trained it in mjlab (MuJoCo Warp, GPU) with BAM M6 XL330 actuators and backlash. We run it on CPU
  MuJoCo with XML PD actuators and a 4-step lag.
- Pollen's own v8 yaw eval (mjlab + BAM + backlash, *not our sim*): in place ±0.6 → +0.52/−0.57, ±1.0 →
  +0.97/−0.87; in place ±0.3 → only +0.08/−0.22.
- Their notes say why small commands stand still: "standing still WAS the reward optimum at 0.3"; in-place
  \|wz\| < 0.4 was never trained.
- So a BAM port should fix speed and turn scaling, but **a small-command dead zone will remain** whatever the
  actuator. *(Wrong on speed; see the Gate A result below. Pollen's own sim also walks forward at 0.43×.)*

**Steps.**
1. **BAM port.** Re-implement the BAM M6 XL330 actuator in numpy: voltage control and its friction terms (Coulomb +
   Stribeck + load-dependent; `vendor/microduck_rl/.../friction_dr_bam.py` subclasses it). Needs the `bam` source
   and XL330 parameters (owner approval). Re-run the M0 tracking battery.
2. **Waypoint layer.** A task-blind, closed-loop "go to relative pose (dx, dy, dθ)" controller on leg odometry
   (turn, walk, turn; `odometry.py` exists). It hides the dead zone and the drift. It is the same bridge
   navigation models use (relative waypoints, "move forward 75 cm"), so stage 3 can talk to it. It knows nothing
   about plates or lamps, like the walker itself.
3. **Re-baseline.** New body = new interface hash = "body v2". Every earlier number becomes "body v1". Re-run
   the bot and the servo on the selection rooms before anything learned.
4. **Only if 1 fails:** fine-tune velstand on CPU (warm start, PPO in MLX), or train a *residual* policy on top
   of the frozen walker. A published residual on a frozen A1 quadruped policy converged in 7.5M steps; see §6
   for our throughput. MuJoCo 3.17 (22 Oct 2026) adds `mujoco.batch` (threaded, GIL-free batched simulation
   with per-sim model fields for domain randomization), the best CPU fit. We run 3.16, so upgrading needs
   approval.

**A finer action interface to check: footsteps.** `apirrone` (the Open Duck Mini author) published a Microduck
footstep policy (`apirrone/microduck-footsteps` on Hugging Face). Its card reports landing within 5–7 mm and 1–2° of the
target in sim (≤ 5 cm forward, 2 cm sideways, 30° turn per step). If it runs on our body, "place the next feet
on the plate" is a cleaner entry than any velocity command. Verify the card, licence and observation layout
before relying on it.

(Note: Open Duck Mini v2 training details, found while researching, describe a different robot and walker;
ours is Pollen's Microduck velstand.)

**Gate.**
- M0 battery: forward and turn ratios within ±25% (M0's own target, never met), median straight-line yaw drift
  < 0.05 rad/s, 0 falls.
- Waypoint test (random 0.2–1.0 m, ±90°): ≥ 95% within 10 cm / 10°, 0 falls. Leg odometry alone drifts
  ~13 cm in 30 s (M3, ×1.3 scale), so tighten this only after measuring the odometry error on the same
  paths.
- Bot and servo on body v2 measured.

**Gate A result and amended gate (2026-11-05 16:40, written before any further walking measurement).**
- **Result:** the BAM port is bit-exact to Pollen's code but fails: forward 0.38×, turn 1.43×, drift +0.18 rad/s,
  0 falls (`results/r8/bam/REPORT.md`). Body v2 is not adopted, and the M0 body stays frozen.
- **Correction:** slow forward walking is the policy's own. Pollen's v8 sim (vx 0.4 → 0.17) gets 0.43×, so the
  forward bar above (±25% of the command) cannot pass on any body. Turning and drift do track in Pollen's sim
  (flat + backlash) but not in ours, with or without BAM.
- **Amended body-v2 gate:** match Pollen's sim, not the command.
  - Forward within ±25% of Pollen's 0.43× (0.32–0.54).
  - Turn ratio 0.75–1.25.
  - Median straight-line drift < 0.05 rad/s.
  - 0 falls.

  Command tracking becomes the waypoint layer's job. Under this bar both current bodies pass forward (XML 0.44,
  BAM 0.38), so turn and drift decide.
- **Prime suspect: backlash.** v8 was trained on `Mjlab-VelStand-Rough-Backlash-MicroDuck` (2° backlash
  hinges on all 14 servos, encoder = main + backlash joint) and evaluated flat + backlash. We run without it. It is
  one concrete difference; the others are diffuse (mjlab solver settings, MuJoCo Warp vs CPU, domain
  randomisation, IMU delay). The next walking step is the backlash port, which needs the owner's go.

## 3. Stage 2: steady eyes (chicken head)

**Facts** (M3.1; development measurements, not saved to a results file):
- Walking straight at 0.3 m/s, camera wobble is 6.7° yaw / 5.1° pitch (std) at 1.86 Hz, peak-to-peak pitch 28°.
- **The trunk wobbles only 2.1° / 0.7°.** The wobble is the head swinging on the neck.
- The head is 280 g, 38% of the robot. Pollen's notes: it "must oscillate while stepping". A tight head-tracking
  reward made their policy stop walking ("standing still scored higher"). Only the slow (DC) part is escapable.
- So a fully stabilized physical head fights this body's balance strategy.

**Steps, cheapest first.**
1. **Value check, no building.** Render four views of the same episodes:
   - head camera (now)
   - head camera with the reflex
   - a trunk-mounted camera (sim only)
   - an oracle-stabilized camera (head position, smoothed orientation)

   Measure the camera's angular wobble and the frame change. Then feed the steadiest stream into the stage 3
   representation gates. The question is not "does the bot improve" (answered: no). It is "does a world model
   or encoder learn the lamp and action effects better from a steady view". If the oracle view doesn't help
   there, stage 2 stops here.
2. **Electronic stabilization (sensor-only).** Render a wider field of view and warp it by the camera rotation
   estimated from gyro + joint encoders. The bot's camera model already matches the simulator's camera to
   0.4° median (M3), so the rotation to undo is known. Rotation-only wobble is fully removable this way.
   Resolution is almost free (§6). This is a chicken head in software.
   - **Phase-locked frames (free).** The wobble is periodic at the stride (1.86 Hz). Frames taken at the same
     gait phase, 1–2 per stride, share nearly the same camera pose. A world model or planner running at
     2–5 Hz can use only those frames.
   - Why the head reflex stalled at 14–35%, as a likely reading: compensation through laggy servos at the
     stride frequency leaves much of the swing, and near the 3.7 Hz step frequency it can amplify it. This is
     an inference from a different robot's servo model (Open Duck Mini), not ours, and M3.1's reflex was
     already phase-compensated. Animal and robot data agree that rotation is the part to cancel:
     - Pigeons stabilize head rotation almost perfectly (gain ≈ 1) and barely stabilize translation.
     - On iCub, feed-forward from the robot's own commands beat gyro feedback by 23%.
     - On the HyQ quadruped, IMU-driven camera stabilization improved mapping accuracy by 25%.
   - We found no study of whether steady footage helps a world model or a VLA, so we'd be measuring it.
3. **Learned chicken walk, last resort.** Fine-tune the walker with a head-stability term that prices only the
   escapable part (Pollen's lesson). CPU PPO, warm start (§6).

**Gate.** Camera wobble at or below trunk level (~2° / 1°), *and* the stage 3 gates improve over the wobbly
stream.

**Not fixed by stage 2:** the lamp is off camera 78% of the time. Steadiness doesn't make the duck look at it;
that's memory or active looking, a stage 3 problem.

## 4. Stage 3: bridging pretrained models to the duck

**Owner's rule (2026-11-05): start from a checkpoint, then train on our own data.** No checkpoint-based world
model was ever given a proper test here. R6 trained LeWM from scratch only, and the DreamZero-style arm never
ran. So every stage 3 route starts from a pretrained checkpoint and is fine-tuned on duck data. The proper test
is a pair: **checkpoint + fine-tune vs the same architecture from scratch**, same data, steps and gates. That
pair is what tells us the checkpoint carried something useful. Candidates that fit on this laptop:

| route | checkpoint | kind | what we fine-tune on duck data | size / licence |
|---|---|---|---|---|
| N | **NoMaD** (19M) / **ViNT** (31M) | navigation policy | waypoint head on servo demos, + lamp/done head | small, MIT |
| L | **LeWM** released checkpoints (`quentinll/lewm-{tworooms,pusht,cube,reacher}`; TwoRoom is a navigation toy) | world model (ViT-Tiny p14 at 224 px, ~18M) | the whole model on duck frames + our actions at 224 px (renders are free); our `lewm/` code exists | ~72 MB each, MIT |
| J | **JEPA-WMs** (`facebook/jepa-wms`: DINOv2-S encoder + predictor; PointMaze / Wall are navigation-like) | world model on a real pretrained encoder | the predictor with a new action input | ~212 MB each, CC-BY-NC |
| V | **V-JEPA 2.1 ViT-L** encoder (+ optionally `V-JEPA2-AC`, Meta's action-conditioned world model, already in MLX) | encoder / world model | our small predictor (3b); later, unfreeze the encoder's last blocks | 1.22 GB MLX port, MIT; AC 2.63 GB |
| W | **NWM CDiT-S** (50M) | navigation world model, relative-pose actions | the predictor on duck rollouts; then a WAM action head (DreamZero's idea at duck scale) | CC BY-NC |

**DreamZero itself** (14B video diffusion, multi-GPU H100s) can't be fine-tuned on this machine. Route W is the
duck-scale version of its idea: a pretrained world model that learns to predict the future and the action
together. A literal DreamZero test would need a cloud GPU (owner's call).

Order: L and N first. They're the cheapest, both MIT, and L directly answers "LeWM from a checkpoint, properly".
J, V and W follow if the representation gates say the latent space is worth planning in.

**The disconnects and their bridges.**

| disconnect | pretrained models | the duck | bridge |
|---|---|---|---|
| visual | natural or robot video, 256–384 px, arm-height or third-person | synthetic rooms, ~0.2 m-high bobbing head camera, 64 px today | frozen encoder at its native resolution (renders are almost free), probed on our frames; stage 2 for steadiness |
| action | arm end-effector deltas, per-embodiment encoders | velocity commands with dead zones | stage 1's waypoint layer: relative pose (dx, dy, dθ) as the action, our own small action encoder |
| time | 2–7 Hz chunks | 50 Hz walker, 10 Hz decisions | plan or act at 2–5 Hz in waypoint chunks; the walker fills in |
| task | goal images, language | "lamp matches target, then say done" | goal images for approach/entry (the step every policy fails); a lamp probe on the latent for done |
| memory | a few frames of context | lamp off camera 78%; plate leaves view during entry | belief tracker (sensors-only, exists) or explicit look-around; don't expect the world model to remember |

**Steps.**
1. **3a. Encoder bake-off.** Linear probes on our frames: lamp state, plate in view, plate bearing/distance.
   - Bars: P2-A′ 0.837, a supervised 64 px CNN 0.953, LeWM 0.503, random features 0.525.
   - Candidates (research report 2026-11-05):

     | encoder | params | weights / MLX route | cost at 240×320 |
     |---|---|---|---|
     | **V-JEPA 2.1 ViT-L/16** (distilled from G) | 300M | Python MLX port `dgrauet/vjepa2-mlx`, weights 1.22 GB; upstream MIT | ~0.2 TFLOP/frame (est.) |
     | V-JEPA 2.1 ViT-B/16 (distilled) | 80M | official `.pt` 1.66 GB (needs conversion); `lukasugar/vjepa2.1-mlx` covers B/L | small |
     | V-JEPA 2.1 ViT-G/16 (RoboJEPA's encoder) | 1.85B | official `.pt` 30.2 GB; unofficial fp16 safetensors 3.69 GB (`mnmly`, written for mlx-swift) | ~1.1 TFLOP/frame (est.) |
     | DINOv2 ViT-S/14 (cheap control; what Meta's small JEPA world models use) | ~21M | HF safetensors | tiny |

     The distilled L is close to G on V-JEPA 2.1's own benchmarks (SSv2 76.5 vs 77.7).
   - Resolution: these encoders were trained at 256–384 px. At 240×320 our ~4 px lamp becomes about one 16 px
     patch; at 64 px the encoder would see a 4×4 grid, out of distribution.
   - Cost note: recordings store 96 px frames only, so native-resolution features need deterministic replays
     (bit-exact with the cold renderer), not a cache read.
2. **3b. Dynamics in the frozen latent (RoboJEPA recipe, our own small predictor).**
   - **RoboJEPA's released predictors** (22M/50M/100M/300M, plus 1B to 8B):
     - Inputs: ViT-G features, 240×320, 2 views, 16-frame clips, all 300 patch tokens per view.
     - Actions: native units zero-padded to 64, with one of 26 per-embodiment slots (12 platforms).
     - Coverage: no legged robot and no navigation evaluation. The closest slot is 1X EVE (head camera + 2
       base-velocity dims, ~0.5% of the data). No instructions exist for adding an embodiment.
     - Their own scaling results:
       - 50–100M learn end-effector *reaching*.
       - Obstacle reaching needs ~300M.
       - *Pushing an object* (a scene-state change) needs ≥ 2B.
       - On the real Franka, 22M reaches 30% grasp success, 8B 67%.
   - **What maps onto the duck.** Walking up to the plate and onto it is ego-centric reaching: make the view
     match a goal view. That's the tier small predictors handle. The lamp change is a scene-state change, so we
     don't ask the world model to predict it; a lamp probe on the latent decides "done".
   - Plan: a small predictor (22M–50M scale) on frozen V-JEPA 2.1-L features (tokens pooled, e.g. 15×20 → 5×5),
     with waypoint actions, trained on duck data.
   - Optional ablation, only on ViT-G features: init from RoboJEPA-22M through the 1X slot vs scratch.
   - Licence: RoboJEPA weights and code are CC BY-NC-SA, so a fine-tuned RoboJEPA predictor can't go into this
     MIT repo (keep it out of git like checkpoints). V-JEPA 2.1 weights are MIT.
   - Gates: R6's G2 (prediction vs copy-last) and G3 (action ranking vs a random-encoder control).
3. **3c. Acting. This must change how actions are produced.**
   - **Latent planning, made laptop-sized.** RoboJEPA's CEM is uniform sampling: 1,024 samples × 10–15
     iterations, horizon 1–5. The 50M model takes ~20 s per iteration on one H100 (eager, dispatch-bound),
     ~20 min per reach episode. That's unusable at 2–5 Hz here.
     - Bridge: rank a handful of *proposed* waypoint chunks (from the servo or a learned policy) with the
       predictor instead of uniform CEM. Their paper lists the missing policy proposal as a limitation.
     - Rough cost at ≤ 64 candidates × horizon 3 × ~25 pooled tokens: ~0.4 TFLOP per replan, tens of ms
       (estimate).
     - Goal image for approach and plate entry; done from the lamp probe.
   - The WAM / PIDM recipe: predict future latents and the next waypoint chunk together, trained on **servo**
     demos. This is `WAM_S2_OPTION.md`, finally run on a representation that passes.
   - **The navigation-model route (likely the best first bet).** Models pretrained for robot navigation
     already solve the embodiment bridge the way stage 1 sets it up:

     | model | params | input | output (the bridge) | evidence on new or legged bodies | weights |
     |---|---|---|---|---|---|
     | **NoMaD** | 19M | 96×96, 3 context frames | 8 waypoints from diffusion; the goal image can be masked to *explore* | exploration 98% success (0.2 collisions) vs 77% for a 335M baseline | MIT |
     | **ViNT** | 31M | 85×64, 5 frames | 5 waypoints + a temporal-distance (arrival) head; a PD controller tracks them at 4 Hz | Go1 deployment; CARLA 40% → 80% after < 1 h of fine-tuning data | MIT |
     | GNM | small | 85×64, 5 frames | waypoints + distance | zero-shot on a low Roomba-like robot with an unseen fisheye camera: 0.93 | MIT |
     | NWM CDiT-S | 50M | 224 px, SD-VAE latents | relative-pose world model; ranks NoMaD samples or plans with CEM | Spot data (SCAND) in training; one real wheel-legged test 4/24 | CC BY-NC (metadata conflicts) |

     Why it fits the failures:
     - **Waypoint chunks instead of 10 Hz velocity commands.** No dithering in the dead zone; stage 1's layer
       tracks them.
     - **A goal image taken from *on the plate*.** Image-goal models drive to where the goal image was taken,
       so the plate leaving view during entry is not a problem by construction.
     - **The arrival head and NoMaD's explore mode** map onto "say done" and "search for the plate".
     - **Low cameras are in their training data** (TurtleBot, LoCoBot), and their 85×64 / 96×96 inputs are close
       to our 64×64 (and renders at any size are free).

     Plan:
     - A zero-shot check on duck frames first (do the distance head and the waypoints point at the plate goal?).
     - Then fine-tune on servo demos with waypoint labels, plus a lamp/done head (NaVILA's stop token and
       CityWalker's arrival head are the same idea).
     - Optional: rank NoMaD's samples with the 3b predictor (NWM's ranking mode: on RECON, ranking 32 NoMaD
       samples improved trajectory error 1.93 → 1.78; full NWM planning reached 1.13).
     - If zero-shot transfer is poor, COMPASS's pattern applies: residual RL in simulation on top of the frozen
       pretrained policy took a wheeled-robot policy from 10% to 91% success on a G1 humanoid.
   - Not recommended: NaVILA (8B, ~1 FPS on a 4090; its language → velocity-for-a-duration parser is the part
     worth keeping), LAPA / UniVLA (7B-class), NavFoM (cameras 0.6–1.5 m high).
   - Baseline rows: servo (31/40), bot (21/40), both re-measured on body v2. Headline on a new reserved slice
     (`make_splits --r8`, slots 13–16; R7 used 9–12).
   - **Gates for every learned stage 3 arm**, written into a protocol file on the selection rooms before
     training:
     - 2 seeds.
     - The R7 bar: ≥ 2 press-needed successes per seed, ≥ 5 over two seeds.
     - No rise in done-without-press.
     - The checkpoint version beats its from-scratch control by a stated margin on *both* seeds.
   - **Labels.** Waypoint labels for fine-tuning come from the simulator's pose in training episodes only.
     They are disclosed as such, like every supplied signal. At test time the policy sees only its own sensors.

## 5. First batch: three independent gates, in parallel, before building anything big

| gate | needs | rough cost | decides |
|---|---|---|---|
| A. BAM port + M0 battery | `bam` source/params (approval) | a day to port, an hour to gate | stage 1 path: port only, or fine-tune too |
| B. steady-camera numbers (head / reflex / trunk / oracle / electronic) | nothing | 1–2 h build + measure | whether stage 2 is worth building, and which kind |
| C. checkpoint checks: encoder probe bake-off, navigation-model zero-shot, LeWM checkpoint vs scratch | weight downloads (approval): LeWM ×4 (~72 MB each, MIT), NoMaD/ViNT (≤ 31M params, MIT), V-JEPA 2.1-L (1.22 GB MLX, MIT), DINOv2-S; ports to MLX | 1–2 days after download | which checkpoint stage 3 builds on, judged against the same model from scratch |

Then: B's best stream × C's winner → the 3b predictor gates and the navigation fine-tune → 3c.

## 6. Numbers measured today (single process, load ~4.9, AC)

- Duck control step (numpy walker MLP + 4 physics steps): **7,690 steps/s**; raw physics 37,590 steps/s.
- Step + head render: 13.8 ms at 64 px, 13.9 at 128, 14.5 at 256, **14.8 at 384**. Resolution is almost free;
  the fixed overhead dominates.
- Walker RL feasibility (upper-bound estimate, not a promise):
  - 4 processes ≈ 30k control steps/s before BAM's extra cost.
  - Pollen's from-scratch gait budget: 4,000–6,000 iterations × 24 steps × 4,096 envs ≈ 0.4–0.6 B steps, so
    ~4–6 h of sim.
  - A warm-start fine-tune of ~500 iterations at 1,024 envs ≈ 12 M steps, so minutes of sim plus PPO updates.
- Disk: 953 GB free; `episodes/` 50 GB.

## 7. Approvals needed (nothing runs without them)

1. The `bam` source and XL330 parameters: read and port, or install.
2. Weight downloads for gate C, all small except V-JEPA:
   - LeWM ×4 (~72 MB each, MIT)
   - NoMaD + ViNT (≤ 31M params each, MIT)
   - DINOv2-S (~21M)
   - V-JEPA 2.1 ViT-L MLX (1.22 GB, MIT)

   Later, for routes J and W: JEPA-WMs (~212 MB each, CC-BY-NC) and NWM CDiT-S (CC BY-NC). Non-commercial
   weights stay out of git, like checkpoints.
3. Any cloud GPU: not needed for anything above, as far as measured. Only a literal DreamZero fine-tune
   would need one.

Licences: RoboJEPA code is CC BY-NC-SA. Using it in a non-commercial lab is fine, but don't copy its code into
this MIT repo; re-implement from the paper or keep it vendored with a notice.

## Research status (2026-11-05)

- RoboJEPA / V-JEPA 2.1: done (§4, 3a–3c). Also found: `facebook/jepa-wms`, small JEPA world models on a
  DINOv2 ViT-S/14 encoder at 224² (PointMaze, Wall, Push-T, Metaworld; ~212 MB each, CC-BY-NC). It's the
  closest small-scale recipe to 3b and includes navigation-like tasks.
- Navigation models: done (§4, 3c). Head stabilization in the navigation literature: almost nothing. One older
  IMU pan-tilt stabilizer on the HyQ quadruped; no paper found that measured whether a steadier camera improves
  navigation. So stage 2's value check is genuinely open.
- Walker and head stabilization: done (§2, §3).
- Still to check before stage 1:
  - the `bam` repo's licence and XL330 parameters;
  - the footstep policy's card;
  - whether published CPU RL claims hold on our body (benchmark first).

  Published budgets for comparison:
  - Microduck gaits: 0.4–0.6 B steps, ~1–2 h on a GPU.
  - Playground Berkeley Humanoid: under 15 min on 2× RTX 4090.
