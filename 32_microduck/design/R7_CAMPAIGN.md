# Round 7 campaign: make the duck press (design, 2026-11-04 evening)

Status: design only. Nothing in this file has been run. Written by the campaign lead from four panel
proposals (perception, dreamzero, wsp, bottleneck) and their adversarial critiques; every number below was
re-read from the files named next to it. Scratch probes live under
`/private/tmp/claude-501/-Users-danielmercado--claude-spares-spare-rsxALG/00c14289-b253-490f-81af-de46a8b4d86a/scratchpad/r7/`
(referred to as `scratch/r7/` below); they are read-only evidence, not project code.

## 0. Summary for the owner

**What we will try.** The panel re-derived where the press chain breaks, and all four critiques agree on
the same two links, neither of which is perception, prediction or camera shake:

1. *Acting on what it sees.* The learned policy sees the plate early (the shadow bot's detector confirms the
   button on the policy's own frames in 14-17 of 18 press-needed rooms, first tick 20-37) and often has it in
   view, close and centred; it still walks on only 10-52% of those ticks, turns toward it at chance
   (0.46-0.53), and never enters it from such a sighting (0 of 3-6 episodes per policy). The reference bot
   walks on 80% and turns toward it on 84% in the same geometry. At the one tick where memory is needed (the
   first blind turn after the lamp check) every checkpoint says "none" in 17-25 of 27 demos and prefers the
   bot's direction at chance; from the next tick on it agrees 97%, because it copies the gyro.
2. *Finishing.* "Done" fires on the clock (median tick 112-127, the teacher's no-press done tick is 106) with
   the lamp in view and misread in 10 of 11 sighted premature dones; and done-after-press is learned in the
   bot's states (teacher-forced 10-22 of 39 press demos) but lost in the learner's own post-press states (0 of
   5). Six lamp-right presses sit in Hr's two confirmation runs with no done after them: a scripted done would
   already give 6/80. So the bar is gameable and needs an intentionality metric, which this plan pre-registers.

So round 7 attacks the state problem directly and measures it with instruments that predict the bar:
(a) a per-tick sensors-only *belief* of where the plate, lamp and mat are (built from the recorded memory
tracks and leg odometry, labels at 2-5 degrees, no simulator truth), fed to the policy as an input and as
auxiliary targets; (b) a one-minute open-loop turn-onset probe and an approach probe as gates; (c) a targeted
on-policy round for the post-press "verify, then done" link; (d) skill selection on the shared sensors-only
map as the required comparison arm. Cheapest decisive tests go first: a pixel-free policy on the belief vector
decides in about two hours whether a 10 Hz command policy can use an explicit map at all.

**Why it might work.** Five independent probes locate the failure at the decision ticks, not in the frames:
the plate is seen, the lamp is read (a 4-number HSV rule reads it at 0.91 and lamp-strong seeds behave exactly
like lamp-weak ones on the bar), and the teacher needs only ~500 of 900 ticks. What has never been learned is
acting on what is seen. The belief labels are rebuildable on 46-75% of set-lamp ticks in every data source,
and the sighting that creates the plate track is inside the training window 84-91% of the time, so the signal
is learnable from the data we have. The owner's intuition that "the solution is in there" is right in this
narrow sense.

**Honest odds** (bar: >= 2 press-needed successes per seed and >= 5 of 80 on the confirmation rooms, no rise
in done-without-press):
- a command-level *pixel* policy (Stage 2) clears it: about 25-30%;
- *some* learned component clears it, including skill selection (Stage 3): about 80%, but most of that is
  scaffolding, and the (skill, lamp-belief) control is there to show how much;
- the owner's four routes on their own this round: LeWM/JEPA or a DreamZero-style world-action model < 5%,
  a pretrained encoder < 10%, weight-space editing of the pixel policy < 5%, the chicken head < 10%. Each is
  kept as a bounded, hard-gated side probe (Stage 4) so the hunch is tested, not argued.

**What it costs.** Core plan (Stages 0-3 and 5): about 22 lane-hours of machine time over 1.5-2 working days
at <= 2 concurrent jobs (longer while projects 20/28 train; load was 9.0 with two `gpu_run` jobs at 21:00, not
four). Optional owner-idea probes (Stage 4): +4-5 lane-hours; the "honest" chicken-head arm and a LeWM-B run
would add ~9 more and are not recommended. Disk: < 5 GB new (< 8 GB with options); no 96 px cache.

**What needs your approval** (full list in section 7): a fresh reserved split drawn before training; a second
look at the 80 confirmation rooms (paired, not a headline); supplied sensors-only perception at test time
(odometry, camera pose, detector, memory) for Stages 2-3, disclosed as such; new code files under
`32_microduck/`; two recording hours (~2 GB each: post-press on-policy round; optional gaze-on round); the
optional DINOv2-small download (Apache-2.0, ~88 MB); an optional Astra budget. No LeWM checkpoint download.

## 1. What the panel established (evidence; all numbers re-read from the files)

**The funnel on the 80 confirmation rooms** (`results/p2/confirm.jsonl`, `confirm_p2_*.json`; pooled over the
8 step-5 runs, 320 press-needed episodes; `scratch/r7/critic_bottleneck/funnel_out.txt`):

| stage | count | share |
|---|---|---|
| early done, no press | 228 | 71% (pooled median tick 219; R3/Hr s0 117-127, E/H 242-338) |
| timed out, no press | 75 | 23% |
| reached the plate (body centre) | 14 | 4.4% (1-3 per run) |
| pressed (plate depressed, intentional or not) | 17 | 5.3% |
| said done after a press | 2 | 0.6% |
| success | 1 | 0.3% (E s0, distractor-dev-0031) |

Per run (press-needed 40; reached / pressed / lamp right at end / said done / done after press; no-press
successes): R3 s0 1/1/1/22/0; 22. R3 s1 3/3/3/20/0; 30. E s0 2/2/2/34/1; 37. E s1 1/1/0/37/1; 35.
H s0 1/2/1/32/0; 30. H s1 1/1/1/30/0; 39. Hr s0 2/2/2/27/0; 30. Hr s1 3/5/4/28/0; 38.

**Reference bot on the same rooms** (`results/m3/eval_final.json`, frozen M3 bot, gaze off): press-needed
27/40 (instances-dev 8/10, arrangements-dev 10/13, appearances-dev 4/10, distractor-dev 5/7), pressed 31,
plate visits 31; no-press 40/40; successes take median 413 ticks (p90 664) of 900. On the 40 selection rooms
(0000-0009 x 4 families): press-needed 15/18 (2/4, 5/5, 4/5, 4/4), no-press 22/22, median 390 ticks.

**Acting on what it sees** (`scratch/r7/critic_bottleneck/approach2_out.txt`, `approach_out.txt`; shadow-dev
episodes, policy driving, bot shadowing, truth read only to score):
- With the plate detected on the current frame, < 1.5 m, within 30 degrees, lamp mismatch known, t > 130:
  P(walk) = R3 s0 0.10, R3 s1 0.42, R2 s0 0.52, R2 s1 0.50; P(turn toward | turning) 0.46-0.53 (chance 0.5);
  entered the plate within 100 ticks of a >= 20-tick near sighting: 0 of 3, 6, 6, 4 episodes. The bot in the
  same geometry: walk 0.80, toward 0.84.
- The shadow bot's memory knows the button at the policy's first lamp check in 14 / 14 / 17 / 16 of 18 rooms
  (`scratch/r7/dreamzero/shadow_failure.json`).
- Turn onset (`scratch/r7/critic_dreamzero/blind_turn_t0*.json`; 27 p2_dev press-needed demos with a blind
  first turn after the lamp check, teacher-forced): at t0 the turn head outputs "none" in 17 (H s1), 23 (Hr s1),
  23 (R2 s1), 24 (E s0) of 27 and the critique reports 23-25 for R3 s1/s0; share of demos with
  p(bot direction) > p(opposite): 0.48 / 0.52 / 0.59 / 0.48 (R3 s1 0.74, R3 s0 0.56; chance 0.5). From t0+1
  the same heads agree with the bot in 23-26 of 27 (gyro copying via `body_vec`).
- A ridge readout of the plate bearing from R3 s1's GRU state: 16.8 degrees in view, 53 degrees out of view,
  72 degrees at the lamp check with 17% within 30 degrees (`scratch/r7/dreamzero/bearing_probe.json`; chance 105).

**Finishing** (`scratch/r7/critic_perception/NOTES.md`, `own_states_lamp_read.json`, `postpress_done_*.json`,
`scratch/r7/critic_dreamzero/follow_teacher.json`):
- Premature done in press-needed rooms fires at median tick 112-127 (the bot's no-press done: 106) with the
  shadow tracker's lamp belief already formed and mismatching in 5/6 (R3 s0) and 6/7 (R3 s1); in 10 of 11
  sighted premature dones the learner's own camera had the lamp in view and the state was read wrong; the
  lamp-strong seed (r3_s1) reads the wrong state at its done tick in 5/6 and says done anyway.
- Teacher-forced on the 80 p2_dev demos (39 press demos), done fires at the bot's post-press done tick for
  hr_s1 22/39, h_s1 18/39, e_s0 18/39, h_s0 11/39, hr_s0 10/39 (R3 s1 2/39); closed loop in the learner's own
  post-press states: 0/5 (Hr s1: 5 presses, 4 lamp-right, all timeouts).
- Counterfactual "done after a lamp-right press" on rooms already looked at: R3 +1/+3, E +1/+0, H +1/+1,
  Hr +2/+4 (Hr would make 6/80). This is why the plan adds an intentionality metric.

**Perception is not the limiter** (`results/lewm/gates.json`, `results/lewm/probes.jsonl`,
`results/p2/lamp_probe.json`, `scratch/r7/perception/probe3_pixel_probes.json`,
`scratch/r7/critic_perception/probe_encoders.jsonl`):
- The bot's 4-number HSV lamp mask reads the lamp at 0.908 on the registered G4 frames; a supervised 64 px CNN
  0.953; random conv features find the plate at AUC 0.79; in-view bearing from image features is 13-16 degrees.
- Frozen-feature lamp probe of the same recipe: seed 0 0.678-0.776, seed 1 0.782-0.882. The lamp-strong and
  lamp-weak seeds have the same press-needed outcomes (successes 0, said-done 22/20, 32/30, 27/28). Raising the
  lamp read 0.70 -> 0.88 bought nothing on the bar.
- LeWM S1 (Round 6): effective rank 27, held-out error ratio 0.753 vs control 0.541, action ranking 0.265 vs
  control 0.283, lamp 0.503 vs 0.525, plate AUC 0.817 vs 0.792. Why (`scratch/r7/perception/probe5b_*`,
  `probe2_*`): 60% of the latent's variance is between rooms (a 30-way room probe reads 0.93; random encoder
  0.42; pixels 0.22); the lamp is 4.3 px wide at 64 px, flips on 0.15% of ticks, the flip is off camera 78% of
  the time and the new state is next seen a median 108 ticks later. No next-frame objective at any resolution
  can see this task variable; SIGReg is blind to a sparse binary direction (0.527 vs 0.528 Gaussian).

**Labels for the belief are rebuildable** (`scratch/r7/dreamzero/odom_label_check.json`, `label_coverage.json`,
`scratch/r7/critic_dreamzero/learnability.json`): leg odometry replayed from the stored `imu_sub` and
`joint_pos` gives pose error 1.2-3.4 cm median (end 2-13 cm), yaw 1.4-2.7 degrees; the plate relative bearing
from the memory track plus the rebuilt pose errs 1.6-1.8 degrees median (approach segments 3.2-4.6, p90 16-25);
a confirmed button track exists on 46-75% of set-lamp ticks in every source. Under the existing R3 sampler
(W=128, burn 32, p_start 0.2) the lamp-check tick is learnable 91% (dev) / 84% (train) and the check-to-press
stretch 98% / 91%: the dreamzero proposal's window-size fallbacks are not needed.

**Geometry the arms must respect**: the straight line from the lamp-check pose to the plate passes within
0.15 m of the partition or occluder in 39% (17/44) of p2_dev press-needed demos and 27% of a train sample
(`learnability.json` `straight_line_min_clearance_m`); the bot path-plans (`refbot.py:358`). Any chooser or
servo needs an obstacle rule, and a learned policy must learn detours from pixels.

**Camera shake** (`RESULTS.md` M3.1, `scratch/r7/perception/probe4_*`, `scratch/r7/critic_wsp/jitter_out.txt`):
walking raises frame change 4x at the policy's 10 Hz, but 75% of it is a rigid 2-D shift and the robot's own
joint+IMU camera model predicts a shift that removes 57% with no tuning; the plate centroid jitters 0.126
half-widths per tick walking straight and 0.028 after a 5-tick mean. The reflex (`gaze.py`) removes 26-35% of
the wobble, nothing at 0.4 m/s, cost the bot 3 falls vs 0 and was off for all 3,345 recorded episodes. The
plate is first seen while standing (tick 20-50), where the reflex is gated off.

## 2. The owner's four ideas: verdicts

Stated plainly, with the critique that downgraded each and what survives.

| idea | panel verdict | what survives in this plan |
|---|---|---|
| **LeWM / JEPA, trained harder or pretrained** | Downgraded by all four critiques. S1's latent is a room code and the task variable (a 4 px lamp that flips off camera once per episode and is re-seen 10 s later) is invisible to any next-frame objective; more steps, 96 px or event-weighted windows do not change that. Pretrained LeWM checkpoints (`quentinll/lewm-*`) were trained on one 2-D toy environment each with that environment's action space at 224 px patch 14 (`design/LEWM_FEASIBILITY.md` section 2); they are not a general vision model and are not requested. | A future "LeWM-B" (room-blind SIGReg, proprio-conditioned predictor, belief-supervised aux heads) is specified with its gates in Stage 4C but **not funded** this round; expected presses < 1 even on a pass. |
| **DreamZero-style world-action model** | Downgraded (dreamzero critique, bottleneck critique). The literal model is 14B/CUDA; a Microduck version would denoise latents that carry no lamp (Round 6) over a 2.5-6 s horizon against a 25-45 s approach; the "task-state WAM" (future beliefs under the action chunk) is kinematics that a bearing-minus-integrated-yaw baseline will tie. `design/WAM_S2_OPTION.md` stays closed. | The transferable part, "supervise the trunk with what happens next", survives as the belief heads (bearing, distance, in-view, age) on the policy trunk in Stage 1C/2A. |
| **Weight-space programming** (rikiyat.github.io blog; our 08 and 25 results) | Downgraded (wsp critique). R3 s1 has no dead or spare units (0 of 16/32/64/64/192/256/128 with std < 0.01), so the blog's "remove an obstacle" edit has nothing to act on; the "compile a controller" mode is a hand-written map-free refbot whose learned-head latch fires falsely in 5/22 (s1) and 10/22 (s0) no-press rooms; 08/25 showed Astra ties a Claude-written program or SGD distillation and loses to DAgger. | (a) a map-free pixel-servo **baseline** row (Stage 1D), labelled supplied capability; (b) an optional Astra round over *named* inputs (lamp belief, bearing, crossed-plate flag) to repair the done/press heads after a reach gate passes (Stage 4D). |
| **Chicken head (gaze stabilisation)** | Downgraded (all four). The plate is first seen while standing, where the reflex is off; in-view localisation is already 13-18 degrees and the failure is inaction, not mislocalisation; the wobble is a rigid shift the robot knows from its own sensors; the reflex cost 3 falls; switching it on at test time is a distribution shift against 1.3M gaze-off frames. It has never been tested for a pixel learner, which is the one honest reason to keep it. | A 10-minute paired pre-gate recording and a no-training test-time replay (Stage 4A); the honest version (gaze-on recording + matched training) only if both pass and lane time remains. |
| **"Pretrained should be better"** (general encoder, DINOv2) | Not applicable as LeWM; as a general encoder it targets in-view perception, which the data say is not the failing stage. | An optional, owner-approved, no-training frozen-feature probe (Stage 4B); a pass earns only a place as the image branch of Stage 2A for one matched run. |

## 3. Pre-registration (written to `results/r7/protocol.json` before any training)

**The bar** (unchanged): on the 80 confirmation rooms (instances-dev 0030-0049, arrangements/appearances/
distractor-dev 0020-0039, set_lamp, k=0), >= 2 press-needed successes per seed and >= 5 over two seeds, with
said-done-without-press in press-needed rooms not above the arm's own base recipe (Hr 27-28/40, R3 20-22/40
from `confirm.jsonl`) and no-press successes not below base - 4. Stage counts reported per seed and per
family: reached plate, pressed, lamp right at end, said done, said done after press (lamp right / wrong).
Never average over appearances-dev (the bot itself drops to 4/10 there).

**Intentionality metric** (new, reported beside every bar count; the bar is gameable by a scripted done over
a wandering policy, section 1). Computed evaluation-side from the sensors-only detector run on the learner's
own frames plus truth for scoring only, as the shadow instruments already do:
1. *intentional reach*: body-centre plate entry within 400 ticks of the detector's first confirmed lamp
   sighting (the bot's own check-to-press stretch is median 210-229 ticks, p90 306-336,
   `scratch/r7/dreamzero/approach_stats.json`), with path length / straight line <= 2.5 over that stretch
   (path from replayed leg odometry); both thresholds are calibrated in Stage 0 so that the bot passes on
   >= 90% of its own press-needed successes, and the calibrated values go into the protocol;
2. *turn onset*: the first turn run (>= 6 ticks) after the lamp check starts toward the shorter bearing to the
   remembered plate;
3. *verified done*: a done after a press counts as verified only if the detector confirms a fresh lamp sighting
   between the press and the done (the bot itself verifies about 78% of the time: 54 of 244 press demos end
   with a done that never re-saw the lamp, `scratch/r7/critic_perception/NOTES.md`; the >= half gate passes it).
A learned claim needs intentional reaches >= half of reaches and verified dones >= half of done-after-press
per seed. Successes that fail these are still counted for the bar and reported as "unverified".

**Open-loop gates that predict the bar** (every learned arm reports both; baselines from the frozen files):
- *Turn-onset probe* (`scratch/r7/critic_dreamzero/blind_turn_t0.py` protocol, 27 p2_dev blind turns, MLX
  inference ~1 min): share of demos with p(bot direction) > p(opposite) at t0, baseline 0.48-0.74; demos where
  the argmax is the bot's direction, baseline 2-5/27. Gate: >= 0.85 and >= 15/27.
- *Bearing read* (only for arms with belief heads): median error at the lamp check <= 30 degrees and >= 60% of
  checks within 30 degrees (baseline 57-72 degrees, 17-27%); out of view <= 30 degrees (baseline 53-65).

**Closed-loop stage gate on the 40 selection rooms** (18 press-needed, 22 no-press; used for checkpoint picks
by the unchanged `p2_select` rule), per seed: reached plate >= 6/18 (base 1-3; bot 18/18), within 0.5 m
>= 9/18 (base 3-5), done after a press with the lamp right in >= half of pressed episodes, said done without
press <= the base recipe's pick on the same rooms (Hr s1 12/18, R3 s1 6/18, from `results/p2/select_*.json`),
no-press successes >= 15/22 (Hr s1 18, R3 s1 19). Note: 30 of the 40 selection rooms also hold p2_dev demos,
so the open-loop and closed-loop gates are not independent, and the selection rooms have been used for 17+
picks; margins are read with that in mind.

**Rooms and looks.** Selection: the 40 selection rooms only. Confirmation: the 80 rooms, once per arm; this is
the project's second look at them (first: step 5), so every confirmation number is a *paired* comparison with
the step-5 rows, not a headline. Headline: a fresh reserved slice, 20 rooms per family (80 rooms), drawn by
new `make_splits.py` SLOT entries (slots 9-12, seeds `base + 1,000,000 * slot + i`, so they overlap nothing)
with `check_splits.py` passing, written in Stage 0 before any training and looked at once in Stage 5 after
methods freeze. The bar is stated in absolute counts against 40 press-needed rooms: the absolute bar is kept if
the fresh draw yields 32-48 press-needed set-lamp rooms (the 80 confirmation rooms had 40), otherwise extra
rooms are drawn in Stage 0 until it does; decided before any training. The 120 test configs per family were spent in M6 (`results/m6/protocol.json`); only
arrangements-dev and appearances-dev 0040-0049 are untouched dev rooms (20), too few for a headline.

**Seeds and power.** Two seeds per arm, never promote on one. With per-episode success p = 0.025,
P(>= 2 of 40) = 0.26 per seed and P(>= 5 of 80) = 0.05; with p = 0.10, 0.92 and 0.90. Two seeds separate a
~4x gain from nothing, not a 2x gain; differences of 1-2 episodes are noise (same recipe spans no-press 7-17/22
and AUC 0.58-0.84 across seeds).

**Controls that must run with each arm**: shuffled-input control for any supplied-input arm (values from a
random other episode's tick), the pixel-free policy for the map-sensor arm, shuffled-label and
(skill, lamp-belief)-conditioned most-frequent-next-skill for the selector, and the frozen bot as the ceiling.

**Disclosure.** Stages 1B, 2 and 3 run the bot's own sensors-only modules at test time (`odometry.py` leg mode,
`bodymodel` camera pose, `perception.detect`, `refbot.Memory`; Stage 3 also its skills, map and budgets). Every
table labels those arms "learned decisions + supplied perception/skills". No simulator truth enters any
controller; truth is used only to score.

## 4. Stages

Each stage: purpose, method, gate, kill rule, cost (lane-hours = hours of one of the two allowed jobs;
wall time is less when both lanes run). Compute flows forward only through passed gates; every attempt,
including kills, gets a ledger row (section 9).

### Stage 0. Protocol, fresh split, instruments (no training; ~1.5 lane-hours, CPU)

- Write `results/r7/protocol.json`: everything in section 3, the baselines quoted from the frozen files, the
  arm list, the lane plan.
- Draw the fresh reserved slice: 4 new SLOT entries in `make_splits.py`, 20 configs per family, smoke-built,
  `check_splits.py` extended to verify them (unique ids and hashes, no seed overlap with train/dev/test);
  `splits/r7_*.json` written and sha256'd into the manifest. Owner approval (section 7, ask 1).
- Instruments (new files, tests at tiny size): `r7_probes.py` (turn-onset, bearing read, approach probe,
  intentionality metric, all from recorded episodes or one short inference pass); the done-gate ceiling is
  computed offline from `confirm_p2_hr_s{0,1}.json` (the six lamp-right unreported presses), not by driving
  the 80 rooms again.
- Pre-flights (code, 1-2 h, no GPU): (i) `p2_train.py --init` is a strict `load_weights` and `p2_pretrain.py`
  exports regress heads, so an encoder-only partial loader and a freeze flag are needed before any init from
  pretraining; (ii) `record.py`: store the bot's odometry pose per tick (`bot_pose` x, y, psi) in new
  recordings only; (iii) the belief label builder and the memory-vector sampler input (Stage 1A); (iv) resume
  verified again at tiny size for every modified trainer (AGENTS.md rule 5), and `smoke.sh` extended.
- Gate: protocol written, split files checked, tests pass. Kill: none; this stage is the entry ticket.

### Stage 1. The cheapest decisive tests (two lanes, ~5 lane-hours, ~3 h wall)

**1A. Belief labels** (~0.5 lane-hour, CPU, no training).
- Method: a per-episode side file (new cache version `p2_belief_v1`, labels only, frames untouched) for all
  3,345 training episodes: replay `Odometry` (leg mode, anchors off; restarted at `takeover_tick` where
  `header.dagger.takeover_fresh` is set) from the stored `imu_sub` and `body[6:20]`; for each class
  {button, lamp (on/off merged), mat, pad_red/green/blue/yellow, ball}: sin/cos of relative bearing, log
  distance, in-view bit (track `last_seen == t`), log ticks-since-seen (cap 300), present flag (confirmed track
  exists), plus the track's most recent sighting tick (window mask); the memory's lamp belief one-hot and age;
  own odometry pose relative to the episode start (x, y, sin, cos); two sensors-only flags: `crossed_button`
  (odometry position within 0.15 m of the remembered button track since the last lamp sighting) and
  `lamp_changed_since_crossing`. Truth is used only to score label error on 40 train + 30 dev demos.
- Gate (pre-registered against `odom_label_check.json`): plate-bearing label error on approach segments median
  <= 6 degrees and p90 <= 30 on train and dev demos; coverage >= 45% of set-lamp ticks in every source;
  >= 80% of check-to-press ticks learnable under the Hr sampler.
- Kill: bearing p90 > 45 degrees on approaches: the replay is wrong; fix it or require `bot_pose` recordings
  before any belief arm.

**1B. Pixel-free policy on the belief vector** (the decisive test; ~1.5 lane-hours).
- Hypothesis: if a 10 Hz command policy with the Hr recipe's heads, labels, sampler and done weight can walk
  to the plate and finish when given an explicit sensors-only map, then the pixel arm (Stage 2A) is worth its
  compute and this run is its mandatory pixel-free control; if it cannot, an explicit map does not fix the
  command policy at this budget and round 7 goes straight to Stage 3.
- Method: `p2_train.py --no-pixels --mem-vector` (new flags): inputs = the ~50-float belief vector, the 34-d
  body vector and the 9-d instruction into the same GRU-256 trunk; heads, loss weights (done positive weight
  50), the Hr stream sampler with its 2,000-step window warm-up, 16k steps, batch 32, seeds 0 and 1 (training
  is minutes: no image encoder). Deployment: `P2Controller` runs `Odometry` (anchors off), `CameraPose`,
  `detect` and `Memory` on its own observations and builds the same vector per tick. Closed loop on the 40
  selection rooms; the turn-onset probe open loop.
- Gate: the closed-loop stage gate of section 3 in at least one seed (reached >= 6/18, within 0.5 m >= 9/18,
  done after press lamp-right >= half of pressed, done-without-press <= 12/18, no-press >= 15/22) and the
  turn-onset gate (>= 0.85, >= 15/27).
- Kill: both seeds fail the reach gate: Stage 2 is cut (no pixel arm), Stage 3 becomes the main arm, and the
  write-up says that an explicit map summary is not enough for the command policy at this budget.

**1C. Belief heads on the pixel recipe, plus the onset-weighting control** (two runs, ~2.2 lane-hours).
- Hypothesis (dreamzero proposal, as narrowed by its critique): supervising the trunk feature y_t with the
  belief targets gives the GRU a persistent bearing; separately, the BC loss is dominated by
  momentum-predictable ticks (one decision tick per blind turn, ~3% of ticks), so up-weighting the bot's
  motion-onset ticks alone may move the turn-onset gate at zero architectural cost.
- Method: run 1 = R3 recipe (binary walk + 3-way turn, leaky, 16k steps, batch 32, seed 1) + four belief heads
  on y_t (per class: sin, cos, log-dist regression + in-view logit; weight 1; masked to ticks whose track's
  most recent sighting is at or before t and inside the window). Run 2 = the same recipe, no belief heads,
  motion-onset ticks (first tick of each stand->turn, turn->walk, stand->walk change in `cmd_bot`) up-weighted
  x20 in the walk/turn losses. One seed each, ~1 h each at today's load (2 invocations with `--max-minutes 40`
  and the verified resume), then the open-loop probes only (no selection, no confirmation): ~1 min each.
- Gate: bearing read (run 1) and turn-onset (both runs) as in section 3.
- Decision rule (pre-registered): bearing passes and onset passes -> keep the heads in Stage 2A; bearing passes,
  onset fails -> belief present but unused by the imitation head (the P2-B' pattern): Stage 2A keeps the heads
  only if 1B passed, else skip to Stage 3; onset passes for run 2 alone -> the loss weighting, not the
  architecture, was the problem: Stage 2A uses onset weighting and drops the heads; bearing fails -> the heads
  cannot learn the bearing from these windows: drop them, report.

**1D. Map-free pixel-servo baseline** (~0.5 lane-hour, CPU, 4 processes; no training).
- Purpose: the project has no map-free reactive baseline for set-lamp; the wsp critique's bounded test
  settles whether a 64 px cyan servo can reach the plate at all under the 900-tick budget, which also calibrates
  the learned arms' reach gate.
- Method: `p2_servo_baseline.py` (~150 lines): cyan mass and column offset at 64 px with a 5-tick mean, head
  pitch 0.45 during the approach, turn-in-place search at 0.55 when the plate is out of view, stall-and-turn
  after 60 ticks without mass growth, straight 55-tick cross with no turning on the plate, 12 wait ticks beyond;
  two latch variants: (i) oracle latch from the task's `needed_presses` (labelled oracle; scores the servo alone)
  and (ii) the bot's geometry-gated lamp perception as the latch (sensors-only, map-free). 18 press-needed +
  22 no-press selection rooms each.
- Gate / reading: reached plate < 6/18 with the oracle latch kills the whole "compile a stage machine into
  weights" route (no stage machine can press what it cannot reach); >= 10/18 means the servo is fine and
  variant (ii) goes into RESULTS as the map-free reactive baseline row beside the bot's 15/18. Neither variant
  is a learned arm or a candidate for the bar.

**1E. Seed stabiliser (optional lane filler; ~1 lane-hour).** `p2_pretrain.py --lamp-weight 3 --steps 6000`
(labels fit on configs 0-199 only). Gate: frozen-feature lamp >= 0.88 on the registered G4 frames (test
200-299). If it passes, it is the shared encoder init of *every* Stage 2 pixel arm (real and shuffled) through
the new partial loader, so attribution stays matched; it is not a path to presses (lamp-strong seeds already
sit at 0 successes).

### Stage 2. The main learned arm: map-sensor pixel policy (conditional on 1B; ~7 lane-hours, +2.5 for 2B)

**2A. Hr recipe + belief vector input (+ belief heads or onset weighting per 1C's rule).**
- Method: Hr (`--stream --stream-reset`, window warm-up 2,000, 16k steps, batch 32, img 64, leaky, binary
  walk + discrete turn, done weight 50) with the ~50-float belief vector appended to the GRU input beside the
  frame, body and instruction. Chosen over R3 because it shows episode endings 4.5x more often, which stage 4
  needs. Seeds 0 and 1. Mandatory controls at the same budget: (i) the identical arm with the vector shuffled
  across episodes, seeds 0 and 1; (ii) the pixel-free policy from 1B (add its second seed if only one ran).
  Four pixel runs at ~1 h each; selection on the 40 rooms (160 episodes, 10-27 min at 4 processes); then one
  pass on the 80 confirmation rooms (5-11 min) for every seed that passes the stage gate, and for the controls
  of any seed that passes, plus the pixel-free policy.
- Gate: section 3's stage gate per seed, then the bar on the 80 rooms with the intentionality metric; the
  shuffled control must stay at <= 2 reached per seed, or the gain is capacity, not content.
- Kill: both seeds fail the reach gate (< 6/18): the pixel policy cannot use the map it is given; report, and
  Stage 3 is the round's result. If the pixel-free policy ties or beats 2A, the pixels add nothing and the
  write-up says so.

**2B. Targeted post-press coverage (only if 2A reaches the plate but loses done-after-press).**
- Hypothesis (perception critique, Astra option 3): done-after-press is learned in the bot's states and lost in
  the learner's own post-press poses; on-policy post-press states with the shadow bot's labels (turn to the
  remembered lamp, verify, done) are the missing data.
- Method: ~1 h recording through `gpu_run` (4 processes x 1 thread), the 2A pick driving set-lamp on train
  configs with beta = 0 and the shadow bot labelling, memory and `bot_pose` recorded per tick (~450-600
  episodes, ~2 GB); keep the episodes in which the learner pressed plus a matched sample; resume 2A's 16k
  checkpoint for +4k steps on the union with windows anchored <= 64 ticks before the learner's press tick (new
  `--event-index`, modelled on `--recovery-index`); both seeds; a control that resumes +4k steps on the old data
  alone.
- Gate: done after a press with the lamp right in >= half of pressed episodes on the selection rooms, verified
  dones >= half, no rise in done-without-press; then the 80 rooms once (these picks replace 2A's in Stage 5).
  Looks rule, all or nothing: a 2A seed that reaches the plate but fails the done-after-press component of the
  stage gate does NOT go to the 80 rooms; only 2B's pick and its +4k-old-data control do, so the arm family
  gets one look, not two.
- Kill: no gain over the +4k control in both seeds: the done decision is not a coverage problem at this budget;
  report it as the post-press link's negative result.

### Stage 3. Skill selection on the shared sensors-only map (the comparison arm; ~3 lane-hours)

Required by the lab plan and by this brief as the comparison; it is also the paused next step in `STATUS.md`.
- **3A. Cheap check** (~1 lane-hour): extract decision points from recorded set-lamp episodes (skill_log /
  activity transitions; train configs only) with the memory snapshot at each decision (named slots: button,
  lamp on/off, mat, pads; bearing, distance, ticks since seen; lamp belief; target); fit a small MLX classifier
  over {scan_here, find(button), go(look point before the button), press_button, look_at_lamp, done}. Controls:
  shuffled-label selector, and most-frequent-next-skill *given (current skill, lamp belief)*: `refbot.set_lamp`
  is a fixed sequence whose only branches depend on the lamp belief (explore until the lamp is seen; press
  while it mismatches the target), so a control without the belief bit lets the learner win by that bit alone. Evaluate all three on the 40 selection rooms with the bot's perception, map and skills
  executing the chosen skill; the frozen bot is the ceiling (15/18). Gate: learned selector >= 50% of the bot's
  press-needed successes and >= 5 above *both* controls. Kill: within 2 episodes of the belief-conditioned
  control: skill selection is scaffolding; 3B is cut to a documentation run (one seed, reported as such).
- **3B. The arm** (~2 lane-hours): the adapter (the learner emits a skill and a named target slot at every
  skill boundary; the shared map, skills and budgets are the bot's; the learner never receives the bot's chosen
  goal or stage); BC on 3A's decision set plus one decision-level DAgger round (~300 set-lamp episodes,
  ~30 min at 4 processes); two seeds; same controls; selection on the 40 rooms, one look at the 80.
- Gate: the bar, plus >= 50% of the bot's 27/40 (>= 14/40) and >= 8 above the belief-conditioned control in
  both seeds; stage counts per transition type (walk->turn, turn->press, press->search; lessons file item 7).
- Kill: within 2 episodes of the belief-conditioned control on both seeds: reported as scaffolding, not learning.

### Stage 4. The owner's ideas as bounded, hard-gated side probes (optional; run in idle lane time)

**4A. Chicken head** (~0.8 lane-hour; honest version +4-5, not recommended).
- Pre-gate recording (10 min, no training): 20 matched set-lamp bot episodes on train configs, gaze on vs off
  (`record.py params={'gaze': True}`, k=6). Gate: per-tick 64 px frame change while walking down >= 20% (off:
  median 15.1 grey levels), plate-sighting share on approaches >= 0.65 (off: 0.70), falls <= 1 of 20.
- Test-time replay (no training): the best Stage 2 pick on the 40 selection rooms with `P2Controller(gaze=True)`
  (already supported), paired with its gaze-off rows. This is a distribution-shift test, expected to fail; it
  is the kill, not the test of the idea. Gate: reach counts not lower, no falls.
- Honest version (only if both pass and lane time remains after Stage 3): ~1 h gaze-on bot recording (~600
  episodes, ~2 GB), the Stage 2A recipe trained on gaze-on vs a matched gaze-off subset (1 seed each first),
  each deployed with its own reflex setting. Gate: reach and within-0.5 m counts on the 18 press-needed
  selection rooms higher by >= 3 for gaze-on; falls not above gaze-off. Kill: within 2 episodes, or falls.
- Why the odds are < 10%: the reflex touches none of the frames where the plate is first seen, the bot gained
  nothing from it, and a software low-pass or the proprio camera model removes more of the jitter than the
  reflex does. The lamp decision happens standing or turning.

**4B. Pretrained encoder probe: DINOv2 ViT-S/14** (~1 lane-hour; needs the download in section 7).
- Method: a ~150-line MLX ViT-S/14 port reading `model.safetensors` with `mx.load` (no torch); the stored
  96 px frames upsampled to 224 px (the lamp is about one 14-px patch); position embeddings interpolated;
  frozen CLS, mean-patch and patch-token linear probes on the registered G4 frames (fit configs 0-199, test
  200-299; same 1,482 lamp / 3,124 plate test frames as `lewm/probes.py`). The downloaded file is untrusted
  data: own empty directory, `python -I`, no code from it.
- Gate: lamp test accuracy >= 0.90 and plate-in-view AUC >= 0.85 (random features 0.79, LeWM 0.82, HSV mask
  0.908, supervised CNN 0.953). A pass earns one 2-seed Stage 2A run with the frozen features as the image
  branch against 2A's own CNN; never a solo policy run (a stage-1-only fix converts early dones into timeouts).
- Kill: either probe fails, or the owner declines the download: no pretrained-encoder work in round 7.

**4C. LeWM-B / WAM: not funded this round.** Specified so the owner can see what it would have to pass:
SIGReg on within-window-centred latents, predictor conditioned on history-frame proprio (never the target
frame's), aux in-view and lamp heads with labels from configs 0-199 only, matched random-encoder control;
gates: 30-way room probe <= 0.50 (S1 0.93), lamp >= 0.90 and plate AUC >= 0.80, action ranking top-1 >= 0.50
and above the matched control by half the gap to the 0.81 pose-oracle, cross-room cam-pitch r^2 >= 0.50.
Cost 4.5 lane-hours plus 1 for the control; expected presses < 1 per seed even on a pass, because the chain
breaks at acting and finishing, not at the latent. The WAM arm's target latents carry no lamp and its horizon
(2.5-6 s) is an order of magnitude short of the approach; it stays closed. If the GPU is idle after Stage 3
and the owner still wants it, it runs under exactly these gates and nothing else.

**4D. Weight-space programming with Astra, over named inputs** (~2 lane-hours; only after a Stage 2A seed
passes the reach gate).
- Method: export the pick's post-GRU features, head weights and the belief-vector layout to numpy text; Astra
  (`../bin/astra_turn`, numpy only, no Metal) gets the 08/25 harness pattern: a 25-call evaluator on the 40
  selection rooms run by Claude through `gpu_run` returning closed-loop stage counts only, keep-if-better,
  audit by hashes as in 25. Optional one-call "read" task first: from the weights, which named input does the
  done head ignore? The final edit is scored once on the 80 rooms beside the unedited pick.
- Gate: press-needed successes on the selection rooms up by >= 3 with no-press down by at most 2, then the
  confirmation bar. Expected (08/25): a tie with a Claude-written rule; the value is the owner's question
  answered, not extra presses. It cannot program navigation from pixels, and it is said so up front.

### Stage 5. Final evaluation

**Arms** (every one labelled by what is supplied):
1. best step-5 pixel policy, Hr s0/s1 and R3 s1: rows already on the 80 rooms (first look), re-run once on the
   fresh slice;
2. Stage 2A (or 2B) map-sensor pixel policy, seeds 0 and 1: learned decisions + supplied perception;
3. its shuffled-vector control, seeds 0 and 1;
4. the pixel-free belief-vector policy, seeds 0 and 1;
5. Stage 3B skill selector, seeds 0 and 1, with the shuffled-label and belief-conditioned controls;
6. the frozen reference bot (ceiling; 27/40 on the 80, re-run on the fresh slice);
7. the map-free servo baseline (variant ii) if it passed 1D;
8. any Stage 4 arm that passed its own gates.

**Procedure.** Matched budgets (16k steps for every trained policy, batch 32, same data index unless the arm
is defined by its data; selectors trained on the same decision set). Checkpoints picked on the 40 selection
rooms only. Each arm looks at the 80 confirmation rooms once (second project look; paired with step 5). When
the method set is frozen (nothing changes after this point), every listed arm runs once on the fresh reserved
slice (80 rooms); that is the headline table. Reported for every arm, seed and family: success, the bar, stage
counts, the intentionality metric, falls, collision ticks, done timing, inference latency with uptime.
Anything that looks wrong on the fresh slice is written down, not tuned.

## 5. Where the compute goes, and why that order

| order | stage | lane-hours | decides |
|---|---|---|---|
| 1 | 0 protocol, split, instruments, pre-flights | 1.5 | nothing runs without it |
| 2 | 1A labels | 0.5 | whether a belief can be supervised at all |
| 2 | 1B pixel-free policy | 1.5 | whether the command policy can use a map (gates Stage 2) |
| 2 | 1C belief heads vs onset weighting | 2.2 | which representation lever, if any, moves turn onset |
| 2 | 1D servo baseline | 0.5 | whether reactive pixel reaching is possible; a baseline row |
| 2 | 1E seed stabiliser (optional) | 1.0 | variance reduction only |
| 3 | 2A map-sensor pixel policy + controls | 7 | the pixel arm's bar |
| 4 | 2B post-press coverage (conditional) | 2.5 | the finishing link |
| 3 | 3A/3B skill selection + controls | 3 | the comparison arm's bar and how much is scaffolding |
| 5 | 4A/4B/4D owner probes (optional) | 4-5 | the owner's hunches, each with a kill |
| 6 | 5 fresh-slice headline | 1.2 | the headline |

Core (1-4 and 6): about 22 lane-hours; with Stage 4: about 27; the honest gaze arm and LeWM-B would add ~9
and are not recommended. Stages 1B/1C/1D can run on two lanes at once (1D is CPU); Stage 3A can fill a lane
while Stage 2 trains.

## 6. Resource plan (AGENTS.md)

- **Jobs:** at most 2 concurrent `gpu_run` jobs for this project, counted by PID (`gpu_run` has 8 slots and
  enforces nothing; 5 jobs ran at once in step 4 by accident). Evaluations: `P2_PROCS=4` with
  `OMP_NUM_THREADS=1` per process; training: 4 threads.
- **Invocations:** <= 45 min each. `p2_train.py --max-minutes 40 --resume` (resume bit-exact; re-verified at
  tiny size after every trainer change: save, kill, resume, loss continues). A 16k-step run takes ~24 min alone
  (11-14 steps/s) and 45-60 min at load 9-11, so plan 2 invocations per run and index every schedule by steps,
  not wall time (lessons file item 11); a run that ends far under its allowance is a bug until shown otherwise
  (item 10).
- **Memory:** every process < 16 GB (the P2 policy trains at a few GB; no LeWM-size model in the core plan).
- **Timing references** (`results/p2/select_*.json`, `confirm_p2_*.json`, per-episode `wall_s` summed and
  divided by 4 processes): selection 160 episodes 10-27 min; confirmation 80 rooms 5-11 min; a 450-600 episode
  recording ~1 h at 4 processes (~9 episodes/min under load).
- **Load now:** 9.0 at 21:00 with two `gpu_run` jobs from project 28 (`r6_train.py`, `r6_run_eval.sh`); the
  brief said four. Start Stage 0 now (CPU); start Stage 1 training when load drops below ~8 or accept ~2x.
- **Disk:** 954 GB free. New: belief side files ~0.3 GB; checkpoints ~0.1 GB per run (runs/ holds all P2 runs
  so far in 1.8 GB); recordings ~2 GB per hour (2B; optional 4A); DINOv2 0.09 GB; results and logs < 0.1 GB.
  Core < 5 GB, with options < 8 GB. No 96 px LeWM cache (37 GB) and no new LeWM cache; the existing 21 GB of
  LeWM caches can be deleted if 4C is not run (ask first).
- **Files written during execution** (all inside `32_microduck/`): `results/r7/protocol.json`,
  `results/r7/ledger.jsonl`, `results/r7/*.json` per run, `splits/r7_*.json` + manifest, `p2_belief_labels.py`,
  `p2_train.py` flags (`--mem-vector`, `--no-pixels`, `--belief-heads`, `--onset-weight`, `--event-index`,
  partial `--init`), `p2_learner.py` (vector input, belief heads, memory-vector builder in `P2Controller`),
  `r7_probes.py`, `p2_servo_baseline.py`, `p2_skillsel.py` (adapter + selector), `record.py` (`bot_pose`),
  tests for each, `RESULTS.md` "Round 7" section, `STATUS.md` lines. Stored episodes stay untouched and
  bit-reproducible. No git commands (the orchestrator commits).

## 7. Owner asks (one line each)

1. Approve drawing a fresh reserved slice (4 new SLOT entries in `make_splits.py`, 20 rooms per family, written
   and checked in Stage 0, looked at once in Stage 5).
2. Approve the second look at the 80 confirmation rooms (one pass per arm, paired with step 5, never a headline).
3. Approve supplied sensors-only perception at test time for Stages 1B/2 (`odometry.py`, `bodymodel` camera
   pose, `perception.detect`, `refbot.Memory`) and the bot's skills/map/budgets for Stage 3, disclosed as
   "learned decisions + supplied perception/skills" in every table.
4. Approve the new code listed in section 6 and the `record.py` `bot_pose` field for new recordings only.
5. Recording budgets: Stage 2B ~1 h on-policy set-lamp recording (~450-600 episodes, ~2 GB), conditional on 2A
   reaching the plate; Stage 4A pre-gate 10 min (~0.1 GB); optional 4A honest arm ~1 h gaze-on (~2 GB).
6. Optional download for Stage 4B: `facebook/dinov2-small` on Hugging Face (DINOv2 ViT-S/14, Apache-2.0,
   `model.safetensors` about 88 MB, 22M parameters), loaded in MLX via `mx.load`, treated as untrusted data.
7. Not requested: the LeWM toy checkpoints `quentinll/lewm-{pusht,cube,tworooms,reacher}` (MIT, 72.3 MB each):
   one toy environment each, other action space, 224 px patch 14, not a general encoder; S1's failure is what
   the latent learned, not convergence.
8. Optional Astra budget for Stage 4D (two codex threads, ~1-2 h wall each, several hundred k tokens each as
   in 08/25), only after a Stage 2A seed passes the reach gate.
9. Scheduling: the core plan is ~22 lane-hours at <= 2 jobs; run Stage 0 now and Stage 1 after projects 20/28
   release the GPU, or accept ~2x slower training.
10. Decision on a grey area: the teacher's `activity` strings (its program state) as an auxiliary *target* in
    an ablation. Default: not used anywhere (stage must never be an input, and it is not needed by any stage
    above).
11. Decision on Stage 4C (LeWM-B): default not funded this round; say if you want it run under its gates
    after Stage 3.

## 8. Risks

- **Belief present but unused** (P2-B' precedent: handing the done head the match belief moved its logit 0.07
  against a 4.0 gap). 1B and 1C's open-loop gates catch it before Stage 2; the shuffled-vector control
  separates content from capacity.
- **Label frame mismatch:** recorded track positions sit in the bot's anchor-corrected frame while replayed
  poses are uncorrected (end error worst 0.37 vs 1.2 m); 1A's label gate and `bot_pose` in new recordings
  bound it; deploy with anchors off to match training.
- **Appearance shift:** the detector's lamp-off recall is 0.50-0.78 on arrangements/appearances-dev and the bot
  gets 4/10 on appearances-dev; supplied-perception arms inherit that ceiling per family; report per family.
- **The done decision:** 2A trains the done head as before; ending exposure (Hr) makes "done fires more" the
  expected effect, so the comparator is Hr's own done-without-press (27-28/40) and a rise fails the arm even
  with more presses. 2B is the designed fix if reaching improves and finishing does not.
- **Obstacles:** 27-39% of straight lines to the plate are blocked; a learned policy must detour from pixels,
  the servo baseline has only a stall-and-turn rule, and the selector inherits the bot's planner.
- **Seed noise and small counts:** differences of 1-2 episodes are noise; judge only on pre-registered gates,
  both seeds, and the paired step-5 rows.
- **Room reuse:** selection rooms used for 17+ picks and every diagnostic; 30 overlap with the p2_dev demos;
  the 80 rooms are on their second look. The fresh slice is the only clean number, and it is used once.
- **Supplied capability dominates:** Stage 3 will likely pass the bar by scaffolding; the belief-conditioned
  control and the per-transition stage counts are what keep the claim honest. The bar itself is gameable by a
  scripted done; the intentionality metric is reported beside every count.
- **Compute contention:** load 9-11 now; 2 invocations per 16k run; do not read partial runs as results.
- **Teacher label errors:** the bot's lamp belief disagrees with the simulator on 2.6-7.5% of known ticks; 81 of
  244 press demos have more than one lamp flip and 54 end with a done that never re-saw the lamp, so the
  post-press demo program is multimodal; truth may only remove labels (amendment 3), never make them.
- **Downloaded weights** (4B): a torch-trained model loaded by hand into MLX; a shape or normalisation slip is
  silent garbage; the probe gate is there to catch it; own directory, `python -I`.

## 9. Reporting rules

- `results/r7/ledger.jsonl`: one row per run or probe, including kills and scratch runs (name, seed, data
  index, steps reached, steps/s, load, wall, gate values, pass/kill, files), appended as each finishes.
- `RESULTS.md` "Round 7" section: the funnel, every stage's table with baselines from the frozen files, what
  worked, what did not and why, surprises, per-family numbers, the intentionality metric beside every bar count,
  disclosures (supplied perception, second look, code versions across runs, job counts).
- `STATUS.md`: one line per milestone with timestamp; what is running; what is next.
- Claims: gates relative to measured baselines only; no "first ever" (wrong twice before); a result on the 80
  rooms is a paired comparison; a headline needs the fresh slice after the method set freezes; label any
  controller that sees simulator truth as an oracle (the 1D oracle latch).
- Demo and checkpoint (AGENTS.md done contract): the best arm's checkpoint copied to `checkpoints/` with a
  reload script, and a clip of one intentional press-needed success (or of the clearest failure, with its
  mechanism) in `media/`.

## 10. Honest framing for the owner

The owner's instinct that the duck is close is supported by the data in one specific sense: the plate is on
the learner's own frames by tick 20-37, the lamp is read well enough, and the teacher needs only ~500 of 900
ticks. What the learner never acquired is the step from seeing to doing: starting a turn toward a remembered
plate, walking at a plate that is in view, and refusing to say done when the lamp it is looking at is wrong.
None of the four routes named in the brief addresses that step: a world model predicts frames the policy
already has, a pretrained encoder sharpens a read that is already sufficient, weight-space edits need a circuit
to edit and R3 has none to spare, and a steadier camera does not change what the duck does with what it sees.
Round 7 therefore spends its first three hours on the two cheapest tests that can say whether an explicit
belief makes the policy act (1B, 1C), its main budget on the arm those tests endorse, and the comparison on the
skill selector the lab plan always called for. If 1B fails on both seeds, the honest round-7 result is that a
10 Hz command policy at this budget does not act on an explicit map either, and the learned content of this
project lives at the skill-selection level.
