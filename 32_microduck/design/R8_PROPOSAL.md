# Round 8 proposal: learn from the servo, not the bot (draft, 2026-11-05; nothing started)

> **Folded into `R8_STAGED_PLAN.md` (2026-11-05, owner's staged direction).** The servo-as-teacher arm survives there as the demo source for stage 3 and a baseline row, run on the new body after stage 1.

## Bottom line

Change the teacher before changing anything else. Train the best pixel learner we have (the R3 discrete-heads
recipe) on demos from the hand-written 64 px servo, with the bot-demo version as the paired control. Then run
DAgger with the servo relabelling the student's own states. Every learned policy so far copied the reference
bot. The bot's actions depend on its map and hidden program stage, and a pixel policy can't recover those from
frames. The servo's actions depend almost entirely on what is in view right now.

## Why this, and why now

- **The task can be solved from 64 px.** A ~150-line servo gets 31/40 press-needed on the fresh slice; the bot
  gets 21/40 (`results/r7/table_fresh.md`). It holds on every family: 7/10, 9/10, 5/7, 10/13 against the bot's
  4/10, 8/10, 1/7, 8/13.
- **Pixel students of the bot don't learn it:** 0 to 2/40. The pixel-free belief policy gets 1 to 4/40.
- **Earlier DAgger failed structurally, not by bad luck.** The shadow bot labels the student's states from a map
  it built along the student's path. Its labels were 1.9% walk, and it said done with the lamp wrong in 18 of 97
  episodes (Phase 2 step 2). A reactive expert labels from the current frames, so its labels on the student's
  states stay valid. That is the textbook condition under which DAgger works.
- **It's the RESULTS.md Round 7 "next" item (3).** This sharpens an existing lead.

## The servo's state (what the student can't see)

The servo is reactive but not stateless. These are the parts the student must infer or be given:

| servo state | what it does | risk for a student |
|---|---|---|
| 5-tick mean of blob mass and column offset | smooths the cue | low: a GRU or a frame stack covers it |
| stall timer (60 ticks without mass growth → 90° turn) | escapes dead ends | medium: a rare, timing-based action |
| **cross counter (55 ticks straight once the blob leaves the frame bottom, then 12 wait ticks)** | **plate entry** | **high: the plate is out of view during the cross. This is exactly the step where every learned policy stalls.** |
| explore walk after a full turn with no sighting | search | low to medium |
| tries (at most 3) and the BeliefTracker lamp memory | the lamp latch and done | high: done depends on lamp memory |

Decision for arm A: the student gets pixels plus the same supplied lamp-belief slots that 1B used. Results are
labelled "learned decisions + supplied lamp perception", as in Round 7. Arm A-px (pixels only) is the harder
second step. Measure cross-phase agreement (student vs servo labels during the 55 cross ticks) as its own probe.
If the student fails, this probe says whether the failure is at plate entry.

**Teacher weakness the student will inherit:** the servo gets only 40/56 no-press rooms (bot 56/56); it presses
when its first full turn misses the lamp. Report both columns. The press-needed target is the servo's 31/40, not
100%.

## Arms (one decisive arm plus its control; not a factorial)

| arm | data | learner | seeds | question |
|---|---|---|---|---|
| A0 control | bot demos, same room set and episode count as A | R3 recipe, same steps | 2 | baseline (R3 s1 exists; re-run both seeds at matched size) |
| **A** | **servo (lamp latch) demos, `record_episode(cold_renderer=True)`** | **R3 recipe + lamp-belief slots** | **2** | **was the teacher the problem?** |
| B (if A clears its gate) | A's data + servo-DAgger rounds (shadow servo along the student's path, β from 0.5 to 0) | same | 2 | does DAgger finally work with a reactive expert? |
| A-px (after A) | servo demos | R3 recipe, pixels only | 2 | is the supplied lamp belief needed? |
| C (optional) | as A or B | + action chunks (k = 8, execute 4) and/or a PIDM-style head (predict blob mass/offset k ticks ahead, condition the action heads on it) | 2 | can a committed chunk carry the blind cross? |

Reading the result: if A beats A0 clearly, the teacher was the problem. If A ≈ A0 ≈ 0, the learner is the problem
even with a reactive teacher. In that case the next round targets the learner (chunks, PIDM, a frozen pretrained
encoder), not the data.

## Gates (draft: fix the numbers before any training, relative to measured baselines)

- **Selection (40 R7 selection rooms, 18 press-needed):** the servo reaches 18/18 and the best pixel arm 1 to 4.
  A passes if it reaches ≥ 9/18 and gets ≥ 4/18 press-needed successes **on both seeds**, and beats A0 by ≥ 3 on
  each seed.
- **Headline:** a new reserved slice. `make_splits.py` seeds are base + 1,000,000 × slot; R7 used slots 9 to 12,
  so an `--r8` option on slots 13 to 16 gives 96 fresh rooms, one look per frozen arm. Bar: ≥ 2 press-needed per
  seed and ≥ 5 over two seeds (the R7 bar), no-press not below the servo's rate − 4, with the bot (21) and the
  servo (31) as rows.

## Cost (from measured rates)

- Servo recording: 300 train rooms × set_lamp. Median servo episode is ~340 ticks (bot ~700), recorder ~38
  ticks/s/proc, 4 procs: roughly 30 to 45 min. One 45-minute guard invocation, with resume.
- Training: same as step 5 (R3 recipe, 16k steps, about 1 to 2 lane-hours per seed with resumes). Two lanes, so
  A0 + A takes about one evening.
- DAgger round (B): ~1 h recording + retrain.

## What the papers give us (read 2026-11-05)

| source | what it is | usable here? |
|---|---|---|
| PIDM, Microsoft, ICML 2026 (arXiv 2601.21718) | predict a future state, then an inverse-dynamics model picks the action; beats BC by 3× (2D nav) and 66% (3D game) in demos needed | **Recipe yes (arm C).** Small and MLX-portable. Read the full paper before building: only the abstract was read. The IDM part can also train on the 616 noisy-recording episodes. |
| LPS, Yonsei/MSR, RSS 2026 (2603.05296) | offline RL: a latent actor steers a frozen one-step flow policy using a Q-critic; beats BC 56% vs 29% (real robot) | **Later.** It improves beyond a teacher using reward, which the sim gives for free. Worth it once a student reaches ~10/40. The code is JAX/CUDA (rule 4), so re-implement in MLX. |
| LeWorldModel (2603.19312) | 15M JEPA from pixels + CEM planning toward a goal image | Already tried: R6 failed all 4 gates (latent = room code, lamp 4.3 px). A from-scratch JEPA on our frames doesn't keep the lamp. |
| RoboJEPA (FAIR, 2610.10515) | 22M to 8B action-conditioned predictors on frozen V-JEPA 2.1 ViT-G; scaling laws; goal-image MPC | No: PyTorch/CUDA, mandatory ViT-G encoder, CC BY-NC-SA, tabletop arms. Lesson: latent planning needs a strong *pretrained* encoder, which LeWM confirms by failure. |
| PRH proof (Balestriero & LeCun, 2610.39302) | theory: JEPAs with "good views" converge to one representation; next-token = "bad views" | No direct use. Loose link: LeWM's next-frame objective is a "bad view" pairing, which fits its room-code collapse. |
| DreamZero (NVIDIA, 14B), GR00T N1.7 (3B), Cosmos3-Edge (4B), LingBot-VLA/VA 2.0, HY-WorldPlay (8B) | world-action models / VLAs / video world models | No: CUDA-only, 3 to 14B, manipulation or video only, no biped. Shared recipe idea: **action chunks**, which go into arm C. |
| GEN-1.5 (Generalist) | closed model; sim demo as an in-context prompt | Concept only, no weights. |
| PERSIST (2603.03482) | video world model with a persistent 3D voxel state | No: a renderer, not a controller. Our BeliefTracker is the hand-built version of its idea. |

## Not this round

Downloads or pretrained encoders (DINOv2 / V-JEPA: still unapproved), any 1B+ model, an LPS critic, 4D weight
edits, LeWM again, chicken head, skill selection. 2B on the 1B pick stays an open, separate ask.

## Order of work (after approval)

1. Fix the gate numbers and the A0 data size in `results/r8/protocol.json` before recording.
2. `make_splits --r8` (slots 13 to 16); `check_splits` passes.
3. Record servo demos (lamp latch) on the train rooms; check that a replay of the recording matches the evaluator
   bit for bit.
4. Record matched bot demos for A0 (or subsample the existing ones to the same count).
5. Train A0 and A, 2 seeds each, in two lanes; select on the 40 rooms; read the gate; add the cross-phase probe.
6. If A passes: B (servo-DAgger). Either way: A-px, then freeze, then the fresh R8 slice.
