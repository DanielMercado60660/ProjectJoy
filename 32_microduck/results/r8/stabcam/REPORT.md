# R8 Gate B: how steady can the camera view get? (32 Microduck, 2026-11-05, 15:08–16:25)

> Saved by the session lead from the builder's final message (subagents can't write report files here).
> **Lead's checks (2026-11-05):**
> - Every table number below matches `wobble.json`.
> - Measurement runs start at 15:49:55, after the protocol was frozen (15:39) and amended (15:45).
> - The bars (2.0° / 1.0°) are the plan's "~2° / 1°" (`design/R8_STAGED_PLAN.md` §3).
> - The head and reflex rows reproduce M3.1's walking average (6.0→3.9° yaw, 5.0→3.7° pitch).
> - The frame set holds 1,824 frames per stream in the `off` files and 2,086 in the `on` files.
> - Lamp in view: 5–9% of frames per stream; plate: 25–32%.
> - `data/r8_stabcam/` is 372 MB in all: `runs/` 209 MB, `runs_nohl/` 49 MB, `frames/` 104 MB, `dev/` 9 MB. All gitignored.

## Verdict

**The pre-registered rule that fired is rule 2.** On the primary (scripted) set, the sensor-only software stabiliser
("eis") passes the gate: 1.23° yaw / 0.84° pitch against bars of 2.0 / 1.0. eis renders the head camera wide and
warps it by the rotation estimated from the gyro, gravity and joint encoders. Neither latency row passes:
- With the sensor estimate 20 ms late: 1.92 / 1.59.
- 20 ms late but extrapolated at the last rotation rate: 1.34 / 1.04, missing the pitch bar by 0.04°.

So software stabilisation needs sensor-to-frame sync well under 20 ms, or a better predictor. That is a hardware
requirement. On the secondary bot episodes, eis (0.87 / 0.61) and the extrapolated row (1.13 / 0.77) pass; the
20 ms late row fails (2.00 / 1.22).

**Reading: stage 2 is worth building as software (a wide lens plus a rotation warp driven by IMU and encoders),
not as a physical chicken head.**
- The `gaze.py` head reflex fails (3.92 / 3.70); this reproduces M3.1.
- eis matches the ORACLE (simulator truth: 1.17 / 0.83; bot 0.86 / 0.61), so sensing isn't the limit.
- eis is steadier than a trunk-mounted camera (2.54 / 0.88).

**Costs:**
- In turns the stabilised view lags the head by about 0.5 s. The smoother is causal and one stride long; the lag
  is visible at 16.5 s in the video.
- It needs a ~140° lens.

**Not measured:** the gate's other half, whether the stage 3 gates improve on the steady stream. The frame set for
that test is saved.

## Numbers

Wobble is the std about a 1 s moving mean, in degrees. Frame change uses the 4A pipeline: 64 px grey, 10 Hz,
walking only. Scripted rows are train-0020..0025 × M3.1's 5 walking segments; bot rows are 9 walking set_lamp
episodes.

| stream | scripted yaw/pitch | gate | bot yaw/pitch | frame change: scripted (bot median) |
|---|---|---|---|---|
| head (as now) | 6.05/5.01 | fail | 7.79/3.71 | 11.68 (13.37) |
| head + reflex | 3.92/3.70 | fail | 4.42/3.28 | 9.87 (11.85) |
| trunk camera (sim only) | 2.54/0.88 | reference | 2.55/0.52 | 7.48 (5.66) |
| ORACLE (truth) | 1.17/0.83 | excluded | 0.86/0.61 | 6.04 (5.30) |
| **eis** | **1.23/0.84** | **pass** | **0.87/0.61** | 6.82 (6.01) |
| eis, 20 ms late | 1.92/1.59 | fail | 2.00/1.22 | 7.65 (6.83) |
| eis, late + extrapolated | 1.34/1.04 | fail | 1.13/0.77 | 7.05 (6.11) |
| eis + reflex | 1.32/0.98 | pass | 1.36/0.79 | 6.38 (5.75) |
| phase-locked head, 1 per stride | 4.52/3.39 | fail | 3.75/2.01 | – |

Other rows (roll, turning in place, two frames per stride, reflex with lag) are in `wobble.json`.

eis usability: 100% of pixels valid. The crop limit was hit on 3.1% of scripted walking steps (14.7% when steering
+0.4) and 0% in bot episodes, within the 5% bar.

## Findings

- **4A reproduced.** Bot head frame change is 13.37 (4A: 13.5). The reflex's −15% fails 4A's −20% bar again;
  eis is −42%.
- **Headlight artefact.** MuJoCo's headlight turns with each camera, so head and eis frames flicker with head
  pitch. With it off: head 11.67, eis 5.71, ORACLE 5.79.
- **Phase-locking** (one frame per left-foot touchdown, detected from encoders and gravity only):
  - The sampled sequence wobbles 4.52/3.39, vs 5.72/4.96 for an unlocked clock at the same rate (bot 3.75/2.01
    vs 7.68/3.68).
  - Change between consecutive frames is unchanged (15.7 vs 15.2).
  - Two frames per stride is worse. On top of eis it adds nothing.
- **Blind scripted walks hit walls** (post hoc analysis, `scripted_per_room.json`):
  - 6 of 30 runs stalled; the gait lost its rhythm and every stream wobbled more (ORACLE 1.74/1.77).
  - In the other 24 runs eis is 1.10/0.61.
  - eis's per-segment pitch failures (straight walking at 0.3 m/s 1.09°, 0.4 m/s 1.70°) are in those rooms, as
    are the ORACLE's.
- **Trunk level is about 2.5° / 0.9° here**, plus 1.9° of roll.

## Files

- **Code:**
  - `r8_stabcam.py`: adds 3 cameras with physics and the bot's own frames bit-identical; the estimator, filters,
    warp, gait-phase detector and episode drivers.
  - `r8_stabcam_{run,analyze,table,perroom,video,frames}.py`.
- **Tests:** `tests/test_r8_stabcam.py`, 7 tests in 4 s. `smoke.sh` passes (170 tests) but took 4 min at load
  ~17, over the 2-minute target.
- **Results:** `protocol.json` (amended once, before measurement), `dev_filters.json`, `dev_grid.txt`,
  `wobble.json`, `wobble_nohl.json`, `scripted_per_room.json`.
- **Demo:** `media/r8_stabcam.mp4`, 56 s.
- **Frame set** (gitignored, `data/r8_stabcam/frames/`, 104 MB; `README.md` beside it):
  - 1,824 paired 224 px frames each for head, trunk, ORACLE and eis; 2,086 each for reflex and eis + reflex.
  - Truth labels.
  - The lamp counts as in view only with its centre in frame and a clear line of sight, so only 5–10% of frames
    have it.

## Compute

About 2.4 process-hours, at most 4 processes per job, all through `r8_job.sh`. No downloads.

## Deviations

- **Filter.** The brief's example (yaw low-pass, τ ≈ 1 s) lost on the dev rooms: it lags turns by 30–45°, so the
  crop hit its limit.
  - I froze instead two stride-length moving averages (0.52 s delay) and a 140° source render.
  - Tuning used train-0000..0007, disjoint from the measurement rooms.
- **Phase-locked control** was amended before measurement. Random pairs with the kept pairs' gap are one stride
  apart, so they are phase-matched by construction; I added fixed-clock controls.
- **Bot episode train-0042** never walked, so it is unscored.
- **`duck.py`** was changed mid-session by the BAM builder (additive, opt-in). I used the default body.

## Open issues

- Sensors are idealised: exact gravity, a noiseless gyro, and no motion blur or rolling shutter.
- Head translation and the turn lag are not removed.
- Bot successes: 5/10 with the reflex vs 7/10 without.

## Next

1. Run the stage 3 representation gates on the frame set: eis vs head vs ORACLE.
2. Measure the real camera's field of view and its sensor-to-frame latency.
3. Try a feed-forward target to cut the turn lag.
