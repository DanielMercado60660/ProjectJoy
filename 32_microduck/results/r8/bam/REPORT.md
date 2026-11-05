# R8 Gate A: BAM M6 actuator port and M0 tracking battery (2026-11-05)

> Saved by the session lead from the builder's final message (subagents can't write report files here).
> **Lead's checks (2026-11-05):**
> - Every body's numbers in the table below match `summary.json`.
> - `protocol.json` was written at 15:39:29. The gate runs start at 15:48; only the disclosed smoke, timing and
>   parity runs came before it.
> - Re-running the XML battery (`gate_xml.json`) matches `results/m0/gate.json` key for key except `wall_s`.
> - `tests/test_r8_bam.py` passes: 9 tests in 2.5 s.
> - `bam_m6.py` carries the Apache-2.0 notice, the Rhoban attribution, the commit pin and the citation.
> - The indexing bug below is confirmed by reading the source at 62bd8ce:
>   - `bam/mujoco.py` builds its friction selector from `actuator.trnid[0]` (a joint id) compared against
>     `efc_id`.
>   - `bam/mjlab.py` notes that "efc.id of such a constraint is the DOF index" and scatters by DOF.
>   - On a floating base, joint id ≠ DOF id.
>
> **Lead's notes (inferences, not the builder's):**
> - **The forward shortfall is the policy's own.** Pollen's v8 eval in their training sim
>   (`vendor/microduck_rl/docs/velstand_policy.md`) gets 0.43× too. No actuator model fixes it, so the
>   forward bar (±25% of the command) cannot pass on any body. An amended body-v2 gate ("match Pollen's sim") is in
>   `design/R8_STAGED_PLAN.md` §2, written before any further walking measurement.
> - **Turning and drift do track in Pollen's sim but not in ours.** v8 was trained on
>   `Mjlab-VelStand-Rough-Backlash-MicroDuck` and evaluated flat + backlash: in place ±0.6 → +0.52/−0.57,
>   ±1.0 → +0.97/−0.87. Backlash is the one concrete difference left.
> - **Commits:** `duck.py` now imports `bam_m6`, so the two must be committed together.
> - **Upstream:** reporting the indexing bug to Rhoban/Pollen is an external post, the owner's call.
> - **Downloads:** they live in the session scratchpad, which is temporary; the port's provenance is pinned in the
>   `bam_m6.py` header.

**Verdict: FAIL. The port matches Pollen's BAM path bit for bit, but it does not close the tracking gap.** The
primary BAM body had 0 falls. Forward speed (0.38× the command), turning (1.43×) and straight-line drift
(+0.18 rad/s) are all outside the pre-registered gate. On this evidence body v2 can't be adopted. The choice is
between porting backlash and then re-measuring, or fine-tuning; this gate can't tell which.

## Files (in `32_microduck/`)

- **`bam_m6.py`:** numpy port of Rhoban bam at commit 62bd8ce. It is Apache-2.0, and the header carries the
  attribution, licence note and citation.
  - Covers the firmware P loop, DC motor with back-EMF, battery sag, an optional current limiter, the M6 friction
    terms, the stiff friction constraint, armature and a 3–6 physics-step target delay.
  - It converts compiled models, so rooms work too.
- **`duck.py` (additive):** `BodyConfig(actuator="bam_m6", bam={...})`; the default stays `"xml"`. Save/restore
  is bit-exact on the BAM body (tested).
- **Scripts:** `r8_bam_defaults.py`, `r8_bam_parity.py` and `r8_bam_gate.py`, a wrapper that leaves the m0 scripts
  and results untouched.
- **Tests:** `tests/test_r8_bam.py` (9 tests).
- **Results** in `results/r8/bam/`: protocol.json, summary.json, gate_*.json, start_*.json, speed.json,
  parity*.json.

## Parity

- **Defaults:** 3 default bodies × 300 steps are identical array for array to a capture taken before the edit.
  Re-running the XML battery reproduces results/m0/gate.json exactly, apart from wall time.
- **Model conversion:** all 481 MjModel arrays are identical to infer_policy's MjSpec route.
- **Replay through bam's own MujocoController:** 2,800 physics steps, bit-identical.
- **Closed loop vs infer_policy.py's own BAM loop:** 700 control steps, bit-identical in observation, action and
  joint positions.
- **Torque level, on 1,400 harvested states, stated tolerance 1e-9 N·m:**
  - max |Δ| = 0 against bam.mujoco;
  - at most 1.4e-17 for the training friction semantics against bam.mjlab's own torch code.

## Gate (pre-registered at 15:39, before any battery ran)

The primary body is infer_policy's numbers (7.4 V, kp 200, no current limiter), plus training's friction semantics,
an actuator lag drawn from 3–6 physics steps, and the 1-step joint-velocity lag (as M0).

| chosen bounds, median (p10–p90) | gate | M0 XML | **BAM** | bam_infer (Pollen's CPU path, no lag) | bam_train |
|---|---|---|---|---|---|
| forward | 0.75–1.25 | 0.44 (0.40–0.48) | **0.38 (0.34–0.44)** | 0.32 | 0.37 |
| backward | – | 0.52 (0.51–0.55) | 0.43 (0.23–0.48) | 0.33 | 0.43 |
| turn, all | 0.75–1.25 | 1.39 (1.21–2.27) | **1.43 (1.22–1.98)** | 1.28 | 1.42 |
| straight drift, rad/s | \|·\| < 0.05 | +0.22 (+0.04–+0.28) | **+0.18 (−0.03–+0.26)** | +0.03 | +0.16 |
| lateral vy ratio | – | 0.17 | 0.17 | 0.12 | 0.16 |
| falls (pads + 40 runs) | 0 | 0 | **0** | 0 | 0 |
| pads reached | – | 16/16 | 16/16 | 10/16 | 16/16 |

All 10 bodies fail; the other variants are in summary.json. bam_train is the current limiter on, the 8.2 V force
range, and the lag re-drawn every physics step.

**Starts from standing** (out of 4 standing states):

| command | XML | BAM |
|---|---|---|
| vx 0.1 | 0 | 0 |
| vx 0.15 | 2 | 0 |
| vx 0.2 | 4 | 4 |
| wz +0.3 | 1 | 0 |
| wz −0.3 | 2 | 1 |
| wz +0.5 | 4 | 1 |
| wz −0.5 | 4 | 4 |
| backward −0.3 | 4 | 1 |
| lateral vy 0.3 | 1 | 0 |

Under BAM the dead zone is larger and lopsided: left turns start less often than right turns.

**Speed** (load ~17.5): XML 3,490 steps/s, BAM 1,756 steps/s (0.50×). The plan's 7,690 steps/s was measured at
load 4.9.

## Reading

- **Predictions:** 2 right (forward fails, as in Pollen's own 0.43×; 0 falls) and 2 wrong (turn and drift were
  expected to pass).
- **At the same lag the actuator physics barely matters:** BAM with lag 4 vs XML with a 4-step lag gives turn 1.41
  vs 1.39 and drift +0.17 vs +0.22.
- **The current limiter never engages** in steady walking.
- **Post-hoc fixed-lag sweep, lags 0–6 (not pre-registered):**
  - the turn ratio runs 1.30 → 1.44 and is never at or below 1.25;
  - drift rises with lag (+0.03 → +0.23) and meets the 0.05 limit only at lag 0, where gaits fail to start
    (10/16 pads);
  - so no reading of the delay unit rescues the gate.

## Findings about Pollen's CPU BAM path

- **Indexing bug:** `bam.mujoco.MujocoController` matches friction rows by joint id, but `efc_id` is a DOF id. On
  this floating-base robot it subtracts the wrong joint's friction, a median 0.0015 N·m or about 7% of the friction
  budget. `bam.mjlab` (the training code) gets it right.
- **Other differences from training:**
  - the quadratic friction term is sign-gated only on the CPU path;
  - training keeps the XL330's 1.75 A current limiter, which infer_policy disables;
  - the force range is set at 7.4 V on CPU vs 8.2 V in training;
  - infer_policy has no actuator delay by default.

## Compute

About 45 CPU-minutes, no GPU. The gate job took 6.6 min and the post-hoc sweep 4 min, each in one r8_job slot.

## Deviations

- protocol.json records a hash of r8_bam_gate.py from before a CLI-only edit. Bodies, metrics and verdict code are
  unchanged.
- **Delay unit:** bam's docs say "simulation steps". mjlab's lag-resampling schedule isn't readable without the
  mjlab source, so I drew one lag per episode; the per-step variant gives the same result.
- **Training commit:** that 62bd8ce is the commit v8 was trained on is inferred from Pollen's lock file, not proven.
- **Downloads:** only into the isolated venv. Paths:
  `/private/tmp/claude-501/-Users-danielmercado-ProjectJoy/763586ee-fb5f-4653-b2fe-46e5c3a47eac/scratchpad/bam_src` and
  `/private/tmp/claude-501/-Users-danielmercado-ProjectJoy/763586ee-fb5f-4653-b2fe-46e5c3a47eac/scratchpad/bam_venv`.
- **Smoke:** `bash smoke.sh`: 170 passed.

## Next

1. **Port backlash, then rerun this battery.** Pollen trained the backlash variant. Load
   robot_allcollisions_backlash.xml; the servo's position feedback and the observed joint position both become
   main + backlash joint.
2. **Not ported or measured:** mjlab's integrator and solver settings, MuJoCo Warp vs CPU MuJoCo, domain
   randomisation, and the IMU observation delay.
3. **Otherwise:** fine-tune velstand or train a residual policy (plan §2 step 4). Report the indexing bug upstream
   to Rhoban and Pollen.
