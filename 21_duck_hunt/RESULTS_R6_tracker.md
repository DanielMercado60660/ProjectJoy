# 21 Duck Hunt, R6: robust duck tracking under scene shifts

Tracker half only (2026-11-04). The flight and hunter files were not touched. Everything here comes from real runs
in this folder; each number names its source file. "Lost" = estimate more than 5 px from the duck
centre (the duck is ~7 px long), as before. 95% CIs are bootstraps over **scenes**, not frames,
since frames within a scene are correlated (2000 resamples, 60 scenes × 40 frames per set).

## Verdict

**Round 2 (official trackers are now the R2 ones):**
- The learned tracker trained with haze in DR loses **0.0–0.5%** on all new pre-registered
  families. On fresh seeds of the diagnosed all2 family it loses **5.5%** (R1: 18.7%), and it stays
  ≤ 0.4% on the older shifts except bg2 (1.3%).
- The R2 classical tracker loses 8–15% on the new families and 30.5% on all2 fresh seeds, and is
  still ~36% at dusk.
- The new families turned out easy for every tracker, the old R5 ones included, so all2_f2 is
  the informative test. Details in "Round 2".

**Round 1** (below):

- **The learned tracker is now robust on every held-out shift tested except the hardest
  combination.** A DuckNet trained on procedurally domain-randomized (DR) scenes (random palettes,
  textures, lighting, random-colour birds and blobs) loses the duck in **0.0–0.2%** of frames on
  all existing held-out sets (was 54–99.7%; photometric-jitter follow-up 0.1–85%). On the fresh,
  pre-registered families it was never trained or tuned on, it loses **0.7%** (city/autumn
  backgrounds), **1.4%** (spatially varying light + haze) and **0.0%** (kites/drones). Only on the
  fresh all-at-once set `all2` is it at **16.8% [10.6–23.8]**. Those losses track haze strength
  (6.5% lost below haze 0.35, 32% above), a contrast compression outside the DR envelope.
- **The classical tracker, given matched help, improved a lot on backgrounds and distractors but
  not on lighting.** Retuned on the same DR scenes, with grey-world normalisation, two duck-template
  cues, a peak detector and the shared re-detecting filter: bg 27.3% → **4.1%**, distract
  50.8% → **24.0%**, all 64.7% → **24.8%**. But dusk got worse (13.2% → **43.8%**) and in-distribution
  `iid` regressed (5.0% → 14.8%). Its held-out behaviour depends heavily on which search run is
  selected (light 11.6–43.8% across runs with similar in-distribution scores).
- Targets. Lost "well under 10%" on background and lighting shifts: **met by the learned tracker**
  (old and fresh), **not by the classical** (bg yes, lighting no). "Large cut on distractors": met by
  both (CNN 70.9% → 0.0%; classical 50.8% → 24.0%). Not met: `all2` for either tracker (16.8% / 24.8%).

## Round 2 (coordinator-approved): haze in DR, new pre-registered held-out families

**Why.** Round 1's only real learned-tracker failure was `all2` (16.8%), which tracked haze, a
contrast compression at a bright mean that DR v1 could not produce. Diagnosing it made all
round-1 sets **dev-visible**, so round 2 needed new held-out families.

**Protocol** (STATUS.md, 16:31, before any DR change):
- *New families*, generator gated on `checkpoints/r6_tracker_FROZEN_r2.json` and rendered only
  after the freeze at 16:45, seeds 400000+:
  - bg3 = ocean or mountain ridges;
  - light3 = local additive sun glare + flare discs;
  - distract3 = paper planes and bats;
  - all3 = all three.
- *Also reported:* the dev-visible `all2` generator at fresh seeds 390000+ (`all2_f2`).
- *Selection rule:* for each tracker, R1 vs R2 by lowest clipped error on 200 new DR-v2 scenes
  (seeds 130000+).
- *Only DR change:* with prob 0.5, a global haze blend toward a random colour
  (h ~ U(0, 0.55), colour value 0.4–1). It uses its own RNG, so each DR v2 scene is the DR v1
  scene of the same seed plus haze (`scene_r6.render_dr2`; test `test_round2_dr_haze_is_dr_plus_haze`).
- *Same recipe as round 1:* CNN nf1, 2000 scenes, 15k steps, 11.0 min. Classical: 1500 configs
  on the first 60 DR-v2 scenes, 10.5 min (slower than round 1 because of extra blur maps and machine load).
- *Selection outcome* (`results/r6_tracker_selection_r2.json`): **R2 wins for both**. Classical
  clipped 3.53 vs R1 3.66 px (lost 18.4% vs 20.3%); CNN 0.40 vs 0.56 px (lost 1.1% vs 2.1%).
- The CNN's filter was re-searched on DR-v2 val (400 configs, `checkpoints/r6_tracker_cnn_r2_kf_params.json`).

**Results** (`results/r6_tracker_eval_r2.json`): lost rate [95% scene-bootstrap CI]. Learned =
R2 CNN + filter; classical = R2 official; R1 columns are the round-1 official trackers (CNN raw,
whose filter rows are within 0.42 points). Mean px errors with CIs are in the JSON.

| set | status | R2 learned | R2 classical | R1 learned (raw) | R1 classical |
|---|---|---|---|---|---|
| bg3 | new held-out | **0.0% [0.0–0.0]** | 8.0% [3.5–13.0] | 0.0% | 9.2% |
| light3 | new held-out | **0.0% [0.0–0.0]** | 14.2% [8.5–20.8] | 0.5% | 16.7% |
| distract3 | new held-out | **0.0% [0.0–0.0]** | 15.1% [8.5–22.6] | 0.0% | 16.9% |
| all3 | new held-out | **0.5% [0.0–1.5]** | 10.5% [5.7–16.1] | 1.1% | 14.1% |
| all2_f2 | dev family, fresh seeds | **5.5% [3.2–8.3]** | 30.5% [22.4–39.2] | 18.7% | 36.2% |
| all2 | dev-visible | 8.3% [4.0–13.8] | 20.7% [14.4–27.2] | 17.0% | 24.8% |
| bg2 / light2 / distract2 | dev-visible | 1.3 / 0.0 / 0.1% | 10.3 / 32.3 / 18.9% | 0.8 / 1.8 / 0.1% | 10.7 / 38.6 / 20.1% |
| bg / light / distract / all | dev-visible (R5 sets) | 0.0 / 0.2 / 0.0 / 0.4% | 2.5 / 36.3 / 34.9 / 22.8% | 0.0 / 0.2 / 0.0 / 0.4% | 4.1 / 43.8 / 24.0 / 24.8% |
| bg_f / light_f / distract_f / all_f | dev-visible | 0.0 / 0.2 / 0.0 / 0.2% | 4.1 / 36.3 / 29.1 / 24.2% | 0.0 / 0.0 / 0.0 / 0.3% | 5.3 / 43.7 / 12.4 / 27.2% |
| iid | train distribution | 0.1% | 14.7% | 0.0% | 14.8% |
| dr2val | R2 training distribution | 0.1% | 10.6% | 1.7% | 11.9% |

Round-2 findings:
- **Haze randomization fixed the haze failure.** On fresh seeds of the all2 generator the learned
  tracker goes from 18.7% to **5.5%**; it reaches 8.3% on the original all2 seeds. light2 goes
  from 1.8% to 0.0%. Nothing else regressed beyond the CIs (bg2 0.8% → 1.3%, within them).
- **The new held-out families turned out easy, so they are a weak test.** The R2 learned tracker
  is at 0.0–0.5% on them. But the *old R5 aug trackers* are too: aug CNN 0.0–2.0%, aug classical
  2.0–5.2% (`cnn_aug_raw`, `classical_aug` in the JSON). Glare, ocean/mountains, planes and bats
  are not confusable with the duck. The informative round-2 number is `all2_f2`: a family that
  was diagnosed, at seeds never seen.
- **The classical tracker gained little from the same help.** all2 24.8% → 20.7%, all2_f2 36.2% → 30.5%,
  and light at about 36%. Old distract got worse (24.0% → 34.9%) and fresh-seed distract_f too
  (12.4% → 29.1%). It is again a single search run picked by in-distribution score, and as in
  round 1 its held-out behaviour moves a lot between runs. On the new families it is 8–15% lost,
  *worse than the R5 aug classical* (2–5%), which keeps a strong colour/luminance prior that these
  families do not challenge.
- Integrity checks:
  - R1 trackers reproduce their round-1 numbers exactly in the round-2 eval (e.g. classical 14.8 / 4.1 /
    43.8 / 24.0%), and flight.py's hash is unchanged.
  - The R2 manifest verifies. The R1 manifest's sha1 differs only for four code files
    (`scene_r6.py`, `track_classical_r6.py`, `train_tracker_r6.py`, `eval_tracker_r6.py`), which
    round 2 extended by adding options. The R1 checkpoints and parameters are unchanged.
- Cosmetic, frozen: the mountain snow caps render as white zigzag bands below the ridge line.
- Demo: `media/r6_tracker_demo_r2.gif` / `_strip.png` (all2, all2_f2, bg3, light3, distract3, all3;
  scene index 3 of each). On distract3 the classical tracker is seen chasing a paper plane.

## Round 1 main table

Learned = **R6 CNN + filter** (`cnn_r6_kf`). Classical = **R6 classical, official** (`classical_r6`,
= v4, chosen by the pre-declared selection rule). The previous best of each (photometric-jitter
"aug" follow-up) is re-run on identical frames for reference. Cells: lost rate [95% CI] · mean px
error [95% CI]; the two reference columns show point estimates only (their CIs are in the JSON).
Source: `results/r6_tracker_eval_fresh.json`.

| set | what is shifted | R6 learned | R6 classical | prev. aug CNN (raw) | prev. aug classical |
|---|---|---|---|---|---|
| drval | DR val (R6 training distribution) | 0.0% [0.0–0.0] · 0.15 [0.14–0.17] | 8.7% [4.1–14.3] · 2.37 [1.34–3.59] | 38.0% · 14.4 | 50.5% · 17.4 |
| iid | original training distribution | 0.0% [0.0–0.0] · 0.11 [0.11–0.12] | 14.8% [8.4–22.1] · 3.72 [2.11–5.48] | 0.0% · 0.09 | 5.0% · 2.67 |
| **existing held-out sets** (same 60 scenes as R5) | | | | | |
| bg | sunset sky, desert/snow | **0.0% [0.0–0.0]** · 0.12 [0.12–0.13] | 4.1% [1.6–7.3] · 1.57 [0.80–2.55] | 40.6% · 14.6 | 27.3% · 10.5 |
| light | dusk + noise | **0.1% [0.0–0.3]** · 0.20 [0.16–0.26] | 43.8% [35.1–52.5] · 12.4 [9.5–15.5] | 0.1% · 0.23 | 13.2% · 6.16 |
| distract | balloons + hawks | **0.0% [0.0–0.0]** · 0.13 [0.13–0.14] | 24.0% [15.8–32.9] · 6.97 [4.25–10.1] | 70.9% · 26.3 | 50.8% · 19.3 |
| all | all three | **0.2% [0.0–0.5]** · 0.28 [0.21–0.36] | 24.8% [18.3–31.4] · 4.84 [3.74–6.01] | 85.0% · 30.3 | 64.7% · 22.4 |
| **fresh seeds, old generators** (rendered after freeze) | | | | | |
| bg_f | sunset, desert/snow | 0.0% [0.0–0.0] · 0.12 [0.12–0.13] | 5.3% [2.5–9.1] · 1.82 [1.01–2.93] | 32.1% · 10.9 | 24.9% · 8.81 |
| light_f | dusk + noise | 0.0% [0.0–0.1] · 0.19 [0.16–0.24] | 43.7% [35.3–52.1] · 13.9 [10.9–17.0] | 0.1% · 0.21 | 12.5% · 5.86 |
| distract_f | balloons + hawks | 0.0% [0.0–0.1] · 0.13 [0.12–0.16] | 12.4% [6.8–18.8] · 3.04 [1.73–4.71] | 71.5% · 25.4 | 43.2% · 14.8 |
| all_f | all three | 0.2% [0.0–0.7] · 0.23 [0.19–0.29] | 27.2% [20.6–34.0] · 5.72 [4.29–7.33] | 88.9% · 34.3 | 67.0% · 24.2 |
| **fresh families** (pre-registered, outside DR, rendered after freeze) | | | | | |
| bg2 | city skyline / autumn canopy | **0.7% [0.1–1.5]** · 0.44 [0.25–0.72] | 10.7% [6.4–15.9] · 3.56 [2.01–5.57] | 54.1% · 16.8 | 9.9% · 4.19 |
| light2 | sun/shadow gain field + haze | **1.4% [0.3–2.8]** · 0.60 [0.24–1.11] | 38.6% [28.9–48.8] · 9.37 [6.67–12.3] | 5.7% · 2.73 | 14.0% · 3.27 |
| distract2 | kites + drones | **0.0% [0.0–0.1]** · 0.13 [0.12–0.15] | 20.1% [12.0–28.5] · 4.88 [2.87–7.04] | 3.5% · 1.05 | 6.3% · 2.53 |
| all2 | all three | **16.8% [10.6–23.8]** · 5.28 [3.15–7.73] | 24.8% [17.3–32.4] · 6.39 [4.26–8.98] | 62.4% · 18.7 | 43.3% · 14.5 |

Median errors: learned 0.10–0.28 px on every set; classical 0.32–0.74 px except light / light_f /
light2 (2.0–3.4 px). The CNN without the filter (`cnn_r6_raw`) is within 0.4 points of `cnn_r6_kf`
everywhere (e.g. all2 17.0%), so the gain comes from the detector, not the filter.

## What changed (both trackers)

**Shared training/tuning scenes: domain randomization** (`scene_r6.py`). Each DR scene
independently randomizes:
- sky top and bottom, ground, tree and cloud colours (random HSV, any hue);
- smooth value-noise texture on sky and ground;
- lighting: global gain 0.3–1.4, per-channel tint 0.7–1.3, offset ±0.08, noise 0–0.07;
- 0–5 distractors: crows, gulls, random-colour birds, "duck-brown" birds with random head and beak
  colours (so only the green head, white ring and orange beak together identify the duck), and
  random-colour ellipse blobs, with straight or wobbling motion.

The duck itself, its trajectories (flight sim) and the 80×80 renderer are unchanged.

By design, DR contains no rectangles or polygons, buildings, kites, drones, spatially varying
illumination or haze, and no hard-coded sunset/snow/desert/balloon/hawk generator.

How far the old held-out sets sit inside DR (5000 sampled DR scenes):
- 9.6% have gain and noise inside the old `light` box (3.1% also blue-tinted).
- 0.06% have a sky bottom inside the old sunset-orange range and 1.4% a top inside the old purple
  range; none have both.
- Random-colour birds and blobs make hawk-like and balloon-like objects possible by chance.

So the old held-out sets are inside the DR envelope, which is why the fresh families exist.

**Learned** (`train_tracker_r6.py`): the same heatmap + sub-cell-offset DuckNet as before, slightly
wider (c = 48) with one dilated layer. Training: 2000 DR scenes (80k frames), 15k steps of batch
64, extra per-sample photometric jitter, h-flips; 9.8 min on the GPU under load. Model selection
on 40 DR val scenes only. A frame-stack variant (nf = 3: current + 2 previous frames) was trained
too; see ablations. The filter (`cnn_r6_kf`) feeds the top-3 heatmap peaks into the same filter
the classical tracker uses. Its 6 parameters were random-searched (400 configs) on DR val
(`checkpoints/r6_tracker_cnn_nf1_kf_params.json`).

**Classical** (`track_classical_r6.py`). It gets the same kind of help, plus more prior information
than the CNN (the renderer's sprite colours and shape):
- *Same scenes:* tuned on the first 60 scenes of the CNN's DR training set, same objective as
  before (mean min(err, 20 px)).
- *Normalisation:* raw or per-frame grey-world (tuned choice), for chroma and background
  subtraction.
- *Cues:*
  - `fg`, the background-subtraction cue (median of the last K frames);
  - `col`, chromaticity distance to the duck colours;
  - `tpl_ch`, a masked per-channel NCC against the rendered duck sprite (2 facings × 4 wing phases),
    invariant to per-channel gain and offset;
  - `tpl_jl` (added in v3), a joint NCC over log intensities, invariant to gain/tint but colour-aware.

  The score is the product of the cues, each raised to a tuned exponent (0 switches a cue off).
- *Detection:* thresholded blobs with area limits (as before) or top-3 local maxima of the smoothed
  score (`det = peaks`, added in v2 after DR-val diagnosis: in noisy scenes the duck blob merges
  with noise and the area cap rejects it). Optional σ = 0.6 pre-denoise of frames and templates (v4).
- *Filter, shared with the CNN:* `run_filter_r6` is the original gated constant-velocity Kalman plus
  re-detection: switch to the global best candidate when it outscores the gated choice by a tuned
  factor.

## Protocol (how held-out integrity was kept)

1. **Pre-registration.** The fresh families were written into STATUS.md at 15:30, before any DR
   design or training. The generator (`scene_r6.fresh_set`) refuses to render until the freeze
   manifest exists.
2. **Freeze.** `checkpoints/r6_tracker_FROZEN.json` (16:06) records the sha1 of the final parameters,
   checkpoints and code. Fresh sets were rendered after it. Their previews were checked for
   rendering bugs, not difficulty; one cosmetic defect was left as frozen: the kite tail "bows" are
   self-intersecting quads and mostly do not draw, so a kite is a two-colour diamond.
3. **Selection only in distribution.** CNN: nf1 vs nf3 on DR val, confirmed on a 200-scene DR
   selection set. Classical: four tuning runs (v1–v4). A rule was fixed in STATUS.md *before*
   looking, after v3 was better on old held-out but worse on DR val: pick the lowest clipped error
   on a new 200-scene DR selection set (`select_r6.py`, seeds 120000+). It picked v2 over v1/v3, then
   v4 when v4 was added (`results/r6_tracker_selection.json`):

   | run | configs | DR-select clipped px | DR-select lost |
   |---|---|---|---|
   | v1 | 500 | 4.12 | 20.7% |
   | v2 | 1500 | 3.19 | 15.4% |
   | v3 | 1500 | 3.38 | 16.1% |
   | v4 | 1500 | 2.72 | 14.0% |

   The CNN (nf1, raw) scores 0.26 px clipped / 0.58% lost on the same set.
4. **What I did look at.** The old held-out sets (bg/light/distract/all) were used for
   *diagnosis* during development. v2's peak detector came from DR-val diagnosis. v3's joint-log
   template came from seeing on the old `distract` set that the per-channel template fires on
   balloon edges. That is a soft leak toward the old sets, which is why the fresh seeds (`*_f`)
   and fresh families (`*2`) are the clean test. The DR spec and the CNN were fixed before any
   held-out number was seen. The flight.py hash is recorded in every result file; the old held-out
   frames are cached, and the aug reference rows reproduce `tracker_eval_aug.json` exactly
   (27.3 / 13.2 / 50.8 / 64.7% classical, 40.6 / 0.1 / 70.9 / 85.0% CNN).

## All variants (ablations), lost rate on each set

Source: `results/r6_tracker_eval_fresh.json` (CIs in the JSON). Non-selected variants are reported for information only.

| set | CNN nf1 (official, raw) | CNN nf3 (frame stack) | CNN no DR distractors | classical v4 (official) | classical v2 | classical v3 | classical, old cues retuned on DR |
|---|---|---|---|---|---|---|---|
| drval | 0.0% | 0.3% | 0.8% | 8.7% | 9.9% | 13.6% | 52.8% |
| iid | 0.0% | 0.0% | 0.0% | 14.8% | 13.1% | 0.8% | 21.5% |
| bg | 0.0% | 0.0% | 0.0% | 4.1% | 5.0% | 3.8% | 26.2% |
| light | 0.2% | 0.0% | 0.4% | 43.8% | 28.3% | 11.6% | 12.5% |
| distract | 0.0% | 0.8% | 1.0% | 24.0% | 54.7% | 25.9% | 52.3% |
| all | 0.4% | 1.0% | 2.1% | 24.8% | 37.5% | 17.8% | 55.7% |
| bg_f | 0.0% | 0.0% | 0.0% | 5.3% | 7.1% | 4.6% | 27.1% |
| light_f | 0.0% | 0.2% | 0.1% | 43.7% | 27.3% | 9.2% | 12.6% |
| distract_f | 0.0% | 0.6% | 0.9% | 12.4% | 43.1% | 14.9% | 50.7% |
| all_f | 0.3% | 0.8% | 2.0% | 27.2% | 34.9% | 18.6% | 59.0% |
| bg2 | 0.8% | 2.7% | 6.2% | 10.7% | 12.3% | 5.9% | 27.6% |
| light2 | 1.8% | 1.0% | 1.5% | 38.6% | 29.6% | 6.0% | 28.5% |
| distract2 | 0.1% | 0.0% | 0.0% | 20.1% | 19.9% | 3.1% | 21.1% |
| all2 | 17.0% | 14.6% | 25.5% | 24.8% | 18.8% | 17.0% | 39.6% |

Ablation findings:
- **DR is what fixed the learned tracker.** Even without DR distractors, the CNN is at 1.0% on the
  old distractor set (aug CNN: 70.9%). The balloon-chasing came from never seeing saturated colours,
  and random palettes already cover them. The DR distractors add robustness on the fresh sets
  (bg2 6.2% → 0.8%, all2 25.5% → 17.0%).
- **Temporal input (3-frame stack) did not help** at equal steps: DR val 0.30 vs 0.16 px clipped,
  selection set 0.39 vs 0.26, and it was slower to train (16.6 vs 9.8 min). It is slightly better
  on all2 (14.6% vs 17.0%), within the CIs.
- **Retuning alone does little for the classical tracker.** The original cue set retuned on DR
  (`oldcues`) is no better than the old aug-tuned one on held-out shifts (bg 26.2%, distract 52.3%).
  The new cues are what moved it.
- **The classical tracker's held-out behaviour is fragile to the tuning outcome.** v2/v3/v4 have
  similar in-distribution scores (selection-set clipped 2.7–3.4 px) but light ranges 9–44% and
  distract 24–55%. The colour-aware v3 is best on held-out (light2 6.0%, distract2 3.1%) but was not
  selected by the in-distribution rule.

## Why the remaining failures happen (`diagnose_r6.py` → `results/r6_tracker_failure_diagnosis.json`)

- **Learned on all2 (16.8%): haze plus duck-coloured clutter.**
  - Scenes with haze < 0.35: 6.5% lost; haze ≥ 0.35: 32.2% lost (scene-level r = 0.58 between haze
    and lost rate). Autumn 25.0% vs city 9.5%.
  - Lost frames have the duck on the darker side of the illumination field (mean gain factor
    0.70 vs 0.81 overall).
  - Haze blends up to 45% toward a pale grey. That compresses contrast at a *bright* mean, which DR
    cannot produce: its offset is only ±0.08 and low gain compresses contrast at a dark mean.
  - Either shift alone is easy (light2 1.4%, bg2 0.7%); the combination is not.
  - Clips: `media/r6_tracker_failure_all2_s41.gif` (autumn, haze 0.42, duck in the shaded half,
    barely visible to the eye) and `..._s27.gif` (city).
- **Classical at dusk (light 43.8%): blind again, as in the original R5 tracker.**
  - In 57% of dusk frames its score has no candidate, and its score maximum is on the duck in
    only 48% (12 scenes checked).
  - The selected v4 uses no colour term, a grey-world motion cue (noise is amplified ×~2 at gain
    0.45) and the per-channel template, whose 25-pixel window cannot beat noise σ 0.03–0.06 at gain
    0.35–0.6.
  - Only 9.6% of DR scenes are that dark and noisy, so the tuning objective barely sees it. The CNN
    copes; the hand-built cues do not.
- **Classical on old distract (24.0%).** v4's lost frames are mostly not on a distractor (159 of
  577), i.e. lost to noise or missed candidates. v3's lost frames are almost all on **hawks**
  (531 of 621): its colour-aware joint template rejects balloons, but a hawk shares the duck's body
  colour and shape.

## Matched-help accounting

| | learned | classical |
|---|---|---|
| training/tuning scenes | 2000 DR scenes (80k frames) | first 60 of the same DR scenes |
| prior information | none beyond labels | duck sprite colours + rendered sprite templates |
| temporal information | single frame (frame stack tried) + filter | background model (K previous frames) + filter |
| filter | shared `run_filter_r6`, 400 configs on DR val | shared `run_filter_r6`, tuned jointly in its search |
| official run compute | 9.8 min GPU training | 3.8 min, 1500 configs × 60 scenes (4 workers) |
| all R6 runs | 4 trainings (nf3, nf1, no-dist, seed-1 replicate): 16.6 + 9.8 + 8.9 + 10.6 = ~46 min | 5 searches (v1–v4, oldcues), 5500 configs: ~22 min |

The classical tracker got more design iterations (4 versions with new cues and detectors) than the
learned one (1 DR spec, 1 architecture). It also got more search configurations, and it is the
only one with a hand-given appearance model.

## Seed replicate (post-freeze robustness check)

A second CNN trained identically with `--seed 1` (10.6 min; DR val clipped 0.164 vs 0.158 px), raw
argmax, same frames (`results/r6_tracker_seed_replicate.json`):

| set | iid | bg | light | distract | all | bg_f | light_f | distract_f | all_f | bg2 | light2 | distract2 | all2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seed 0 (official) | 0.0% | 0.0% | 0.2% | 0.0% | 0.4% | 0.0% | 0.0% | 0.0% | 0.3% | 0.8% | 1.8% | 0.1% | 17.0% |
| seed 1 | 0.0% | 0.0% | 0.3% | 0.0% | 0.5% | 0.0% | 0.2% | 0.0% | 0.1% | 1.9% | 1.3% | 0.0% | 18.5% |

The learned tracker's held-out result is not a lucky seed. The classical tracker shows the
opposite: its search runs v2/v3/v4 differ by up to 32 points on the same held-out set.

## What worked
- Procedural DR with *generic* randomization (no held-out family generators) closed every
  single-factor gap for the learned tracker, including fresh families that are structurally
  absent from DR (buildings, foliage canopies, spatial illumination, haze, kites, drones).
- Shape-and-colour template cues invariant to gain and tint lifted the classical tracker on
  backgrounds (27% → 4%) and distractors (51% → 24%).
- Caching the old held-out frames and re-running the R5 aug trackers as reference rows made the
  comparison exact, and protects it from edits to flight.py by the agent working on flight
  (scene.py draws the duck's trajectory from flight.py; its hash was unchanged throughout this round).

## What didn't
- Temporal input for the CNN (no gain, slower).
- Pre-denoising for the classical tracker (offered in v4; the search did not pick it).
- The classical tracker at dusk; selecting a classical configuration by in-distribution score does
  not predict its held-out robustness.
- `all2` for both trackers (16.8% / 24.8%).

## Surprises
- Kites and drones (fresh distractors) were easy for every tracker, even the old aug ones (3.5% / 6.3%).
  Novel shapes that do not look like a duck are not confusing; duck-coloured birds and saturated
  blobs were.
- The CNN without DR distractors was already immune to balloons.
- With the colour-aware template, the classical tracker's errors on distract move from balloons to
  hawks, like the R5 colour-prior tracker. Body colour and shape are the duck's least distinctive
  features, and the hand-built cue still weights them heavily.

## Unfinished / not measured
- Round 2 addressed `all2` for the learned tracker (8.3%, fresh seeds 5.5%). Still open: the classical
  tracker on all2 (20.7%) and at dusk (~36%), and a harder held-out family than round 2's (which proved easy for all trackers).
- Classical dusk robustness (a noise-aware detector or a DR objective weighted toward dark scenes;
  not tried).
- Not built: a recurrent/heatmap-confidence learned tracker with explicit re-detection beyond the
  shared filter, and motion priors from the flight dynamics beyond the constant-velocity Kalman.
- Inference speed; occlusion; a duck leaving the frame (it never does).
- Kite tails do not render (cosmetic, frozen).

## Decisions (made by the coordinator, 2026-11-04 16:30)

1. **Classical headline: v4**, the pre-registered rule's pick. v3 stays in the tables, labelled as
   designed after viewing held-out results.
2. **Round 2 for all2: done**, see "Round 2" above.
3. **Disk: nothing deleted**; the owner decides at commit time. Sizes of the R6 artifacts:

   | what | size |
   |---|---|
   | `checkpoints/r6_tracker_cache/` (cached frames; regenerated on demand) | 850 MB after round 2 |
   | `checkpoints/r6_tracker_*.safetensors` (5 CNNs incl. round 2) | 7.3 MB |
   | `results/r6_tracker_eval_fresh_tracks_*.npz`, `r6_tracker_eval_r2_tracks_*.npz` | 4.0 MB, 5.1 MB |
   | `results/r6_tracker_eval_v1/v2/v3_tracks_*.npz` (development evals) | 3.4 MB |
   | `media/r6_tracker_*` | 16 MB |

4. README.md / RESULTS.md are left untouched; the coordinator merges.

## How to run (from `21_duck_hunt/`)

```sh
export OMP_NUM_THREADS=4
../bin/gpu_run ../.venv/bin/python train_tracker_r6.py --nf 1 --out checkpoints/r6_tracker_cnn_nf1      # ~10 min
../bin/gpu_run ../.venv/bin/python train_tracker_r6.py --nf 3 --out checkpoints/r6_tracker_cnn_nf3      # ablation
../bin/gpu_run ../.venv/bin/python train_tracker_r6.py --nf 1 --dr nodist --out checkpoints/r6_tracker_cnn_nf1_nodist
OMP_NUM_THREADS=1 ../bin/gpu_run ../.venv/bin/python track_classical_r6.py --tune --n_random 1000 --n_local 500 --out r6_tracker_classical_v4
OMP_NUM_THREADS=1 ../bin/gpu_run ../.venv/bin/python track_classical_r6.py --tune --cues old         # ablation
OMP_NUM_THREADS=1 ../bin/gpu_run ../.venv/bin/python select_r6.py                                     # in-distribution selection
OMP_NUM_THREADS=1 ../bin/gpu_run ../.venv/bin/python eval_tracker_r6.py --fresh --tag r6_tracker_eval_fresh \
    --cnn checkpoints/r6_tracker_cnn_nf1 --kf_params checkpoints/r6_tracker_cnn_nf1_kf_params.json \
    --extra_classical classical_r6_v2=checkpoints/r6_tracker_classical_v2.json classical_r6_v3=checkpoints/r6_tracker_classical_v3.json \
    --extra_cnn cnn_r6nf3_raw=checkpoints/r6_tracker_cnn_nf3 cnn_r6nodist_raw=checkpoints/r6_tracker_cnn_nf1_nodist   # ~8 min
../.venv/bin/python diagnose_r6.py
../.venv/bin/python replicate_r6.py
../.venv/bin/python demo_tracker_r6.py --tag r6_tracker_eval_fresh      # media/r6_tracker_demo.gif + _strip.png
# round 2
../bin/gpu_run ../.venv/bin/python train_tracker_r6.py --nf 1 --dr haze --out checkpoints/r6_tracker_cnn_r2
OMP_NUM_THREADS=1 ../bin/gpu_run ../.venv/bin/python track_classical_r6.py --tune --dr haze --n_random 1000 --n_local 500 --out r6_tracker_classical_r2
OMP_NUM_THREADS=1 ../bin/gpu_run ../.venv/bin/python select_r6_round2.py
OMP_NUM_THREADS=1 ../bin/gpu_run ../.venv/bin/python eval_tracker_r6.py --fresh --fresh3 --tag r6_tracker_eval_r2 \
    --classical checkpoints/r6_tracker_classical_r2.json --cnn checkpoints/r6_tracker_cnn_r2 --kf_params checkpoints/r6_tracker_cnn_r2_kf_params.json \
    --extra_classical classical_r1=checkpoints/r6_tracker_classical.json classical_r6_v3=checkpoints/r6_tracker_classical_v3.json \
    --extra_cnn cnn_r1_raw=checkpoints/r6_tracker_cnn_nf1                                                   # ~11 min
../.venv/bin/python demo_tracker_r6.py --tag r6_tracker_eval_r2 --conds all2 all2_f2 bg3 light3 distract3 all3 --out media/r6_tracker_demo_r2
./smoke.sh                                                               # includes tests/test_r6_tracker.py (5 tests)
```
**Caution:** `track_classical_r6.py --tune` without `--out` writes `checkpoints/r6_tracker_classical.json`
and would overwrite the frozen official parameters; always pass `--out`.
The tuning runs v1–v3 were made with intermediate versions of `track_classical_r6.py`. Each later
version only added options, and earlier parameter files still run unchanged: v1/v2/v3 have no
`a_tplj`, `det` or `blur` keys, which default to off/blob/off. The v1/v2 searches drew from smaller
search spaces, so re-running `--tune` today does not reproduce them byte-for-byte.

## Files
| file | what |
|---|---|
| `scene_r6.py` | DR generator, cached old held-out sets, gated fresh-family generator |
| `train_tracker_r6.py` | DuckNetT (frame-stack option), DR training, peaks |
| `track_classical_r6.py` | R6 classical cues, detector, shared re-detecting filter, tuning |
| `eval_tracker_r6.py` | evaluation with scene-bootstrap CIs, filter tuning for the CNN |
| `select_r6.py` | 200-scene in-distribution selection |
| `diagnose_r6.py`, `replicate_r6.py`, `demo_tracker_r6.py` | diagnosis, seed replicate, demo |
| `tests/test_r6_tracker.py` | 5 smoke tests (~6 s) |
| `checkpoints/r6_tracker_cnn_nf1.*` (+`_kf_params.json`) | official learned tracker |
| `checkpoints/r6_tracker_classical.json` (= `_v4.json`) | official classical parameters; `_v1/_v2/_v3/_oldcues` variants |
| `checkpoints/r6_tracker_FROZEN.json`, `_FROZEN_r2.json` | freeze manifests (sha1), round 1 and round 2 |
| `checkpoints/r6_tracker_cnn_r2.*` (+`_kf_params.json`), `checkpoints/r6_tracker_classical_r2.json` | **round-2 official trackers** |
| `select_r6_round2.py`, `results/r6_tracker_selection_r2.json`, `results/r6_tracker_eval_r2.json` (+ tracks) | round-2 selection and evaluation |
| `checkpoints/r6_tracker_cache/` | 850 MB of cached frames (old held-out, DR val, fresh sets); safe to delete, regenerated on demand |
| `results/r6_tracker_eval_fresh.json` (+ `_tracks_*.npz`) | final evaluation; `r6_tracker_eval_v1/v2/v3*` are the development evals |
| `results/r6_tracker_selection.json`, `r6_tracker_failure_diagnosis.json`, `r6_tracker_seed_replicate.json` | selection, diagnosis, replicate |
| `results/r6_tracker_classical_*_tuning.json`, `*_train_log.jsonl`, `*_stdout.txt` | tuning and training records |
| `media/r6_tracker_demo.gif`, `_strip.png` | side by side on bg / light / distract / bg2 / light2 / all2 (scene index 3 of each, not picked) |
| `media/r6_tracker_failure_all2_s41.gif`, `_s27.gif` | the learned tracker's worst all2 scenes |
