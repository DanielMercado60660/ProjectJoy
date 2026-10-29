# Self-Healing Pixels: results

**Status:** done. First satisfying version: one target (lizard) that regrows after held-out damage, with a growth-only baseline, 2 seeds each.

## Setup (what was actually run)
- **Target:** 🦎 lizard from Apple Color Emoji, rendered with PIL, 40×40, padded to 72×72, RGB premultiplied by alpha.
- **Model:** 16-channel state; fixed identity + Sobel-x/y perception (48 features); per-cell MLP 48→128 (ReLU)→16,
  last layer zero-initialised, no bias (8,320 parameters); 50% stochastic fire rate; alive mask = 3×3 max-pooled alpha > 0.1, applied before and after each step.
- **Training (both conditions identical except damage; seeds 0 and 1):** sample pool of 1024, batch 8, 64–96 NCA steps per iteration,
  Adam lr 2e-3 → 2e-4 at iteration 2000, per-variable gradient normalisation, 8000 iterations.
  The highest-loss sample in each batch is replaced by the seed.
  - `lizard_regen`, `lizard_regen_s1`: the 3 lowest-loss samples of each batch get a random **circular** wound (centre in [-0.5,0.5]², radius 0.1–0.4 in [-1,1] grid coordinates).
  - `lizard_persist`, `lizard_persist_s1` (**baseline, growth-only**): same pool training, never damaged.
- **Eval protocol** (`eval.py`, 32 trials per cell, fixed seeds, identical masks for both models): grow 200 steps from the seed → erase cells → run 400 steps.
  Damage types: `circle` (training distribution) and three **held-out** types never seen in training: `half` (left/right/top/bottom half of the grid, about 49% of the body),
  `stripes` (3-cell-wide stripes every 8 cells, either orientation, about 37% of the body), `blobs` (smoothed-noise blobs, about 37% of the body).
- **Recovery threshold:** a trial counts as recovered when its MSE gets back to `pre + 0.2·(post − pre)`, i.e. 80% of the damage-induced excess error is gone,
  judged against that trial's own pre-damage MSE. "Stable" also requires it to still be under the threshold at +400.
  I picked this after a first try with `2 × pre-damage MSE` turned out to be meaningless: the damage itself often only doubles the
  whole-canvas MSE (the canvas is mostly empty background), so trials were counted as "recovered" at step 0.
- MSE is RGBA MSE over the full 72×72 canvas. For scale: an empty canvas scores **0.0271**.

## Headline
Two training seeds per condition (0 and 1); 32 eval trials per row; identical damage masks and fire-mask seeds for every model.

| | growth-only baseline (seed 0 / seed 1) | regeneration-trained (seed 0 / seed 1) |
|---|---|---|
| MSE 400 steps after held-out damage (half / stripes / blobs) | 0.0094 / 0.0089 / 0.0114 · 0.0230 / 0.0227 / 0.0224 (all trials, mostly dead seeds; **0.0108 / 0.0144 / 0.0145 on the 8–12 trials that grew**) | **0.0053 / 0.0042 / 0.0043 · 0.0034 / 0.0032 / 0.0028** |
| damage-induced error still left at +400 (held-out, range over types and seeds; ratio of means, grown trials) | 53–124% (per-trial median 42–92%) | **14–25%** (per-trial median 12–25%) |
| trials reaching the 80% recovery threshold (held-out) | 0–19% (seed 0); seed 1: 0–25% among runs that grew | **44–75% (seed 0), 75–100% (seed 1)** |
| median recovery time when it happens | 40–52 steps (seed 0; not meaningful for seed 1, where damage barely changes an already-poor state) | 22–50 steps |
| undamaged MSE at step 2000 (stability) | **0.0018** / 0.0211 (seed dies in ~70% of runs) | 0.0024 / 0.0019 |
| MSE after 10 repeated wounds | 0.61 / 10.7 (explodes; seed-1 mean is driven by 4 of 32 trials, its median is 0.027 = dead) | **0.015 / 0.013** (slow fade) |

For scale: an empty canvas scores 0.0271, and both models start from about 0.011–0.014 right after a held-out wound.

**Reading it:**
- **Regeneration generalises to damage it never saw.** After half-cuts, stripes and blobs (training only used circles), the regen models get back to 0.0028–0.0053 and leave only 14–25% of the damage error, for both seeds.
  The growth-only baselines fill in a bit and then drift *up* again (see the curves): they regrow tissue in the wrong places. On average they keep 53–124% of the damage error (ratio of means); the typical (median) baseline trial keeps 42–92%, still about 2–7× the regen models (12–25%). The means are pulled up by a few bad trials.
- **The ordering is the same for both seeds** on every damage type and every metric except one: undamaged accuracy. On the undamaged lizard, the seed-0 baseline is slightly sharper than the regen models (0.0018 vs 0.0024). Training on wounds costs a little sharpness, as the Distill paper notes.
- **Recovery time is not where they differ.** When a baseline trial does recover, it takes about as long as a regen trial (tens of steps). The difference is whether it recovers at all.
- **Long-rollout stability without damage:** the regen models and the seed-0 baseline are flat from step 200 to 2000 (worst trial ≤ 0.0034, no explosions). The seed-1 baseline is "stable" only because its seed mostly dies (see failure 6).
- **Repeated wounds:** both baselines explode (worst trial 4.9 and 310, i.e. 180× and 11,000× an empty canvas). Both regen models degrade slowly instead (0.003–0.004 → 0.013–0.015).

### Full tables (`results/eval_lizard.json`, regenerate with `python report.py lizard`)
Damage → recovery (mean of 32 trials; ± is std over trials):

| damage | model | pre-damage MSE | right after damage | +50 | +100 | +400 | excess error left at +400 | recovered (80%) | recovered & stable | median steps to recover |
|---|---|---|---|---|---|---|---|---|---|---|
| circle (train) | lizard_persist | 0.0020 | 0.0072 | 0.0069 | 0.0081 | 0.0085 ± 0.0075 | 125% | 16% | 12% | 38 |
| circle (train) | lizard_regen | 0.0023 | 0.0074 | 0.0033 | 0.0033 | 0.0033 ± 0.0012 | 20% | 62% | 47% | 36 |
| circle (train) | lizard_persist_s1 | 0.0190 | 0.0203 | 0.0204 | 0.0204 | 0.0204 ± 0.0095 | 107% | 72% | 66% | 0 |
| circle (train) | lizard_regen_s1 | 0.0015 | 0.0067 | 0.0024 | 0.0022 | 0.0023 ± 0.0011 | 15% | 75% | 59% | 32 |
| half (held-out) | lizard_persist | 0.0018 | 0.0140 | 0.0086 | 0.0086 | 0.0094 ± 0.0043 | 62% | 19% | 16% | 52 |
| half (held-out) | lizard_regen | 0.0024 | 0.0143 | 0.0058 | 0.0053 | 0.0053 ± 0.0019 | 25% | 44% | 41% | 50 |
| half (held-out) | lizard_persist_s1 | 0.0216 | 0.0243 | 0.0229 | 0.0230 | 0.0230 ± 0.0073 | 53% | 81% | 81% | 0 |
| half (held-out) | lizard_regen_s1 | 0.0016 | 0.0139 | 0.0041 | 0.0033 | 0.0034 ± 0.0011 | 15% | 75% | 66% | 48 |
| stripes (held-out) | lizard_persist | 0.0017 | 0.0110 | 0.0070 | 0.0084 | 0.0089 ± 0.0041 | 77% | 0% | 0% | n/a |
| stripes (held-out) | lizard_regen | 0.0023 | 0.0114 | 0.0041 | 0.0042 | 0.0042 ± 0.0014 | 21% | 75% | 69% | 27 |
| stripes (held-out) | lizard_persist_s1 | 0.0197 | 0.0223 | 0.0225 | 0.0226 | 0.0227 ± 0.0064 | 114% | 66% | 66% | 0 |
| stripes (held-out) | lizard_regen_s1 | 0.0017 | 0.0110 | 0.0030 | 0.0031 | 0.0032 ± 0.0016 | 16% | 100% | 91% | 22 |
| blobs (held-out) | lizard_persist | 0.0021 | 0.0113 | 0.0087 | 0.0104 | 0.0114 ± 0.0068 | 101% | 16% | 12% | 40 |
| blobs (held-out) | lizard_regen | 0.0023 | 0.0115 | 0.0044 | 0.0042 | 0.0043 ± 0.0014 | 21% | 72% | 69% | 42 |
| blobs (held-out) | lizard_persist_s1 | 0.0191 | 0.0217 | 0.0217 | 0.0222 | 0.0224 ± 0.0077 | 124% | 62% | 62% | 0 |
| blobs (held-out) | lizard_regen_s1 | 0.0015 | 0.0110 | 0.0030 | 0.0027 | 0.0028 ± 0.0009 | 14% | 88% | 81% | 32 |

> **Reviewer note on the recovery columns.** The 80% threshold was chosen after seeing data (the first rule, 2× pre-damage MSE, counted trials as recovered at step 0). The current rule still does that whenever damage barely changes the MSE: for `lizard_persist_s1`, 20–24 of 32 held-out trials "recover" at step 0 because the seed had died and there was nothing to damage. Its 62–81% recovered / median 0 above are that artifact. Use the grown-only table below for seed 1, and the threshold-free comparison in "Reviewer check" for the headline.

**Only trials where the seed actually grew (pre-damage MSE < half of an empty canvas):**

| damage | model | grown | pre | right after damage | +400 | excess error left | recovered (80%) | recovered & stable |
|---|---|---|---|---|---|---|---|---|
| circle | lizard_persist | 100% | 0.0020 | 0.0072 | 0.0085 | 125% | 16% | 12% |
| circle | lizard_regen | 100% | 0.0023 | 0.0074 | 0.0033 | 20% | 62% | 47% |
| circle | lizard_persist_s1 | 38% | 0.0056 | 0.0091 | 0.0093 | 107% | 25% | 8% |
| circle | lizard_regen_s1 | 100% | 0.0015 | 0.0067 | 0.0023 | 15% | 75% | 59% |
| half | lizard_persist | 100% | 0.0018 | 0.0140 | 0.0094 | 62% | 19% | 16% |
| half | lizard_regen | 100% | 0.0024 | 0.0143 | 0.0053 | 25% | 44% | 41% |
| half | lizard_persist_s1 | 25% | 0.0051 | 0.0158 | 0.0108 | 53% | 25% | 25% |
| half | lizard_regen_s1 | 100% | 0.0016 | 0.0139 | 0.0034 | 15% | 75% | 66% |
| stripes | lizard_persist | 100% | 0.0017 | 0.0110 | 0.0089 | 77% | 0% | 0% |
| stripes | lizard_regen | 100% | 0.0023 | 0.0114 | 0.0042 | 21% | 75% | 69% |
| stripes | lizard_persist_s1 | 34% | 0.0057 | 0.0133 | 0.0144 | 114% | 0% | 0% |
| stripes | lizard_regen_s1 | 100% | 0.0017 | 0.0110 | 0.0032 | 16% | 100% | 91% |
| blobs | lizard_persist | 100% | 0.0021 | 0.0113 | 0.0114 | 101% | 16% | 12% |
| blobs | lizard_regen | 100% | 0.0023 | 0.0115 | 0.0043 | 21% | 72% | 69% |
| blobs | lizard_persist_s1 | 38% | 0.0057 | 0.0128 | 0.0145 | 124% | 0% | 0% |
| blobs | lizard_regen_s1 | 100% | 0.0015 | 0.0110 | 0.0028 | 14% | 88% | 81% |

**Long rollout from the seed, no damage:**

| model | seeds that grew (by step 200) | MSE@100 | @200 | @500 | @1000 | @2000 | worst trial @2000 |
|---|---|---|---|---|---|---|---|
| lizard_persist | 100% | 0.0017 | 0.0017 | 0.0017 | 0.0017 | 0.0018 | 0.0034 |
| lizard_regen | 100% | 0.0025 | 0.0023 | 0.0024 | 0.0024 | 0.0024 | 0.0033 |
| lizard_persist_s1 | 28% | 0.0212 | 0.0211 | 0.0211 | 0.0211 | 0.0211 | 0.0271 |
| lizard_regen_s1 | 100% | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0019 | 0.0028 |

**Repeated held-out wounds (10 random half/stripes/blobs wounds, 150 steps apart):**

| model | MSE after wound 1 | 3 | 5 | 10 | worst trial after 10 | trials < 0.01 after 10 |
|---|---|---|---|---|---|---|
| lizard_persist | 0.0084 | 0.0253 | 0.0598 | 0.6087 | 4.8838 | 0% |
| lizard_regen | 0.0044 | 0.0078 | 0.0104 | 0.0152 | 0.0229 | 6% |
| lizard_persist_s1 | 0.0252 | 0.0706 | 0.1313 | 10.7344 | 310.2599 | 0% |
| lizard_regen_s1 | 0.0028 | 0.0057 | 0.0082 | 0.0133 | 0.0241 | 31% |

### Reviewer check: threshold-free, grown trials only (added by the cross-reviewer)
Recomputed from the checkpoints with `eval.py`'s own functions and seeds (32 trials, seed 123). Every mean in the tables above reproduced to 4 decimals. No recovery threshold is used here. Only trials where the seed grew are counted. The 95% CI is a bootstrap over trials. "Paired" counts trials with the same damage mask where regen ends lower than the baseline (both grown).

| seed | damage | baseline grown | baseline MSE@400 [95% CI] | regen MSE@400 [95% CI] | mean MSE over +1..+400: baseline / regen | paired: regen lower |
|---|---|---|---|---|---|---|
| 0 | half | 32/32 | 0.0094 [0.0080, 0.0109] | 0.0053 [0.0047, 0.0060] | 0.0097 / 0.0060 | 21/32 |
| 0 | stripes | 32/32 | 0.0089 [0.0075, 0.0104] | 0.0042 [0.0038, 0.0047] | 0.0088 / 0.0044 | 29/32 |
| 0 | blobs | 32/32 | 0.0114 [0.0092, 0.0138] | 0.0043 [0.0038, 0.0048] | 0.0109 / 0.0045 | 28/32 |
| 1 | half | 8/32 | 0.0108 [0.0082, 0.0135] | 0.0034 [0.0031, 0.0038] | 0.0119 / 0.0040 | 8/8 |
| 1 | stripes | 11/32 | 0.0144 [0.0123, 0.0168] | 0.0032 [0.0028, 0.0038] | 0.0162 / 0.0033 | 11/11 |
| 1 | blobs | 12/32 | 0.0145 [0.0105, 0.0189] | 0.0028 [0.0025, 0.0031] | 0.0149 / 0.0030 | 12/12 |

The regen advantage holds without the threshold, with non-overlapping CIs on every held-out type for both seeds. Two caveats. On half-cuts the seed-0 baseline often does fine: regen wins only 21/32 paired trials (see `filmstrip_half.png`, where the baseline regrows a plausible lizard with a scar). And the seed-1 comparison rests on only 8–12 grown baseline trials, and those grown lizards were already worse before damage (pre-damage MSE about 0.005 vs 0.0016).

## Gallery (bonus, regeneration-trained only, 4000 iterations, seed 0, 16 eval trials per row vs 32 for the lizard)
`media/gallery.gif`: each organism is grown, has its top half removed (held-out), then gets random blobs (held-out).

| organism | seeds that grow | undamaged MSE @2000 | held-out damage, grown trials only: excess error left at +400 | recovered (80%) | empty-canvas MSE |
|---|---|---|---|---|---|
| butterfly 🦋 | 62–81% of eval runs (100% in another 16-run batch) | 0.0047 | 6–13% | 82–100% | 0.0765 |
| octopus 🐙 | 100% (64/64 eval runs + 16 long runs) | 0.0034 | 6–12% | 94–100% | 0.0813 |

Under repeated wounds the octopus fades slowly, like the lizard (0.0067 → 0.028 after 10 wounds). The butterfly's repeated-wound numbers (0.024 → 0.038) are dominated by dead seeds, so they are not comparable.

The butterfly and octopus regrow cleanly from half-cuts and blobs. Both are compact, roughly symmetric shapes, and they leave less damage error than the lizard (6–13% vs 14–25%). But **it also has a stillbirth problem**: in about 20–40% of runs the single seed cell dies before growing.
So stillbirth is not unique to the growth-only baseline. The regen models only escaped it on the lizard (0 deaths in 16 + 32×5 runs per lizard regen model).
This weakens my hypothesis in failure 6 that damage training protects growth-from-seed. A shorter run (4000 vs 8000 iterations) and a different shape may matter just as much. Not resolved.
Full numbers: `results/tables_butterfly.md`, `results/eval_butterfly.json`, `results/tables_octopus.md`, `results/eval_octopus.json`.

## Demos
- `media/grow_damage_regrow.gif` / `.mp4`: side by side, same seed and same fire masks: grow → left half removed (held-out) → random blobs (held-out) → circle.
- `media/repeated_damage.gif`: 10 random held-out wounds, one every 150 steps.
- `media/filmstrip_{circle,half,stripes,blobs}.png`: grown | damaged | +25 | +50 | +100 | +200 | +400.
- `media/recovery_curves_lizard.png`: mean MSE curves for every damage type, the 2000-step rollout and repeated wounds.
- `media/demo.html`: interactive and self-contained (712 KB, weights inlined, plain JS, about 4.5 ms per step measured in the Claude desktop browser pane, which is Chromium). Drag to erase, "Cut in half", switch models.
  The JS port was checked against MLX on a fixed input with every cell firing: max abs error 6.8e-6 (`node tests/check_demo_js.mjs media/demo.html`).
  I also checked it in a real browser: a diagonal erase stroke took MSE 0.0104 → 0.0042 after 300 steps.

## Good failures (saved clips in `media/failures/`, listed in `index.json`)
1. **Growth-only "ghost twin" / doubled body** (`filmstrip_stripes.png`, `lizard_persist_*`): after stripes, the baseline regrows a second, parallel body and extra legs beside the remnants.
   It never learned what to do with a partial body, so each surviving fragment acts like a fresh seed and grows its own lizard. The pieces overlap.
   **Seen again in seed 1:** `lizard_persist_s1_stripes_*.gif` regrows three side-by-side bodies.
2. **Growth-only scar tissue:** after half-cuts the wound edge fills with saturated off-colour cells (blue, yellow, black) that never resolve, and the error keeps creeping up (0.010 → 0.034 over 600 steps in the worst trial). **Also seen in seed 1** (`lizard_persist_s1_half_*.gif`: a frozen blue/cyan scar, 0.021 → 0.032).
3. **Growth-only explosion under repeated wounding:** with 10 held-out wounds the seed-0 baseline's mean MSE goes 0.008 → 0.025 → 0.06 → **0.61** (worst trial 4.9, about 180× an empty canvas). The seed-1 baseline goes to 10.7 (worst trial 310). It becomes a chaotic many-headed mass (`repeated_damage.gif`).
4. **Regen "extra legs":** after stripes or blobs, the regeneration-trained lizard sometimes regrows a **longer body with an extra pair of legs** and stays that way: a stable but wrong attractor (`lizard_regen_stripes_*`, `lizard_regen_blobs_*`).
   Local rules can tell "there should be a leg here" from what the neighbours look like, but nothing counts legs.
   Seed 1's worst case is milder, the reverse problem: after blobs it regrows the body but **leaves out a hind leg** (`lizard_regen_s1_blobs_*.gif`, MSE 0.0054 vs 0.0017 before damage).
5. **Regen slow fade under repeated wounding:** instead of exploding, the regen model heals each wound slightly incompletely, and the errors compound.
   After 10 wounds it has shrunk to a faint fragment (MSE 0.0044 → 0.015 for seed 0 and 0.0028 → 0.013 for seed 1; an empty canvas is 0.027). So it degrades gracefully but it does degrade.
6. **Baseline seed 1 "stillbirth": it forgot how to be born** (`media/failures/stillbirth_grid.png`). The seed-1 growth-only run trained normally until about iteration 5000 (rolling-median batch loss 0.0030).
   Then its loss jumped to about 0.0057 between iterations 5150 and 5325 and stayed near 0.006 to iteration 8000 (reviewer-corrected; the first draft said 3500 and 0.007). The jump is not at a resume boundary (resumes were at 4000 and 7000), so the stop/resume is an unlikely direct cause (not ruled out: the pool for this run was not kept, so the jump cannot be replayed). The final model grows the lizard from a single cell in only **4 of 16** runs (25–38% in the eval). In the rest the seed dies within about 20 steps, leaving an empty canvas.
   When it does grow, the lizard looks fine. My reading, a hypothesis I have not tested: in pool training, 7 of every 8 samples are already-grown lizards, so "keep a lizard alive" dominates the gradient.
   Only one sample per batch exercises "grow from a single cell", and that skill can be overwritten late in training. The regen runs keep practising growth from fragments, and both regen seeds grew in 16/16 runs.
   (The butterfly gallery model shows the same failure at a lower rate even with damage training. See Gallery.)
   It was caught only because of the second seed. A single-seed comparison would have missed it, and this is why the baseline columns above look so different between seeds.
   Even restricted to runs where the seed grew, this baseline leaves 53–124% of the damage error (table above).
7. **Early-training snapshots** (`lizard_persist_step1000`, `lizard_regen_step1000`): at iteration 1000 the pool-trained baseline already grows a **second lizard** out of its own outline during a 2000-step rollout (`media/failures/lizard_persist_step1000_long_rollout_*.gif`).

## Surprises and notes
- **MLX GPU matmul precision:** on this M5 Max, float32 matmuls on the GPU differ from a float64 reference by about 1.6e-3 relative (the CPU stream gives 2e-7).
  The JS self-check reference is therefore computed on the CPU stream. Training is unaffected, but anyone comparing ports should know.
- GPU contention: 12 builders shared the GPU. Iterations took 0.13 s alone but 0.33–0.95 s shared. To keep every run segment under 45 minutes, runs were stopped at a checkpoint and resumed
  (weights, Adam state, the full 1024-sample pool, numpy RNG state and step are checkpointed; the MLX RNG is re-seeded with seed+step on resume, so resumed runs are not bit-identical to uninterrupted ones).
  Resume was verified before the long runs (stop at 30 → resume to 60; loss continued, `tests/test_smoke.py` covers it).
- Damage briefly makes things *worse* for about 5 steps (MSE spikes above the post-damage value for both models) before regrowth starts. The wound edge becomes "active" and overshoots.
- `../bin/gpu_run` reads the list of slots once at startup. When the lab went from 4 to 8 slots at about 05:59, my already-waiting jobs kept waiting for the old 4, so I killed and relaunched them (no training progress was lost).
- `mx.compile` on the per-step helpers gave no speed-up here (matmul-bound and GPU-contended), so it is not used.

## Not measured / limitations
- Only one target has a baseline comparison (lizard). Gallery models are regen-only, one seed, 4000 iterations.
- Recovery time depends on the chosen threshold (80% of excess error removed). Only mean curves are saved (`results/curves_lizard.json`). For another threshold, change `HEAL_FRAC` in `eval.py` and rerun (about 3 min).
- Two training seeds per condition is still a small sample. The two baseline seeds behave very differently (failure 6), so baseline numbers should be read per seed, not averaged.
- No rotation/scale/timing variations (the "Go deeper" stage) were attempted.

## Next steps
- Train with mixed damage shapes (half-cuts, stripes) to fix the "extra legs" attractor, and with repeated wounds per pool sample to stop the slow fade.
- Longer training / lr schedule tuning: the loss plateaued at about 0.0025–0.0035 (regen) and 0.0015 (seed-0 baseline) after about 4000 iterations.
- Guard growth-from-seed in pool training (for example 2 seeds per batch, or checkpoint selection on a seed-growth metric) to prevent the stillbirth failure.
- Port demo.html's update loop to WebGL for bigger grids on iPad.
