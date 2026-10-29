# 14 · Camouflage Wars: results (first satisfying version)

Every number below comes from `results/eval.json` / `results/eval.md` (produced by `evaluate.py`),
`results/extra.json` (`analysis_extra.py`), `baseline_choice.json` (`tune_baseline.py`) or
`checkpoints/unet_log.json` (`train.py`); the reviewer's stronger baseline rows come from
`review/baseline_variant_test.json`. One training run, seed 0. Data seeds: train 1, val 2,
test_id 3, test_ood 4.

## Setup in one paragraph
96×96 procedural scenes with 1–3 blob creatures (11.5% of pixels on average). Training uses six
texture families; **`test_ood` uses three other families (marble, ripples, streaks) for both the
background and the creature skin**, so the segmenter has never seen any of those textures. 75% of
creatures wear the background's own family with parameters moved by a camouflage knob δ ~ U[0,1]
(δ≈0: same colours, scale and orientation, only a new phase/seed); 25% wear an unrelated family.
The U-Net (2.76 M params) trained for 5,500 steps × batch 32 (176k scene samples from 24k
scenes, with flips/rotations and colour augmentation) in **17.3 min under load** (load average
4.8 at start, ~13.6 at the end). The checkpoint was chosen on seen-family `val` (IoU 0.893).
The baseline is classical: per-pixel local texture features with a robust per-image
background model and Otsu threshold. Its variant (outlier vs 2-means) and window σ were tuned on
the same `val` split (best: outlier, σ = 2, val scene IoU 0.468). **Reviewer addition:** that
shipped baseline leaves a 2–3 px halo around every blob (the σ-pooled features bleed past the
boundary; red rings in `media/ood_hardest.png`) and keeps tiny components. Adding a mask erosion
and re-tuning σ on the same `val` split (`review/baseline_erode.py`: σ = 3, erode 3 px, val
scene IoU 0.548) gives a stronger classical baseline, reported in the headline table.

## Headline (1000 scenes per split)

| split | method | scene IoU | pixel IoU | det, isolated creatures* | coverage ≥ 0.5 | det, all creatures* | FP components / scene |
|---|---|---|---|---|---|---|---|
| test_id (seen families) | **U-Net** | **0.896** | 0.892 | **0.934** | 0.951 | 0.738 | 0.03 |
| test_id | baseline (shipped, σ 2) | 0.485 | 0.412 | 0.584 | 0.745 | 0.456 | 3.26 |
| test_id | baseline (reviewer, σ 3 + erode 3) | 0.556 | 0.529 | 0.627 | 0.707 | 0.508 | 0.56 |
| **test_ood (unseen families)** | **U-Net** | **0.832** | 0.827 | **0.898** | 0.923 | 0.717 | 0.30 |
| test_ood | baseline (shipped, σ 2) | 0.433 | 0.375 | 0.532 | 0.756 | 0.421 | 3.77 |
| test_ood | **baseline (reviewer, σ 3 + erode 3)** | **0.527** | 0.496 | **0.589** | 0.717 | 0.481 | 0.98 |

Against the stronger, val-tuned baseline the U-Net's held-out margin is 0.31 scene IoU (0.832 vs
0.527) and 0.31 isolated detection (0.898 vs 0.589), not the 0.40 / 0.37 the shipped baseline
suggested. Coverage ≥ 0.5 is not precision-aware (a method that over-predicts scores well on it),
which is why the eroded baseline's coverage drops while every IoU-based number rises. The
breakdown tables below still use the shipped baseline; the reviewer baseline's test_ood det(iso)
by δ is other family 0.796, δ 0.75–1 0.765, 0.5–0.75 0.675, 0.25–0.5 0.466, δ < 0.25 0.145.

\*Detected = instance IoU ≥ 0.5, where the instance is the union of predicted connected components
touching the creature. **About one third of creatures touch or overlap another creature.** A
semantic mask merges those into one blob, so "det, all creatures" mostly measures merging, not
seeing (U-Net coverage is still 0.95). For that reason the difficulty breakdowns below use
isolated creatures for detection and coverage for all creatures.

Going from seen to unseen texture families costs the U-Net 0.064 scene IoU (0.896 → 0.832), costs
3.6 points of isolated detection, and raises false-positive blobs tenfold (0.03 → 0.30 per scene,
worst on marble backgrounds: 0.46). On the unseen families it still beats both classical
baselines by a wide margin (0.832 vs 0.527 / 0.433 scene IoU).

## Broken down by how well the skin matches the background

**By the camouflage knob δ (test_ood, unseen families)**

| creature skin | n (isolated) | U-Net det(iso) | baseline det(iso) | U-Net cov | baseline cov |
|---|---|---|---|---|---|
| other family | 525 (333) | 0.979 | 0.715 | 0.998 | 0.941 |
| same family, δ 0.75–1 | 373 (251) | 0.964 | 0.697 | 0.973 | 0.912 |
| same family, δ 0.5–0.75 | 384 (249) | 0.948 | 0.647 | 0.987 | 0.854 |
| same family, δ 0.25–0.5 | 372 (253) | 0.937 | 0.387 | 0.978 | 0.661 |
| same family, δ < 0.25 | 361 (234) | **0.620** | **0.128** | 0.637 | 0.321 |

Finer slices of the δ < 0.25 bin, isolated creatures (`results/extra.json`):

| δ | test_id U-Net | test_id baseline | test_ood U-Net | test_ood baseline |
|---|---|---|---|---|
| 0–0.05 | 0.33 (n=43) | 0.00 | 0.15 (n=55) | 0.07 |
| 0.05–0.1 | 0.68 (n=41) | 0.05 | 0.60 (n=53) | 0.04 |
| 0.1–0.25 | 0.90 (n=125) | 0.28 | 0.83 (n=126) | 0.19 |

**By measured texture-statistics distance** between skin and background (colour mean/std,
oriented gradient energy, fine-detail energy). This descriptor is not an input to either
detector, but it is not independent of them (see caveat). Quartile edges are fixed on the training split, which is why the OOD bins are
unequal. **Caveat:** this descriptor is not independent of the baseline. It belongs to the same
feature family as `baseline.features` (colour mean/std, oriented gradient energy), so "closest
match" partly means "least visible to the baseline's own features", and the Q1 gap partly flatters
the U-Net. The δ axis above is the independent difficulty axis.

| split | quartile | n (isolated) | U-Net det(iso) | baseline det(iso) | U-Net cov | baseline cov |
|---|---|---|---|---|---|---|
| test_id | Q1 closest match | 462 (280) | 0.796 | 0.221 | 0.814 | 0.325 |
| test_id | Q2 | 519 (337) | 0.961 | 0.540 | 0.983 | 0.765 |
| test_id | Q3 | 512 (329) | 0.979 | 0.714 | 0.996 | 0.896 |
| test_id | Q4 least match | 507 (326) | 0.979 | 0.810 | 0.998 | 0.953 |
| test_ood | Q1 closest match | 324 (217) | 0.673 | 0.134 | 0.670 | 0.302 |
| test_ood | Q2 | 417 (279) | 0.900 | 0.455 | 0.930 | 0.700 |
| test_ood | Q3 | 488 (318) | 0.965 | 0.623 | 0.982 | 0.857 |
| test_ood | Q4 least match | 786 (506) | 0.953 | 0.688 | 0.987 | 0.911 |

Per held-out skin family (test_ood, isolated detection, U-Net vs baseline): marble 0.898 vs 0.494,
ripples 0.889 vs 0.547, streaks 0.908 vs 0.551. Full tables are in `results/eval.md`.

## Speed (busy machine; load average ≈ 13 during eval)
U-Net inference 0.44–0.52 ms/scene (batch 64, MLX GPU); shipped baseline 2.5–3.4 ms/scene
(NumPy/SciPy, one core) across the seed-0 and seed-1 eval runs (`results/*/eval.json`). Training: 5.3 steps/s at batch 32. These are under-load numbers only.

## Seed replicate (run after the write-up; `checkpoints/unet_seed1*`, `results/seed1/`)
I trained the same config with `--seed 1` (different init and batch order; same data), 5,500 steps in
12.4 min. The load was dropping by then, so this run was faster. The demo still uses the seed-0
checkpoint.

| metric (U-Net) | seed 0 | seed 1 |
|---|---|---|
| val scene IoU (selection) | 0.893 | 0.887 |
| test_id scene IoU | 0.896 | 0.894 |
| test_ood scene IoU | 0.832 | 0.839 |
| test_ood det, isolated | 0.898 | 0.908 |
| test_ood det(iso), same family δ < 0.25 | 0.620 | 0.620 |
| test_ood FP components / scene | 0.30 | 0.31 |

The two seeds agree to within about 0.01, so the headline gaps (about 0.3 scene IoU over the
reviewer's val-tuned baseline, 0.4 over the shipped one, and the 0.06 seen-to-unseen drop) are not seed noise. The two δ < 0.25 values are both
145/234 by coincidence; instance IoU in that bin differs (0.437 vs 0.452). Two seeds are not a
confidence interval.

## Demo (`media/`, produced by `demo.py` from `checkpoints/unet`; the detrended image is from `analysis_extra.py`)
- `ood_hardest.png` / `ood_hardest.gif`: the 8 held-out scenes whose best-hidden creature is
  closest to its background in texture statistics. Rows: scene, ground truth, U-Net, baseline.
  Green = hit, red = false alarm, blue = miss. The GIF cycles through the four rows.
- `ood_random.png`, `id_random.png`: 8 random scenes from each test split.
- `flat_ripples_scene943_detrended.png`: see Surprises.

## What worked
- A 2.8 M-param U-Net trained in 17 minutes carries over to texture families it never saw: the
  scene IoU drop is small (0.064). The texture-matching task seems to be learned as "find the
  region whose local statistics or phase disagree with its surroundings", not "memorise skins".
- The U-Net's biggest margin over the baseline is in camouflage proper. For same-family skins with
  δ in 0.25–0.5 on unseen families, it detects 94% of isolated creatures against the
  shipped baseline's 39% (47% for the reviewer's eroded baseline).
- Both difficulty axes (the generation knob δ and the measured statistic distance) order the
  difficulty roughly the same way for both methods (U-Net test_ood Q3 0.965 vs Q4 0.953 is the
  one inversion).

## What didn't / limitations
- **Near-perfect camouflage is mostly undetectable**: for δ < 0.05, U-Net detection is 0.33 on seen
  families and 0.15 on unseen ones. Often there is nothing to see (`ood_hardest.png`, column 2),
  and that is by design.
- **Overlapping creatures merge.** This is a semantic segmenter with no instance separation, so
  instance-level detection over all creatures is only about 0.72–0.74. An instance head
  (centre/offset or a boundary class) is the fix.
- The false-positive rate rises tenfold on unseen families (0.03 → 0.30 blobs/scene). Marble
  backgrounds trigger blobs most often.
- The shipped baseline fires on texture cells and edges (3–4 FP blobs per scene) and leaves a halo
  around each blob. The reviewer found most of that is post-processing: eroding the mask by 3 px
  (σ 3, tuned on val) cuts it to 0.6–1.0 FP blobs per scene and lifts held-out scene IoU
  0.433 → 0.527. The U-Net's tenfold FP rise from seen to unseen families still stands.
- Seed spread: see the seed-replicate section (two seeds only).
- **Not measured:** adversarial camouflage generator (go-deeper), transfer to detectors not used
  during camouflage training (go-deeper; there is no camouflage training yet), CNN vs compact ViT,
  idle-machine speed.

## Surprises
- In a scene that looks perfectly flat (`ood_hardest.png` column 4: low-contrast ripples
  background, creature at δ = 0.013), the U-Net still segments part of the creature (IoU 0.44).
  Blurring and removing the illumination plane (`media/flat_ripples_scene943_detrended.png`)
  shows why: the background has a faint ripple pattern, and the creature's patch breaks its phase
  (different ripple centres). That is a legitimate seam cue, not a renderer bug. Compositing is
  hard-masked and every nuisance (illumination, blur, noise) is applied to the whole scene after
  compositing, so no halo or noise-level cue exists.
- On seen families, even δ < 0.05 creatures are found 33% of the time. I expected the periodic
  families (which leave a phase seam at the blob boundary even with identical parameters) to
  account for most of these hits, but they don't. For δ < 0.1, isolated-creature detection by
  family is checker 0.61 (n=18), plaid 0.53 (19), spots 0.50 (8), stripes 0.46 (13), cells 0.43
  (14), fbm 0.42 (12), and 0.31–0.39 on the unseen families. The hits are spread across all
  families. The likely cues are the phase or seed discontinuity at the boundary plus a small
  colour offset (δ shifts the palette linearly). The sample sizes are small. These numbers are in
  `results/extra.json` under `det_iso_delta_lt_0.1_by_skin_family`.
- Most of the gap between scene IoU (0.90) and all-creature detection (0.74) was creature
  merging, not misses. This only showed up once detection was split into isolated and touching
  creatures.

## Next steps
1. Instance separation (boundary class or centre heatmap) so touching creatures count separately.
2. Go deeper: a constrained skin generator (palette and frequency bounded, opaque, fixed shape)
   trained to minimise this U-Net's recall. Then measure transfer to a held-out detector, e.g.
   the classical baseline or a second architecture trained on clean data.
3. Seeds ×3 and a wider/deeper receptive field to see whether the OOD false positives fall.

## Follow-up 1 (round 3, 2026-10-22 20:04–20:30): the painter — one round of the arms race

The first version above is only a segmenter. This follow-up builds one small round of the
project's namesake: a camouflage **painter** that picks each creature's skin to fool the trained
U-Net, then measures how much that hurts the attacked U-Net and whether the camouflage
**transfers** to detectors the painter never queried. Code: `painter.py` (search and all scoring),
`painter_analysis.py`, `painter_demo.py`. Numbers: `results/painter/summary.json` / `summary.md`,
`results/painter/analysis.json`; per-candidate scores in `results/painter/scores.npz`.

**Setup.** 400 new scenes, **held-out families only** (marble / ripples / streaks for background
and skin), seeds 9,000,000+ (no overlap with any existing split). One creature per scene (blob
radius 10–18), skin from the background's own family. The camouflage knob is fixed per scene,
δ ∈ {0.25, 0.5, 0.75, 1.0}, with 100 scenes each. For each scene the painter draws **K = 64 candidate skins** from the
generator itself (`perturb_params(bg, δ)` + `render_texture`, one RNG stream per candidate). The
background, blob mask, illumination, blur and sensor-noise field stay identical across the 64
candidates, so the candidates differ only in the skin. Every candidate is scored by four detectors:
`unet0` (the attacked frozen seed-0 U-Net), `unet1` (seed-1 U-Net, never queried by the unet0
painter), `base` (shipped classical baseline, outlier σ 2) and `base_er` (reviewer's eroded
baseline, σ 3, erode 3). A painter that attacks detector D picks the candidate that minimises D's
IoU (soft IoU for the U-Nets, hard IoU for the baselines). "Random skin" is the mean over all 64
draws (the expected score of one random draw at the same δ); the single draw 0 gives the same numbers
within 0.01 (e.g. unet0 0.866 vs 0.875), and so does one seeded random draw per scene (unet0 0.885,
unet1 0.895, base 0.404, base_er 0.580; reviewer, `review/painter_followup_trace.py`). All
off-diagonal cells use the same pick per scene; nothing is re-selected for the evaluated detector.

**Why this design (constraints).** The painter is a random search of size K inside the generator's
own δ-conditional distribution. It cannot change the shape, the opacity or the compositing. It
cannot leave the generator's palette or parameter ranges, and it cannot pick a different δ.
Painted and random skins therefore sit at exactly the same knob value. I rejected a free
parameter-space optimiser because `perturb_params` lets a
painter pick the "random" colour equal to the background colour, and orientation is periodic
(mod π) or ignored for some families. Any param-distance budget is gameable that way: it
"deletes the difference", which §14 rules out. The price is a weak attacker, best-of-64 with no
gradient steps or evolution.

Run: 3.0 min wall time under load (load average 14.3 at the end). Rendering took 21 s, the U-Net scoring
about 1 min and the two baselines about 1.6 min.

### Headline: painter × detector matrix (all 400 held-out scenes, mean scene IoU; det = IoU ≥ 0.5)

| evaluated detector ↓ / skin → | random | painted vs **unet0** | vs unet1 | vs base | vs base_er | closest texture-stat draw |
|---|---|---|---|---|---|---|
| **unet0** (attacked) | 0.875 (det 0.97) | **0.603** (det 0.73) | 0.686 | 0.822 | 0.828 | 0.842 |
| **unet1** (transfer) | 0.886 (det 0.97) | **0.692** (det 0.83) | 0.625 | 0.841 | 0.843 | 0.857 |
| base (shipped) | 0.401 (det 0.42) | 0.346 (det 0.30) | 0.351 | 0.145 | 0.173 | 0.247 |
| base_er (reviewer) | 0.575 (det 0.64) | 0.507 (det 0.50) | 0.500 | 0.283 | 0.242 | 0.416 |
| stat distance skin↔bg | 1.061 | 0.976 | 1.018 | 0.737 | 0.765 | (minimised) |

(`results/painter/summary.md`, section "all". The diagonal is white-box, and part of each diagonal drop
is min-of-64 selection on that detector's own idiosyncratic errors. The off-diagonal cells are the
honest transfer numbers.)

**(1) Drop at matched knob.** Against the attacked U-Net0 the painter lowers IoU from 0.875 to
0.603 (−0.27). Detection falls from 97% to 73%. Broken down by δ (`summary.md`):

| δ | unet0 random → painted (det) | unet1 random → painted-vs-unet0 (det) | base_er random → painted-vs-unet0 |
|---|---|---|---|
| 0.25 | 0.837 → **0.446** (0.97 → 0.49) | 0.847 → 0.507 (0.97 → 0.60) | 0.413 → 0.311 |
| 0.5 | 0.869 → 0.600 (0.96 → 0.75) | 0.881 → 0.709 (0.97 → 0.90) | 0.564 → 0.478 |
| 0.75 | 0.897 → 0.683 (0.99 → 0.84) | 0.909 → 0.767 (0.99 → 0.93) | 0.635 → 0.587 |
| 1.0 | 0.897 → 0.683 (0.97 → 0.85) | 0.907 → 0.784 (0.97 → 0.91) | 0.688 → 0.651 |

Even at δ = 1 (colours, scale and orientation re-drawn as far as the generator allows), U-Net0
misses 15% of best-of-64 painted creatures, against 3% of random ones.

**(2) Transfer: the Measure item.**
- **To the other U-Net: strong.** Camouflage painted against seed 0 drops seed 1 from 0.886 to
  0.692. That is 74% of the drop a painter attacking seed 1 directly achieves (0.886 → 0.625). Transfer is roughly symmetric:
  painting against seed 1 drops seed 0 to 0.686, 70% of the direct attack (74% the other way). Same architecture and
  data, different init: the two U-Nets share most of their blind spots.
- **To the classical baselines: weak.** Painted-vs-U-Net0 skins lower the shipped baseline by only
  0.055 (0.401 → 0.346) and the eroded one by 0.068 (0.575 → 0.507). Those are about 21% and 20% of
  what a painter attacking each baseline directly achieves (→ 0.145 and → 0.242). The reverse is
  also weak: skins painted against either baseline cost the U-Nets only about 0.05.
- **What the painter exploits differs by detector.** The model-free painter that just picks the
  draw closest to the background in the texture-stat descriptor (the "closest texture-stat draw" column) is
  a decent attack on the baselines (0.401 → 0.247, 0.575 → 0.416), which is expected because that
  descriptor shares the baselines' feature family. It barely touches the U-Nets (0.875 → 0.842).
  The U-Net painter's skins are only slightly closer in texture stats (1.061 → 0.976), so what
  fools the U-Nets is mostly *not* "closer by the classical texture statistics".
- Of the 107 painted creatures U-Net0 misses (IoU < 0.5), U-Net1 still finds 52%, the eroded
  baseline 33% and the shipped baseline 18% (`results/painter/analysis.json`).

**What the painted skins look like** (`media/painter_best.png`: for each δ, the scene with the
largest U-Net0 drop; `media/painter_grid.png`: 8 random scenes, random vs painted). Averaged
over all scenes (`analysis.json`), the U-Net0 painter mostly picks draws whose **mean colour
matches the surrounding background**: the gap between the mean colour inside the creature and a
6-px ring around it falls from 0.240 to 0.161 (RGB, 0–1), and it is smaller than draw 0's gap in 73% of scenes (reviewer). The skin's texture contrast does not
fall (per-channel std inside 0.065 → 0.074). The worst cases in `painter_best.png` are a different,
more interesting failure. At δ = 0.75 the U-Net0-best skin on yellow/navy streaks is a nearly
flat dark-grey patch, obvious to a human. U-Net0 gets IoU 0.06, while U-Net1 (0.69) and the eroded
baseline (0.45) find it. At δ = 1.0 a pale, washed-out marble blob gets IoU 0.00 from *both*
U-Nets, while the eroded baseline segments it at 0.66. Both U-Nets put a false-positive blob
elsewhere in that scene instead. So in these hand-picked worst cases what the painter finds is not
camouflage in the human sense. It looks like a U-Net blind spot for low-contrast or flat skins on
high-contrast held-out backgrounds. The aggregate does **not** support "flat skins" as the general
mechanism: the painted skin is flatter than draw 0 in only 40% of scenes, and fewer painted skins than
random ones have std below half the background's (13% vs 19%, `analysis.json`). So the two examples are
anecdotes, not the typical pick. Within a δ budget, "looking different" is not enough to be seen by a
texture-trained CNN.

**Surprise: part of the attack is non-local.** The background more than 3 px from the creature is
pixel-identical across the 64 candidates of a scene (reviewer check: max difference 0/255 beyond 3 px,
at most 1/255 at 3 px, because the σ = 0.5 blur reaches 2 px; `review/painter_followup_check.py`).
Even so, painted skins roughly triple U-Net0's **false positives far from the creature** (more than
3 px away) (`analysis.json`, keys `nonlocal/*`):

| | recall inside creature | FP pixels > 3 px from creature / scene | scenes with ≥ 20 such FP px |
|---|---|---|---|
| unet0, random skin (draw 0) | 0.954 | 64 | 29% |
| unet0, painted vs unet0 | 0.756 | **176** | **58%** |
| unet1, random skin (draw 0) | 0.956 | 49 | 23% |
| unet1, painted vs unet0 | 0.791 | 98 | 38% |

In IoU terms the far false positives are a minority of the drop. Removing them from the predictions
leaves a U-Net0 drop of 0.207 out of 0.263, so they account for about **21%** of the attack on U-Net0.
For the transferred attack on U-Net1 the share is 8% (0.174 of 0.189;
`results/painter/review_nonlocal_decomp.json`, reviewer). Most of the drop is missed creature pixels
(recall 0.95 → 0.76). The painter scores candidates by whole-image soft IoU, so it is free to select
skins that cause far false positives, and part of the U-Net0 share is that selection. Still, the skin
changes predictions in untouched background regions. The likely channels are the U-Net's
whole-image receptive field (5 levels on 96 px) and GroupNorm, whose statistics are pooled over the
whole image. These are global shortcuts, not camouflage in the biological sense. They show up as the
spurious red blobs in `media/painter_grid.png` and `painter_best.png`. Part of this transfers to
the seed-1 U-Net, which shares the architecture. I did not test the GroupNorm hypothesis directly
(not measured).

### (3) One round of retraining (`run_painter_round1.sh`, `run_painter_round1_control.sh`; numbers in `results/painter_round1/compare.json`)
- **Painted training data, seen families only:** `painter.py --families train --no_baselines`
  with 1,000 new training-family scenes (seeds 8,000,000+), K = 32, painted against U-Net0
  (`results/painter_train/summary.md`). On seen families the same painter is weaker: 0.939 → 0.774,
  det 1.00 → 0.93. K differs from the held-out run, so this is not a matched comparison.
- **Fine-tune:** `painter_finetune.py` takes U-Net0 and trains 1,500 steps (2.5 min) on 50% original
  training scenes and 50% painted creatures, lr 3e-4 with warmup and cosine, last checkpoint, no
  selection. Output: `checkpoints/painter_round1/unet_ft.*`.
- **Control:** an identical fine-tune on the same 1,000 scenes, but with the random skin draw instead
  of the painter's pick (`checkpoints/painter_round1/unet_ft_control.*`). Without this control,
  "more training on single-creature scenes" would be mistaken for adversarial hardening.
- **Re-attack:** a fresh painter (K = 64) against each fine-tuned model, on the same 400
  held-out scenes and the same candidate skins:

| model | val IoU (seen) | random skin | stale picks (painted vs original U-Net0) | **fresh re-attack** (det) | U-Net1 on the fresh picks |
|---|---|---|---|---|---|
| original U-Net0 | 0.893 | 0.875 | 0.603 (white-box) | **0.603** (0.73) | 0.692 |
| fine-tuned, random skins (control) | 0.894 | 0.889 | 0.628 | **0.608** (0.73) | 0.696 |
| fine-tuned, painted skins | 0.888 | 0.888 | 0.683 | **0.650** (0.77) | 0.710 |

- **Reading:** retraining on painted creatures buys **+0.042 IoU under a fresh attack** over the
  control (paired over the 400 scenes: 0.0425 ± 0.006 SE). That SE covers scenes only, from one
  fine-tune each, so training-seed variance is not measured. The cost is 0.006 seen-family val IoU.
  Most of the attack survives: the defended model still loses 0.24 IoU to a fresh best-of-64 painter.
  So in round 1 the painter stays well ahead.
- The "stale picks" column flatters retraining. Any model other than the attacked one scores
  about 0.63–0.69 on those picks (U-Net1 gets 0.692), so only the fresh re-attack column measures
  robustness. The fresh picks against the defended model cost U-Net1 slightly less (0.710 vs 0.692), but
  they are also weaker overall (re-attack 0.650 vs 0.603), so this is not evidence of less transfer.
- The δ breakdown is in `compare.json`. The gain over the control is +0.03 to +0.06 in every δ bin
  (e.g. δ = 0.25: 0.474 vs 0.429).

**Caveats.**
- The diagonal (white-box) drops include the min-of-K selection effect on the attacked model's
  own noise. The cross-seed numbers carry the claim.
- K = 64 random search is a weak painter. A gradient or evolutionary painter would push all
  numbers further, and transfer might change with it.
- One creature per scene, so these IoUs are not comparable to the multi-creature headline table above.
- 100 scenes per δ bin. Per-bin standard errors were not computed. The only formal SE is the
  paired fine-tune-vs-control one above.
- Not measured: a learned (gradient or evolutionary) painter, painting outside the generator's
  distribution, multi-creature painted scenes, a second retraining round, retraining seeds, a
  painter against a different architecture (e.g. a ViT), and idle-machine timings (everything
  here ran under load, load average 14–33).

**Cross-review (follow-up 1).** Every number in this section traces to `results/painter*/` or
`checkpoints/painter_round1/*_log.json`. Two claims were weakened: the non-local false positives are
about 21% of the U-Net0 IoU drop, not half, and "flat skins" describes the hand-picked worst cases,
not the typical pick.

**Next steps (follow-up).** Swap the best-of-K painter for CMA-ES over the perturbation latent, with
a hard perceptual-difference budget. Try a U-Net with local normalisation (or no GroupNorm) to test
whether the non-local false positives go away. Iterate painter ↔ detector for 3–5 rounds and
track transfer to the frozen baselines each round.
