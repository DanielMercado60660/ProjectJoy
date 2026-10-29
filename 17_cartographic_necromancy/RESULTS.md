# Results: Cartographic Necromancy (first satisfying version)

All numbers come from `results/eval.json` and `results/eval_table.md`, produced by `evaluate.py --ckpt checkpoints/unet`
from one 12-minute training run (`results/train_log.txt`). Ten projects were training at the same time
(load average about 4–10 during this project's run and eval). This project records no speed numbers apart from that wall time.

## Setup
- **Maps:** vector road graphs rasterized at 64×64. Roads are 2 px wide and arterials 3 px. Train/val style is `grid`
  (a jittered grid, 20% of edges dropped, plus 1–2 curved arterials). Road pixels cover about 27% of the image, so the maps are dense.
- **Ruin:** one axis-aligned rectangle per map, at least 3 px from the image border, touching at least 2 boundary road pixels.
- **U-Net:** 1.21M parameters (32/64/128/128 channels, bottleneck 8×8). Input is the masked raster plus the mask.
  Loss is BCE on hole pixels only, with AdamW and cosine LR 2e-3. The run did 7 367 steps × batch 64 in 720 s, and the
  EMA hole BCE fell from 0.66 to 0.247. Checkpoint: `checkpoints/unet.safetensors` and `.json`.
- **Composite before scoring:** every method keeps the known pixels and only fills the hole.
- **Held-out tests (500 maps each, seeds disjoint from train and val):** `test_id` (grid style, 12–24 px holes),
  `test_bighole` (grid style, **28–34 px** holes, never seen in training), `test_delaunay` (**random planar Delaunay style**,
  never seen in training).

## Metrics (all computed through the hole only)
Stubs are the 8-connected road components on the 1-px ring just outside the hole. Connectivity is labelled on
(hole ∪ ring) for both GT and the restoration, so a path that goes around the hole never counts.
I didn't skeletonize because skimage isn't installed. A thresholded mask with 8-connectivity gives the same component structure.
- `partition_exact`: the restoration groups the stubs exactly as the GT does. `stub_correct`: the share of stubs whose connected set matches GT exactly.
- `pair_recall` / `pair_precision` / `pair_f1`: over stub pairs that are connected through the hole.
- `continuity`: the share of stubs that continue into the hole in GT and also continue in the restoration (boundary continuity).
- `hole_iou`: strict pixel IoU inside the hole. `relaxed_f1`: the same comparison with a 1-px tolerance, the usual choice for thin roads.
- **Primary metric, `correct_and_faithful`:** `partition_exact` AND relaxed precision ≥ 0.8 AND relaxed recall ≥ 0.8.
  Every method, the U-Net threshold included, is tuned on val for this metric.
- `plausible_but_wrong`: pixels are faithful (relaxed P and R ≥ 0.8) but the partition is wrong.
- `dangling_per_map`: restored components inside the hole that touch no stub.

## Baselines (all tuned on the 300 val maps, grids in `evaluate.py`)
- `stub_lines`: direction-aware straight-line reconnection. 240 configs (128 in the first pass). Best: `w_ang=4, thr=3.0, ext=1.0, lw=3`.
  Each stub's direction runs from the centroid of nearby known road to the stub. My first version used PCA, which
  broke at junctions just outside the hole. The method pairs stubs greedily by min cost (distance/size + w·misalignment),
  then extends unmatched stubs straight until they hit restored road or leave the hole.
- `diffusion`: Laplace fill plus a threshold. 28 configs. Best: `iters=1500, thr=0.15`.
- `fill_all`: a sanity reference that paints the whole hole as road.
- U-Net threshold: tuned over {0.05 … 0.7}. Best: 0.2, an interior point of the grid.
- The first tuning pass hit grid edges for `stub_lines` and diffusion. I widened the grids and re-ran; the first-pass table is
  kept in `results/eval_table_grid1.md`. After widening, `stub_lines` val C&F rose from 0.247 to 0.297. Its best `thr=3.0` still
  sits on the grid edge, so it may not be fully tuned. *Reviewer:* extending thr to {4.5, 6, 10, ∞} and w_ang to 8 gives val
  0.303 (best `w_ang=8, thr=4.5`, interior in thr), test_id 0.300; optimal max-weight matching instead of greedy does not help
  (val 0.297). The real gap was roads along the hole edge; see the `stub_lines + edge fill` column in the headline.

## Headline (test sets, `results/eval_table.md`)

| test set | U-Net C&F | stub_lines + edge fill C&F (reviewer) | stub_lines C&F (builder) | diffusion C&F | fill_all C&F |
|---|---|---|---|---|---|
| test_id (in-distribution) | **0.726** | 0.354 | 0.284 | 0.080 | 0.016 |
| test_bighole (held-out 28–34 px) | **0.678** | 0.162 | 0.116 | 0.008 | 0.000 |
| test_delaunay (held-out style) | 0.202 | **0.250** | 0.220 | 0.140 | 0.030 |

**Reviewer update:** the strongest baseline is now `stub_lines + edge fill` (`review_edgefill.py`,
tuned on val only, `results/review/edgefill.json`): the same straight-line pairing plus each stub dilated 2 px into the
hole, which restores roads that run along the hole edge. It lifts the baseline by 0.03–0.07 on every test set. The U-Net
still wins on grid maps by about 2× (0.726 vs 0.354) and about 4× on big holes (0.678 vs 0.162). On Delaunay it now
loses by 0.048 (0.202 vs 0.250, about 2 standard errors), not "narrowly". Under looser or stricter faithfulness
thresholds (0.7 or 0.9 instead of 0.8) the ranking does not change; see `results/review/review.json`.

Full table (all metrics):

| set | method | correct_and_faithful | partition_exact | stub_correct | pair_recall | pair_precision | pair_f1 | continuity | hole_iou | relaxed_f1 | plausible_but_wrong | dangling_per_map |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val | fill_all | 0.013 | 0.670 | 0.664 | 1.000 | 0.846 | 0.917 | 1.000 | 0.286 | 0.712 | 0.000 | 0.000 |
| val | diffusion | 0.080 | 0.430 | 0.458 | 0.818 | 0.869 | 0.843 | 1.000 | 0.338 | 0.779 | 0.093 | 0.000 |
| val | stub_lines | 0.297 | 0.467 | 0.470 | 0.804 | 0.882 | 0.841 | 0.997 | 0.374 | 0.809 | 0.163 | 0.000 |
| val | unet | 0.770 | 0.787 | 0.790 | 0.995 | 0.902 | 0.946 | 0.998 | 0.636 | 0.957 | 0.107 | 0.010 |
| test_id | fill_all | 0.016 | 0.646 | 0.630 | 1.000 | 0.834 | 0.910 | 1.000 | 0.295 | 0.717 | 0.002 | 0.000 |
| test_id | diffusion | 0.080 | 0.464 | 0.469 | 0.845 | 0.854 | 0.849 | 1.000 | 0.347 | 0.783 | 0.064 | 0.000 |
| test_id | stub_lines | 0.284 | 0.438 | 0.424 | 0.801 | 0.853 | 0.827 | 0.998 | 0.383 | 0.810 | 0.152 | 0.000 |
| test_id | unet | 0.726 | 0.752 | 0.740 | 0.990 | 0.877 | 0.930 | 0.999 | 0.637 | 0.953 | 0.172 | 0.010 |
| test_bighole | fill_all | 0.000 | 0.660 | 0.646 | 1.000 | 0.887 | 0.940 | 1.000 | 0.298 | 0.726 | 0.000 | 0.000 |
| test_bighole | diffusion | 0.008 | 0.334 | 0.337 | 0.812 | 0.895 | 0.852 | 1.000 | 0.327 | 0.756 | 0.006 | 0.000 |
| test_bighole | stub_lines | 0.116 | 0.402 | 0.407 | 0.851 | 0.901 | 0.875 | 0.998 | 0.323 | 0.738 | 0.068 | 0.000 |
| test_bighole | unet | 0.678 | 0.728 | 0.724 | 0.993 | 0.912 | 0.951 | 1.000 | 0.542 | 0.925 | 0.194 | 0.050 |
| test_delaunay | fill_all | 0.030 | 0.632 | 0.563 | 1.000 | 0.765 | 0.867 | 1.000 | 0.216 | 0.555 | 0.016 | 0.000 |
| test_delaunay | diffusion | 0.140 | 0.518 | 0.475 | 0.889 | 0.787 | 0.835 | 1.000 | 0.318 | 0.754 | 0.086 | 0.000 |
| test_delaunay | stub_lines | 0.220 | 0.448 | 0.358 | 0.677 | 0.822 | 0.742 | 0.999 | 0.313 | 0.724 | 0.100 | 0.000 |
| test_delaunay | unet | 0.202 | 0.616 | 0.569 | 0.871 | 0.843 | 0.857 | 0.992 | 0.337 | 0.731 | 0.132 | 0.280 |

## What worked
- On grid maps the U-Net restores correct topology with faithful pixels about 2× as often as the strongest tuned
  baseline (0.726 vs 0.354 C&F on test_id for the reviewer's `stub_lines + edge fill`; the builder's `stub_lines` had 0.284, i.e. 2.6×).
  It also generalizes to holes larger than any it trained on (0.678 vs 0.162). On test_id its hole IoU is 0.64, against 0.38 for the best baseline.
- Boundary continuity is about 1.0 for every method. Every method continues nearly every stub into the hole, so
  continuity alone says almost nothing; the errors are all in what the stubs connect to.

## What didn't / surprises
1. **Connectivity alone is gameable.** `fill_all` (paint the whole hole) gets pair recall 1.0, pair F1 0.91 and an
   exact partition on 65% of test_id maps. In a dense grid, nearly every stub that enters a hole meets every other at
   some junction, so "connect everything" is usually the right topology. Connectivity therefore has to be paired with pixel
   faithfulness, which is why the primary metric requires both; `fill_all` drops to 0.016 on it. The plain pair-F1 ranking
   (U-Net 0.93 > fill_all 0.91 > stub_lines 0.83) would have made the trivial baseline look second best.
2. **The U-Net over-connects.** Its pair recall is 0.99 but its precision only 0.88, and its tuned threshold is low (0.2).
   That threshold maximizes the share of maps whose connectivity is exactly right: at 0.5 the val C&F would be 0.35
   instead of 0.77, because roads break up. Its typical failure is *plausible but wrong*. It draws a clean grid cross or an
   extra stub-to-stub link where the GT has a jog, a T-junction or a dead end (17% of test_id maps, 19% of bighole). See
   `media/plausible_but_wrong_unet.png` (the PNG selects on relaxed F1 ≥ 0.8, a slightly looser criterion than the table's P and R ≥ 0.8): the pixels look like a road map, but the link structure is invented. Several of these
   cases have a road running *along* the hole boundary, which shows up as one long stub. That is a real edge case of the stub definition.
3. **Style shift breaks the U-Net.** On Delaunay maps (sparser, arbitrary angles, never seen in training) the U-Net falls to
   0.202 C&F, under the straight-line baseline (0.220; 0.250 for the reviewer's `stub_lines + edge fill`, a gap of about 2 SE). It also leaves 0.28 dangling blobs per map, against 0 for the
   baselines (`media/strip_test_delaunay.png`). It learned "grid priors", not "continue roads along their direction". Its
   partition_exact (0.616) is the best among the non-trivial methods but below `fill_all`'s 0.632. Much of it comes from blobby over-filling, the same trick `fill_all` uses.
4. My first stub-direction estimate (PCA) was wrong at junctions near the hole. It was fixed before tuning (see Baselines).

## Not measured
- Terrain/river consistency (the "Go deeper" stage): not measured.
- Multiple valid completions or ambiguity: not measured. Each hole has one GT.
- Blob-shaped (non-rectangular) holes: not measured.
- Seeds: one training seed only, so no error bars. Test sets have 500 maps, so the sampling standard error of a C&F rate is
  about 0.02 (binomial).
- Speed and latency: not measured. Eval took 91 s of wall time under load.

## Next steps
- Train on mixed styles (grid and Delaunay) and with blob holes. Check whether the Delaunay gap closes.
- Add a topology-aware loss term (e.g. a connectivity penalty on stub pairs) to cut the over-connection.
- Sample multiple completions (diffusion model) for genuinely ambiguous holes, and score them as a set.
