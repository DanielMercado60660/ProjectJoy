# 14 · Camouflage Wars (first satisfying version: the segmenter)

Procedurally generated 96×96 scenes: a textured background with 1–3 blob "creatures" painted in a
skin that may closely imitate the background. A tiny MLX U-Net learns to segment the creatures; it
is tested on **texture families it never saw** (neither as background nor as skin) and compared with
a classical texture-statistics baseline. Results are broken down by how well the creature's skin
matches the background.

Follow-up 1 adds one small round of the arms race: a best-of-K camouflage painter against the frozen
U-Net, with transfer to a second-seed U-Net and both classical baselines (see below and RESULTS.md).
A learned (gradient-trained) pattern generator is still not built.

## The scenes (`scenes.py`)
- Nine texture families, each a parametric scalar field coloured with a random two-colour palette:
  - training families: `stripes, checker, spots, fbm, cells, plaid`
  - **held-out families**: `marble, ripples, streaks` (used only in `test_ood`)
- Creature shape: radial-Fourier blob, radius 8–18 px. Creatures may overlap.
- Creature skin: 25% "other family" (an independent texture from another family of the same split);
  75% "same family" = the background's own family with its parameters (colours, scale,
  orientation, sharpness) moved away by a camouflage knob **δ ~ U[0,1]**. δ = 0 means identical
  parameters with only a new phase/seed: near-perfect camouflage, often invisible by design.
- Nuisances on the whole scene: illumination gradient, slight blur, sensor noise.
- Difficulty is also measured independently of δ: `texture_stats` describes each full texture
  field (colour mean/std, oriented gradient energy, fine-detail energy) and the creature's
  **stat distance** is the standardised distance between its skin and background descriptors.
  Quartile edges are fixed from the training split. (This descriptor shares a feature family with
  the baseline, so the δ knob is the independent difficulty axis.)

## Methods
- **U-Net** (`model.py`, `train.py`): 5 levels, widths 24-48-96-192-192, GroupNorm + GELU,
  2.76 M params, BCE + soft-Dice, AdamW (lr 2e-3, warmup + cosine), batch 32, flips/rot90 plus
  channel-permutation / gain colour augmentation. Checkpoint chosen by IoU on `val` (seen families).
- **Classical baseline** (`baseline.py`): per-pixel local features (Gaussian-pooled colour, local
  colour std, 4 oriented gradient energies). Two variants: per-image 2-means (creature cluster =
  the one touching the border least) and a robust background model (median/MAD outlier score with
  an Otsu threshold). Both are cleaned by an opening and removal of components < 30 px. Variant and
  window σ are tuned on the seen-family `val` split (`tune_baseline.py` → `baseline_choice.json`).
  Cross-review added a stronger variant (mask eroded 3 px, σ 3, also tuned on `val`) in `review/`;
  RESULTS.md reports both.

## Metrics (`metrics.py`)
- scene IoU (foreground IoU per scene, averaged), pooled pixel IoU
- per creature: instance IoU between the creature and the union of predicted connected components
  touching it; **detected** if instance IoU ≥ 0.5
- false-positive components per scene (≥ 10 px, < 10% on any creature)
- breakdowns by δ bin / "other family", by stat-distance quartile, and by skin family

## Commands (run from this folder)
```bash
../.venv/bin/python make_data.py                     # data/: 24k train, 500 val, 1000 test_id, 1000 test_ood (~45 s)
../.venv/bin/python tune_baseline.py                 # baseline_choice.json (val only)
../bin/gpu_run ../.venv/bin/python train.py --steps 5500 --eval_every 500 --max_minutes 18.5 --out checkpoints/unet
../.venv/bin/python evaluate.py                      # results/eval.json, results/eval.md
# seed replicate: train.py ... --seed 1 --max_minutes 19 --out checkpoints/unet_seed1; evaluate.py --ckpt checkpoints/unet_seed1 --out results/seed1
../.venv/bin/python analysis_extra.py                # results/extra.json (δ<0.25 slices, FP by background family)
../.venv/bin/python demo.py                          # media/*.png, media/ood_hardest.gif (reloads checkpoints/unet)
../.venv/bin/python review/baseline_variants.py      # reviewer: min-area / multi-scale baseline sweep (val)
../.venv/bin/python review/baseline_erode.py         # reviewer: erosion sweep (val) -> review/baseline_erode.json
../.venv/bin/python review/baseline_variant_test.py  # reviewer: score val-chosen variants on test splits
./smoke.sh                                           # tiny end-to-end check, ~20 s, writes to a temp dir
```

### Follow-up 1 (round 3): the painter — one round of the arms race
A camouflage painter picks each creature's skin (best of K draws from the generator at a fixed
camouflage knob δ, so the shape, opacity, palette and δ are unchanged) to minimise the frozen U-Net's IoU.
The same candidates are scored by the seed-1 U-Net and both classical baselines, which gives the transfer matrix.
Held-out families only. Results in RESULTS.md, section "Follow-up 1".
```bash
../bin/gpu_run ../.venv/bin/python painter.py --n_per_delta 100 --k 64 --out results/painter   # ~3 min -> results/painter/{summary.md,summary.json,scores.npz,images.npz}
../.venv/bin/python painter_analysis.py results/painter      # colour-gap / skin-contrast stats -> results/painter/analysis.json
../.venv/bin/python painter_demo.py                          # media/painter_best.png, media/painter_grid.png
../bin/gpu_run ./run_painter_round1.sh                       # retraining round: painted TRAIN-family data (results/painter_train/),
                                                             # fine-tune U-Net0 (checkpoints/painter_round1/), re-attack held-out (results/painter_round1/)
../bin/gpu_run ./run_painter_round1_control.sh               # control: same fine-tune on random skins + re-attack -> results/painter_round1_control/
../.venv/bin/python painter_round1_compare.py                # orig vs fine-tuned vs control -> results/painter_round1/compare.json
./smoke_painter.sh                                           # painter pipeline at tiny size, ~6 s, temp dir only
```
All painter runs are deterministic (scene seeds 9,000,000+ held-out, 8,000,000+ train-family; one RNG
stream per candidate skin). All `images.npz` files are gitignored, including `results/painter/images.npz`
(48 MB), which `painter_analysis.py` and `painter_demo.py` need. Regenerate it first with the
`painter.py --n_per_delta 100 --k 64 --out results/painter` command above (deterministic, ~3 min; the
the reviewer re-rendered one scene and it matched the stored images bit for bit). `painter_finetune.py`
also needs `data/` (`make_data.py`) and `results/painter_train/images.npz` (first line of `run_painter_round1.sh`).
Reviewer checks for this follow-up: `review/painter_followup_trace.py`, `review/painter_followup_check.py`.

## Layout
```
scenes.py          texture families, blobs, camouflage knob, difficulty descriptor
make_data.py       fixed-seed splits -> data/ (940 MB, gitignored; regenerate in ~45 s;
                   chunking depends on --workers, so keep the default 4 to reproduce exactly)
model.py           tiny U-Net + save/load
train.py           training loop (val = seen families only)
baseline.py        classical segmenters; tune_baseline.py picks variant/σ on val
metrics.py         IoU, detection, false positives, difficulty breakdowns
evaluate.py        test_id + test_ood evaluation -> results/
analysis_extra.py  extra slices -> results/extra.json
demo.py            PNG strips + GIF from the checkpoint -> media/
checkpoints/       unet.safetensors/.json (best val), unet_last.*, unet_log.json, train.log
results/           eval.json, eval.md, per-scene IoU arrays
review/            cross-review baseline scripts + JSON results
painter.py         follow-up: best-of-K skin painter + 4-detector scoring -> results/painter*/
painter_analysis.py, painter_demo.py, painter_finetune.py, painter_round1_compare.py,
run_painter_round1.sh, run_painter_round1_control.sh, smoke_painter.sh
results/painter*/  checkpoints/painter_round1/  media/painter_best.png, media/painter_grid.png
media/             ood_random.png, ood_hardest.png, ood_hardest.gif, id_random.png
```
