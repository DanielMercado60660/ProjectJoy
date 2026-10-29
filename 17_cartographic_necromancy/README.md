# 17 · Cartographic Necromancy: restore maps from ruins

This project generates synthetic road networks as vector graphs, rasterizes them to 64×64, and erases a rectangle
(the "ruin"). A small MLX U-Net then inpaints the missing roads. Scoring looks at **structure** first: are the
road stubs that cross the hole boundary connected through the hole the same way they are in the ground truth?

## Run it

```bash
cd 17_cartographic_necromancy
./smoke.sh                                                    # end-to-end at tiny size, ~25 s

# main training run (12 min wall-clock cap, through the shared GPU lock)
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python train.py --minutes 12 --n_train 20000 --out checkpoints/unet

# tune every method on validation, then score on 3 held-out test sets -> results/eval.json, results/eval_table.md
OMP_NUM_THREADS=4 ../.venv/bin/python evaluate.py --ckpt checkpoints/unet

# demo PNG strips from the saved restorations -> media/
../.venv/bin/python demo.py

# reviewer checks: wider stub_lines grid + optimal matching, alt faithfulness thresholds, edge-road maps
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python review_check.py      # -> results/review/review.json (~20 s)
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python review_edgefill.py   # -> results/review/edgefill.json (~40 s)
```

## Files

| file | what |
|---|---|
| `maps.py` | map generators (`grid`: perturbed grid + curved arterials; `delaunay`: random planar graph), rasterizer, holes, seed ranges |
| `metrics.py` | stub extraction, through-the-hole connectivity, pair recall/precision, exact partition, continuity, IoU, relaxed P/R |
| `baselines.py` | `stub_lines` (direction-aware straight-line pairing + extension), `diffusion` (Laplace fill + threshold) |
| `train.py` | U-Net (32-64-128-128 channels, 3 downsamples), training loop, `load_model`, `predict_prob` |
| `evaluate.py` | val tuning (baseline grids + U-Net threshold), test scoring, per-sample dumps |
| `demo.py` | PNG strips: ruin, diffusion, stub_lines, U-Net, ground truth. Stub colour shows whether connectivity is correct |
| `smoke.sh` | tiny end-to-end check |
| `review_check.py`, `review_edgefill.py` | cross-review checks; `stub_lines + edge fill` is the strongest baseline |
| `checkpoints/unet.{safetensors,json}` | trained weights + config; `unet_trainlog.json` has the loss curve |
| `results/` | `train_log.txt`, `eval_log.txt`, `eval.json`, `eval_table.md` (final), `eval_table_grid1.md` (first, narrower tuning grids), `samples_<set>.npz` (restorations + per-sample scores) |
| `media/` | `strip_<set>.png`, `plausible_but_wrong_<method>.png` |

## Data and splits
- Train: grid style, seeds 0–19 999, holes drawn on the fly (12–24 px per side, random flips and transposes).
- Val: grid style, seeds 1 000 000+, fixed holes 12–24 px (300 maps). All tuning happens here.
- Test (500 maps each, seeds 2 000 000+ in disjoint blocks): `test_id` (grid, 12–24),
  `test_bighole` (grid, **28–34 px holes, never seen in training**), `test_delaunay` (**held-out map style**, 12–24).
- Each hole must touch at least 2 road pixels on its boundary ring. The vector graph is kept in `build_set(...)["graphs"]`.
