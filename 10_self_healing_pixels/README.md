# 10 · Self-Healing Pixels

A **neural cellular automaton** (NCA) that grows a lizard emoji from a single cell, and, if it was trained
for it, grows the missing parts back after you cut pieces off. Every cell runs the same tiny neural network
(48 → 128 → 16) and sees only its 3×3 neighbourhood. This follows *Growing Neural Cellular Automata*
(Mordvintsev et al., Distill 2020), written in MLX.

The comparison: the **regeneration-trained** NCA (sample pool + circular wounds during training) against a
**growth-only** NCA (same pool, same steps, same seed, never damaged). Both are tested on damage they never
saw: half-cuts, stripes and random blobs.

Results and numbers: [RESULTS.md](RESULTS.md).

## Look at it
- `media/demo.html`: open it in any browser (works offline, also on iPad). Drag on the creature to erase
  cells and watch it regrow. You can switch between the regeneration-trained and growth-only models.
- `media/grow_damage_regrow.gif` (and `.mp4`): side by side, growth-only on the left and regeneration-trained
  on the right, same seed and same damage: grow → left half cut off → random blobs → circle.
- `media/filmstrip_<damage>.png`: grown | damaged | +25 | +50 | +100 | +200 | +400 steps, for both models.
- `media/recovery_curves_lizard.png`: MSE vs steps after each damage type, plus the 2000-step rollout.
- `media/repeated_damage.gif`: 10 random held-out wounds in a row. The baseline explodes and the regen model slowly fades.
- `media/failures/`: clips of the strange cases, plus `stillbirth_grid.png` (see RESULTS.md).
- `media/gallery.gif`: other organisms (butterfly, octopus) cut and regrowing, if their runs finished.

## Run it
All commands from this folder (`10_self_healing_pixels/`).

```bash
# smoke test (< 1 min): tiny training run, checkpoint/resume, eval pieces, JS port check
../.venv/bin/python -m pytest -q

# train (about 30-45 min each on a shared M5 Max; resumable with --resume)
../bin/gpu_run ../.venv/bin/python train.py --mode regen   --steps 8000 --ckpt_every 1000 --out checkpoints/lizard_regen
../bin/gpu_run ../.venv/bin/python train.py --mode persist --steps 8000 --ckpt_every 1000 --out checkpoints/lizard_persist

# seed-1 replicates: same commands with --seed 1 --out checkpoints/lizard_{regen,persist}_s1

# evaluate all four on the same damage protocol -> results/eval_lizard.json, media/recovery_curves_lizard.png
../bin/gpu_run ../.venv/bin/python eval.py --models lizard_persist lizard_regen lizard_persist_s1 lizard_regen_s1 --trials 32
../.venv/bin/python report.py lizard        # markdown tables used in RESULTS.md

# demos: GIF/MP4, repeated-damage GIF, filmstrips, failure clips, demo.html (seed-0 pair)
../.venv/bin/python make_demo.py --models lizard_persist lizard_regen
# 16-seed "does it even grow" grid for all four models
../.venv/bin/python make_demo.py --models lizard_persist lizard_regen lizard_persist_s1 lizard_regen_s1 --only birth
# gallery of other organisms (regen-only, 4000 iterations each)
logs/run_gallery.sh butterfly 4000 3000
../.venv/bin/python make_gallery.py --runs lizard_regen:lizard butterfly_regen:butterfly octopus_regen:octopus

# check the JavaScript port in demo.html against MLX (headless, needs node)
node tests/check_demo_js.mjs media/demo.html
```

Under GPU contention the runs here were stopped at a checkpoint and continued with `--resume` (same command plus
`--resume`); see the `=== ... ===` lines in `logs/train_*.log`.

Modes in `train.py`: `grow` (always restart from the seed, no pool), `persist` (sample pool, no damage: the
growth-only baseline), `regen` (sample pool, and the 3 lowest-loss samples in each batch of 8 get a random
circular wound).

## Files
| file | what |
|---|---|
| `nca.py` | target rendering (Apple Color Emoji → 40×40 premultiplied RGBA, padded to 72×72), the NCA model, JSON export |
| `damage.py` | damage masks: `circle` (training) and held-out `half`, `stripes`, `blobs` |
| `train.py` | pool training with per-variable gradient normalisation, checkpoint/resume (weights, Adam state, pool, numpy RNG, step, config) |
| `eval.py` | damage/recovery protocol and long-rollout stability |
| `make_demo.py` | GIF/MP4, filmstrips, failure clips, `media/demo.html` |
| `demo_template.html` | the interactive page; the JS update rule mirrors `nca.py`, and weights are inlined at build time |
| `report.py` | prints the markdown tables in RESULTS.md from `results/eval_<tag>.json` |
| `make_gallery.py` | multi-organism GIF (one regen model per target) |
| `logs/run_split.sh`, `logs/split_*.sh`, `logs/run_gallery.sh` | helpers that stop a run at a checkpoint and `--resume` it, to keep each segment under 45 min on the shared GPU |
| `checkpoints/lizard_*_step1000/` | early-training snapshots used for some failure clips |
| `tests/` | pytest smoke tests and the node-based check of the JS port |
| `checkpoints/<run>/model.safetensors` | final weights. `ckpt/` holds the resumable state, but the 325 MB `pool.npy` is kept only for `lizard_regen` and `lizard_persist` (and is git-ignored); the other runs can't be `--resume`d as-is |
| `results/` | eval JSON and mean curves |
| `logs/` | training logs |
