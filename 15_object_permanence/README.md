# 15 · Object Permanence Olympics: Where Did It Go?

Coloured balls roll and bounce in a 64×64 box. Opaque grey walls hide them for several frames.
A learned tracker (per-frame CNN + GRU, MLX) watches the video and must report, **for every ball
and every frame including the hidden ones**, a 2-D Gaussian: a mean position and a full 2×2
covariance. It is compared with a classical baseline (colour-threshold detector + constant-velocity
Kalman filter) that sees the same frames and outputs the same Gaussians.

Results, numbers and discussion are in [RESULTS.md](RESULTS.md).

## Demos (in `media/`)
- `demo_base_*.gif` / `demo_hard_*.gif`: side-by-side replays (GRU left, Kalman+bounce right) on
  held-out long-occlusion test sequences. The dotted ring is the true position of a hidden ball;
  `+` is each tracker's mean; the solid and faint ellipses are its 1σ and 2σ regions.
  `*_random` is sequence #0 (not cherry-picked). `*_gru_better` and `*_kalman_better` are the
  sequences, out of the first 300, where the two trackers' hidden-frame errors differ most in each direction.
- `demo_hard_crossing_swap_abl_ident.gif`: the "good failure". The hard model learned that balls bounce
  off each other, so when identical balls pass *through* each other (a world it never saw) it swaps their identities.
- `demo.html`: self-contained "shell game" replay (no network). Pick a sequence and press Play.
  In shell-game mode the replay pauses while a ball is hidden. Click your guess, then see where it really is and
  where the GRU and the Kalman filter thought it was.
- `curves_base_wide.png`: the control model trained on wider walls, evaluated on test / test_long.
- `curves_hard_ablation.png`: the same curves for the hard model on the factor-ablation worlds.
  `curves_hard_stage1.png` shows the hard model after training stage 1 (10.5k steps), kept for comparison.
- `curves_base.png` / `curves_hard.png`: error, predicted σ, ellipse coverage vs. frames since the
  ball disappeared, plus calibration (reliability) plots.

## How to run
All commands run from this folder with the shared venv (`PY=../.venv/bin/python`).

```bash
./smoke.sh                                    # < 1 min: pytest + tiny train/eval/GIF end to end
../.venv/bin/python -m pytest -q              # unit checks only

# 1. data (fixed seeds; generated on first use, cached in data/*.npz)
../.venv/bin/python sim.py --splits train val test test_long hard_train hard_val hard_test hard_test_long

# 2. tune the Kalman baseline on val (q, r grid; selection = mean NLL, same objective as the net)
OMP_NUM_THREADS=1 ../.venv/bin/python tune_kalman.py --variant base
OMP_NUM_THREADS=1 ../.venv/bin/python tune_kalman.py --variant hard

# 3. train (≈30–40 min each on a shared M5 Max; resumes from <out>/last.* automatically)
../bin/gpu_run ../.venv/bin/python train.py --variant base --steps 12000 --out checkpoints/base
../bin/gpu_run ../.venv/bin/python train.py --variant hard --steps 12000 --out checkpoints/hard

# 4. evaluate on identical rendered test frames -> results/eval_*.json, media/curves_*.png
../.venv/bin/python evaluate.py --variant base --ckpt checkpoints/base/final
../.venv/bin/python evaluate.py --variant hard --ckpt checkpoints/hard/final

# 5. demos
../.venv/bin/python demo.py --variant base --ckpt checkpoints/base/final
../.venv/bin/python demo.py --variant hard --ckpt checkpoints/hard/final
../.venv/bin/python demo.py --variant hard --ckpt checkpoints/hard/final --split abl_ident --crossing_swap --n_pool 100
../.venv/bin/python build_demo_html.py

# 6. extra analyses
../.venv/bin/python evaluate.py --variant hard --ckpt checkpoints/hard/final \
    --splits abl_ident abl_ident_coll abl_ident_turn hard_test --tag hard_ablation   # factor ablation
../.venv/bin/python analyze_shape.py          # is the predicted mass behind the wall?
../.venv/bin/python identity_by_contact.py     # ID swaps on sequences where balls touch vs not (51% vs 5%)
../.venv/bin/python make_tables.py base       # markdown tables used in RESULTS.md
# control: train on walls covering test_long widths
../bin/gpu_run ../.venv/bin/python train.py --variant base --train_split train_wide --steps 12000 --max_minutes 40 --out checkpoints/base_wide
../.venv/bin/python evaluate.py --variant base --ckpt checkpoints/base_wide/final --tag base_wide
```

Training notes: the hard model was trained in two stages (10.5k steps, stopped by the 40-minute guard,
then `train.py --variant hard --steps 20000 --max_minutes 40` resumed from it with a warm restart).
The stage-1 weights are in `checkpoints/hard_stage1/`.
Exact history (from `logs/`): base ran `--steps 20000` for its first 1500 steps, then resumed with
`--steps 12000 --max_minutes 40`; hard stage 1 was `--steps 12000 --max_minutes 40` (stopped at 10,564; stage 2
resumed from the step-10,500 `last` checkpoint). The commands above therefore reproduce the recipe, not the
bit-identical checkpoints.

## File layout
| file | what |
|---|---|
| `sim.py` | ball/occluder simulator, split definitions (seeds), numpy renderer, occlusion-run helper |
| `model.py` | `Tracker`: CoordConv CNN encoder → GRU (forward-only/causal) → per-slot Gaussian head; NLL |
| `train.py` | training loop (GPU renderer twin of `sim.render`, mirror augmentation, checkpoint/resume) |
| `kalman.py` | baseline: colour detector (wall-aware), CV Kalman filter with optional bounce reflection, Hungarian association for identical balls |
| `tune_kalman.py` | grid search of Kalman q/r on val |
| `metrics.py` | error, Gaussian NLL, 1σ/2σ coverage, occlusion-length buckets, reappearance, identity switches |
| `evaluate.py` | runs all trackers on the same frames, writes JSON + curves |
| `analyze_shape.py` | where the predicted probability mass sits relative to the hiding wall |
| `identity_by_contact.py` | identity switches split by ball contact on the ablation worlds (added by the cross-reviewer; reproduces `results/identity_by_contact.json` exactly) |
| `make_tables.py` | prints the RESULTS.md tables from `results/eval_*.json` |
| `demo.py`, `build_demo_html.py` | GIFs and the shell-game page |
| `test_pipeline.py`, `smoke.sh` | tests / end-to-end smoke |
| `data/` | fixed splits (state only; frames are rendered deterministically on demand) |
| `checkpoints/{base,hard}/final.*` | trained weights + optimizer state + config (`last.*` = resumable); `failed_*` = weights of the two documented failed runs (weights only, not loadable with `load_model`; kept as evidence) |
| `results/` | tuning + evaluation JSON |
| `logs/` | training logs |

## Round 5 follow-up: bind an identity, keep several possibilities, spend one look

Results: [RESULTS.md](RESULTS.md) "Round 5" (TL;DR at its top).
Designs: [DESIGN_R5.md](DESIGN_R5.md) (turn 1), [DESIGN_R5_T2.md](DESIGN_R5_T2.md) (turn 2). These are the commands that were actually run
(2026-10-24, AC power), copied from the `gpu_run` headers in `logs/r5/`. Pipe each one to the log named.

```bash
# Checks (NumPy only, ~2 s): 17 tests
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest -q test_r5_numpy.py test_r5_turn2_numpy.py
../bin/gpu_run bash ./smoke_r5.sh --turn2            # tiny end-to-end incl. SIGKILL/resume, ~9 s

# Turn 2 (final) -------------------------------------------------------------------------------
../bin/gpu_run ../.venv/bin/python -u r5_cue_probe.py --checkpoint checkpoints/r5_ac/last.pkl \
  --out results/r5/t2_probe_turn1.json --n 32 --max-minutes 1                     # logs/r5/t2_probe_turn1.log
../bin/gpu_run ../.venv/bin/python -u r5_tune.py --turn2 --out results/r5/t2_tuning_ac.json \
  --screen-n 24 --confirm-n 64 --particles 512 --max-minutes 8                    # logs/r5/t2_tune_ac.log, 4.4 min
# three resumable <=35-min segments (32.6 / 30.0 / 23.7 min); 2nd and 3rd add --resume
../bin/gpu_run ../.venv/bin/python -u r5_turn2_train.py --warm checkpoints/hard/final --out checkpoints/r5_t2_ac \
  --steps 120000 --stop-after 40000 --batch 8 --lr 0.0003 --seed 52024 --save-every 1000 \
  --val-count 128 --max-minutes 35                                                 # logs/r5/t2_train_ac_1.log
#   ... --stop-after 80000 ... --resume   (t2_train_ac_2.log);  --stop-after 120000 ... --resume  (t2_train_ac_3.log)
../bin/gpu_run ../.venv/bin/python -u r5_cue_probe.py --checkpoint checkpoints/r5_t2_ac/best.pkl \
  --out results/r5/t2_probe_selected.json --n 32 --max-minutes 2 --require-equivariant
../bin/gpu_run ../.venv/bin/python -u r5_evaluate.py --checkpoint checkpoints/r5_t2_ac/best.pkl \
  --tuning results/r5/t2_tuning_ac.json --thresholds results/r5/t2_thresholds_ac.json --calibrate --n 32 --max-minutes 2
../bin/gpu_run ../.venv/bin/python -u r5_evaluate.py --checkpoint checkpoints/r5_t2_ac/best.pkl \
  --tuning results/r5/t2_tuning_ac.json --thresholds results/r5/t2_thresholds_ac.json \
  --split turn2_fresh --n 512 --out results/r5/t2_fresh_ac.json --max-minutes 6   # 127 s
../bin/gpu_run ../.venv/bin/python -u r5_evaluate.py --checkpoint checkpoints/r5_t2_ac/best.pkl \
  --tuning results/r5/t2_tuning_ac.json --thresholds results/r5/t2_thresholds_ac.json \
  --split hard_test --n 64 --out results/r5/t2_legacy_ac.json --max-minutes 2
../.venv/bin/python r5_demo.py --evaluation results/r5/t2_fresh_ac.json --out media/r5_t2_ac --max-minutes .5

# Reviewer: strengthened particle filter (4096 particles, val-only retune) + diagnostics -----------
../bin/gpu_run ../.venv/bin/python review_r5.py --pf-scaling --pf-sweep    # ~2.5 min CPU: review_{diagnostics,pf_scaling,pf4096_sweep}.json + strips
#   (results/r5/review_pf4096_tuning.json = the sweep's selection: q=.3 r=1 miss=.35 P=4096 bw=.5; Kalman unchanged)
../bin/gpu_run ../.venv/bin/python -u r5_evaluate.py --tuning results/r5/review_pf4096_tuning.json \
  --thresholds results/r5/review_pf4096_thresholds.json --calibrate --n 32 --max-minutes 2
../bin/gpu_run ../.venv/bin/python -u r5_evaluate.py --tuning results/r5/review_pf4096_tuning.json \
  --thresholds results/r5/review_pf4096_thresholds.json --split turn2_fresh --n 512 \
  --out results/r5/review_pf4096_fresh.json --max-minutes 12                       # 7.8 min, classical only
../bin/gpu_run ../.venv/bin/python -u r5_evaluate.py --tuning results/r5/review_pf4096_tuning.json \
  --thresholds results/r5/review_pf4096_thresholds.json --split hard_test --n 64 \
  --out results/r5/review_pf4096_legacy.json --max-minutes 3

# Turn 1 (preliminary) -------------------------------------------------------------------------
../bin/gpu_run bash ./smoke_r5.sh
../bin/gpu_run ../.venv/bin/python -u r5_tune.py --out results/r5/tuning_ac.json --screen-n 16 --confirm-n 48 --particles 512 --max-minutes 8
../bin/gpu_run ../.venv/bin/python -u r5_train.py --out checkpoints/r5_ac --steps 6000 --batch 8 --save-every 100 --val-count 16 --max-minutes 35
../bin/gpu_run ../.venv/bin/python -u r5_evaluate.py --checkpoint checkpoints/r5_ac/last.pkl --tuning results/r5/tuning_ac.json \
  --thresholds results/r5/thresholds_ac.json --calibrate --n 16 --max-minutes 3
../bin/gpu_run ../.venv/bin/python -u r5_evaluate.py --checkpoint checkpoints/r5_ac/last.pkl --tuning results/r5/tuning_ac.json \
  --thresholds results/r5/thresholds_ac.json --split fresh --n 128 --out results/r5/fresh_ac.json --max-minutes 8
../bin/gpu_run ../.venv/bin/python -u r5_evaluate.py --checkpoint checkpoints/r5_ac/last.pkl --tuning results/r5/tuning_ac.json \
  --thresholds results/r5/thresholds_ac.json --split hard_test --n 64 --out results/r5/legacy_ac.json --max-minutes 5
```

Code: turn 1 `r5_{data,metrics,classical,model,train,tune,evaluate,demo,load}.py`; turn 2 `r5_turn2_{model,train,utils}.py`
and `r5_cue_probe.py`. `r5_mlx_check.py` and `r5_resume_check.py` are runner-only checks, and `review_r5.py` holds the
reviewer diagnostics. Artifacts live in `results/r5/`, `logs/r5/`, `checkpoints/r5_*` and `media/r5_*`. The selected
turn-2 checkpoint is `checkpoints/r5_t2_ac/best.pkl` (paired step 100000; reload it with `r5_load.load_pair`). The
round-5 demos are `media/r5_t2_ac_*.png` and `media/r5_t2_review_*.png`. Original commands, checkpoints, results and
demos are intact.
