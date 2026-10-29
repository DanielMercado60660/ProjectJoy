# 13 · Daydream Engine: a neural network that imagines physics

**Round 5 (2026-10-24): a world model that knows when to distrust itself.** Known wall physics
plus a learned pairwise ball-ball correction, as a 5-member ensemble. Planning that penalises
ensemble disagreement raises real success from 43.5% to 54.5% (256 shots) and from 57% to 75%
(4,096 shots), and cuts the fooled rate 2–3×. The learned mean model is no better than a fitted
soft-contact physics model, so the gain comes from knowing when to distrust the model. Numbers,
caveats and sources: [RESULTS.md — Round 5](RESULTS.md#round-5-a-world-model-that-knows-when-to-distrust-itself);
design: [DESIGN_R5.md](DESIGN_R5.md). Demo: `media/r5_t2_20261024T161011Z_demo.gif` (a fooled
mean-model plan), plots `media/r5_t2_20261024T161011Z_{pressure,forecast,cost}.png`.

Runner fix: `r5_core.pair_features` applies `stop_gradient` to the pair-gate inputs, because the
`sqrt` of the zero self-distance gave a NaN VJP (`0*inf`) at training step 2.

Round-5 commands as run for turn 2 (from this folder; `RUN=r5_t2_20261024T161011Z`, turn-1 data in
`results/r5/r5_20261024T152040Z/data`; each GPU/heavy step goes through `../bin/gpu_run`):
```bash
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLBACKEND=Agg
../.venv/bin/python -m pytest -q test_r5_numpy.py test_r5_turn2_numpy.py      # NumPy tests (<1 s)
../bin/gpu_run bash smoke_r5.sh                                               # MLX/VJP/resume smoke, ~17 s
D=results/r5/r5_20261024T152040Z/data; RUN=r5_t2_20261024T161011Z
../bin/gpu_run ../.venv/bin/python r5_soft_contact.py --data $D --out results/r5/$RUN/soft_fit.json \
    --proposals 192 --refine 64 --shortlist 32 --samples 2048 --fit-horizon 5 --val-scenes 128 --horizon 40 --max-minutes 5
for F in raw geometry; do                                                     # 14 min each
  ../bin/gpu_run ../.venv/bin/python train_r5.py --data $D --out checkpoints/${RUN}_$F --features $F \
    --members 5 --hidden 64 --depth 3 --batch 256 --unroll 5 --seed 53010 --collision-weight 3 \
    --contact-fraction 0.75 --lr 0.0008 --decay-steps 100000 --min-lr-fraction 0.1 --steps 2000000 \
    --checkpoint-every 5000 --history-every 100 --max-minutes 14
done
../bin/gpu_run ../.venv/bin/python experiment_r5.py tune --data $D --fit results/r5/r5_20261024T152040Z/fit.json \
    --soft-fit results/r5/$RUN/soft_fit.json --reference-selection results/r5/r5_20261024T152040Z/selection.json \
    --checkpoints checkpoints/r5_20261024T152040Z_h32/step_00100000_86f61496 checkpoints/${RUN}_raw checkpoints/${RUN}_geometry \
    --out results/r5/$RUN/selection.json --forecast-val-scenes 128 --tasks 64 --tune-seed 53003 --shots 256 \
    --horizon 40 --betas 0 0.25 0.5 1 2 --chunk 512 --max-minutes 3
../bin/gpu_run ../.venv/bin/python experiment_r5.py eval --selection results/r5/$RUN/selection.json \
    --out results/r5/$RUN/headline --turn2 --tasks 200 --forecast-scenes 256 --horizon 40 --shots 64 256 1024 4096 \
    --cem-iters 1 2 4 --population 64 --matched-cost --chunk 512 --seed 62001 --action-seed 62002 \
    --label turn2_headline --max-minutes 35                                    # ~22 min
../bin/gpu_run ../.venv/bin/python render_r5.py --eval results/r5/$RUN/headline \
    --selection results/r5/$RUN/selection.json --prefix media/$RUN --max-minutes 0.8
```
Round-5 commands as run for turn 1 (`RUN=r5_20261024T152040Z`; copied from the `gpu_run` headers in
`logs/r5/r5_20261024T152040Z/`):
```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLBACKEND=Agg
RUN=r5_20261024T152040Z
../.venv/bin/python -m pytest -q test_r5_numpy.py                             # 27 passed in 0.32 s
../bin/gpu_run bash smoke_r5.sh                                               # run twice: first failed, second SMOKE_R5_OK
../bin/gpu_run ../.venv/bin/python experiment_r5.py prepare --out results/r5/$RUN/data \
    --scenes 2000 --val-scenes 256 --frames 60 --max-minutes 0.7
../bin/gpu_run ../.venv/bin/python experiment_r5.py fit --data results/r5/$RUN/data --out results/r5/$RUN/fit.json \
    --candidates 80 --refine 16 --samples 2048 --horizon 3 --max-minutes 1.5
for H in 32 64; do                                                            # 5 min each
  ../bin/gpu_run ../.venv/bin/python train_r5.py --data results/r5/$RUN/data --out checkpoints/${RUN}_h$H \
    --members 5 --hidden $H --batch 128 --unroll 3 --seed 51010 --lr 0.001 --steps 100000 \
    --checkpoint-every 250 --max-minutes 5
done
../bin/gpu_run ../.venv/bin/python experiment_r5.py tune --data results/r5/$RUN/data --fit results/r5/$RUN/fit.json \
    --checkpoints checkpoints/${RUN}_h32 checkpoints/${RUN}_h64 --out results/r5/$RUN/selection.json \
    --tasks 24 --shots 256 --horizon 40 --betas 0 0.25 0.5 1 2 --chunk 512 --max-minutes 2.5
../bin/gpu_run ../.venv/bin/python experiment_r5.py eval --selection results/r5/$RUN/selection.json \
    --out results/r5/$RUN/headline --tasks 200 --forecast-scenes 256 --horizon 40 --shots 64 256 1024 4096 \
    --cem-iters 1 2 4 --population 64 --matched-cost --chunk 512 --seed 51901 --action-seed 51902 \
    --label fresh_headline --max-minutes 14
../bin/gpu_run ../.venv/bin/python experiment_r5.py eval --selection results/r5/$RUN/selection.json \
    --out results/r5/$RUN/old_secondary --tasks 32 --forecast-scenes 128 --horizon 40 --shots 64 256 1024 4096 \
    --cem-iters 1 2 4 --population 64 --matched-cost --chunk 512 --legacy-data data/test.npz \
    --seed 1 --action-seed 2 --label old_scenes_secondary --max-minutes 5
../bin/gpu_run ../.venv/bin/python render_r5.py --eval results/r5/$RUN/headline \
    --selection results/r5/$RUN/selection.json --prefix media/$RUN --max-minutes 0.8
```
Round-5 files:
`r5_core.py` (simulator glue, features, seeds), `r5_model.py`/`train_r5.py` (MLX ensemble),
`r5_soft_contact.py` (classical fit), `r5_planning.py` (random/CEM, risk score), `experiment_r5.py`
(prepare/fit/tune/eval), `render_r5.py`, `smoke_r5.sh`, `test_r5_*numpy.py`. Artifacts live in
`results/r5/`, `checkpoints/r5_*`, `logs/r5/` and `media/r5_*`.

A tiny 2D world of three balls bouncing in a box under gravity, and an MLX MLP that learns to
predict it. Given only the first frame of a held-out scene, the model "daydreams" the next
second open loop (feeding its own predictions back in). We measure how its error grows with
horizon and around bounces, compare it against three cheap physics baselines, and use it as the
only world model for a planner that picks a launch velocity. Then we check the plan in the real
simulator to measure the gap between imagined and realized outcomes.

Results and numbers: [RESULTS.md](RESULTS.md). Demo: `media/real_vs_imagined.gif`.

## The world (`sim.py`)
- 3 balls, radius 0.06, in a unit box, gravity 3.0 (box units/s²), downward.
- Walls are inelastic (restitution 0.85). Ball-ball collisions are elastic, equal masses.
- Frames are 20 ms apart, with 10 semi-implicit Euler substeps per frame. The state is
  `(x, y, vx, vy)` per ball, 12 numbers in all.
- Initial conditions: uniform non-overlapping positions, velocities uniform in [-1.5, 1.5]².
  Train: 20,000 scenes × 100 frames (seed 0). Test: 2,000 held-out scenes (seed 1).
  About 13% of frames contain a wall bounce and 3.5% a ball-ball collision.
- The simulator logs a per-frame collision flag (wall or ball-ball) for the bounce analysis.

## The model (`model.py`, `train.py`)
- MLP 12 → 4×512 (SiLU) → 12, 0.8 M parameters. It predicts a normalized state delta
  (residual).
- Multi-step loss: each batch has 1,024 random windows. The model rolls out K steps open loop
  from each window's first frame, and the loss is the mean squared error over all K predicted
  frames, in units of state std. K ramps from 1 to 25 over the first 30% of training. The
  learning rate follows a cosine schedule driven by whichever budget (steps or wall clock) is
  further along.
- One main run, capped at 16 minutes through `../bin/gpu_run`.

## Baselines (cheap, no learning)
- **Constant velocity:** straight lines, with velocities frozen.
- **Ballistic, no collisions:** exact free fall under gravity. It ignores walls and other balls.
  It is exact until the first bounce.
- **Walls-only physics** (added in review, `sim.walls_only`): the true integrator with gravity
  and wall bounces at the true restitution, but no ball-ball collisions. It is exact until two
  balls touch, and it is the strongest baseline. The GIF's baseline panel is still ballistic.

## Commands (run from this folder)
```bash
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLBACKEND=Agg
../.venv/bin/python sim.py                       # data/train.npz, data/test.npz (~7 s)
../bin/gpu_run ../.venv/bin/python train.py --steps 60000 --batch 1024 --unroll 25 \
    --max-minutes 16 --log-every 1000 --out checkpoints/dynamics > logs/train.log
../.venv/bin/python eval.py                      # results/eval.json, media/horizon_error.png
../.venv/bin/python plan.py                      # results/plan.json, media/plan_examples.png
../.venv/bin/python make_gif.py                  # media/real_vs_imagined.gif (+ _strip.png)
./smoke.sh                                       # end-to-end at tiny size, ~10 s, prints SMOKE OK
# ablation (one-step loss, same 16-min budget) and its robust comparison:
../bin/gpu_run ../.venv/bin/python train.py --steps 1000000 --batch 1024 --unroll 1 \
    --max-minutes 16 --log-every 5000 --out checkpoints/ablation_unroll1 > logs/ablation_unroll1.log
../.venv/bin/python eval.py --ckpt checkpoints/ablation_unroll1 --out results/eval_ablation_unroll1.json --plot ""
../.venv/bin/python plan.py --ckpt checkpoints/ablation_unroll1 --out results/plan_ablation_unroll1.json --plot ""
../.venv/bin/python ablation_summary.py          # results/ablation_summary.json (median error, divergence)
```
`eval.py`, `plan.py` and `make_gif.py` all reload `checkpoints/dynamics.{safetensors,json}`.

## Layout
```
sim.py          simulator, collision flags, the three baselines, dataset generation
model.py        Dynamics MLP, normalization, open-loop rollout, save/load
train.py        multi-step-loss training with a wall-clock budget
eval.py         forecast error by horizon, split by bounces, one-step error by event type
plan.py         closed-loop launch-to-target task: imagined vs realized
make_gif.py     real | imagined | baseline side-by-side GIF
ablation_summary.py  median error / divergence: multi-step vs one-step checkpoints
smoke.sh        tiny end-to-end check (writes to smoke_out/, deleted on success)
data/           generated datasets (regenerable with sim.py)
checkpoints/    dynamics.{safetensors,json,opt.safetensors}: main model (weights, config/step/log, optimizer)
                ablation_unroll1.*: one-step-loss ablation
results/        eval.json, plan.json (+ ablation results)
media/          GIF, strip PNG, horizon plot, planning examples
logs/           training log
```

## Round 5, turn 3: classical risk control

The new control ranks five existing validated spring fits, tunes their disagreement penalty
on the same validation tasks as the frozen hybrid, then compares six planners on 200 fresh
tasks at 256/4096 shots. Results are in RESULTS.md ("Turn 3: classical risk-aware control"). The exact setup, smoke, tuning and
evaluation commands are below; heavy steps use `gpu_run`.
`r5_classical_risk.py tune` writes the classical selection; its `eval` command writes resumable
paired receipts and Holm-adjusted comparisons. `test_r5_turn3_numpy.py` checks this without MLX.
Only the retained final turn-2 checkpoint is required; the reviewed outputs remain frozen.

Turn-3 commands as run (from this folder; copied from the `gpu_run` headers in
`logs/r5/r5_t3_20261024T172531Z/`):
```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLBACKEND=Agg
RUN=r5_t3_20261024T172531Z; FROZEN=results/r5/r5_t2_20261024T161011Z/selection.json
../.venv/bin/python -m pytest -q                                              # 52 passed
../bin/gpu_run bash smoke_r5.sh                                               # SMOKE_R5_OK
../bin/gpu_run ../.venv/bin/python r5_classical_risk.py tune --selection $FROZEN \
    --out results/r5/$RUN/classical_selection.json --shots 256 --max-minutes 2
../bin/gpu_run ../.venv/bin/python r5_classical_risk.py eval --selection $FROZEN \
    --classical-selection results/r5/$RUN/classical_selection.json --out results/r5/$RUN/headline \
    --tasks 200 --shots 256 4096 --chunk 512 --seed 73001 --action-seed 73002 --target-seed 73101 \
    --max-minutes 25
```
