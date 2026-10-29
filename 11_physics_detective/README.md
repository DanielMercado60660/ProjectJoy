# 11 · The Physics Detective

A hidden world (a block on a table with secret Coulomb friction), a robot finger that can push it
with a chosen force and duration, and a noisy camera that only sees the first metre of the table.
The detective's job is to learn the hidden law with as few experiments as possible and to say
honestly how sure it is.

What's here:

* **Simulator** (`physics.py`): analytic numpy Coulomb-friction block (static break-free threshold,
  kinetic sliding, stopping), camera with ±8 cm noise at 10 Hz for 2 s, and a 1 m field of view.
  Readings past the edge come back as "out of view" (censored). Too soft a push and nothing moves;
  too hard and the block leaves the picture before you see it stop.
* **Exact Bayesian inference** (`inference.py`): grid posterior over μ_k (1500 points) with the
  censored (Tobit) likelihood.
* **Baselines**: a conventional least-squares fitter with Gauss–Newton standard errors
  (`inference.lsq_fit`), random probes, and a fixed "textbook" protocol (8 N for 0.3 s, repeated).
* **Active experiment selection**: choose the next push (F, T) from a 150-push menu by maximising
  expected information gain (nested Monte Carlo over 64 posterior quantile particles).
* **ML element** (`sbi.py`): an MLX amortized posterior network (Deep-Sets encoder over the set of
  experiments, then an 8-component Gaussian mixture over logit μ_k) trained only on simulations.
  It's compared with the exact posterior for accuracy, calibration and speed (`eval_sbi.py`).
* **Go deeper** (`deeper.py`):
  * *World 2, two frictions*: μ_k and μ_s are both hidden. Do active probes separate them?
  * *World 3, uncalibrated pusher*: mass m and actuator gain k are hidden. Pushes only identify k/m,
    so the posterior is a **ridge**. We report k/m and the ridge rather than a fake point estimate,
    show least squares landing at arbitrary ridge points, and show that one spring-scale weighing
    breaks the tie (an EIG detective chooses to weigh on its own).
* **Demos** (`media/`): `detective.gif` / `.mp4` (active vs random detective on the same world,
  with posterior and EIG map), `calibration.png`, `active_vs_random.png`, `active_choices.png`,
  `world2_two_frictions.png`, `world3_ridge.png`, and **`demo.html`**, a self-contained browser
  game. You pick pushes, the exact posterior and the EIG map update live in JS, then you make your
  accusation.

## How to run

All commands run from this folder and use the shared venv. Long runs go through the GPU/CPU slot
lock.

```bash
# smoke check: unit tests plus a tiny end-to-end pipeline (< 2 min)
./smoke.sh                      # or: ../.venv/bin/python -m pytest -q tests

# 1. active vs random vs fixed over 500 hidden worlds x 15 pushes (~25 min under load, 4 procs)
../bin/gpu_run ../.venv/bin/python run_active.py --worlds 500 --steps 15 --procs 4 --out results/active.npz

# 2. train the amortized MLX posterior network (60k steps, about 15-25 min; resumable from checkpoints/sbi)
../bin/gpu_run ../.venv/bin/python sbi.py --steps 60000 --ckpt checkpoints/sbi

# 3. evaluate the network against the exact posterior (fresh data + the detectives' sequences)
../bin/gpu_run ../.venv/bin/python eval_sbi.py --ckpt checkpoints/sbi   # ~2 min

# 4. go deeper: two frictions (200 worlds x 12 pushes) and the non-identifiable ridge
../bin/gpu_run ../.venv/bin/python deeper.py world2 --worlds 200 --steps 12
../.venv/bin/python deeper.py world3

# 5. figures + summary numbers (results/summary.json), ridge figure, animation (reloads checkpoints/sbi)
../.venv/bin/python analyze.py
../.venv/bin/python plot_world3.py
../.venv/bin/python make_demo.py --mu 0.23 --pushes 8 --ckpt checkpoints/sbi

# 6. play: open media/demo.html in any browser (no network needed)
```

## File layout

```
physics.py        hidden world: analytic block/friction sim, camera noise + field of view, likelihood
inference.py      grid posterior, EIG experiment selection, least-squares baseline, strategies
run_active.py     main experiment: active / random / fixed over many hidden worlds -> results/active.npz
sbi.py            MLX amortized posterior network: data generator, model, resumable training
eval_sbi.py       network vs exact posterior: accuracy, calibration, speed -> results/sbi_eval.json
deeper.py         world 2 (mu_k + mu_s) and world 3 (k/m ridge, spring-scale tie-breaker)
analyze.py        summary.json + figures from results/
plot_world3.py    ridge figure for world 3
check_finegrid.py coverage re-check of the active detective on a 30k-point grid
make_demo.py      animation media/detective.{gif,mp4} (loads checkpoints/sbi)
media/demo.html   self-contained detective game (pure JS, same physics and exact posterior)
tests/            pytest unit and pipeline tests
smoke.sh          < 2 min end-to-end check
checkpoints/sbi/  model.safetensors, opt.safetensors, config.json, state.json (step, RNG, loss log)
results/          raw run outputs (npz/json), summary.json; world2_coarsegrid.npz = the preserved coarse-grid run
logs/             run logs
```


## Round 5: probe to place

The follow-up asks whether EIG or task-directed probes help a block stop inside a
5 cm target zone, including worlds with an unmodelled actuator gain error. It
compares both with tuned least-squares and censoring-aware Tobit controllers,
prior-only action, and two explicitly labelled oracles. Turn 2 adds the unknown
gain to the model. This round is CPU-only NumPy/SciPy; it does not use the
earlier SBI network. Results are in
[RESULTS.md](RESULTS.md#round-5-probe-to-accomplish-something).

Below are the commands actually run (2026-10-24, AC power), copied from the
first line of each `logs/r5/*.log`. Prefix CPU-heavy runs with
`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMBA_NUM_THREADS=4`.

```bash
# Smoke: CPU tests (30 pass, 1 MLX deselected), tiny tune/eval, kill/resume, offline demos (<2 min)
./smoke_r5.sh

# Turn 1 — validation tuning of LSQ/Tobit (matched worlds only, 23 s)
../bin/gpu_run ../.venv/bin/python run_r5.py tune --worlds 128 --seed 6202611 --grid-size 6001 \
  --checkpoint checkpoints/r5_tune_main.json --out results/r5/tuning_main.json --max-minutes 3

# Turn 1 — headline: 512 fresh worlds x 2 families (481 s)
../bin/gpu_run ../.venv/bin/python run_r5.py evaluate --worlds 512 --seed 6202612 --grid-size 6001 \
  --particles 64 --fantasies 16 --design-stride 1 --tuning results/r5/tuning_main.json \
  --checkpoint checkpoints/r5_main.json --out results/r5/main_summary.json --max-minutes 20

# Turn 1 — secondary split with the original seed-2026 friction values
../bin/gpu_run ../.venv/bin/python run_r5.py evaluate --worlds 128 --seed 6202613 --legacy-mu \
  --grid-size 6001 --particles 64 --fantasies 16 --design-stride 1 --tuning results/r5/tuning_main.json \
  --checkpoint checkpoints/r5_legacy.json --out results/r5/legacy_summary.json --max-minutes 4

# Turn 1 — AC numerical check: 2x grid, particles and fantasies on the first 128 headline worlds
../bin/gpu_run ../.venv/bin/python run_r5.py evaluate --worlds 128 --seed 6202612 --grid-size 12001 \
  --particles 128 --fantasies 32 --design-stride 1 --tuning results/r5/tuning_main.json \
  --checkpoint checkpoints/r5_ac_resolution.json --out results/r5/ac_resolution_summary.json --max-minutes 20

# Turn 1 — offline demo from the checkpoint
../.venv/bin/python r5_report.py --checkpoint checkpoints/r5_main.json \
  --summary results/r5/main_report_summary.json --out media/r5_main_demo.html --max-minutes 1

# Turn 2 — validation selection (new seed). This hit its 10-min guard at 22/32,
# so the same command was resumed with --resume --max-minutes 8 (910 s total)
../bin/gpu_run ../.venv/bin/python run_r5_turn2.py tune --worlds 32 --seed 8202611 \
  --baseline-tuning results/r5/tuning_main.json --mu-size 2401 --gain-size 641 \
  --checkpoint checkpoints/r5_turn2_tune.json --out results/r5/turn2_tuning.json --max-minutes 10

# Turn 2 — fresh test: 128 worlds x 2 families (1104 s)
../bin/gpu_run ../.venv/bin/python run_r5_turn2.py evaluate --worlds 128 --seed 8202612 \
  --baseline-tuning results/r5/tuning_main.json --mu-size 2401 --gain-size 641 \
  --tuning results/r5/turn2_tuning.json --checkpoint checkpoints/r5_turn2_main.json \
  --out results/r5/turn2_main_summary.json --max-minutes 40

# Turn 2 — AC check: same first 32 test worlds, joint grid doubled (850 s)
../bin/gpu_run ../.venv/bin/python run_r5_turn2.py evaluate --worlds 32 --seed 8202612 \
  --baseline-tuning results/r5/tuning_main.json --tuning results/r5/turn2_tuning.json \
  --mu-size 4801 --gain-size 1281 --resolution-check --checkpoint checkpoints/r5_turn2_resolution.json \
  --out results/r5/turn2_resolution_summary.json --max-minutes 40

# Turn 2 — offline demo (recovered and remaining-failure worlds)
../.venv/bin/python r5_report_turn2.py --checkpoint checkpoints/r5_turn2_main.json \
  --summary results/r5/turn2_main_report_summary.json --out media/r5_turn2_main_demo.html --max-minutes 1
```

Round-5 files: `r5_core.py` (task, policies, fitters), `run_r5.py` (tuning,
resumable evaluation, metrics), `r5_report.py` (offline demo), `r5_gain.py`
(joint friction/gain posterior, gain-MLE, same-history control),
`run_r5_turn2.py`, `r5_report_turn2.py`, and `diagnose_r5_turn2.py`
(exploratory same-history controller swaps on reused turn-1 test worlds, written
to `results/r5/turn2_diagnostics.json`). Tests are `tests/test_r5.py` and
`tests/test_r5_turn2.py`, and the smoke script is `smoke_r5.sh`. Artifacts live
under `results/r5/`, `logs/r5/`, `checkpoints/r5_*` and `media/r5_*`. Earlier
commands and exhibit artifacts are unchanged.
