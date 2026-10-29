# 08 · Tiny Grand Prix — teach a neural network to race

A top-down 2D racing toy. A tiny car (kinematic bicycle model with a grip limit) drives
procedurally generated closed tracks using only **10 wall-distance rays, its speed and its
previous action**. A 13→32→32→2 MLP (~1.6k weights) learns to steer, throttle and brake with
two methods, **evolution strategies (ES)** and **PPO in MLX**, and is compared with a tuned
hand-written ray-following controller (same inputs) and a **privileged** pure-pursuit
controller (sees the true centerline) on **held-out tracks it never trained on**, at several
grip levels.

Results, numbers and discussion: [RESULTS.md](RESULTS.md). Live demo: open
`media/demo.html` in any browser (self-contained, no network): pick a held-out track, change
grip, race the policies yourself with the arrow keys.

## How to run (from this folder)

```bash
PY=../.venv/bin/python
# tests / smoke check (< 1 min): sim physics, hack guards, tiny ES+PPO runs with resume, JS parity
$PY -m pytest -q

# 1. tune the matched-information baseline on training tracks (~8 min on a busy machine)
../bin/gpu_run $PY tune_baseline.py
../bin/gpu_run $PY tune_baseline.py --grip_lo 0.5 --grip_hi 1.1 --out checkpoints/ray_follower_dr.json
# 2. train everything one job at a time (ES 30 min, PPO 25 min, then grip-randomized variants)
./train_all.sh            # or individually:
../bin/gpu_run $PY train_es.py  --gens 300 --max_minutes 30 --tag es
../bin/gpu_run $PY train_ppo.py --minutes 25 --tag ppo
#    resume either one:  --resume checkpoints/<tag>_state.npz
# reward-hacking exhibit: ES with soft walls (walls push back instead of ending the episode)
../bin/gpu_run $PY train_es.py --gens 150 --max_minutes 15 --soft_walls --tag es_soft
# extra seeds (seed-variance check): ppo_s1/2, es_s1/2
./train_seeds.sh
# grip-randomized follow-up: 3 seeds (r0 = bit-identical rerun of seed 0, so it can be re-selected),
# same step budget as seed 0, a policy snapshot at every validation step (--snapshots)
for s in 1 2; do ../bin/gpu_run $PY train_ppo.py --minutes 40 --max_updates 2386 --grip_lo 0.5 --grip_hi 1.1 --seed $s --tag ppo_dr_s$s --snapshots; done
../bin/gpu_run $PY train_ppo.py --minutes 40 --max_updates 2386 --grip_lo 0.5 --grip_hi 1.1 --seed 0 --tag ppo_dr_r0 --snapshots
for s in 1 2; do ../bin/gpu_run $PY train_es.py --gens 276 --max_minutes 40 --grip_lo 0.5 --grip_hi 1.1 --seed $s --tag es_dr_s$s --snapshots; done
../bin/gpu_run $PY train_es.py --gens 276 --max_minutes 40 --grip_lo 0.5 --grip_hi 1.1 --seed 0 --tag es_dr_r0 --snapshots
#   re-select every snapshot on randomized-grip validation -> checkpoints/<tag>_rg_best.npz, results/dr_seeds/selection.json
../bin/gpu_run $PY select_dr.py ppo_dr_s1 ppo_dr_s2 ppo_dr_r0 es_dr_s1 es_dr_s2 es_dr_r0 --extra ppo_dr es_dr
../bin/gpu_run $PY eval.py --controllers ppo_dr ppo_dr_s1 ppo_dr_s2 ppo_dr_r0_rg ppo_dr_s1_rg ppo_dr_s2_rg \
    es_dr es_dr_s1 es_dr_s2 es_dr_r0_rg es_dr_s1_rg es_dr_s2_rg ray_dr \
    --splits heldout hard --grips 1.0 0.8 0.7 0.6 0.5 --modes strict --out results/dr_seeds
$PY dr_seeds_table.py                # -> results/dr_seeds/seeds_table.md
# 3. evaluate on held-out / hard / train tracks at grip 1.0 .. 0.5 -> results/
../bin/gpu_run $PY eval.py          # all controllers, splits, grips 1.0..0.5, strict + respawn
$PY hack_check.py                   # soft-wall hack vs real rules        -> results/hack.json
$PY diagnose_grip.py                # why less grip helps on hard tracks   -> results/diagnose_grip.json
# 4. demos
$PY plot_curves.py                                   # media/learning_curves.png
$PY plot_lines.py --controllers es ppo ray pursuit --out media/racing_lines.png
$PY render.py --split heldout --tracks 0 2 5 7 --controllers es ppo ray --out media/heldout_race.mp4
$PY render.py --split heldout --tracks 0 2 5 7 --controllers es ppo ray --out media/heldout_race.gif --cell 300 --every 2 --fps 15
$PY render.py --split hard --tracks 0 1 2 3 --controllers es ppo ray --out media/hard_tracks.mp4
$PY render.py --split heldout --tracks 2 7 --controllers es ppo ray pursuit --grip 0.7 --out media/low_grip_0.7.mp4
$PY render.py --mode soft --controllers es_soft ray --tracks 2 7 --out media/hack_wall_riding.mp4
$PY make_demo_html.py                                # media/demo.html
```

Set `NUMBA_NUM_THREADS=4` on this shared machine (`train_all.sh` does).

## File layout

| path | what |
|---|---|
| `racing/track.py` | procedural tracks (periodic spline through perturbed points, rejection-sampled), splits by seed: `train` 0.., `heldout` 1,000,000+, `hard` 2,000,000+ (narrower, twistier) |
| `racing/sim.py` | numba vectorized car sim: physics, signed progress along centerline, crash detection, ray casting |
| `racing/policy.py` | the shared MLP format (flat weight vector) + numpy forward |
| `racing/baselines.py` | `RayFollower` (matched info) and `PurePursuit` (privileged) |
| `racing/rollout.py` | the 60 s training episode + fitness used by ES, baseline tuning and validation |
| `racing/evaluate.py` | one-lap time trials (strict and respawn rules) + wall-hugging diagnostics |
| `racing/controllers.py` | load a controller by name |
| `train_es.py`, `train_ppo.py`, `tune_baseline.py`, `train_all.sh` | training |
| `eval.py` | full evaluation → `results/eval.json`, `results/eval_table.md`, `results/eval_per_track.npz` |
| `render.py` | MP4/GIF race videos with rays |
| `web/sim.js`, `web/demo_template.html`, `make_demo_html.py` | JS port of the sim + policies → `media/demo.html` |
| `plot_curves.py`, `plot_lines.py` | learning curves; speed-colored racing lines |
| `hack_check.py`, `diagnose_grip.py` | reward-hacking exhibit; hard-track/grip diagnostic |
| `train_seeds.sh` | extra seeds for the variance check |
| `select_dr.py`, `dr_seeds_table.py` | grip-randomized follow-up: re-select checkpoints on randomized-grip validation; seed × grip tables |
| `checkpoints/` | `es_best.npz`, `ppo_best.npz` (+ `_dr` grip-randomized, `es_soft` hack demo, `_s1/_s2` seeds): policies selected on validation tracks (what eval/demos use); `*_final.npz`: last weights; `*_state.npz`: full resume state (weights, optimizer, step, RNG, config); `ray_follower*.json`: tuned baselines; `{ppo,es}_dr_{s1,s2,r0}_*`: grip-randomized seeds (`r0` = bit-identical rerun of seed 0), `*_snapshots.npz` = weights at every validation step, `*_rg_best.npz` = chosen on randomized-grip validation |
| `logs/` | training logs and per-generation/update histories |
| `media/` | `demo.html` (live, interactive), `heldout_race.mp4/.gif`, `hard_tracks.mp4`, `low_grip_0.7.mp4`, `hack_wall_riding.mp4` + `_lines.png`, `racing_lines.png`, `learning_curves.png` |
| `results/` | `eval.json`, `eval_table.md`, `eval_per_track.npz`, `hack.json`, `diagnose_grip.json`; `seeds/` (ES/PPO seed 2); `dr_seeds/` (grip-randomized seeds × selection rule: `seeds_table.md`, `selection.json`, `eval*.{json,md,npz}`) |
| `tests/test_smoke.py` | pytest smoke suite |

## Round 5: discover grip while driving

The new experiment adds invisible wet patches and mid-lap changes, paired GRU /
memoryless PPO, an extensively tuned online grip estimator, full hard-track
validation, fresh held-out seeds, and recovery-time scoring with failure coverage.
The original exhibit above remains unchanged.

[DESIGN_R5.md](DESIGN_R5.md) is the experiment specification; the exact commands
that were run are listed below. Seed 0 is complete: FF finishes **383/384**, RNN **370/384**, and
adaptive classical **382/384** on the fresh test. Recurrence did not help here;
the current sensor packet already exposes slip. Classical has the best capped
recovery (0.670 s, 98.18% event coverage). See the full accounting and limitations
in [Round-5 results](RESULTS.md#round-5-discover-the-roads-grip-while-driving).

Open [the offline race demo](media/r5_seed0_demo.html), which retains RNN failures,
and inspect [recovery curves](media/r5_turn2_v2_recovery.png) and
[validation curves](media/r5_turn2_v2_validation.png). The
[NumPy audit](results/r5/r5_turn2_analysis_v2.json) verifies receipt aggregates,
checkpoint selection, and exact frozen-policy replays; no test-based reselection
was made. No new training is queued for Turn 2.

```bash
# NumPy-only checks (no MLX imports)
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  ../.venv/bin/python -m pytest -q tests/test_r5_numpy.py tests/test_r5_analysis.py --basetemp=results/r5/pytest_tmp/run
# Claude's complete tiny MLX/NumPy pipeline, including hard-kill resume (<2 minutes)
../bin/gpu_run bash ./smoke_r5.sh
```

**Commands actually run for seed 0 (2026-10-24, on AC, from this folder, threads capped at 4).**
Order and timings come from the logs in `logs/r5/`:

```bash
# 1. Smoke (16:06, PASS, disposable tag r5_smoke_20261024_160652)
../bin/gpu_run bash ./smoke_r5.sh 2>&1 | tee logs/r5/claude_smoke.log
# 2. First full start FAILED at once: the original track generator could not build
#    hard training seed 30049 (traceback kept at the top of both console logs).
../bin/gpu_run ../.venv/bin/python train_r5.py --tag r5_seed0 --seed 0 --max-minutes 18 2>&1 | tee -a logs/r5/r5_seed0_train.console.log
../bin/gpu_run ../.venv/bin/python tune_r5.py  --tag r5_seed0 --seed 0 --max-minutes 12 2>&1 | tee -a logs/r5/r5_seed0_baseline.console.log
#    Fix (Astra turn 1b, before any driving result): R5-only deterministic retry in
#    r5_env.make_r5_track. Only hard seed 2030049 is replaced (effective seed 4814659841,
#    retry 1); all 438 full-size R5 tracks now build; legacy 100/50 tracks untouched.
#    Audit: results/r5/r5_turn1b_track_audit.json.
# 3. Smoke again (16:36, PASS, tag r5_smoke_20261024_163601)
../bin/gpu_run bash ./smoke_r5.sh 2>&1 | tee logs/r5/claude_smoke2.log
# 4. Paired GRU/FF PPO: 4436 updates, 36.3M env steps per arm, 1076.9 s, stop wall_guard
../bin/gpu_run ../.venv/bin/python train_r5.py --tag r5_seed0 --seed 0 --max-minutes 18 2>&1 | tee -a logs/r5/r5_seed0_train.console.log
# 5. CEM tuning of the classical controller: 627 generations, 205M transitions, 716.4 s, wall_guard
../bin/gpu_run ../.venv/bin/python tune_r5.py --tag r5_seed0 --seed 0 --max-minutes 12 2>&1 | tee -a logs/r5/r5_seed0_baseline.console.log
# 6. Fresh test (3.4 s) and secondary legacy split (3.7 s, AC-only step)
../bin/gpu_run ../.venv/bin/python r5_eval.py --tag r5_seed0 --split test --n 64 --max-minutes 2.5 2>&1 | tee logs/r5/claude_test.log
../bin/gpu_run ../.venv/bin/python r5_eval.py --tag r5_seed0 --split legacy --n 50 --max-minutes 2.5 2>&1 | tee logs/r5/claude_legacy.log
# 7. Offline demo -> media/r5_seed0_demo.html (no log file was kept for this step)
../bin/gpu_run ../.venv/bin/python demo_r5.py --tag r5_seed0 --max-minutes .5
# 8. Turn-2 NumPy audit/plots (Astra, no gpu_run) -> results/r5/r5_turn2_analysis_v2.json,
#    media/r5_turn2_v2_*.png. Command line rebuilt from analyze_r5.py's flags; the log
#    does not record the exact invocation.
MPLCONFIGDIR=results/r5/mpl_cache ../.venv/bin/python analyze_r5.py --tag r5_seed0 --max-minutes 1 \
  --output results/r5/r5_turn2_analysis_v2.json --media-prefix media/r5_turn2_v2
```

The planned AC seeds 1/2 were **not** run; all Round-5 numbers are one seed. The reviewer
re-ran the 23 NumPy tests and `analyze_r5.py` (output to a scratch path) and got an identical
audit. `analyze_r5.py` needs `checkpoints/r5_seed0_state.npz` (364 KB, the final rolling
state); `.gitignore` now keeps that one state file so the audit reruns from a checkout.

To reproduce the completed analysis (~4–15 s, no MLX), run the timestamped
commands below from this folder; they preserve existing receipts and keep CPU
threads at four. The analysis uses `analyze_r5.py` and the local final resume
state, without changing selected checkpoints.

```bash
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
export MPLCONFIGDIR=results/r5/mpl_cache
R5_ANALYSIS_ID=r5_turn2_verify_$(date +%Y%m%d_%H%M%S)
# measured 9.89 s; healthy: 23 passed, including all 438 layouts
../.venv/bin/python -m pytest -q tests/test_r5_numpy.py tests/test_r5_analysis.py \
  --basetemp="results/r5/pytest_tmp/$R5_ANALYSIS_ID"
# measured 3.77 s warm / 13.28 s first font-cache build; healthy: three replay_verified records (384 each), then ANALYSIS_PASS
../.venv/bin/python analyze_r5.py --tag r5_seed0 --max-minutes 1 \
  --output "results/r5/${R5_ANALYSIS_ID}.json" --media-prefix "media/${R5_ANALYSIS_ID}"
```

Round-5 files: `r5_env.py`, `r5_policy.py`, `r5_baseline.py`, `r5_eval.py`;
`train_r5.py` / `tune_r5.py`; `bench_r5.py`, `check_r5_mlx.py`,
`check_r5_kill_resume.py`; `demo_r5.py` generates `media/r5_seed0_demo.html` from
selected checkpoints and its matching evaluation receipt. Artifacts stay in
`results/r5/`, `logs/r5/`, `checkpoints/r5_*`, and `media/r5_*`. Only selected exports
need committing; rolling resume files and disposable smoke checkpoints are ignored.
