# 24 The Floor Is Lava: quadruped stepping stones

A Unitree Go1 (MuJoCo Menagerie model, 12 position-controlled joints) has to cross a short
sequence of six broad stepping stones over a lava floor. Touching the floor with any body part ends
the episode as a failure. This is a "first satisfying version": full
state observations, plus the positions of the next few stones.

Built from a working locomotion controller up:
1. **Body:** Go1, picked after a speed smoke test (Go1 ~100k physics steps/s standing, single
   core, load average 3.8 at the time; A1 ~89k, Barkour ~78k; gymnasium Ant-v5 ~8.8k env steps/s). Simulation runs at
   4 ms with 50 Hz control.
2. **Flat-ground gait:** a Cartesian-space CPG (foot trajectories plus 2-link IK) with a heading
   term that steers toward the centre line of the stone ahead.
3. **Stones:** random layouts (six stones 0.35–0.55 m long and 0.55–0.80 m wide, gaps 0.05–0.20 m,
   lateral offsets of ±8 cm, tops 8–12 cm above the lava). Stones are moved and resized at runtime,
   with no recompile.
4. **Learned method:** residual PPO in MLX on top of the tuned CPG. The policy outputs 12
   joint-target offsets (±0.25 rad), a gait-frequency modulation (×(1±0.4)) and a stride-length
   modulation (×(1±0.5)). The observation is full state plus the next 4 stones. Training uses a gap
   curriculum.
5. **Baselines (scripted, no learning):** (a) the **open-loop CPG** trot, tuned by random search
   plus CEM on training layouts. Its only feedback is the heading term. (b) **CPG + scripted
   foothold planner**: at each foot's lift-off it predicts the touchdown point and, if that point
   is in a gap, shifts the step onto the nearest stone. It uses the same stone information as the
   policy and was tuned the same way.

Held-out layouts use seeds `10_000_000 + i`; training only ever samples seeds below `10_000_000`.

## How to run
All commands run from this folder. Heavy jobs go through the shared GPU/CPU slot lock.
```bash
# smoke check of the whole pipeline at tiny size (~10-25 s)
./smoke.sh                                   # also runs: ../.venv/bin/python -m pytest -q tests

# 1. tune the open-loop CPG baseline on training layouts (~7 min, 4 workers)
../bin/gpu_run ../.venv/bin/python tune_cpg.py --n_random 500 --n_layouts 24 --cem_gens 14 --cem_pop 48 \
    --final_layouts 128 --out checkpoints/cpg_params.json
# 2. tune the CPG + foothold-planner baseline (round 1; rounds 2-4 and CPG round 2 in logs/*.sh)
../bin/gpu_run ../.venv/bin/python tune_cpg.py --planner --warm checkpoints/cpg_params.json --n_random 500 \
    --n_layouts 24 --cem_gens 14 --cem_pop 48 --final_layouts 128 --out checkpoints/planner_params.json
#    logs/after_blind.sh (planner round 2), logs/planner3.sh, logs/planner4.sh, logs/cpg2.sh: later rounds + their evals
# 3. residual PPO (one 42-min run was used; --resume continues a run from its checkpoint)
../bin/gpu_run ../.venv/bin/python ppo.py --run main --minutes 42
cp checkpoints/main_latest.npz checkpoints/policy.npz          # = checkpoints/main_run1_end.npz
#    ablation: same config with the stone observations zeroed
../bin/gpu_run ../.venv/bin/python ppo.py --run blind --blind --minutes 42
# 4. held-out evaluation (all methods on the same layouts and seeds; clean + push test)
../bin/gpu_run logs/eval_run1.sh      # full gaps, gaps<=0.125, flat-walkway push
../bin/gpu_run logs/eval_extra.sh     # blind policy, masked stones, 80/120 N pushes, out-of-range gaps <=0.25
# 5. demo videos on held-out layouts (reloads checkpoints/policy.npz)
../.venv/bin/python render_demo.py --policy checkpoints/policy.npz --planner checkpoints/planner4_params.json
../.venv/bin/python render_demo.py --policy checkpoints/policy.npz --planner checkpoints/planner4_params.json --index 140 --tag ppo_failure
# 6. reviewer checks: gait/exploit probe, PPO-gait baseline without residuals,
#    planner round 5 with a widened gait range, 80 N stones push for planner r4, energy per second
../.venv/bin/python review_probe.py --n 40                     # -> results/review_probe.json (~10 s)
../bin/gpu_run ../.venv/bin/python evaluate.py --policy "" --cpg checkpoints/review_cpg_ppogait.json \
    --planner checkpoints/review_planner_ppogait.json --n 500 --out results/review_eval_ppogait_full.json
../bin/gpu_run logs/review_planner5.sh                          # ~10 min: tune + held-out eval
../bin/gpu_run ../.venv/bin/python review_fast_corner.py      # fastest open-loop gait on flat (~1 min)
../bin/gpu_run logs/review_planner_fast.sh                      # ~4 min: planner restricted to fast gaits + eval
../bin/gpu_run ../.venv/bin/python evaluate.py --policy "" --planner checkpoints/planner4_params.json \
    --n 500 --push 80 --out results/review_eval_planner4_push80_full.json
../.venv/bin/python review_energy.py eval_planner4_full         # J/m and W on jointly completed layouts
# 7. training-curve plots and result tables
../.venv/bin/python plot_curves.py
../.venv/bin/python make_tables.py    # -> results/summary.md
```

## Files
| file | what |
|---|---|
| `lava_env.py` | Go1 + lava floor + stones, layout generator, train/held-out seed split, lava/flip detection |
| `task.py` | CPG trot (IK, heading term), scripted foothold planner, observation, reward, termination, `run_episode` |
| `tune_cpg.py` | random search + CEM + re-evaluation tuner for both scripted baselines (training layouts only) |
| `ppo.py` | residual PPO in MLX: 4 worker processes, gap curriculum, checkpoint/resume |
| `evaluate.py` | held-out evaluation with Wilson/bootstrap CIs, push test, completion by gap size |
| `render_demo.py`, `render_util.py` | offscreen MuJoCo rendering to MP4/GIF |
| `plot_curves.py` | training curves to `media/training_curves.png` (and `_blind.png`) |
| `make_tables.py` | collects `results/eval_*.json` into `results/summary.md` |
| `smoke.sh`, `tests/` | end-to-end smoke check and unit tests |
| `checkpoints/` | `cpg_params.json`/`cpg2_params.json` (open-loop CPG, tuning rounds 1–2), `planner*_params.json` (planner rounds 1–4), `policy.npz` (final policy), `main_*.npz`, `blind_latest.npz` (ablation) |
| `results/` | evaluation JSON (summary + per-episode) |
| `media/` | demo videos (policy vs baselines on the same held-out layout), curves |
| `logs/` | training CSVs and run logs |
| `review_probe.py`, `review_tune_wide.py`, `review_fast_corner.py`, `review_energy.py` | reviewer checks; outputs `results/review_*.json`, `checkpoints/review_*.json` |

`.gitignore` (added in review) keeps these on disk but out of the repo: the byte-identical copies
of `policy.npz` (`main_latest.npz`, `main_run1_end.npz`), the intermediate `main_it800.npz`, smoke
outputs, per-config tuning logs (`checkpoints/*_params_log.json`) and per-episode evaluation dumps
(`results/*_episodes.json`, ~20 MB). None is needed for the demo, the evaluation summaries or
`make_tables.py`; `review_energy.py` and the paired per-layout analyses need the episode dumps, so
re-run the evaluation to regenerate them. `checkpoints/policy.npz` (2.2 MB) also holds the Adam state so
`--resume` works; the weights alone are ~0.7 MB.
