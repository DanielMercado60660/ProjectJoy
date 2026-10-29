# 25 Neural Skill Transplantation

Can two policies that learned different skills be combined into one network without retraining
everything? This exhibit builds a **first satisfying version**: two
same-architecture specialists with identical observation and action conventions, an explicit
policy-switching baseline (the ceiling), naive weight averaging (the floor), and a first test of
merging, distillation and fine-tuning as ways to put both skills in one network.

Results, numbers and caveats: **[RESULTS.md](RESULTS.md)**.

## Setup

* **Body:** gymnasium `HalfCheetah-v5` (17-d obs, 6-d action, 1000-step episodes, no termination).
* **Two skills, same body, same conventions:**
  `fwd`: r = +x_velocity − 0.1·|a|² (the standard reward), `bwd`: r = −x_velocity − 0.1·|a|².
  These are deliberately *conflicting* skills (the same joints must do opposite things).
  Follow-up pair (compatible skills): `fwd` vs `cruise`: r = +x_velocity − 1.0·|a|² (low-energy forward running),
  selected with `SKILL_PAIR=fwd_cruise`.
* **One architecture:** input = [normalized obs (17), task one-hot (2)], actor and critic are separate
  tanh MLPs 19→128→128→(6 | 1), tanh-bounded action mean, state-independent log-std. The
  observation normalizer is frozen after the shared base phase, so every branch, merge and
  fine-tune sees the same input convention.
* **Shared start θ0, two conditions:**
  * `pre`: θ0 = a short multitask PPO pretraining (0.5M env steps, 8 fwd + 8 bwd envs).
  * `init`: θ0 = the same random initialization, no pretraining (same frozen normalizer).
* **Specialists:** PPO (MLX) from θ0 on one task for 2M env steps each (16 envs, task one-hot set).
  Snapshots at 125k/250k/500k/1M steps give a branch-length study.
* **Combiners** (all training-free unless stated):
  * oracle switch (task label picks the specialist) — **ceiling**, 2× parameters
  * weight interpolation (1−α)θ_fwd + αθ_bwd; α = 0.5 = naive averaging — **floor**
  * task arithmetic θ0 + λ(τ_fwd + τ_bwd), τ = θ_spec − θ0
  * TIES-merging (top-20% trim, sign election, disjoint mean) and DARE (drop 90%, rescale)
  * λ for each merge is chosen on **validation** reset seeds, results are on **held-out eval** seeds
  * DAgger distillation of both specialists into one network (200k env steps, no rewards)
  * multitask PPO fine-tuning from θ0 / the average / the task-arithmetic merge (1M env steps)
  * multitask PPO from scratch with the full matched budget (4.5M = 0.5M + 2×2M)
  * sequential transfer: specialist of one task → PPO on the other (1M)
* **Scripted baseline:** an open-loop sinusoidal gait (CPG, 19 params) tuned per task with CEM on
  training seeds only — tells you how much of a merged policy's return any closed-loop skill explains.

## How to run

All heavy runs go through the shared GPU lock. Seeds run in a pool of 4 single-thread workers.

```bash
cd 25_skill_transplantation
./smoke.sh                                   # whole pipeline at tiny size + pytest (92 s at load avg ~40; 52 s at ~16 in review)
../bin/gpu_run ./run_main.sh                 # stage train + merge, seeds 0-3   (~25 min under load)
../bin/gpu_run ./run_adapt.sh                # stage adapt (fine-tune / distill / transfer / scratch) + CPG
../bin/gpu_run ../.venv/bin/python lambda_fine.py --seeds 0 1 2 3   # extends the merge lambda grid down to 0
../bin/gpu_run ./run_pair2.sh                # follow-up: compatible pair fwd vs low-energy cruise (train + merge)
../bin/gpu_run ./run_pair2_adapt.sh          # follow-up: adapt stage (distill / fine-tune / transfer) for that pair
../.venv/bin/python report.py                # results/summary.md, results/summary.json, media/*.png
SKILL_PAIR=fwd_cruise REPORT_RES=results/fwd_cruise ../.venv/bin/python report.py   # results/fwd_cruise/summary_fwd_cruise.md, media/*_fwd_cruise.png
../.venv/bin/python render_demo.py --seed 0 --cond pre   # media/demo_grid.{mp4,gif}, checkpoints/demo/
../.venv/bin/python -m pytest -q tests      # merge identities, MLX == numpy forward, reward/normalizer
# cross-review additions
../bin/gpu_run ../.venv/bin/python cpg_wide.py                              # CPG re-tuned in a wider box, 2 restarts -> results/cpg_wide.json (~2 min)
../bin/gpu_run ../.venv/bin/python review_checks.py                         # single-specialist baseline (val-picked), zero-grad check, seed banks, per-seed 90% steps, demo reload -> results/review_checks.json
SKILL_PAIR=fwd_cruise ../bin/gpu_run ../.venv/bin/python review_checks.py   # same for the follow-up pair -> results/fwd_cruise/review_checks_fwd_cruise.json
```

Individual stages: `../.venv/bin/python pipeline.py --stage {train,merge,adapt} --seeds 0 1 2 3`.
Every PPO run checkpoints (weights, Adam state, step, RNG, normalizer, config) and resumes; the
adapt stage also skips runs already recorded in `results/adapt_s<seed>.json`.

## Files

| file | what |
|---|---|
| `common.py` | env, task rewards, network params as numpy dicts, numpy forward, normalizer, evaluation |
| `ppo.py` | PPO with MLX updates and numpy rollouts, checkpoint/resume |
| `merge.py` | interpolation, task arithmetic, TIES, DARE, task-vector statistics |
| `pipeline.py` | the experiment: `train`, `merge`, `adapt` stages per seed |
| `cpg.py` | scripted CPG gait baseline, CEM-tuned per task |
| `cpg_wide.py` | review: CPG baseline re-tuned (box edge in `cpg.py` was binding), wider box + 2 CEM restarts |
| `review_checks.py` | review: single-specialist baseline picked on val seeds, label-column zero-gradient check, seed-bank disjointness, per-seed steps to 90%, demo reload check |
| `lambda_fine.py` | follow-up: merge λ grid extended to {0, 0.1, 0.2}, re-selected on validation seeds |
| `run_pair2.sh` | follow-up: same train + merge on a compatible pair (`SKILL_PAIR=fwd_cruise`) |
| `run_pair2_adapt.sh` | follow-up: adapt stage on the compatible pair |
| `report.py` | aggregation over seeds → `results/summary.md`, plots in `media/` |
| `render_demo.py` | reloads checkpoints, rebuilds merges, renders the demo grid |
| `run_main.sh`, `run_adapt.sh` | the two gpu_run jobs |
| `smoke.sh`, `tests/` | smoke test and unit tests |
| `checkpoints/s<seed>/` | `base` (θ0 of `pre`), `init`, `{pre,init}_{fwd,bwd}` specialists (+ `_snap<steps>`), `ft_*` fine-tunes, `distill_*` |
| `checkpoints/fwd_cruise/s<seed>/` | the same layout for the fwd vs cruise follow-up |
| `checkpoints/demo/` | the averaged and best-merge networks used in the demo |
| `results/` | per-seed JSON (`train_`, `merge_`, `adapt_`, `lambda_fine_`), `cpg.json`, `summary.md/json`; `results/fwd_cruise/` for the follow-up |
| `media/` | `demo_grid.mp4/.gif`, `interpolation.png`, `branch_length.png`, `adaptation.png`, `training_curves.png` |
| `logs/` | job logs and the checkpoint/resume test |

**Repo size:** `checkpoints/**/*_state.pkl` (Adam moments, RNG and history for resume, ~58 MB of the
~107 MB) are excluded by the project `.gitignore` but kept on disk. The demo, `report.py`,
`lambda_fine.py` and `review_checks.py` only read the `.npz` weights. Without the `.pkl` files,
`ppo.train` cannot skip finished runs, so rerunning `run_*.sh` retrains from scratch instead of
returning the saved result.

## Round 5: sequential three-skill repertoire

Round-5 work is independent of the earlier commands. [DESIGN_R5.md](DESIGN_R5.md) defines
speed-target tasks, frozen-trunk low-rank adapters, replay DAgger, cumulative joint PPO,
CPG tuning, switch failures and teacher-inclusive cost accounting. Results and the reviewer's
TL;DR are in the **Round 5** section of [RESULTS.md](RESULTS.md).

Turn 1 (preliminary, under-budgeted; run 2026-10-24 on AC, one `gpu_run` slot per command):

```bash
../.venv/bin/python -m pytest -q tests/test_r5_numpy.py  # NumPy only
../bin/gpu_run ./smoke_r5.sh                           # runner: all new paths + SIGKILL/resume
../bin/gpu_run ../.venv/bin/python r5.py train --tag main_s0 --seed 0 --max-minutes 25
../bin/gpu_run ../.venv/bin/python r5.py cpg --tag main_s0 --seed 0 --max-minutes 5
../bin/gpu_run ../.venv/bin/python r5.py evaluate --tag main_s0 --seed 0 --max-minutes 2
../bin/gpu_run ../.venv/bin/python r5.py replay-demo --tag main_s0 --seed 0 --max-minutes 0.5
```

`r5_core.py`: NumPy protocol, metrics and CPU rollout mirror; `r5_train.py`: MLX training/checkpoints;
`r5.py`: orchestration, CPG and evaluation; `r5_report.py`: report/PNG/offline animation;
`r5_gpu_checks.py`: runner-only numerical and killed-process resume checks.
New outputs live in `checkpoints/r5_<tag>/`, `results/r5/<tag>/`, `logs/r5/`, and `media/r5_<tag>_*`.
The `tiny` profile and `cpucheck*` tags are pipeline checks, never headline learning results.

### Round 5, turn 2

The completed `main_s0..s2` runs are **preliminary, under-budgeted**. Turn 2 enlarges the
matched acquisition budget to 15,728,640 credits/arm, restores stronger PPO settings,
adds dense training-only speed feedback and validation-selected deployment, and requires
qualified teachers. The paired CPG policies are reused unchanged. New headline resets use
9M; old 7M results are secondary.

Commands run for turn 2 (2026-10-24, AC, from this folder, three seeds in parallel; the orchestrator's
invocations, with output captured in `logs/r5/claude_t2b_s*.out` and `claude_t2c_s*.out`):

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
../.venv/bin/python -m pytest -q tests/test_r5_numpy.py           # 20 passed
../bin/gpu_run ./smoke_r5.sh 2>&1 | tee logs/r5/smoke_t2.log       # R5_SMOKE_OK, 27 s
for s in 0 1 2; do ../bin/gpu_run ./run_r5_t2.sh $s > logs/r5/claude_t2b_s$s.out 2>&1 & done; wait
# all three hit the 40-min train guard (exit 75, R5_PARTIAL) in stage 3; rerun resumes and finishes:
for s in 0 1 2; do ../bin/gpu_run ./run_r5_t2.sh $s > logs/r5/claude_t2c_s$s.out 2>&1 & done; wait
../.venv/bin/python r5_review_t2.py | tee logs/r5/review_t2.log    # reviewer: 3-seed aggregate + pooled-CPG check, ~10 s, numpy only
```

The very first launch (`claude_t2_s*.out`) exited at once because `run_r5_t2.sh` called `rg`,
which is not on PATH; the AC check now uses `grep`. Each seed took ~40 + ~18 min across the two
invocations. Outputs: `results/r5/t2_s<seed>/` (`summary.md`, `eval.json`, `training.json`,
`qualification.json`, `cpg.json`, `demo_reload.json`), checkpoints in `checkpoints/r5_t2_s<seed>/`,
demo `media/r5_t2_s<seed>_demo.html` (offline stick-figure animation of each method's worst test
transition) and `media/r5_t2_s<seed>_transitions.png`, reviewer output `results/r5/review_t2/review.json`.

`r5_audit_t2.py` produced the retrospective/validation-only audit in
`results/r5/turn2_audit/turn1.json`; turn-1 sources and handoff were copied into that audit's
`source_turn1/` directory before editing. New job outputs use `t2_s0..s2`; exit 76 means
an unqualified teacher, exit 75 a resumable time guard. Neither is a completed experiment.
