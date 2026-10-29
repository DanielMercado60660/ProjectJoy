# 03 · The Machine That Gets Bored

A small creature in a nine-room house full of switches, levers, a key, a locked door, a box
in a doorway, and a button that opens a door on the other side of the house. **There is no reward
and no goal.** The only drive is an intrinsic exploration bonus. Does a curious agent find more
rooms and mechanisms than a random walk with the same 512-action budget? What happens when you
add a "noisy TV" to the house?

Results and the write-up are in [RESULTS.md](RESULTS.md).

## The world (`env.py`)

```
###################      l lever      o lamp      b button     k key
#l...o#.....#.....#      B box        L locked door (needs key)
#.....#....b#.....#      S door opened by a button   _ open doorway
#....._.....S.....#      T noisy-TV screen (only in the --noisy_tv world)
#.....#.....#.....#
#.....#.....#....k#      start: centre of the top-left room
###_#####B#####_###
#.....#o....#.....#
#.....#.....#.....#
#....T#.....#.....#
#....T#.....#.....#
#....T#..l..#.....#
###L###########S###
#.....#.....#....o#
#.....#.....#.....#
#....._.....#....l#
#.....#.....#.....#
#b....#.....#.....#
###################
```

* 3×3 rooms with 5×5 interiors (19×19 grid, 233 walkable cells). Each room has its own floor colour.
* Observation: a **7×7 egocentric window** (one-hot cell type, room colour and TV pixel per cell),
  plus "holding key" and facing direction: 1622 binary features. No position and no memory.
* Actions: up, down, left, right, interact (with the faced cell). Episodes last 512 actions and
  then reset to the start.
* 8 **useful interactions** (easy to hard): lever A lights a lamp; button B opens the door next to
  it; push the box out of a doorway; pick up the key; unlock the locked door; button E (bottom-left)
  opens a door at the far right; lever C (behind the box); lever D (only reachable after the far door).
* `--noisy_tv`: three screen tiles in room R10 show fresh random colours every step.
* Numba-vectorised: about 850k random env steps per second on 4 threads.

## Agents

| agent | what drives it |
|---|---|
| `random` | uniform random actions (the baseline) |
| `count` | PPO + `1/sqrt(N(hash(obs)))`, with counts on a 64-bit hash of the egocentric observation |
| `rnd` | PPO + Random Network Distillation (Burda et al. 2018): fixed random target net, trained predictor, observation normalisation, intrinsic reward divided by the running std of intrinsic returns, non-episodic intrinsic value head + extrinsic head (the extrinsic reward is always zero) |
| `rnd_ep` | "RND × episodic count": PPO + episodic novelty (1/√n of this observation's hash *within the current episode*) × a lifelong RND multiplier clipped to [1, 5] (after Badia et al. 2020, "Never Give Up") |
| `ep` | ablation: the episodic part of `rnd_ep` alone |
| `ngu` | noisy-TV fix, "NGU-style": like `rnd_ep`, but episodic novelty is a k-nearest-neighbour similarity in a 32-d embedding trained only by an inverse-dynamics model (predict the action from two consecutive embeddings), so it has no reason to encode things the agent cannot control |
| `disagree` | noisy-TV fix: PPO + variance across an ensemble of 5 forward models (Pathak et al. 2019) that predict a fixed random embedding of the next observation |

PPO (MLX): MLP 1622→256→256 with policy, intrinsic-value and extrinsic-value heads; 128 envs ×
128-step rollouts, 4 epochs × 4 minibatches, clip 0.1, γ_int 0.99, λ 0.95, Adam 2.5e-4,
entropy 0.001. The extrinsic head is kept, as in Burda et al., but its advantage weight is 0
because the extrinsic reward is always zero (see RESULTS.md: weighting it by 2 just injected noise).
The RND predictor trains on a random 25% of each minibatch.

## How to run

All commands run from this folder.

```bash
# smoke test (< 2 min): unit tests, tiny train, save/kill/resume, eval, render
./smoke.sh
../.venv/bin/python -m pytest -q

# train one agent (long runs go through the shared GPU lock)
../bin/gpu_run ../.venv/bin/python train.py --agent rnd --seed 0 --n_envs 128 --total_steps 6000000 --out runs/rnd_base_s0
#   add --noisy_tv for the noisy-TV world; --resume runs/<name>/ckpt.pkl to continue a run

# sweep several (agent, seed) pairs in one GPU slot, then evaluate each
../bin/gpu_run ./sweep.sh base 6000000 rnd:0 rnd:1 rnd:2 count:0 count:1 count:2
# or: put lines like "tv 6000000 ngu:1" in runs/queue.txt and drain them 3 at a time in one slot
../bin/gpu_run ./worker.sh 36 3

# evaluate (64 fresh 512-action episodes, fixed eval seed)
../.venv/bin/python evaluate.py --ckpt checkpoints/rnd_base_s0.pkl --out runs/eval_x.json
../.venv/bin/python evaluate.py --random --seed 10000 --out runs/eval_random_base_s0.json

# figures + summary table (reads runs/)
../.venv/bin/python plots.py

# demo GIFs from saved checkpoints: random walk + one panel per checkpoint, side by side
../.venv/bin/python render.py --ckpt checkpoints/rnd_ep_base_s0.pkl --label "PPO + RND x episodic count" --out media/explorer_vs_random.gif
../.venv/bin/python render.py --ckpt checkpoints/rnd_base_s2_leverspam.pkl --label "PPO + RND (seed 2): lever spam" --out media/failure_lever_spam_rnd.gif
../.venv/bin/python render.py --noisy_tv --ckpt checkpoints/rnd_ep_tv_s0.pkl checkpoints/ngu_tv_s0.pkl checkpoints/disagree_tv_s0.pkl \
    --label "RND x episodic count" "NGU-style (inv.-dyn. kNN)" "Ensemble disagreement" --out media/noisy_tv.gif

# how interesting does each trained bonus find the TV? (same random-walk states for all agents)
# reads runs/*_tv_s*/ckpt.pkl, so it needs the (git-ignored) runs/ folder; output is in results/tv_diagnosis.json
../.venv/bin/python diagnose_tv.py

# markdown tables used in RESULTS.md
../.venv/bin/python table.py base; ../.venv/bin/python table.py tv
```

## Layout

```
env.py        world, mechanics, metrics (numba)          agents.py   policy, RND, disagreement, hash counter
train.py      PPO training loop, checkpoint/resume       evaluate.py fixed-budget evaluation
render.py     side-by-side heatmap GIF                   plots.py    training curves, interaction timeline, summary
diagnose_tv.py  bonus on TV vs elsewhere                 table.py    markdown tables for RESULTS.md
sweep.sh      train+eval several runs in one GPU slot    worker.sh   drains runs/queue.txt, 3 runs in parallel
smoke.sh      end-to-end smoke check
tests/        pytest: mechanics, obs, training, exact resume
runs/         everything a run writes: ckpt.pkl, log.jsonl per update, episodes.jsonl, eval JSONs
              (git-ignored at the repo root, about 1 GB; plots.py and table.py read from here)
results/      committed copy of the small outputs: every eval JSON, summary.json, tv_diagnosis.json,
              and per-run log.jsonl + config.json
checkpoints/  final checkpoints used for the demos
media/        GIFs and figures
```

## Round 5: curiosity that becomes useful later

The follow-up compares learning-progress PPO, the existing explorers, and a tuned frontier/BFS
baseline, then transfers identical action/observation replay to seven surprise tasks.
Both turns are complete; see [RESULTS.md, Round 5](RESULTS.md#round-5-curiosity-that-becomes-useful-later). Headline (Turn 2): RND × episodic replay solves 64% of
surprise tasks in the plain house (random walk 16%, empty replay 7%), the noisy TV drops it to 16%, the
learning-progress explorer (LPM) collapses and stays at the empty-replay floor, and a frontier/BFS
explorer solves 98–100%. [DESIGN_R5.md](DESIGN_R5.md) is the Turn-1 protocol;
[DESIGN_R5_T2.md](DESIGN_R5_T2.md) the corrected shared planner and LPM. Checkpoints: Turn 1 `checkpoints/r5_r5_ac/`, Turn 2 `checkpoints/r5_t2_ac/`.

Commands actually run on 2026-10-24 (AC power; logs in `logs/r5/claude_*.log`):

```bash
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
# Turn 1 (preliminary; 3,145,728 steps/cell, seeds 510-512)
../bin/gpu_run ./smoke_r5.sh                                                     # claude_smoke1.log
../bin/gpu_run ../.venv/bin/python run_r5.py tune --tag r5_tune --max-minutes 3  # claude_tune.log
for s in 510 511 512; do
  ../bin/gpu_run ../.venv/bin/python run_r5.py run --tag r5_ac --seed $s --steps 3145728 --episodes 16 \
    --selection results/r5/r5_tune/selection.json --max-minutes 42               # claude_run_s$s.log
done
# Turn 2 (headline; 5,996,544 steps/cell, seeds 610-612, test seeds 85400-85415)
../bin/gpu_run ./smoke_r5.sh                                                     # claude_t2_smoke.log (24 tests + 5 SMOKE_METHOD_OK)
../bin/gpu_run ../.venv/bin/python diagnose_r5_t2.py --mlx-probe --max-minutes 1 --out results/r5/t2_entropy_probe.json
../bin/gpu_run ../.venv/bin/python run_r5_t2.py tune --tag t2_tune --max-minutes 8
../bin/gpu_run ../.venv/bin/python run_r5_t2.py batch --tag t2_ac --selection results/r5/t2_tune/selection.json \
  --workers 2 --steps 5996544 --wave 1 --max-minutes 40                         # ALL_COMPLETE in 25.9 min; waves 2-3 not needed
../.venv/bin/python report_r5.py --turn2 --directory results/r5/t2_ac --seeds 610 611 612 --out results/r5/t2_ac/summary.md
../.venv/bin/python render_r5.py --receipt results/r5/t2_ac/lpm_tv_s610.json --out media/r5_t2_lpm_tv.gif
../.venv/bin/python render_r5.py --receipt results/r5/t2_ac/frontier_tv_s610.json --out media/r5_t2_frontier_tv.gif
../bin/gpu_run ../.venv/bin/python evaluate_r5.py --turn2 --method lpm --world tv \
  --checkpoint checkpoints/r5_t2_ac/lpm_tv_s610.pkl --episodes 16 --max-minutes 2 --out results/r5/t2_ac/lpm_tv_s610_reload.json
# Reviewer: NumPy-only LPM diagnosis receipts (<1 s, no MLX)
../.venv/bin/python review_r5_lpm.py
```

The Turn-1 report, GIF and reload commands (`results/r5/r5_ac/summary.md`, `media/r5_ac_*.gif`,
`lp_tv_s510_reload.json`) followed the Turn-1 plan; their exact command lines were not logged.

```bash
# NumPy-only checks; does not import MLX. Existing full tests include MLX.
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ../.venv/bin/python -m pytest -q tests/test_r5_numpy.py tests/test_r5_t2_numpy.py --basetemp=logs/r5/pytest_t2_runner
# Claude/Metal runner only: tiny pipeline plus real kill/resume and demo generation.
../bin/gpu_run ./smoke_r5.sh
```

New files: `r5_core.py` (observation-only mapping, frontier and replay planner), `train_r5.py`
(MLX progress/PPO and full checkpoints), `classical_r5.py` (matched classical step budgets),
`evaluate_r5.py` (transfer/control trials), `run_r5.py` (bounded tuning/run scheduler),
`report_r5.py` (complete matched tables), `render_r5.py` (headless replay GIFs),
`smoke_r5.py` / `smoke_r5.sh`, and `tests/test_r5_numpy.py`.
Turn 2 adds `planner_r5_t2.py` (replay-derived map, mechanism state and explicit oracle),
`progress_r5_t2.py` (bounded error queue and adaptive scaling), `run_r5_t2.py` (isolated,
bounded parallel waves), `diagnose_r5_t2.py` (receipt audit and optional MLX entropy probe),
and `tests/test_r5_t2_numpy.py`.
Reviewer: `review_r5_lpm.py` (LPM training-phase summary, NumPy forward-pass probe, discovery-vs-task correlation).
Outputs are isolated in `results/r5/`, `logs/r5/`, `checkpoints/r5_<tag>/`, and `media/r5_*`.
