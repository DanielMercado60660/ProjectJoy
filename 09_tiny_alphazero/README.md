# 09 · Tiny AlphaZero: Connect Four from self-play

A small AlphaZero-style system for Connect Four (7 columns × 6 rows) that starts knowing only the rules.
A residual policy/value CNN (MLX) is trained purely from self-play games searched with PUCT
Monte Carlo tree search. Each generation is checkpointed and then played against a fixed suite of
classical opponents and against earlier generations.

**Results, numbers and caveats are in [RESULTS.md](RESULTS.md).**

## What's here

| file | what it does |
|---|---|
| `c4.py` | Connect Four on int64 bitboards (numba): moves, win and draw detection, features, plus the **fixed opponent suite**: `random`, `greedy` (win > block > avoid giving a win > random), `uct<n>` (plain MCTS with *n* random-rollout simulations), `ab<d>` (depth-*d* alpha-beta with a window-count heuristic, random tie-breaks) |
| `mcts.py` | **Game-parallel batched PUCT MCTS**: one tree per game in flat numba arrays. Each simulation step selects a leaf in *every* game (numba, 4 threads), evaluates all leaves in **one** MLX call, then expands and backs up. Dirichlet noise at the root for self-play. |
| `net.py` | `PVNet`: 64-channel, 5-block residual CNN (376k params), policy head (7 logits) + value head (tanh) |
| `train.py` | AlphaZero loop: self-play (512 concurrent games, 100–200 sims/move, optional random opening plies) → replay buffer (250k positions) → AdamW training with random left-right flip augmentation → generational checkpoint + full resumable state. Inline eval vs greedy/ab4/uct1000 every 5 gens. |
| `arena.py` | Agents and a lock-step arena: every match is the 49 two-ply openings × both colours = 98 games; Bradley–Terry Elo fit |
| `evaluate.py` | Suite eval per generation, generation round robin, matched-budget comparison, Elo table, plots (results cached in `results/matches.json`) |
| `viz.py` | Game GIFs (with the net's visit distribution under the board) and the **visible search tree** (interactive HTML + PNG) |
| `export_web.py`, `build_demo.py`, `web/` | Export a checkpoint to JSON (BatchNorm folded into convs) and build the self-contained `media/demo.html` (JS rules + forward pass + MCTS). `web/check_js.js` checks the JS forward pass against MLX. |
| `slot.sh`, `eval_slot.sh`, `final_eval.sh` | GPU-slot sessions: train N min + evaluate M min; a standalone suite-evaluation session; the final evaluation (suite, latency, matched budgets, generation round robin, Elo, plots, report) |
| `seeds_chain.sh`, `seeds_eval.py` | Seed replication (RESULTS §10): phases B and C for a new training seed after phase A, with seed 0's exact per-phase arguments; suite / head-to-head evaluation with the same protocol, cached separately in `results/seeds/matches.json` so seed 0's cache and Elo fit are untouched |
| `solver.py` | **Exact solver** (RESULTS §11): Pons-style negamax alpha-beta on the same bitboards, only non-losing moves searched, move ordering by winning spots created then centre-first, transposition table (lower/upper bounds, mirror-canonical key), null-window iterative search, weak (win/draw/loss) or strong (exact score, win distance) solving, node budget; plus a plain full-minimax `brute_force` reference |
| `eval_solver.py` | Held-out test positions (random play and net play, fixed seeds), agent moves on them, a 4-process solve job, move accuracy vs perfect play, the 2-ply opening table (`results/solver/`, `media/solver_openings.png`) |
| `tests/` | pytest: rules (wins in all directions, no wrap-around, draws, mirror), batched MCTS (finds win / block, illegal moves, noise), stale-weights guard, end-to-end tiny pipeline with resume, solver vs brute-force minimax (`tests/test_solver.py`) |

Layout: `checkpoints/main/gen_XXX.{safetensors,json}` (one per generation, 0–191; weights + config; **gen_191 is the final net**),
`checkpoints/noflip/` (the no-augmentation ablation, gens 0–42),
`checkpoints/main/state/` (latest model + optimizer + replay buffer + RNG for resuming),
`checkpoints/seed1/`, `checkpoints/seed2/` (training seeds 1 and 2, same three-phase schedule, gens 0–191),
`checkpoints/noflip_seed1/`, `checkpoints/noflip_seed2/` (no-flip ablation, seeds 1 and 2, gens 0–40),
`logs/` (per-generation JSONL; `logs/seeds/` holds the seed-replication job logs), `results/` (match cache, Elo; `results/seeds/` for the seed replication), `media/` (demo, GIFs, plots, tree).

**Checkpoints in git.** All 192 + 43 generations and the resumable state (about 370 MB) exist only on the
machine that trained them. Git holds a representative subset (about 17 MB, see `.gitignore`):
`checkpoints/main/gen_{000,002,005,010,020,040,070,100,136,150,170,191}` + `config.json`. That covers the
final net, every command below that loads a checkpoint (demo, GIFs, search trees, the generation round robin)
and a spread of earlier generations to play against. `checkpoints/main/state/` and `checkpoints/noflip/` are
local only (the ablation's numbers are in `logs/noflip*.jsonl`). In a fresh clone, `--gens every5`/`every10`
and `final_eval.sh` only see the committed generations; their cached results are in `results/` either way.
For the seed replication only each new run's final generation is committed (`seed1/gen_191`, `seed2/gen_191`,
`noflip_seed1/gen_040`, `noflip_seed2/gen_040`, about 6 MB); its numbers are in `logs/seed*.jsonl` and `results/seeds/`.

## How to run

All commands from this folder, with the shared venv. Long jobs go through the shared GPU lock.

```bash
# tests (~25 s): rules, MCTS, end-to-end tiny training + resume
../.venv/bin/python -m pytest -q

# train (resumes automatically from checkpoints/<run>/state if present).
# Without state/ (e.g. a fresh clone) this starts a NEW run and overwrites gen_000.. in that folder,
# so use a new run name there:  --run main2
NUMBA_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python train.py --run main --minutes 44
# or one slot session = train 36 min + 8 min of evaluation. The main run was made of four sessions:
../bin/gpu_run bash slot.sh 36 8                                                         # gens 1-100
TRAIN_ARGS="--sims 200 --lr 1e-3" ../bin/gpu_run bash slot.sh 36 8                       # gens 101-136
TRAIN_ARGS="--sims 200 --lr 1e-3 --random-plies 4 --temp-moves 16" ../bin/gpu_run bash slot.sh 36 8   # 137-165, again -> 191
RUN=noflip TRAIN_ARGS="--no-flip" ../bin/gpu_run bash slot.sh 36 1                       # ablation

# seed replication (RESULTS §10): phase A for seed S, then phases B and C chained (each phase ≤ 44 min; optional STOP=HH:MM ends training by then)
NUMBA_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python train.py --run seed1 --seed 1 --gens 100 --minutes 44 > logs/seeds/seed1_A.log
echo 191 > logs/seeds/c_target; bash seeds_chain.sh 1
NUMBA_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python train.py --run noflip_seed1 --seed 1 --no-flip --gens 40
../bin/gpu_run ../.venv/bin/python seeds_eval.py suite --runs seed1,seed2 --gens 40,70,100,136,170,191
../bin/gpu_run ../.venv/bin/python seeds_eval.py h2h --pairs seed1/191:seed1/136,seed1/191:main/191
../.venv/bin/python seeds_eval.py table --runs main,seed1,seed2 --gens 100,136,191
../.venv/bin/python seeds_eval.py logs --runs main,seed1,seed2,noflip,noflip_seed1,noflip_seed2

# evaluation (each command caches results and can be resumed)
../bin/gpu_run ../.venv/bin/python evaluate.py baselines              # suite round robin (Elo anchors)
../bin/gpu_run ../.venv/bin/python evaluate.py suite --gens every5    # every 5th generation vs the suite
../bin/gpu_run ../.venv/bin/python evaluate.py rr --gens 0,5,10,20    # generation round robin
../bin/gpu_run ../.venv/bin/python evaluate.py latency --gen <G>      # single-game ms/move, time-matched rollout budgets
../bin/gpu_run ../.venv/bin/python evaluate.py budget --gen <G>       # policy-only vs search-only vs combined
../.venv/bin/python evaluate.py elo && ../.venv/bin/python evaluate.py plots && ../.venv/bin/python evaluate.py report
../bin/gpu_run bash final_eval.sh 191                                 # all of the above for the final net

# demos (each well under a minute)
../.venv/bin/python viz.py gif --a net:checkpoints/main/gen_191:200 --b ab8 --names "AlphaZero gen 191 (200 sims),alpha-beta depth 8" --out media/game_final_vs_ab8.gif
../.venv/bin/python viz.py gif --a net:checkpoints/main/gen_005:200 --b net:checkpoints/main/gen_191:200 --out media/game_gen5_vs_final.gif
../.venv/bin/python viz.py tree --ckpt checkpoints/main/gen_191 --sims 800 --out media/search_tree
../.venv/bin/python build_demo.py checkpoints/main/gen_191 media/demo.html
../.venv/bin/python export_web.py checkpoints/main/gen_191 web/weights.json

# solver follow-up (RESULTS §11): verify, build the held-out set, collect agent moves, solve, report.
# The main solve job (25 openings, 1,000 positions, 4 first moves) took ~12 min with 4 processes on a busy machine.
../bin/gpu_run ../.venv/bin/python eval_solver.py verify --max-empties 16 --per-empties 100   # vs brute force -> results/solver/verify_bruteforce.json
../.venv/bin/python eval_solver.py positions      # 2 x 500 positions -> results/solver/positions.json (seconds)
../.venv/bin/python eval_solver.py agents         # policy argmax, net+MCTS 100/800, ab8 moves -> results/solver/agent_moves.json (~30 s)
NUMBA_NUM_THREADS=1 ../bin/gpu_run ../.venv/bin/python eval_solver.py solve --procs 4 --minutes 40 --ply1-nodes 8e9   # resumable -> results/solver/solved.jsonl
NUMBA_NUM_THREADS=1 ../bin/gpu_run ../.venv/bin/python eval_solver.py solve --strong --procs 4 --minutes 15    # exact scores after 1. d (~1.5 min)
NUMBA_NUM_THREADS=1 ../bin/gpu_run ../.venv/bin/python eval_solver.py solve --strong1 --procs 4 --minutes 5 --strong-nodes 4e9   # exact first-move scores (~6 min)
NUMBA_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python eval_solver.py protocol    # replay gen191@100 vs ab8, uct5000 and cross with opening values (~1 min)
../.venv/bin/python eval_solver.py report         # -> results/solver/report.md, summary.json, media/solver_openings.png
node web/check_js.js web/weights.json   # JS forward pass vs MLX
```

`media/demo.html` opens offline in any browser (iPad included): pick a colour and a thinking budget
(policy only, 32, 100 or 400 simulations) and play; the panel shows the net's prior and the
search's visit share for its last move.

## Media

| file | what |
|---|---|
| `media/demo.html` | play against the final net, offline (iPad-friendly) |
| `media/game_final_vs_ab8.gif`, `media/game_ab8_vs_final.gif` | final net vs alpha-beta depth 8, from each side |
| `media/game_gen5_vs_final.gif` | generation 5 vs generation 191 |
| `media/search_tree.html/.png`, `media/search_tree_gen100.html/.png` | visible MCTS trees (visit counts, Q, priors), auto-picked positions |
| `media/suite_vs_generation.png`, `media/elo_vs_generation.png` | progress per generation |
| `media/budget_comparison.png` | policy only vs search without a net vs net + search, by simulation budget |
| `media/training_curves.png` | losses, self-play outcomes and game length (shows the opening collapse) |
| `media/ablation_flip.png`, `media/inline_eval.png` | flip-augmentation ablation; inline eval during training |
| `media/solver_openings.png` | the 49 two-ply openings solved: first-player win / draw / loss (RESULTS §11) |

## Round 5: learn when to think longer

A frozen gen-191 network plus a mistake head allocates search against entropy,
value/visit disagreement, a tuned alpha-beta rule and logistic regression.
**The learned allocator did not beat fixed MCTS:** main equal-simulation errors were
11/11; the larger run was 61/60, at 3.69× the decision time. At matched time the
larger learned/fixed counts were 73/60. Game scores were 0.49494/0.49242 for the two
learned heads. These are nested-data runs, not independent replications.

**Turn-2 follow-up (12,480 fresh solver-labelled boards, fixed PUCT and every gate's
floor/cap retuned on 1,536 fresh validation boards):** at matched time, fixed made 548
mistakes and every adaptive signal made more (learned 568, rule 584). At equal simulations,
a 6-ply alpha-beta disagreement rule cut mistakes to 445 (−19%, CI 13–25%). Its lookahead
isn't counted as simulations, and in lockstep-batched MCTS the uneven budgets cost
1.7–3.6× the wall time. The rule's gain is new in this test (turn-1 rules selected by
calibration showed none). The learned head's equal-simulation gain (517) is borderline.

See [RESULTS.md](RESULTS.md#round-5-learn-when-to-think-longer) for every signal's
rates, calibration, game intervals, provenance and baseline-tuning limitations.
The original [design](DESIGN_R5.md) is retained. All commands below were
run on 2026-10-24 (AC); the log each one wrote is named.

```bash
# Turn 1, as run by Claude (all through gpu_run; do not rerun on the same run names):
../bin/gpu_run bash ./smoke_r5.sh > logs/r5/claude_smoke.log 2>&1                    # 11.7 s
../bin/gpu_run ../.venv/bin/python r5_run.py train --run main --max-minutes 24 > logs/r5/main_train.log 2>&1
../bin/gpu_run ../.venv/bin/python r5_run.py evaluate --run main --max-minutes 8 > logs/r5/main_eval.log 2>&1
../bin/gpu_run ../.venv/bin/python r5_run.py train --run ac_large --max-minutes 44 --data-minutes 28 \
    --holdout 1536 --min-train 4096 > logs/r5/ac_large_train.log 2>&1
../bin/gpu_run ../.venv/bin/python r5_run.py evaluate --run ac_large --max-minutes 12 \
    --holdout 1536 --min-train 4096 > logs/r5/ac_large_eval.log 2>&1

# CPU-only checks and receipt audit (~3–10 s); no MLX imported.
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
../.venv/bin/python -m pytest -q tests/test_r5.py tests/test_c4.py tests/test_solver.py
../.venv/bin/python r5_analyze.py
../.venv/bin/python r5_plot.py
../.venv/bin/python r5_report.py
../.venv/bin/python r5_run.py demo --run main --demo-out media/r5_main_review.html
../.venv/bin/python r5_run.py demo --run ac_large --demo-out media/r5_ac_large_review.html

# Turn-2 follow-up, as run (AC, both gpu_run): smoke 1.9 s, then 358 s, 12,480 test boards.
# Writes results/r5/turn2_floor/ and media/r5_turn2_floor.html. A rerun ADDS test blocks.
../bin/gpu_run ../.venv/bin/python r5_followup.py --smoke --max-minutes 1 > logs/r5/turn2_smoke.log 2>&1
../bin/gpu_run ../.venv/bin/python r5_followup.py --max-minutes 6 > logs/r5/turn2_floor.log 2>&1
```

Open the [summary plot](media/r5_review.png), [main report/demo](media/r5_main_review.html),
[larger report/demo](media/r5_ac_large_review.html) or the
[follow-up demo](media/r5_turn2_floor.html) offline. The review HTML shows
calibration, actual game costs and failure boards; the follow-up demo shows the 14-arm
table and 12 learned-arm failure boards. Silent original demos had been
written successfully; the renderer now prints a `demo_written` receipt. Original
media and measured outputs remain intact.

`r5_core.py`, `r5_data.py`, `r5_baselines.py` implement CPU mechanics;
`r5_mlx.py` implements the head/resumable training; `r5_run.py` orchestrates.
`r5_analyze.py` audits saved records, `r5_plot.py` plots them, `r5_report.py` generates
the derived report, and `r5_followup.py` tunes frozen-model search on fresh positions.
Evidence lives in `results/r5/{main,ac_large,turn2_floor}/` and `results/r5/turn2_analysis.json`;
logs in `logs/r5/`; selected checkpoints in `checkpoints/r5_{main,ac_large}/selected/`.
Only selected weights/configuration/normalizers/baselines are needed for reproduction;
alternating resume slots and smoke artifacts stay local. The follow-up creates no new weights.
