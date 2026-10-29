# 05 · The Dungeon That Hates You

A small level-design loop: a recurrent PPO navigator learns to escape 13×13 dungeons while seeing only
a 5×5 window around itself. An evolutionary designer then breeds dungeons that this navigator
fails in. Every level is BFS-checked during breeding, so the designer can never just wall off the
exit. The designer's levels then go into a curriculum, and the result is compared against a
fine-tune with the same compute and no designer levels.

Results and numbers: [RESULTS.md](RESULTS.md). Demos are in `media/`. Open `media/demo.html` in
any browser; it works offline. Its JS GRU port reproduces the Python greedy trajectories exactly.
Then look at `media/rogues_gallery.png`, `media/before_after.gif` and `media/curriculum_rounds.png`.

## Pieces

| piece | what it is |
|---|---|
| maze families | `dfs` (perfect maze), `braid` (dfs with loops), `rooms` (rooms and corridors) for **training**; `caves` (cellular automata) and `spiral` (concentric rings or serpentines) **held out** and never trained on. Every level is BFS-verified. Start and exit are ≥6 steps apart. |
| environment | vectorised numpy, 4 absolute moves, 200-step limit. Observation: a 5×5 wall window, a 5×5 exit window (the exit is invisible until it is within 2 cells), the previous action, and a "bumped" flag. Reward is `1 - 0.9 t/T` on escape. Training adds a 0.01 first-visit bonus as shaping. |
| scripted baselines (same 5×5 view, same step limit) | `random_walk`; `wall_follow` (left-hand rule); `tremaux` (depth-first exploration with perfect memory of visited cells). All three head for the exit by the shortest known path once they have seen it. |
| reference | `bfs_oracle`: full map, shortest path (efficiency 1 by definition) |
| learned navigator | MLP encoder → GRU(128) → policy/value heads, PPO in MLX (`train_solver.py`) |
| designer | (μ+λ)-style evolution over wall layouts plus start/exit (`dungeon/designer.py`). Fitness is the navigator's regret against BFS, `R(BFS) - R(solver)`, averaged over 8 sampled rollouts. Mutations are rejected unless the exit stays reachable. |
| curriculum | fine-tune the navigator on 50% family levels + 50% designer levels, against a matched-compute control fine-tuned on family levels only. Three rounds, 6 training-seed chains per arm (seed 0 in `pipeline2.sh`, seeds 11-15 in `pipeline3.sh`). |
| ablation | `--memoryless`: the same PPO with the GRU state zeroed before every step |

## How to run

All commands run from this folder and use the shared venv.

```bash
# smoke test (< 1 min): unit tests + tiny PPO run + tiny designer run + quick eval
./smoke.sh
../.venv/bin/python -m pytest -q tests      # unit tests only (incl. JS-vs-Python parity via node)

# the full experiment, under the shared GPU lock. Finished stages are skipped, so a rerun resumes.
# Each script stops starting new stages after 36 min.
../bin/gpu_run bash pipeline.sh  > logs/pipeline.out  2>&1   # round 1: v1, designer, curriculum vs control (took 6 min)
../bin/gpu_run bash pipeline2.sh > logs/pipeline2.out 2>&1   # rounds 2-3, seed replicates, memoryless ablation,
                                                             # exploitability runs, min-dist-12 designer (took 33 min)
# seed replication of curriculum rounds 1-3 (one chain per seed; ~5-9 min each on an idle GPU)
for s in 11 12 13 14 15; do ../bin/gpu_run bash pipeline3.sh $s > logs/pipeline3_r$s.out 2>&1; done
../.venv/bin/python evaluate.py --out eval_seeds --designer-runs v1_s1 v1_md12_s7 ctl1_s2 --n-arch 372 \
  --solvers solver_v1 solver_{ctl,cur}{1,2,3} solver_{ctl,cur}{1,2,3}_r1{1,2,3,4,5}   # bash brace expansion
../.venv/bin/python plot_seeds.py        # -> media/curriculum_rounds_seeds.png

# individual stages
../.venv/bin/python train_solver.py --name solver_v1 --steps 12000000 --envs 512 --minibatches 2
../.venv/bin/python evolve_designer.py --solver checkpoints/solver_v1 --out v1_s1 --seed 1 --gens 100
../.venv/bin/python evolve_designer.py --solver wall_follow --out wall_follow_s0 --gens 100
../.venv/bin/python evaluate.py          # -> results/eval.json, results/eval_tables.md
../.venv/bin/python train_solver.py --name solver_ff --memoryless --steps 12000000 --envs 512 --minibatches 2
../.venv/bin/python evolve_designer.py --solver checkpoints/solver_v1 --out v1_md12_s7 --seed 7 --min-dist 12
../.venv/bin/python make_curriculum.py results/curriculum_r1.npz v1_s0   # designer archive -> curriculum levels
../.venv/bin/python evaluate.py --tables-only --out eval_r2             # rebuild markdown from json
../.venv/bin/python make_media.py        # -> media/*.png, media/escape.gif, media/before_after.gif
../.venv/bin/python export_web.py        # -> media/demo.html (embeds the GRU weights + preset dungeons)
```

`train_solver.py` resumes on its own: rerunning the same command continues from
`checkpoints/<name>/`, which holds the weights, Adam state, update count, numpy RNG state, and
config.

## Layout

```
dungeon/mazes.py          families, BFS, level statistics (dead ends, loops, ...)
dungeon/env.py            vectorised partial-view maze env
dungeon/scripted.py       random walk / wall follower / Trémaux (numba) + metrics (success, efficiency, SPL)
dungeon/policy.py         GRU actor-critic (MLX) + batched runner
dungeon/rollout.py        run a policy on a fixed batch of levels
dungeon/data.py           cached seeded train pool / test sets (data/*.npz)
dungeon/designer.py       mutation, BFS constraint, regret fitness, evolution loop
dungeon/web_core.js       JS port of BFS, scripted solvers and the GRU forward pass (for demo.html)
dungeon/demo_template.html
train_solver.py           PPO training (checkpoint / resume)
evolve_designer.py        designer runs -> results/designer_<name>.{npz,json}, logs/designer_<name>.jsonl
evaluate.py               all solvers × all level sets
make_media.py             gallery, nightmares, GIF, curves
export_web.py             builds media/demo.html
pipeline.sh, pipeline2.sh round 1 / rounds 2-3 + ablations, resumable
pipeline3.sh SEED         one extra seed chain of curriculum rounds 1-3 (own designer runs + archives), resumable
plot_seeds.py             per-seed curriculum-vs-control plot
make_curriculum.py        designer archives -> curriculum level file
smoke.sh, tests/          smoke + unit tests
checkpoints/solver_*/     weights.safetensors, opt.safetensors, meta.json, policy.json
                          (v1 = families; ctlN/curN = control/curriculum round N; *_r11.._r15 = extra seed
                           chains (only round-3 weights of new chains are committed, see .gitignore);
                           ff = memoryless ablation; smoke_solver = from smoke.sh)
results/                  eval.json/eval_tables.md (round 1), eval_r2.json/eval_r2_tables.md (round 2+),
                          eval_seeds.json/eval_seeds_tables.md (6 seed chains x 3 rounds), curriculum_r{2,3}_rS.npz,
                          designer_<run>.{npz,json} populations + archives, curriculum_r*.npz
media/                    demo outputs
```
