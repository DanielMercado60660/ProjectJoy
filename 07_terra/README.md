# 07 · Terra: the RL Pathfinder

A terrain-routing playground. It generates procedural maps (fBm elevation, rivers that follow
downhill flow, lakes, bridges, forest and rock) and defines **one explicit cost**. Every planner
uses that cost: Dijkstra, A*, weighted A*, **D\* Lite** (incremental replanning), and
**A\* with a learned heuristic** (an MLX U-Net). There are two "the world changed" scenarios:

1. **Bridge collapse.** The agent follows its optimal route, and the bridge it is about to
   cross disappears. D\* Lite repairs its search tree. A\* and Dijkstra replan from scratch.
2. **Fog of war (partial information).** The agent knows the elevation map, but terrain type is
   revealed only within a sensing radius. The same D\* Lite agent runs with three priors for
   unknown cells: optimistic (free space), a **learned prior** (MLX U-Net, elevation → P(water),
   expected cost), and the generator's own hydrology model (a *reference*, not a fair
   competitor: it is exactly how the maps were generated).

Media: `media/demo.html` (interactive), `media/demo_session.gif` (a recorded scripted session of
the demo), `media/planner_routes.png`, `media/bridge_collapse.{png,gif}`,
`media/heuristic_tradeoff.png`, `media/heuristic_fields.png`, `media/fog_priors.png`,
`media/fog_walk.gif`.

The demo is **`media/demo.html`**, a self-contained, editable terrain editor that needs no
network. You paint water, bridges, forest, rock, or raise and lower the ground, and the
Dijkstra / A\* / weighted A\* / D\* Lite routes update live. Each edit reports how many nodes
D\* Lite repaired and how many A\* expanded from scratch. You can walk the agent and collapse
the next bridge on its route. The JS planners are a port of the Python ones, and a test checks
them for exact parity.

## The cost (shared by everything)

```
c(a, b) = d · m̄ · (1 + |z_b − z_a| / d)
  d  = 1 (orthogonal) or √2 (diagonal)
  m̄  = mean multiplier of a and b: grass 1, forest 2, rock 3, bridge 1; water impassable
  z  = elevation × 24 (cell units)
  no corner cutting (a diagonal step needs both side cells passable)
```
The cost is symmetric and ≥ d, so the octile distance is an admissible, consistent heuristic.
It is defined once in `terra/cost.py` and ported to `web/terra_core.js`.
`tests/test_js_parity.py` runs the JS in node and checks that edge costs match Python
bit-for-bit.

## How to run

All commands run from this folder and use the shared venv.

```bash
# tests + tiny end-to-end pipeline (~1 min)
./smoke.sh
../.venv/bin/python -m pytest -q            # unit tests only (~15 s)
../.venv/bin/python stress_dstar.py         # D* Lite vs Dijkstra, 4610 replans (~30 s)

# learned heuristic: data (3000 train maps x 4 goals, exact cost-to-go labels) + training
../bin/gpu_run bash pipeline.sh             # make_dataset.py then train.py (~15 min)
../bin/gpu_run ../.venv/bin/python train_prior.py --steps 3000   # learned fog prior (~6 min)

# experiments (each writes results/*.json and media/*)
../.venv/bin/python run_bridge.py --maps 150 --per_map 2    # bridge collapse: D* Lite vs A*
../.venv/bin/python eval_heuristic.py --maps 100 --per_map 3  # learned h vs A*/WA* (held-out)
../.venv/bin/python run_fog.py --maps 100 --per_map 2       # partial information

# or everything after training in one GPU-slot reservation:
../bin/gpu_run bash pipeline2.sh            # eval, bridge, prior, fog, browser-sized net, demo
../.venv/bin/python summarize_results.py    # markdown tables from results/*.json

# demo
../.venv/bin/python build_demo.py           # -> media/demo.html (open in any browser)
../.venv/bin/python reproduce_demo.py       # reload checkpoints, redo all figures + demo
```
Open `media/demo.html#selftest` to run a scripted interaction in the page. It walks the agent,
collapses a bridge, and paints with every tool, checking D\* Lite against Dijkstra after each
change. It prints `SELFTEST PASS`.

## File layout

```
terra/terrain.py     procedural maps: fBm, priority-flood, D8 flow accumulation, rivers, lakes, bridges
terra/cost.py        THE cost definition, cost.Graph (adjacency + local rebuild after edits)
terra/planners.py    Dijkstra, A* / weighted A*, D* Lite (Koenig & Likhachev 2002, Fig. 3, with k_m)
terra/scenarios.py   bridge-collapse scenario + benchmark
terra/fog.py         partial-information scenario and priors
terra/learn.py       features, exact cost-to-go labels (scipy), MLX U-Net (HNet)
terra/viz.py         terrain rendering helpers
make_dataset.py      data/train.npz, data/val.npz (seeds 0..2999 train, 500000+ val; test 900000+)
train.py             learned heuristic training (checkpoint/resume)
train_prior.py       learned terrain prior for the fog scenario (checkpoint/resume)
eval_heuristic.py    held-out speed/optimality tradeoff, planner route figure
run_bridge.py        bridge-collapse benchmark + figure + GIF
run_fog.py           fog benchmark + figure + GIF
stress_dstar.py      D* Lite vs Dijkstra stress test (4610 replans)
summarize_results.py markdown tables from results/*.json
pipeline.sh / pipeline2.sh  data+training / all remaining runs, each in one gpu_run slot
reproduce_demo.py    reload checkpoints, regenerate figures + demo.html
build_demo.py        builds media/demo.html from web/*
web/terra_core.js    JS port: cost, Dijkstra/A*/D* Lite, generator (tested in node)
web/demo_ui.js       demo UI (painting, live replanning, agent, learned-h in a Web Worker)
web/hnet.js          JS forward pass of the MLX U-Net (float16 weights baked by build_demo.py)
tests/               pytest: planners, D* Lite vs Dijkstra under random edits, JS parity, scenarios
checkpoints/         hnet.* (evaluated learned heuristic, base 32), hnet16.* (browser-sized, baked
                     into demo.html), prior.* (learned fog prior); each with optimizer state + step + RNG
data/                generated training data (665 MB, regenerable with make_dataset.py; gitignored)
results/             JSON outputs of the experiments
media/               demo.html, PNGs, GIFs
```
