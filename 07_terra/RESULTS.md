# Terra: results

Every number here comes from a real run on **held-out maps**. Training used map seeds
0–2999, validation 500000+, and all tests 900000+. `summarize_results.py` prints the tables
from `results/*.json`. Grids are 96×96 and 8-connected, with one shared cost (see README).

The machine was an M5 Max shared with 12 other builders, so **wall-clock latencies are noisy**.
Node-expansion counts don't depend on load. All planners are plain Python + `heapq` written
in the same style. Bridge-replan latencies are best-of-3 with the GC off; other timings are
single runs.

## What was built (first satisfying version: done)

- **An editable terrain grid where planners pick visibly different routes and replan after a
  bridge disappears:** `media/demo.html`, a self-contained page with no network use. A
  recorded session is in `media/demo_session.gif`. You paint water, bridges, forest, rock, or
  raise and lower the ground. Dijkstra, A\*, weighted A\*, D\* Lite and learned-h A\* update
  live. The **Collapse next bridge** button with **▶ Walk** shows D\* Lite repairing its tree
  while A\* re-searches from scratch.
- **Static views:** `media/planner_routes.png` (same map, 6 planners, expanded-node haze),
  `media/bridge_collapse.png/.gif`, `media/fog_priors.png`, `media/fog_walk.gif`.
- **Correctness checks:** 37 pytest tests.
  - D\* Lite matches Dijkstra after random terrain edits and agent moves. A separate stress run
    found 0 mismatches in 4,610 replans on 300 maps (`stress_dstar.py`, about 30 s).
  - A\* matches Dijkstra, and the octile heuristic is checked admissible on every cell.
  - Dijkstra matches scipy's `csgraph` as an independent oracle.
  - The **JS port matches Python bit-for-bit** on every edge cost and on optimal costs; JS
    D\* Lite matches JS Dijkstra under edits.
  - The **JS U-Net matches MLX** (float16 weights; mean |Δr| < 0.01).
  - `demo.html#selftest` scripts the UI and checks 18 D\* Lite = Dijkstra points (PASS).

## 1. Bridge collapse: D\* Lite vs replanning from scratch

The agent walks its optimal route. When it is 6 steps before the first bridge on that route,
the bridge turns to water. Both agents replan from the same cell. Sample: 150 held-out maps × 2
queries = 300 collapses.

| planner | initial plan: expanded (median) | replan: expanded (median / mean) | replan ms (median / mean) |
|---|---|---|---|
| D\* Lite (incremental repair) | 3164 | **556** / 1171 | 2.39 / 5.08 |
| A\* from scratch (octile h) | 3403 | 2130 / 2587 | **1.76** / 2.70 |
| Dijkstra from scratch | 6135 | 4508 / 4573 | 4.57 / 4.57 |

- **D\* Lite and A\* agree.** D\* Lite's repaired cost equals A\*'s fresh optimum in **300/300**
  cases.
- **Failure rate is 1/300 (0.3%).** In that one case the collapse cut the goal off completely,
  and both planners correctly report "no path".
- **D\* Lite expands fewer nodes in 92% of replans** (3.8× fewer at the median). It is
  **faster in wall-clock in only 33%**. The reason is cost per expansion: 4.3 µs for D\* Lite vs
  1.0 µs for A\*. Each D\* Lite expansion re-keys all 8 neighbours. A raise (an underconsistent
  node) also recomputes rhs by scanning each neighbour's successors.
- **The win depends on the "shadow" of the bridge.** When the detour is small (below the median
  detour ratio), D\* Lite repairs 125 nodes against A\*'s 1311 (10×). When the detour is large,
  the collapse invalidates a big part of the backward tree: 1244 vs 2654, barely 2×, and slower
  in Python.
- **Late information costs travel.** The planned route cost 156.8. The realized cost (walk +
  detour) was 203.3. An omniscient planner that knew the bridge was gone would have paid 180.1.
  Realized/omniscient averages **1.134**.
- **Initial-plan latency is not comparable.** D\* Lite's initial ms (8.9) is a single run while
  A\* is best-of-3, so compare expansions instead (3164 vs 3403).
- **The figure is a favourable example** (added by the reviewer). `media/bridge_collapse.png/.gif`
  (map 900003, the demo's "Lake crossing") shows 258 vs 3794 expansions, about 15×; the median
  across the 300 cases is 3.8×, and D\* Lite is slower in wall-clock in 67% of them.

## 2. Learned heuristic (MLX) vs admissible A\* and weighted A\*

**Model.** A U-Net (`terra/learn.py`, 1.97 M parameters, 8 input channels: terrain plus
goal-distance features) predicts the log-ratio of the true cost-to-go to the octile distance.
Labels are exact Dijkstra-from-goal fields: 3000 maps × 4 goals.

Training ran 8000 steps at batch 16 with AdamW and cosine decay, taking 12.1 min with
checkpoint/resume. Validation mean |log-ratio error| is 0.109. Predictions are clipped at ≥
octile, because the true cost is never below it, so the clip is free information.

**Use.** At query time A\* uses `h = octile + a·(h_learned − octile)`. This A\* never re-opens
closed nodes, for WA\* as well as learned-h. With an inconsistent heuristic, that is one of the
two sources of suboptimality; the other is overestimation. Sample: **298 held-out
queries** (100 maps × 3; 2 unreachable queries skipped).

| method | expanded / A\* (mean) | expanded (median) | cost above optimal: mean | p95 | max | search ms (median) | total ms incl. inference |
|---|---|---|---|---|---|---|---|
| Dijkstra | 2.231 | 5312 | 0 | 0 | 0 | 5.29 | 5.29 |
| A\* (octile, admissible) | 1.000 | 2486 | 0 | 0 | 0 | 2.60 | 2.60 |
| WA\* w=1.25 | 0.725 | 1890 | 0.43% | 1.41% | 4.85% | 1.68 | 1.68 |
| WA\* w=1.5 | 0.548 | 1418 | 1.77% | 4.34% | 9.34% | 1.23 | 1.23 |
| WA\* w=2 | 0.346 | 732 | 4.94% | 11.70% | 36.6% | 0.63 | 0.63 |
| WA\* w=3 | 0.206 | 261 | 13.59% | 31.31% | 99.6% | 0.22 | 0.22 |
| Learned a=0.5 | 0.580 | 1426 | 1.36% | 2.98% | 78.3% | 1.64 | 14.78 |
| Learned a=0.75 | 0.412 | 951 | 2.77% | 7.13% | 80.2% | 0.90 | 13.86 |
| **Learned a=1** | **0.296** | 580 | **4.65%** | 10.97% | 86.4% | 0.54 | 13.44 |
| Learned a=1.5 | 0.204 | 338 | 8.05% | 18.92% | 101.5% | 0.28 | 13.06 |
| Oracle h\* (exact cost-to-go) | 0.037 | 74 | 0 | 0 | 0 | 0.14 | – |

The trade-off curve is in `media/heuristic_tradeoff.png`; the predicted fields are in
`media/heuristic_fields.png`.

**What this shows (honestly):**
- **The fair baseline is weighted A\*,** which just inflates the octile heuristic. Comparing to
  plain A\* alone would flatter the network.
- **At the low-suboptimality end it only matches.** Learned a=0.5 (0.58×, +1.4%) sits on top
  of WA\* w=1.5 (0.55×, +1.8%).
- **In the middle and fast end it beats the WA\* curve.**
  - At about +4.7–4.9% mean cost, learned a=1 expands 0.296× A\* vs 0.346× for WA\* w=2: 15%
    fewer nodes, with a median of 580 vs 732.
  - At about 0.20× expansions, the learned heuristic costs +8.1% against +13.6% for WA\* w=3.
- **The tail is the catch.** WA\* has a guarantee (cost ≤ w × optimal); the learned
  heuristic has none. Its worst query is **+86%** even at a=1, and +78% at a=0.5. See the
  failure analysis below.
- **End-to-end it is slower.** One MLX forward pass took a median 12.6 ms on the shared GPU,
  while an entire admissible A\* search takes 2.6 ms. So for a *single* query on a 96×96 grid,
  learned-h A\* is about 5× slower in wall-clock than plain A\*.
  - The forward pass gives h for **every start** to that goal, so it pays off only when amortized
    (many agents or queries per goal), on larger maps, or where node expansion is expensive (the
    expansion count is what this table measures).
  - The perfect heuristic (0.037×) shows lots of headroom left.

**The good kind of failure: the network misjudges connectivity.** `media/learned_failure.png`
shows held-out map 900062, query 4759→1734. The learned h is about **2× too high** over the
whole region east of a river. The network "believed" that region connected to the goal only
via a long southern detour, and missed the open passage north of the lake. A\* trusted it, never
explored north, and returned a route costing **204 instead of 110 (+86%)**. This is the
expected weakness of a CNN predicting a global, topological quantity (which side of a river
connects where) from local features: most of the field is right, and one wrong wall costs a lot.

## 3. Partial information ("fog of war"): where learning helps

**Setup.**
- **What the agent knows.** It knows the elevation (a topo map). Terrain type (water, bridges,
  forest, rock) is revealed only within a sensing radius R of where it has been.
- **How it plans.** The same D\* Lite agent and the same true cost run with three priors for
  unknown cells:
  - **optimistic:** unknown cells count as grass (the textbook free-space assumption).
  - **learned:** an MLX U-Net (`train_prior.py`, 3000 steps, 8.3 min) predicts P(water)
    and the expected land multiplier from elevation and slope only.
  - **hydrology:** the generator's own river/lake model run on the known elevation.
    *Caveat: this is literally how the maps were made, so it is an oracle-style reference,
    not a fair competitor.*
- **Soft water.** Predicted water is "soft" (cost ×12, not impassable), so a prior can never
  make the goal look unreachable.
- **Learned prior quality (validation).** Water IoU 0.43, accuracy 93.8%, land-multiplier
  MAE 0.45.
- **Sample.** 100 held-out maps × 2 queries, R=5.

| prior | failure rate | travel / optimal: mean | median | p95 | cells observed (mean) | replans (mean) |
|---|---|---|---|---|---|---|
| optimistic | 0% | 1.556 | 1.249 | 2.975 | 1294 | 89 |
| **learned (MLX)** | 0% | **1.345** | **1.126** | 2.207 | 1177 | 100 |
| hydrology (oracle-ish) | 0% | 1.242 | 1.079 | 2.065 | 1129 | 75 |

- **The learned prior closes about two-thirds of the gap.** It cuts mean excess travel from
  +55.6% (optimistic) to +34.5%; the generator-model reference reaches +24.2%.
- **It is not uniformly better.** It wins on 62% of queries and loses on 38%. Likely causes,
  not checked one by one: avoiding a valley that turns out to be dry, or detouring around a
  predicted river that a hidden bridge actually crosses.
- **The radius sweep (40 maps × 1 query, mean travel/optimal) shows the same ordering.** The
  learned prior's advantage shrinks as sensing grows, as it should:

| sensing radius | optimistic | learned | hydrology |
|---|---|---|---|
| 2 | 1.745 | 1.494 | 1.243 |
| 4 | 1.691 | 1.423 | 1.172 |
| 8 | 1.462 | 1.246 | 1.182 |
| 12 | 1.327 | 1.178 | 1.113 |

**Replanning compute under fog (D\* Lite's home turf).** Changes are always next to the agent,
which is exactly where D\* Lite's backward search tree is cheapest to repair.

| prior | D\* Lite: expansions per episode (mean) | A\* from scratch after every change | D\* Lite ms / episode (median) | A\* ms / episode (median) | per replan (median): D\* Lite vs A\* |
|---|---|---|---|---|---|
| optimistic | 8,102 | 76,737 | 25 | 64 | 0.37 vs 0.93 ms |
| learned | 7,432 | 199,883 | 29 | 202 | 0.36 vs 2.20 ms |
| hydrology | 5,988 | 97,812 | 22 | 74 | 0.35 vs 1.13 ms |

With non-uniform belief costs, the octile heuristic is weak, so A\* re-search gets expensive
(27× more expansions with the learned prior). D\* Lite only repairs the area around the agent.

`media/fog_priors.png` is a **hand-picked illustrative case** (map 900028), chosen from
`results/fog.json` because the priors differ there. The optimistic agent walks into a
lake-bounded pocket and travels 4.69× optimal, the learned prior 1.09×, hydrology 1.01×. The
aggregate table above is the honest summary. `media/fog_walk.gif` shows optimistic and learned
side by side.

## Measure items

| item | status |
|---|---|
| Travel cost | Measured: bridge collapse (realized/omniscient 1.134) and fog (table above). |
| Observation cost | **Partly measured.** Observation here is *passive* (everything within R is seen for free), so the reported "cells observed" is a footprint, not a paid cost. Costly *active* observations (the doc's "Go deeper") are **not measured**. |
| Failure rate | Bridge: 0.3% (goal truly cut off). Fog: 0% for all priors. Learned heuristic: 0 failures, since A\* stays complete with any finite h, but the cost tail is above. |
| Replanning latency | Bridge: D\* Lite median 2.39 ms vs A\* 1.76 ms. Fog: 0.37 vs 0.93 ms per replan. Expansions in the tables. |

## Surprises

1. **D\* Lite "wins" on expansions but often loses on the clock** in the bridge scenario. The
   textbook claim is about expansions. In Python, each D\* Lite expansion does about 4× more
   work. The scenario also matters: a bridge collapse ahead of the agent invalidates a large
   shadow of the backward tree. Under fog, where changes are local to the agent, D\* Lite wins
   on both counts.
2. **A heuristic net that is right "on average" can be very wrong on topology** (the +86% case
   above).
3. **The learned fog prior is mediocre at its own task** (water IoU 0.43) and still useful:
   travel drops from 1.56× to 1.35× optimal. Rivers are one cell wide, and flow accumulation is
   a global computation, so the net mostly learns "valleys and basins are wet". That is enough
   to stop the agent from betting on dry valley floors.

## Next steps

- A Web Worker that runs D\* Lite in JS for 256² maps, and real DEM tiles.
- Learned heuristic:
  - Add topological inputs (connected components of land, distance-to-bridge), or a
    multi-scale/global model, to fix the connectivity failure.
  - Train with an asymmetric loss that penalizes overestimates, to shrink the tail.
  - Batch inference across goals to amortize the forward pass.
- Fog:
  - Costly active observations ("scan" actions with a price) and a learned policy deciding
    when to scan, which is the doc's "Go deeper" question.
  - Calibrate the soft-water penalty.

## Gaps / caveats

- Nothing was missing from the venv.
- Wall-clock numbers come from a shared, loaded machine. The GPU queue (8 slots, about 25
  waiters) delayed the experiments by about 90 min but did not change their size.
- `hnet16` (base 16, 0.49 M parameters, used inside demo.html for speed) is a smaller sibling
  of the evaluated model. Its own held-out numbers are below.

## Appendix: the browser-sized heuristic (hnet16)

Same data and recipe, base width 16 instead of 32, 6000 steps (8.9 min). Validation |log-ratio error| is
0.102 (vs 0.109 for the base-32 model). It was evaluated on the same 298 held-out queries
(`results/heuristic_hnet16.json`):

| method | expanded / A\* | cost above optimal: mean | p95 | max |
|---|---|---|---|---|
| Learned a=0.5 (hnet16) | 0.583 | 1.14% | 3.25% | 19.1% |
| Learned a=1.0 (hnet16) | 0.299 | 4.87% | 12.40% | 88.2% |
| Learned a=1.5 (hnet16) | 0.209 | 8.73% | 21.65% | 106.2% |

- **Capacity is not the bottleneck.** A model with 4× fewer parameters lands on the same curve.
- **Same tail.** The +88% worst case is the same connectivity failure mode.
- **In the browser** (JS forward pass in a Web Worker, headless Chrome on this machine) it takes
  about 0.9–1.1 s per map+goal, against about 3.6 s for the base-32 model. That is why the demo
  bakes hnet16 (1.6 MB page instead of 5.4 MB).
- **Timing noise.** The search-ms columns in that JSON were measured during a busier period (A\*
  median 6.8 ms vs 2.6 ms in the main run). This shows how noisy wall-clock is here; the
  expansion counts are identical.
