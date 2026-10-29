# R5 Turn 2: remove the transfer bottleneck, then test error-model progress

**Hypotheses.** Composable replay knowledge will convert better exploration into more task
success. A calibrated log-error model with adaptive reward scaling will retain useful discovery
while reducing TV capture; failure of either claim remains a reportable result.

**Evidence.** Turn 1's perfect frontier explorer got 9 rooms/8 mechanisms but only 2/7 tasks.
Its recorded tour needed 150–295 actions for the other five tasks (first TV trial); increasing
replay to 512 recovered every plan. The remembered-map graph preserves acquisition history,
so it cannot freely combine paths. Old LP also failed exploration; its shrinking raw-MSE
signal was divided by a cumulative return scale still dominated by early training. These are
separate failures. Policy entropy was not logged; the new runner probe supplies it.

**Transfer repair.** All methods still transfer exactly one 512-action observation/action replay.
A shared map builder records first-seen tiles and infers a switch/lever effect only when a
unique observed activation explains a later door/lamp change. Ambiguous links are omitted.
A common symbolic planner composes navigation, key pickup, key-consuming door unlock,
button/lever use and box pushes. It receives generic affordances, never the simulator map or
unobserved mechanism links. The physical task resets; key and mechanism states are simulated
in the plan. Unknown tiles are blocked. Box pushes must increase navigable area, which is an
access heuristic, not a general Sokoban solver. First-seen dynamic tiles may still misestimate
reset state when their original state was never observed: actual execution scores such errors.

Keep 128 physical task actions; allow 100,000 search expansions counting navigation cells and
macro states. Same cap, knowledge format and seeded fallback for all methods; failures cost 128.
An explicit oracle supplies the true initial map and causal links to the **same planner**, solely
as a privileged diagnostic. Oracle and complete-replay witness plans now solve all 14 world/task
pairs in 5–94 actions. Ordinary frontier validation solves 28/28 tasks in each house. These are
planner checks, not new learned headline results. See `results/r5/t2_diagnosis.json` and
`results/r5/t2_planner_validation_{plain,tv}.json`.

**New curiosity method (`lpm`).** Following the error-model formulation in
[LPM, equations 1–3](https://arxiv.org/html/2509.25438v2#S3), use
`g(o,a) - log(MSE(f(o,a), phi(o_next)) + 1e-6)`. The separate `g` regresses a bounded queue
of detached **pre-update** log-error labels; reward is scored before fitting current data.
The fixed random embedding is an adaptation, not a full paper reproduction. Raw observations
include TV pixels. Signed rewards are symmetrically clipped; no positive rectification or
TV mask. Warm up window+2 updates, then divide by EMA reward SD with floor .1 and cap ±5.
Log entropy, calibration, signed reward fractions, scale and TV/non-TV means. The original
`lp` remains unchanged and reloadable; improvement over its old budget is not a matched ablation.

**Validation only.** LPM window {1,4} × entropy {.001,.01}; 1,048,576 steps/world/candidate,
pilot seed 619. Frontier mechanism priority {0,6,12} × frontier priority {0,2}. Select by the
existing discovery score on seeds 64600–64603, not by task performance. Other learned methods
retain their settings. No warm starts. New training seeds 610–612; headline seeds 85400–85415
and policy RNG 83519; old 72500–72503 are secondary. Held-out randomness, not novel layouts.

**Budget and metrics.** Match round 1's exact 5,996,544 steps/method/world/seed, including
actual simulator steps for nonlearning baselines. Use 128 environments for every training cell.
Measured Turn-1 busy totals imply ~53 min/seed if merely doubled sequentially; instead allow two
isolated processes with two threads each under one GPU slot, across at most three 40-minute
resumable waves. Preparation ≤12 min; total ≤132 min. Speedup and neural performance remain
unmeasured. Battery minimum: one seed, 1,048,576 steps, reduced tuning, ≤35 min total.
Report discovery/coverage/TV, task success/capped and success-only steps, planning coverage,
per-task outcomes and empty replay. Means/SD over three fitted training seeds; baseline repeats
are not independent replicas. No incomplete or unequal-budget cells enter the headline.
