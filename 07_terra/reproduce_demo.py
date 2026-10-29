"""Reload the saved checkpoints and regenerate every demo artifact (no training, ~1 min).

  ../.venv/bin/python reproduce_demo.py
media/planner_routes.png  (learned heuristic checkpoint: checkpoints/hnet.*)
media/fog_priors.png, media/fog_walk.gif  (learned prior checkpoint: checkpoints/prior.*)
media/bridge_collapse.png/.gif, media/demo.html
"""
import numpy as np
from terra import terrain, cost, scenarios
import eval_heuristic, run_fog, run_bridge, build_demo, train_prior

model, cfg = eval_heuristic.load_model("hnet")  # the evaluated model (base 32)
print("loaded heuristic checkpoint, step", cfg.get("step"), "val |logratio err|", cfg.get("val_mae"))
eval_heuristic.routes_figure(model, "media/planner_routes.png")
eval_heuristic.field_figure(model, "media/heuristic_fields.png")
eval_heuristic.failure_figure(model, "media/learned_failure.png", 900062, 4759, 1734)

import os
if os.path.exists("checkpoints/prior.safetensors"):
    prior, pcfg = train_prior.load_prior("prior")
    print("loaded prior checkpoint, step", pcfg.get("step"), "val water IoU", pcfg.get("val_water_iou"))
    seed, s, g = run_fog.FIG_QUERY
    T = terrain.generate(seed, 96)
    priors, pw = run_fog.get_priors(T, prior)
    run_fog.figure(T, s, g, priors, pw, "media/fog_priors.png", "media/fog_walk.gif", 5)
else:
    print("NOTE: checkpoints/prior.safetensors missing -> skipping fog figures")

T = terrain.generate(900003, 96); G = cost.Graph(T["elev"], T["kind"])
q = scenarios.pick_bridge_query(T, G, np.random.default_rng(1), min_dist=50)
run_bridge.figure(T, q[0], q[1], 6, "media/bridge_collapse.png", "media/bridge_collapse.gif")
build_demo.main()
print("done")
