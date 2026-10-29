"""Build media/demo.html: a self-contained terrarium (sim + hand-written + learned brains in JS).

  ../.venv/bin/python make_demo_html.py --nets checkpoints/ppo1 checkpoints/ppo_nocompass checkpoints/es1
"""
import argparse, json, os
import numpy as np
import sim
from policy import Policy, load_weights

LABELS = {"ppo1": "learned: PPO shared policy", "ppo2": "learned: PPO + starvation penalty",
          "ppo_nocompass": "learned: PPO, no compass (early best)", "es1": "learned: evolution strategies"}

ap = argparse.ArgumentParser()
ap.add_argument("--nets", nargs="*", default=["checkpoints/ppo1"])
ap.add_argument("--out", default="media/demo.html")
a = ap.parse_args()

layouts = []
for seed, fam in [(50_001, "heldout"), (50_003, "heldout"), (2, "train"), (50_007, "heldout"), (5, "train"), (50_010, "heldout")]:
    obst, food, nest, _ = sim.make_layout(seed, fam)
    layouts.append({"obst": obst.ravel().astype(int).tolist(), "food": food.ravel().astype(int).tolist(),
                    "nest": [int(nest[0]), int(nest[1])]})
nets = {}
for path in a.nets:
    wpath = os.path.join(path, "model_best.safetensors")  # same rule as evaluate.load_policy
    if not os.path.exists(wpath):
        wpath = os.path.join(path, "model.safetensors")
    if not os.path.exists(wpath):
        print("skip", path); continue
    cfg = json.load(open(os.path.join(path, "config.json")))
    m = load_weights(Policy(hid=cfg.get("hid", 128)), wpath)
    net = {"no_compass": bool(cfg.get("no_compass", False))}
    for name in ("a1", "a2", "move", "dep"):
        layer = getattr(m, name)
        net[name] = {"w": np.round(np.array(layer.weight), 4).tolist(), "b": np.round(np.array(layer.bias), 4).tolist()}
    nets[LABELS.get(os.path.basename(path.rstrip("/")), os.path.basename(path))] = net
notes = ("Layouts cycle through unseen worlds (held-out family: food 24-30 cells from the nest, long walls). "
         "The browser sim is a line-by-line port of sim.py; actions are sampled with Math.random, so runs "
         "differ from the Python evaluation, but the rules and the policies are the same.")
data = "const DATA = " + json.dumps({"layouts": layouts, "nets": nets, "notes": notes}, separators=(",", ":")) + ";"
html = open("demo_template.html").read().replace("/*__DATA__*/", data)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, "w").write(html)
print("wrote", a.out, f"{len(html) / 1e3:.0f} kB", "nets:", list(nets))
