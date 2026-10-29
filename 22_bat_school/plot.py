"""Training curves and the pulses-vs-success trade-off plot."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = [json.loads(l) for l in open("checkpoints/bat_log.jsonl")]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
x = [r["steps"] / 1e6 for r in log]
for a, key, name in zip(ax, ["success", "pulses_per_step", "coll_per_step"],
                        ["success (training mazes, last 400 eps)", "pulses per step", "collisions per step"]):
    a.plot(x, [r[key] for r in log], color="#2d6fa3")
    a.set_title(name, fontsize=9); a.set_xlabel("env steps (M)", fontsize=8); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("media/training_curves.png", dpi=110)

ev = json.load(open("results/eval.json"))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for a, size in zip(axs, ["size6", "size7"]):
    t = ev["tables"][size]
    for prefix, color, label in [("wallfollow_k", "#999999", "wall follower, pulse every k"),
                                 ("bat_forced_k", "#e0883a", "learned moves, forced pulse every k")]:
        ks = sorted(k for k in t if k.startswith(prefix))
        a.plot([t[k]["pulses_per_step"] for k in ks], [t[k]["success"] for k in ks], "o-", color=color, label=label)
        for k in ks:
            a.annotate(k[-2:], (t[k]["pulses_per_step"], t[k]["success"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    if "wallfollow_event" in t:   # reviewer baseline: scripted "pulse only when a decision is due"
        e = t["wallfollow_event"]
        a.plot([e["pulses_per_step"]], [e["success"]], "D", ms=8, color="#555555", label="wall follower, pulse only when a decision is due")
    b = t["bat_learned"]
    a.plot([b["pulses_per_step"]], [b["success"]], "*", ms=14, color="#2d6fa3", label="learned (chooses when to pulse)")
    r = t["random_walk"]
    a.axhline(r["success"], color="#cc4444", ls=":", lw=1, label="random walk")
    a.set_xlabel("pulses per step"); a.set_ylabel("completion on held-out mazes")
    a.set_title(f"held-out {size[-1]}x{size[-1]} mazes (n={b['n']})", fontsize=9); a.grid(alpha=0.3)
axs[0].legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig("media/pulse_tradeoff.png", dpi=110)
print("wrote media/training_curves.png media/pulse_tradeoff.png")
