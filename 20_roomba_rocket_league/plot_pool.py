"""Follow-up (round 3): pool training curve, goal rate per pool opponent (training episodes, last 400 mixed).

    ../.venv/bin/python plot_pool.py logs/pool_mixed.jsonl media/pool/pool_training_curve.png
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = [json.loads(l) for l in open(sys.argv[1])]
r4 = [json.loads(l) for l in open("logs/run4_tanh_relp.jsonl")]
fig, ax = plt.subplots(figsize=(7, 3.8))
x = [e["steps"] / 1e6 for e in r]
for k, c in [("goalie", "#2b6cb0"), ("goalie_rand", "#805ad5"), ("chaser", "#dd6b20")]:
    ax.plot(x, [e.get(f"goal_{k}") for e in r], label=f"pool run: goal vs {k}", color=c, lw=1)
ax.plot(x, [e["conceded"] + e["own_goal"] for e in r], label="pool run: conceded+own (all)", color="#c53030", lw=1)
ax.plot([e["steps"] / 1e6 for e in r4], [e["goal"] for e in r4], label="run 4: goal vs goalie", color="#999", lw=1, ls="--")
ax.plot(x, [e["curriculum"] for e in r], label="pool run curriculum", color="#555", ls=":")
ax.set_xlabel("env steps (M)"); ax.set_ylabel("fraction of recent training episodes")
ax.set_title("Opponent-pool PPO vs run 4 (training episodes, 10 min wall each)")
ax.legend(frameon=False, fontsize=7); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=120)
print("wrote", sys.argv[2])
