"""Plot the training log: ../.venv/bin/python plot_train.py logs/train.jsonl media/training_curve.png"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "logs/train.jsonl"
dst = sys.argv[2] if len(sys.argv) > 2 else "media/training_curve.png"
r = [json.loads(l) for l in open(src)]
x = [e["steps"] / 1e6 for e in r]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(x, [e["goal"] for e in r], label="goal", color="#2b6cb0")
ax[0].plot(x, [e["own_goal"] for e in r], label="own goal", color="#c53030")
ax[0].plot(x, [e["curriculum"] for e in r], label="curriculum (0 near goal -> 1 anywhere)", color="#888", ls="--")
ax[0].set_xlabel("env steps (M)"); ax[0].set_ylabel("fraction of last 400 training episodes")
ax[0].set_title("PPO vs scripted goalie (training episodes)"); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(x, [e["std"] for e in r], color="#2f855a")
ax[1].set_xlabel("env steps (M)"); ax[1].set_title("policy action std")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(dst, dpi=120)
print("wrote", dst)
