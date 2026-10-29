"""Turn results/tournament.json + the training log into markdown tables (results/tables.md) and plots.

  ../.venv/bin/python make_report.py
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/tournament.json"))
LOG = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/main_log.jsonl"
out = []


def pct(k, n):
    return f"{100 * k / n:.1f}%"


def ci(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


out.append("## Round robin (row = side A; W/L/D from A's view; 95% Wilson CI on A's win and loss rates)\n")
out.append("| A | B | n | A win | 95% CI | A loss | 95% CI | draw | score (W-L)/n | tips A/B | ring-outs A/B | mean bout s |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in T["rr"]:
    out.append(f"| {r['a']} | {r['b']} | {r['n']} | {pct(r['W'], r['n'])} | {ci(r['win_ci'])} | {pct(r['L'], r['n'])} | "
               f"{ci(r['loss_ci'])} | {pct(r['D'], r['n'])} | {r['score']:+.3f} | {r['tip_a']}/{r['tip_b']} | "
               f"{r['out_a']}/{r['out_b']} | {r['mean_len'] * 0.04:.1f} |")

# standings: mean score across all opponents
ents = sorted({r["a"] for r in T["rr"]} | {r["b"] for r in T["rr"]})
score = {e: [] for e in ents}
for r in T["rr"]:
    score[r["a"]].append(r["score"]); score[r["b"]].append(-r["score"])
out.append("\n## Standings (mean score over the other 5 entrants; +1 = always wins)\n")
out.append("| entrant | mean score |")
out.append("|---|---|")
for e in sorted(ents, key=lambda e: -np.mean(score[e])):
    out.append(f"| {e} | {np.mean(score[e]):+.3f} |")

snaps = [f"snap{k}" for k in T["snap_idx"]]
lab = {s: f"{s} (upd {u})" for s, u in zip(snaps, T["snap_updates"])}
lab["ppo"] = f"final (upd {T['train_updates']})"
out.append("\n## Cross-play: final policy vs archived snapshots (final's view)\n")
out.append("| opponent | n | final W | L | D | score |")
out.append("|---|---|---|---|---|---|")
for r in T["xp"]:
    if r["a"] == "ppo":
        out.append(f"| {lab[r['b']]} | {r['n']} | {pct(r['W'], r['n'])} | {pct(r['L'], r['n'])} | {pct(r['D'], r['n'])} | {r['score']:+.3f} |")
out.append("\n## Snapshot vs snapshot score matrix (row's score vs column, (W-L)/n)\n")
M = {s: {t: "" for t in snaps} for s in snaps}
for r in T["xp"]:
    if r["a"] in M and r["b"] in M:
        M[r["a"]][r["b"]] = f"{r['score']:+.2f}"; M[r["b"]][r["a"]] = f"{-r['score']:+.2f}"
out.append("| | " + " | ".join(lab[s] for s in snaps) + " |")
out.append("|---" * (len(snaps) + 1) + "|")
for s in snaps:
    out.append(f"| {lab[s]} | " + " | ".join(M[s][t] or "-" for t in snaps) + " |")
out.append("\n## Snapshots vs scripts (snapshot's score; W/L/D %)\n")
scripts = ["charger", "flanker", "circler", "counter"]
out.append("| snapshot | " + " | ".join(scripts) + " |")
out.append("|---" * 5 + "|")
for s in snaps:
    cells = []
    for sc in scripts:
        r = [x for x in T["snapvs"] if x["a"] == s and x["b"] == sc][0]
        cells.append(f"{r['score']:+.2f} ({pct(r['W'], r['n'])}/{pct(r['L'], r['n'])}/{pct(r['D'], r['n'])})")
    out.append(f"| {lab[s]} | " + " | ".join(cells) + " |")

out.append("\n## Robustness (training: ring R 0.9-1.1 m, friction x0.8-1.2). PPO's score vs each script; "
           "last column = mean decisive % (wins + losses) of the 6 script-vs-script pairs (how decisive the game is under that shift)\n")
conds = sorted({(r["R"], r["fric"]) for r in T["rob"]}, key=lambda c: (c != (1.0, 1.0), c))
out.append("| R | friction | " + " | ".join(f"PPO vs {s} W/L/D" for s in scripts) + " | PPO mean score | script pairs mean decisive % |")
out.append("|---" * (len(scripts) + 4) + "|")
for R, f in conds:
    rs = [[x for x in T["rob"] if x["R"] == R and x["fric"] == f and x["b"] == s][0] for s in scripts]
    ss = [x for x in T["rob_s"] if x["R"] == R and x["fric"] == f]
    dec = np.mean([(x["W"] + x["L"]) / x["n"] for x in ss])
    out.append(f"| {R} | {f} | " + " | ".join(f"{pct(r['W'], r['n'])}/{pct(r['L'], r['n'])}/{pct(r['D'], r['n'])}" for r in rs)
               + f" | {np.mean([r['score'] for r in rs]):+.3f} | {100 * dec:.1f}% |")

out.append("\n## Diversity: behaviour descriptors (20 bouts vs tuned charger, same starts for every policy) and action disagreement\n")
out.append("| policy | mean fwd cmd | mean abs turn cmd | frac reversing | mean own edge margin (m) | mean opp dist (m) |")
out.append("|---|---|---|---|---|---|")
for s in snaps + ["ppo"]:
    b = T["behaviour"][s]
    out.append(f"| {lab[s]} | {b['mean_fwd']:+.2f} | {b['mean_abs_turn']:.2f} | {b['frac_reverse']:.2f} | "
               f"{b['mean_edge_margin']:.2f} | {b['mean_opp_dist']:.2f} |")
ad = T["action_disagreement"]
cons = [ad.get(f"{snaps[i]}|{snaps[i + 1]}") for i in range(len(snaps) - 1)]
out.append(f"\nMean |Δ action| between policies on a fixed probe set of {T['probe_n']} observations "
           f"(actions in [-1,1], so 0 = identical, ~0.67 = unrelated uniform-random actions):\n")
out.append("- consecutive snapshots: " + ", ".join(f"{lab[snaps[i]]}→{snaps[i + 1]}: {c:.3f}" for i, c in enumerate(cons)))
out.append(f"- all snapshot pairs mean: {np.mean([v for k, v in ad.items() if 'ppo' not in k]):.3f}; "
           f"max: {max(v for k, v in ad.items() if 'ppo' not in k):.3f}")
out.append(f"- final vs first snapshot: {ad.get(snaps[0] + '|ppo', float('nan')):.3f}")
open("results/tables.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))

# training curve
L = [json.loads(l) for l in open(LOG)]
x = [d["elapsed"] / 60 for d in L]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for k, c in (("charger", "#c0392b"), ("flanker", "#2471a3"), ("circler", "#229954")):
    ax[0].plot(x, [d[f"win_{k}"][0] if d[f"win_{k}"][0] is not None else np.nan for d in L], c=c, label=f"win vs {k}")
    ax[0].plot(x, [d[f"win_{k}"][1] if d[f"win_{k}"][1] is not None else np.nan for d in L], c=c, ls=":", label=f"loss vs {k}")
ax[0].set_xlabel("training minutes"); ax[0].set_ylabel("rate (last 300 bouts, stochastic policy)")
ax[0].legend(fontsize=7, ncol=2); ax[0].set_title("training-time win/loss vs training scripts")
ax[1].plot(x, [d["kl"] for d in L], lw=0.6, label="approx KL")
ax[1].plot(x, [d["clipfrac"] for d in L], lw=0.6, label="clip frac")
ax[1].set_xlabel("training minutes"); ax[1].legend(fontsize=7); ax[1].set_title("PPO diagnostics")
plt.tight_layout(); plt.savefig("media/train_curve.png", dpi=110)
print("wrote results/tables.md, media/train_curve.png")
