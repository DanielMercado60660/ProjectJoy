"""Plot for the round-3 structural follow-up -> media/struct_followup.png (reads results/struct/eval.json)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(os.path.join(HERE, "results", "struct", "eval.json")))
levels = list(E["levels"])
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
cols = {"0.25MPa": "C0", "0.5MPa": "C2", "0.125MPa": "C3"}
for k in levels:
    L = E["levels"][k]
    sp = sorted(L["hand_best_per_span"], key=lambda s: float(s[:-2]))
    ax[0].plot([float(s[:-2]) for s in sp], [L["hand_best_per_span"][s] for s in sp], "o-", c=cols.get(k, "k"),
               label=f"hand grid, sigma {k}")
    sc = L["cma_seeds"]
    ax[0].scatter(sc["span_cm"], sc["train_fit"], marker="*", s=120, c=cols.get(k, "k"), edgecolor="k", zorder=5)
ax[0].set_xlabel("span (cm)"); ax[0].set_ylabel("training mean range (m)")
ax[0].set_title("best training range by span (line: hand grid; stars: CMA-ES seeds)")
ax[0].legend(fontsize=8)

key = "0.25MPa" if "0.25MPa" in E["levels"] else levels[0]
D = E["levels"][key]["designs"]
for name, c in (("cma_best", "C0"), ("hand_retrim", "C1"), ("random_best", "C7"), ("old_cma_best", "C3")):
    if name in D:
        r = np.sort(D[name]["heldout_ranges"])
        ax[1].step(r, np.arange(1, len(r) + 1) / len(r), c=c, where="post",
                   label=f"{name}: mean {D[name]['heldout']['mean']:.1f} m, {D[name]['heldout']['breaks']} breaks")
ax[1].set_xlabel("held-out range (m)"); ax[1].set_ylabel("fraction of 64 launches")
ax[1].set_title(f"held-out ECDF, struct model sigma = {key} (headline)"); ax[1].legend(fontsize=8)

for name, c in (("cma_best", "C0"), ("hand_retrim", "C1"), ("old_cma_best", "C3"), ("old_baseline", "C7")):
    if name not in D:
        continue
    d = D[name]["design"]
    b, ch = d["span_m"], d["chord_m"]
    ax[2].add_patch(plt.Rectangle((-b / 2, -ch), b, ch, fill=False, ec=c, lw=2,
                                  label=f"{name}: {b*100:.0f}x{ch*100:.1f} cm, n_safe {d['safe_load_factor']:.1f}"))
ax[2].set_xlim(-0.3, 0.3); ax[2].set_ylim(-0.2, 0.05); ax[2].set_aspect("equal")
ax[2].set_title("wing planforms (top view, m)"); ax[2].legend(fontsize=8, loc="lower center")
fig.tight_layout()
out = os.path.join(HERE, "media", "struct_followup.png")
fig.savefig(out, dpi=110)
print("wrote", out)
