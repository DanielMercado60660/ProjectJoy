"""Bridge-collapse benchmark (D* Lite vs A*/Dijkstra replanning from scratch) + figures.

  ../.venv/bin/python run_bridge.py                 # 150 held-out maps x 2 queries
  ../.venv/bin/python run_bridge.py --maps 10       # quick
Writes results/bridge.json and media/bridge_collapse.png / .gif
"""
import argparse, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from terra import terrain, cost, planners, scenarios, viz
from terra.terrain import WATER


def summarize(rows):
    ok = [r for r in rows if not r["failed"]]
    S = {"queries": len(rows), "failed_no_path_after_collapse": sum(r["failed"] for r in rows),
         "dstar_astar_consistent": all(r["consistent"] for r in rows)}
    for m in ("dstar", "astar", "dijkstra"):
        for k in ("init_expanded", "init_ms", "replan_expanded", "replan_ms"):
            v = np.array([r[m][k] for r in ok])
            S[f"{m}_{k}_median"] = float(np.median(v))
            S[f"{m}_{k}_mean"] = float(np.mean(v))
    re_d = np.array([r["dstar"]["replan_expanded"] for r in ok])
    re_a = np.array([r["astar"]["replan_expanded"] for r in ok])
    ms_d = np.array([r["dstar"]["replan_ms"] for r in ok])
    ms_a = np.array([r["astar"]["replan_ms"] for r in ok])
    S["frac_dstar_fewer_expansions_than_astar"] = float(np.mean(re_d < re_a))
    S["frac_dstar_faster_than_astar"] = float(np.mean(ms_d < ms_a))
    S["dstar_us_per_expansion"] = float(ms_d.sum() / re_d.sum() * 1e3)
    S["astar_us_per_expansion"] = float(ms_a.sum() / re_a.sum() * 1e3)
    # travel cost: planned vs realized (late information) vs omniscient
    S["mean_planned_cost"] = float(np.mean([r["planned_cost"] for r in ok]))
    S["mean_travel_cost"] = float(np.mean([r["dstar"]["travel_cost"] for r in ok]))
    S["mean_omniscient_cost"] = float(np.mean([r["omniscient_cost"] for r in ok]))
    S["mean_travel_over_omniscient"] = float(np.mean([r["dstar"]["travel_cost"] / r["omniscient_cost"] for r in ok]))
    # where does D* Lite win? split by how much of the old tree the collapse invalidates
    detour = np.array([r["omniscient_cost"] / r["planned_cost"] for r in ok])
    small = detour < np.median(detour)
    S["dstar_replan_expanded_median_small_detour"] = float(np.median(re_d[small]))
    S["astar_replan_expanded_median_small_detour"] = float(np.median(re_a[small]))
    S["dstar_replan_expanded_median_large_detour"] = float(np.median(re_d[~small]))
    S["astar_replan_expanded_median_large_detour"] = float(np.median(re_a[~small]))
    return S


def figure(T, s, g, warn, path_png, path_gif):
    rec = scenarios.bridge_collapse(T, s, g, warn, record=True, reps=1)
    G = cost.Graph(T["elev"], T["kind"])
    w = G.w
    # expansions for the replan: D* Lite (trace) vs A* from scratch
    D = planners.DStarLite(G, s, g); D.compute()
    D.move_to(rec["collapse_pos"])
    ch = G.set_cells(T["bridges"][rec["bridge"]], kind_value=WATER)
    D.trace = []
    D.update_vertices(ch); D.compute()
    gr, gc = divmod(g, w)
    a1 = planners.astar(G, rec["collapse_pos"], g, cost.octile_field(G.h, w, gr, gc).ravel(), keep_closed=True)
    a_closed = np.flatnonzero(np.frombuffer(a1.closed, np.uint8))
    rgb0 = viz.terrain_rgb(T["elev"], T["kind"])
    rgb1 = viz.terrain_rgb(G.elev, G.kind)
    br = T["bridges"][rec["bridge"]]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.8))
    ax = axs[0]
    ax.imshow(rgb0)
    viz.plot_path(ax, rec["path0"], w, "#ffd400", label="initial optimal route")
    ax.scatter([c for _, c in br], [r for r, _ in br], s=30, facecolors="none", edgecolors="#ff3b30", lw=1.5)
    ax.set_title("1. Plan: route uses the bridge (circled)")
    ax = axs[1]
    ax.imshow(viz.overlay_cells(rgb1, set(D.trace), w, (1, 0.15, 0.1), 0.55))
    viz.plot_path(ax, rec["traj"], w, "#ffffff", label="walked")
    viz.plot_path(ax, rec["path1"], w, "#ff3b30", label="D* Lite replan")
    ax.set_title(f"2. Bridge gone: D* Lite repairs {len(D.trace)} nodes")
    ax = axs[2]
    ax.imshow(viz.overlay_cells(rgb1, a_closed, w, (0.1, 0.6, 1.0), 0.55))
    viz.plot_path(ax, rec["traj"], w, "#ffffff")
    viz.plot_path(ax, a1.path, w, "#ffd400", label="A* replan (same cost)")
    ax.set_title(f"3. A* from scratch expands {a1.expanded} nodes")
    for ax in axs:
        viz.mark(ax, s, w, "#ffffff", "o"); viz.mark(ax, g, w, "#ff2bd6", "*", 220)
        viz.mark(ax, rec["collapse_pos"], w, "#ff3b30", "X", 120)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="lower left", fontsize=8, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(path_png, dpi=110)
    plt.close(fig)

    # GIF: agent walks, bridge collapses, agent follows the new route
    frames = []
    walk = rec["traj"] + rec["path1"][1:]
    cut = len(rec["traj"]) - 1
    for k in range(0, len(walk), 2):
        collapsed = k >= cut
        fig, ax = plt.subplots(figsize=(5.2, 5.4))
        ax.imshow(rgb1 if collapsed else rgb0)
        if collapsed:
            ax.imshow(np.zeros_like(rgb1[..., 0]), alpha=0)
            viz.plot_path(ax, rec["path0"][cut:], w, "#ffd400", lw=1.2, ls="--")
            viz.plot_path(ax, rec["path1"], w, "#ff3b30")
            ax.set_title(f"bridge collapsed! D* Lite replan: {len(D.trace)} nodes\n(A* from scratch: {a1.expanded})", fontsize=10)
        else:
            viz.plot_path(ax, rec["path0"][k:], w, "#ffd400")
            ax.set_title("following the optimal route...", fontsize=10)
        viz.plot_path(ax, walk[:k + 1], w, "#ffffff", lw=1.6)
        viz.mark(ax, walk[k], w, "#ffffff", "o", 120)
        viz.mark(ax, g, w, "#ff2bd6", "*", 220)
        ax.set_xticks([]); ax.set_yticks([])
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
        if k <= cut < k + 2:
            frames += [frames[-1]] * 8  # pause at the collapse
    frames += [frames[-1]] * 10
    imageio.mimsave(path_gif, frames, duration=0.08, loop=0)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=150)
    ap.add_argument("--per_map", type=int, default=2)
    ap.add_argument("--warn", type=int, default=6)
    ap.add_argument("--fig_seed", type=int, default=900003)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True); os.makedirs("media", exist_ok=True)
    rows = scenarios.run_bridge_benchmark(range(900000, 900000 + a.maps), per_map=a.per_map, warn=a.warn)
    S = summarize(rows)
    with open("results/bridge.json", "w") as f:
        json.dump({"summary": S, "rows": rows, "args": vars(a)}, f, indent=1)
    for k, v in S.items():
        print(f"{k:55s} {v}")
    # showcase figure: a held-out map where the query crosses a bridge
    T = terrain.generate(a.fig_seed, 96)
    G = cost.Graph(T["elev"], T["kind"])
    q = scenarios.pick_bridge_query(T, G, np.random.default_rng(1), min_dist=50)
    figure(T, q[0], q[1], a.warn, "media/bridge_collapse.png", "media/bridge_collapse.gif")


if __name__ == "__main__":
    main()
