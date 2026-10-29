"""Held-out evaluation of the learned heuristic vs admissible A*, weighted A*, Dijkstra.

  ../.venv/bin/python eval_heuristic.py --maps 100 --per_map 3
Test maps use seeds 900000+ (training used 0..2999, validation 500000+).
Writes results/heuristic.json, media/heuristic_tradeoff.png, media/planner_routes.png
"""
import argparse, json, math, os, time
import numpy as np
import mlx.core as mx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from terra import terrain, cost, planners, learn, viz

WEIGHTS = [1.25, 1.5, 2.0, 3.0, 5.0]
ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5]


def load_model(tag="hnet"):
    cfg = json.load(open(f"checkpoints/{tag}.json"))
    model = learn.HNet(base=cfg["base"])
    model.load_weights(f"checkpoints/{tag}.safetensors")
    model.eval()
    return model, cfg


def queries(T, G, rng, k, min_dist=30):
    kind = T["kind"].ravel()
    passable = np.flatnonzero(kind != terrain.WATER)
    out = []
    for _ in range(500):
        if len(out) >= k:
            break
        s, g = int(rng.choice(passable)), int(rng.choice(passable))
        if G.h_octile(s, g) < min_dist:
            continue
        out.append((s, g))
    return out


def run(maps, per_map, model, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    # warm up the model (compile kernels) so inference timing is steady-state
    T = terrain.generate(1, 96)
    for _ in range(3):
        learn.predict_h(model, T["elev"], T["kind"], 10, 10)
    for ms in range(900000, 900000 + maps):
        T = terrain.generate(ms, 96)
        G = cost.Graph(T["elev"], T["kind"])
        st = learn.static_features(T["elev"], T["kind"])
        for s, g in queries(T, G, rng, per_map):
            gr, gc = divmod(g, G.w)
            ref = planners.dijkstra(G, s, g)
            if not ref.path:
                rows.append({"map": ms, "unreachable": True})
                continue
            row = {"map": ms, "s": s, "g": g, "opt": ref.cost, "unreachable": False,
                   "Dijkstra": (ref.expanded, ref.cost, ref.ms)}
            ho = cost.octile_field(G.h, G.w, gr, gc).ravel().tolist()
            r = planners.astar(G, s, g, ho)
            row["A*"] = (r.expanded, r.cost, r.ms)
            for w in WEIGHTS:
                r = planners.astar(G, s, g, ho, weight=w)
                row[f"WA* w={w}"] = (r.expanded, r.cost, r.ms)
            t0 = time.perf_counter()
            hl, hoct = learn.predict_h(model, T["elev"], T["kind"], gr, gc, static=st)
            row["infer_ms"] = (time.perf_counter() - t0) * 1e3
            hl, hoct = hl.ravel(), hoct.ravel()
            for a in ALPHAS:
                h = (hoct + a * (hl - hoct)).tolist()
                r = planners.astar(G, s, g, h)
                row[f"Learned a={a}"] = (r.expanded, r.cost, r.ms)
            # perfect heuristic (exact cost-to-go) as the headroom bound
            ctg = learn.cost_to_go(T["elev"], T["kind"], (gr, gc)).ravel()
            ctg = np.where(np.isfinite(ctg), ctg, 1e9).tolist()
            r = planners.astar(G, s, g, ctg)
            row["Oracle h*"] = (r.expanded, r.cost, r.ms)
            rows.append(row)
    return rows


def summarize(rows):
    ok = [r for r in rows if not r["unreachable"]]
    methods = [k for k in ok[0] if isinstance(ok[0][k], tuple)]
    S = {"queries": len(ok), "unreachable_queries_skipped": len(rows) - len(ok),
         "infer_ms_median": float(np.median([r["infer_ms"] for r in ok]))}
    base = np.array([r["A*"][0] for r in ok], float)
    for m in methods:
        e = np.array([r[m][0] for r in ok], float)
        sub = np.array([r[m][1] / r["opt"] - 1 for r in ok])
        ms = np.array([r[m][2] for r in ok])
        S[m] = {"expanded_mean": float(e.mean()), "expanded_median": float(np.median(e)),
                "expanded_vs_astar_mean_ratio": float(np.mean(e / base)),
                "subopt_mean_pct": float(100 * sub.mean()), "subopt_p95_pct": float(100 * np.percentile(sub, 95)),
                "subopt_max_pct": float(100 * sub.max()), "frac_optimal": float(np.mean(sub < 1e-9)),
                "search_ms_median": float(np.median(ms)), "failures": int(np.sum(~np.isfinite(sub)))}
        if m.startswith("Learned"):  # end-to-end wall clock includes the network forward pass
            tot = ms + np.array([r["infer_ms"] for r in ok])
            S[m]["total_ms_median_incl_inference"] = float(np.median(tot))
        else:
            S[m]["total_ms_median_incl_inference"] = float(np.median(ms))
    return S


def tradeoff_plot(S, path):
    fig, ax = plt.subplots(figsize=(7.5, 5))

    def pts(prefix):
        ks = [k for k in S if isinstance(S[k], dict) and k.startswith(prefix)]
        return ([S[k]["expanded_vs_astar_mean_ratio"] for k in ks], [S[k]["subopt_mean_pct"] for k in ks], ks)

    for prefix, col, lab in (("WA*", "#00a0c0", "weighted A* (inflate octile)"), ("Learned", "#d0209a", "learned heuristic (MLX U-Net)")):
        x, y, ks = pts(prefix)
        o = np.argsort(x)
        ax.plot(np.array(x)[o], np.array(y)[o], "o-", color=col, label=lab)
        for xi, yi, k in zip(x, y, ks):
            ax.annotate(k.split(" ")[-1], (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8, color=col)
    for k, mk, col in (("A*", "s", "k"), ("Dijkstra", "D", "gray"), ("Oracle h*", "*", "gold")):
        ax.scatter([S[k]["expanded_vs_astar_mean_ratio"]], [S[k]["subopt_mean_pct"]], marker=mk, s=90, c=col, edgecolors="k", label=k, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("nodes expanded / admissible A* (mean ratio, log)")
    ax.set_ylabel("path cost above optimal (mean %)")
    ax.set_title(f"Held-out maps: speed vs optimality ({S['queries']} queries)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def routes_figure(model, path, seed=900007):
    """Same map, same start/goal: which cells each planner expands and which route it picks."""
    T = terrain.generate(seed, 96)
    G = cost.Graph(T["elev"], T["kind"])
    rng = np.random.default_rng(3)
    s, g = queries(T, G, rng, 1, min_dist=70)[0]
    gr, gc = divmod(g, G.w)
    ho = cost.octile_field(G.h, G.w, gr, gc).ravel()
    hl, _ = learn.predict_h(model, T["elev"], T["kind"], gr, gc)
    D = planners.DStarLite(G, s, g); D.trace = []; D.compute()
    runs = [("Dijkstra", planners.dijkstra(G, s, g, keep_closed=True), None),
            ("A*", planners.astar(G, s, g, ho, keep_closed=True), None),
            ("D* Lite", None, D),
            ("Weighted A* (w=2)", planners.astar(G, s, g, ho, weight=2.0, keep_closed=True), None),
            ("Greedy", planners.astar(G, s, g, ho * 1e3, keep_closed=True), None),
            ("Learned-h A*", planners.astar(G, s, g, hl.ravel(), keep_closed=True), None)]
    opt = runs[0][1].cost
    rgb = viz.terrain_rgb(T["elev"], T["kind"])
    fig, axs = plt.subplots(2, 3, figsize=(15, 10.4))
    for ax, (name, r, d) in zip(axs.flat, runs):
        if d is not None:
            nodes, p, c, e = set(d.trace), d.path(), d.cost_to_goal, len(d.trace)
        else:
            nodes = np.flatnonzero(np.frombuffer(r.closed, np.uint8)); p, c, e = r.path, r.cost, r.expanded
        ax.imshow(viz.overlay_cells(rgb, nodes, G.w, (1.0, 1.0, 1.0), 0.45))
        viz.plot_path(ax, p, G.w, viz.ROUTE_COLORS[name])
        viz.mark(ax, s, G.w, "#ffffff"); viz.mark(ax, g, G.w, "#ff2bd6", "*", 220)
        ax.set_title(f"{name}: cost {c:.1f} ({100 * (c / opt - 1):+.1f}%), expanded {e}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Same held-out map, same cost definition. White haze = expanded nodes.", fontsize=12)
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close(fig)


def field_figure(model, path, seed=900011):
    """What the network predicts: octile vs learned vs exact cost-to-go, for one goal."""
    T = terrain.generate(seed, 96)
    G = cost.Graph(T["elev"], T["kind"])
    gr, gc = 30, 64
    while T["kind"][gr, gc] == terrain.WATER:
        gc += 1
    hl, ho = learn.predict_h(model, T["elev"], T["kind"], gr, gc)
    ctg = learn.cost_to_go(T["elev"], T["kind"], (gr, gc))
    water = T["kind"] == terrain.WATER
    vmax = np.nanpercentile(np.where(np.isfinite(ctg), ctg, np.nan), 99)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.2))
    axs[0].imshow(viz.terrain_rgb(T["elev"], T["kind"])); axs[0].set_title(f"held-out map {seed}, goal ★")
    for ax, f, t in ((axs[1], ho, "octile h (admissible A*)"), (axs[2], hl, "learned h (MLX U-Net)"),
                     (axs[3], ctg, "exact cost-to-go h* (Dijkstra)")):
        im = ax.imshow(np.ma.masked_where(water | ~np.isfinite(ctg), f), cmap="magma", vmin=0, vmax=vmax)
        ax.contour(np.where(np.isfinite(f), f, vmax), levels=12, colors="w", linewidths=0.4, alpha=0.6)
        ax.set_title(t)
    rel = np.abs(hl - ctg)[~water & np.isfinite(ctg)] / np.maximum(ctg[~water & np.isfinite(ctg)], 1)
    axs[2].set_xlabel(f"median |h - h*| / h* = {100 * np.median(rel):.1f}%")
    for ax in axs:
        viz.mark(ax, gr * 96 + gc, 96, "#ff2bd6", "*", 220); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close(fig)


def failure_figure(model, path, seed, s, g):
    """The worst held-out query for the learned heuristic: where its guess misleads A*."""
    T = terrain.generate(seed, 96)
    G = cost.Graph(T["elev"], T["kind"])
    gr, gc = divmod(g, 96)
    hl, ho = learn.predict_h(model, T["elev"], T["kind"], gr, gc)
    ctg = learn.cost_to_go(T["elev"], T["kind"], (gr, gc))
    opt = planners.dijkstra(G, s, g)
    lr = planners.astar(G, s, g, hl.ravel(), keep_closed=True)
    rgb = viz.terrain_rgb(T["elev"], T["kind"])
    water = T["kind"] == terrain.WATER
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
    axs[0].imshow(viz.overlay_cells(rgb, np.flatnonzero(np.frombuffer(lr.closed, np.uint8)), 96, (1, 1, 1), 0.45))
    viz.plot_path(axs[0], opt.path, 96, "#ffd400", label=f"optimal ({opt.cost:.0f})")
    viz.plot_path(axs[0], lr.path, 96, "#c77dff", label=f"learned-h A* ({lr.cost:.0f}, +{100 * (lr.cost / opt.cost - 1):.0f}%)")
    axs[0].legend(loc="lower left", fontsize=8)
    axs[0].set_title(f"held-out map {seed}: worst case for the learned h")
    ratio = np.where(water | ~np.isfinite(ctg), np.nan, hl / np.maximum(ctg, 1e-9))
    im = axs[1].imshow(ratio, cmap="coolwarm", vmin=0.5, vmax=1.5)
    plt.colorbar(im, ax=axs[1], fraction=0.046)
    axs[1].set_title("learned h / true h*  (red = overestimate)")
    vmax = np.nanpercentile(np.where(np.isfinite(ctg), ctg, np.nan), 99)
    axs[2].imshow(np.ma.masked_where(water | ~np.isfinite(ctg), ctg), cmap="magma", vmin=0, vmax=vmax)
    axs[2].set_title("true cost-to-go h*")
    for ax in axs:
        viz.mark(ax, s, 96, "#ffffff"); viz.mark(ax, g, 96, "#ff2bd6", "*", 220)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close(fig)
    return opt.cost, lr.cost, float(hl.ravel()[s] / ctg.ravel()[s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=100)
    ap.add_argument("--per_map", type=int, default=3)
    ap.add_argument("--tag", default="hnet")
    ap.add_argument("--out", default="results/heuristic.json")
    ap.add_argument("--no_figs", action="store_true")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True); os.makedirs("media", exist_ok=True)
    model, cfg = load_model(a.tag)
    rows = run(a.maps, a.per_map, model)
    S = summarize(rows)
    json.dump({"summary": S, "rows": rows, "model_cfg": cfg, "args": vars(a)}, open(a.out, "w"), indent=1)
    print(f"queries {S['queries']}, inference median {S['infer_ms_median']:.2f} ms")
    print(f"{'method':18s} {'exp/A*':>7s} {'exp med':>8s} {'subopt%':>8s} {'p95%':>7s} {'max%':>7s} {'opt frac':>8s} {'ms med':>7s} {'tot ms':>7s}")
    for k, v in S.items():
        if isinstance(v, dict):
            print(f"{k:18s} {v['expanded_vs_astar_mean_ratio']:7.3f} {v['expanded_median']:8.0f} {v['subopt_mean_pct']:8.2f} "
                  f"{v['subopt_p95_pct']:7.2f} {v['subopt_max_pct']:7.2f} {v['frac_optimal']:8.2f} {v['search_ms_median']:7.2f} {v['total_ms_median_incl_inference']:7.2f}")
    if not a.no_figs:
        tradeoff_plot(S, "media/heuristic_tradeoff.png")
        routes_figure(model, "media/planner_routes.png")
        field_figure(model, "media/heuristic_fields.png")


if __name__ == "__main__":
    main()
