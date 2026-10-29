"""Partial-information benchmark: same agent (D* Lite on a belief map), same sensing radius,
three priors for unknown terrain. Also D* Lite vs A*-from-scratch replanning compute.

  ../.venv/bin/python run_fog.py --maps 100 --per_map 2
Writes results/fog.json, media/fog_priors.png, media/fog_walk.gif
"""
import argparse, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from terra import terrain, cost, planners, fog, viz
import train_prior

PRIORS = ["optimistic", "learned", "hydrology"]
# showcase query, picked from results/fog.json as a case where the priors behave differently
FIG_QUERY = (900028, 4664, 8990)


def get_priors(T, model):
    p, ml = train_prior.predict(model, T["elev"])
    return {"optimistic": fog.prior_optimistic(T["elev"]),
            "learned": fog.prior_from_probs(p, ml),
            "hydrology": fog.prior_hydrology(T["elev"])}, p


def queries(T, G, rng, k, min_dist=40):
    passable = np.flatnonzero(T["kind"].ravel() != terrain.WATER)
    out = []
    for _ in range(300):
        if len(out) >= k:
            break
        s, g = int(rng.choice(passable)), int(rng.choice(passable))
        if G.h_octile(s, g) >= min_dist:
            ref = planners.dijkstra(G, s, g)
            if ref.path:
                out.append((s, g, ref.cost))
    return out


def summarize(rows, R):
    S = {}
    for pr in PRIORS:
        rs = [r[pr] for r in rows]
        ok = [x for x in rs if not x["failed"]]
        ratio = np.array([x["travel_cost"] / r["opt"] for x, r in zip(rs, rows) if not x["failed"]])
        S[pr] = {"queries": len(rs), "failure_rate": 1 - len(ok) / len(rs),
                 "travel_over_opt_mean": float(ratio.mean()), "travel_over_opt_median": float(np.median(ratio)),
                 "travel_over_opt_p95": float(np.percentile(ratio, 95)),
                 "observed_cells_mean": float(np.mean([x["observed_cells"] for x in ok])),
                 "steps_mean": float(np.mean([x["steps"] for x in ok])),
                 "replans_mean": float(np.mean([x["replans"] for x in ok]))}
        if "astar_expanded" in ok[0] and ok[0]["astar_expanded"]:
            S[pr].update({
                "dstar_expanded_total_mean": float(np.mean([x["dstar_expanded"] for x in ok])),
                "astar_expanded_total_mean": float(np.mean([x["astar_expanded"] for x in ok])),
                "dstar_ms_total_median": float(np.median([x["dstar_ms"] for x in ok])),
                "astar_ms_total_median": float(np.median([x["astar_ms"] for x in ok])),
                "dstar_ms_per_replan_median": float(np.median([x["dstar_ms"] / max(1, x["replans"]) for x in ok])),
                "astar_ms_per_replan_median": float(np.median([x["astar_ms"] / max(1, x["replans"]) for x in ok]))})
    # paired comparison learned vs optimistic
    a = np.array([r["learned"]["travel_cost"] / r["opt"] for r in rows])
    b = np.array([r["optimistic"]["travel_cost"] / r["opt"] for r in rows])
    S["learned_better_than_optimistic_frac"] = float(np.mean(a < b - 1e-9))
    S["learned_worse_than_optimistic_frac"] = float(np.mean(a > b + 1e-9))
    S["R"] = R
    return S


def figure(T, s, g, priors, pwater, path_png, path_gif, R):
    rgb = viz.terrain_rgb(T["elev"], T["kind"])
    w = T["kind"].shape[1]
    G = cost.Graph(T["elev"], T["kind"])
    opt = planners.dijkstra(G, s, g)
    runs = {pr: fog.run_fog(T, s, g, priors[pr], R=R, compare_astar=False, record=True) for pr in PRIORS}
    fig, axs = plt.subplots(1, 4, figsize=(21, 5.6))
    axs[0].imshow(rgb); axs[0].imshow(pwater, cmap="Blues", alpha=0.0)
    im = axs[0].imshow(pwater, cmap="Blues", vmin=0, vmax=1)
    viz.plot_path(axs[0], opt.path, w, "#ffd400")
    axs[0].set_title(f"learned P(water) from elevation only\n(yellow: omniscient optimum, cost {opt.cost:.0f})", fontsize=10)
    for ax, pr in zip(axs[1:], PRIORS):
        r = runs[pr]
        img = rgb.copy()
        img[~r["known"]] = 0.35 * img[~r["known"]] + 0.65 * np.array([0.12, 0.12, 0.14])
        ax.imshow(img)
        viz.plot_path(ax, r["traj"], w, "#ff3b30")
        lab = "hydrology (generator reference, not a competitor)" if pr == "hydrology" else f"{pr} prior"
        ax.set_title(f"{lab}: travel {r['travel_cost']:.0f} ({r['travel_cost'] / opt.cost:.2f}× opt)\n"
                     f"{r['observed_cells']} cells observed, {r['steps']} steps", fontsize=10)
    for ax in axs:
        viz.mark(ax, s, w, "#ffffff"); viz.mark(ax, g, w, "#ff2bd6", "*", 220)
        ax.set_xticks([]); ax.set_yticks([])
    seed = next((q[0] for q in [FIG_QUERY] if q[1] == s and q[2] == g), None)
    fig.suptitle(f"HAND-PICKED illustrative case{f' (held-out map {seed})' if seed else ''}, chosen because the priors "
                 f"differ here; it is not typical. Aggregate over 200 queries: see RESULTS.md.", fontsize=11)
    plt.tight_layout(); fig.subplots_adjust(top=0.82); plt.savefig(path_png, dpi=95); plt.close(fig)
    # GIF: optimistic vs learned walking side by side, fog revealed as they go
    known_hist = {}
    for pr in ("optimistic", "learned"):
        kn = np.zeros(T["kind"].shape, bool); hist = []
        offs = fog.disk_offsets(R)
        for u in runs[pr]["traj"]:
            r0, c0 = divmod(u, w)
            for dr, dc in offs:
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < kn.shape[0] and 0 <= cc < w:
                    kn[rr, cc] = True
            hist.append(kn.copy())
        known_hist[pr] = hist
    n = max(len(runs["optimistic"]["traj"]), len(runs["learned"]["traj"]))
    frames = []
    for k in list(range(0, n, 3)) + [n - 1] * 12:
        fig, axs = plt.subplots(1, 2, figsize=(9.6, 5.2))
        for ax, pr in zip(axs, ("optimistic", "learned")):
            tr = runs[pr]["traj"]; kk = min(k, len(tr) - 1)
            kn = known_hist[pr][kk]
            img = rgb.copy(); img[~kn] = 0.3 * img[~kn] + 0.7 * np.array([0.12, 0.12, 0.14])
            ax.imshow(img)
            viz.plot_path(ax, tr[:kk + 1], w, "#ff3b30", lw=1.8)
            viz.mark(ax, tr[kk], w, "#ffffff", "o", 80); viz.mark(ax, g, w, "#ff2bd6", "*", 180)
            done = " (arrived)" if kk == len(tr) - 1 else ""
            ax.set_title(f"{pr} prior{done}", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Fog of war, sensing radius {R}. Same D* Lite agent; only the prior differs.", fontsize=10)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    imageio.mimsave(path_gif, frames, duration=0.07, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=100)
    ap.add_argument("--per_map", type=int, default=2)
    ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--sweep_maps", type=int, default=40)
    ap.add_argument("--tag", default="prior")
    ap.add_argument("--out", default="results/fog.json")
    ap.add_argument("--no_figs", action="store_true")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True); os.makedirs("media", exist_ok=True)
    model, pcfg = train_prior.load_prior(a.tag)
    rng = np.random.default_rng(0)
    rows = []
    for ms in range(900000, 900000 + a.maps):
        T = terrain.generate(ms, 96)
        G = cost.Graph(T["elev"], T["kind"])
        priors, _ = get_priors(T, model)
        for s, g, opt in queries(T, G, rng, a.per_map):
            row = {"map": ms, "s": s, "g": g, "opt": opt}
            for pr in PRIORS:
                row[pr] = fog.run_fog(T, s, g, priors[pr], R=a.R)
            rows.append(row)
    S = summarize(rows, a.R)
    # radius sweep (travel cost only)
    sweep = {}
    for R in ((2, 4, 8, 12) if a.sweep_maps else ()):
        rr = []
        rng2 = np.random.default_rng(1)
        for ms in range(900000, 900000 + a.sweep_maps):
            T = terrain.generate(ms, 96); G = cost.Graph(T["elev"], T["kind"])
            priors, _ = get_priors(T, model)
            for s, g, opt in queries(T, G, rng2, 1):
                row = {"opt": opt}
                for pr in PRIORS:
                    row[pr] = fog.run_fog(T, s, g, priors[pr], R=R, compare_astar=False)
                rr.append(row)
        sweep[R] = {pr: float(np.mean([r[pr]["travel_cost"] / r["opt"] for r in rr if not r[pr]["failed"]])) for pr in PRIORS}
        sweep[R]["n"] = len(rr)
        print("R", R, sweep[R], flush=True)
    S["radius_sweep_travel_over_opt_mean"] = sweep
    json.dump({"summary": S, "rows": rows, "prior_cfg": pcfg, "args": vars(a)}, open(a.out, "w"), indent=1)
    print(json.dumps(S, indent=1))
    if a.no_figs:
        return
    seed, s, g = FIG_QUERY
    T = terrain.generate(seed, 96)
    priors, pw = get_priors(T, model)
    figure(T, s, g, priors, pw, "media/fog_priors.png", "media/fog_walk.gif", a.R)


if __name__ == "__main__":
    main()
