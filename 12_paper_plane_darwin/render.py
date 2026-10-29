"""Render the demo media from checkpoints/designs.json, checkpoints/search.npz and results/eval.json.

Usage:  NUMBA_NUM_THREADS=4 ../.venv/bin/python render.py [--no-gif]
Writes media/*.png and media/flight.gif.
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sim
from search import DT_SOUND, DT_CHEAP, ME_RANGE

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "media")
# reference categorical palette (light mode), fixed order: entity -> colour never changes
COL = {"baseline_hand": "#eb6834", "cma_best": "#2a78d6", "mapel_best": "#1baf7a",
       "random_best": "#6250d6", "cheap_sim_best": "#e34948"}
LABEL = {"baseline_hand": "hand-designed baseline", "cma_best": "CMA-ES best",
         "mapel_best": "MAP-Elites best", "random_best": "random search best (matched budget)",
         "cheap_sim_best": "CMA-ES on cheap Euler sim"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "lines.linewidth": 2})


def load():
    with open(os.path.join(HERE, "checkpoints", "designs.json")) as fh:
        designs = json.load(fh)
    with open(os.path.join(HERE, "results", "eval.json")) as fh:
        ev = json.load(fh)
    ck = np.load(os.path.join(HERE, "checkpoints", "search.npz"))
    return designs, ev, ck


def planform(ax, genes, color, title):
    u = sim.genes_to_phys_units(genes)
    b, c, tr, lt, mb, xb = u[:6]
    P = sim.decode(np.asarray(genes, dtype=np.float64))
    st = P[5]
    ct = P[7]
    bt = st / ct
    # body x forward = up on the page; wing AC at 0
    ax.add_patch(plt.Rectangle((-b / 2, -0.75 * c), b, c, fc=color, ec=INK, alpha=0.35, lw=1))
    ax.add_patch(plt.Rectangle((-bt / 2, -lt - 0.75 * ct), bt, ct, fc=color, ec=INK, alpha=0.35, lw=1))
    ax.plot([0, 0], [xb + 0.02, -lt - 0.75 * ct], color=INK, lw=2)
    if mb > 1e-4:
        ax.scatter([0], [xb], s=20 + 3000 * mb, color=INK2, zorder=3)
    ax.scatter([0], [P[16]], marker="+", s=120, color=INK, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-0.27, 0.27)
    ax.set_ylim(-0.68, 0.24)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    d = sim.describe(genes)
    ax.text(-0.26, -0.46, f"span {b*100:.0f} cm, chord {c*100:.1f} cm (AR {d['aspect_ratio']:.1f})\n"
            f"tail {tr*100:.0f}% of wing at {lt*100:.0f} cm, ballast {mb*1000:.1f} g\n"
            f"incidence wing {u[6]:+.1f}°, tail {u[7]:+.1f}°\nmass {d['mass_g']:.1f} g, static margin {d['static_margin']:.2f}",
            fontsize=7.5, color=INK2, va="top")


def fig_planforms(designs):
    names = ["baseline_hand", "cma_best", "mapel_best", "archive_floater", "archive_smooth", "archive_porpoise"]
    names = [n for n in names if n in designs]
    fig, axs = plt.subplots(1, len(names), figsize=(2.6 * len(names), 4.4))
    for ax, n in zip(np.atleast_1d(axs), names):
        planform(ax, np.array(designs[n]["genes"]), COL.get(n, "#52514e"), LABEL.get(n, n.replace("_", " ")))
    fig.suptitle("Planforms (top view, nose up; + = CG, dot = nose ballast)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "planforms.png"), dpi=130)
    plt.close(fig)


def fig_paths(designs):
    hc, hg = sim.heldout_conditions(n=64)
    nom_c, nom_g = sim.still_air()
    k = 7  # one gusty held-out launch for the second panel
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.4))
    for ax, (cond, gust, ttl) in zip(axs, [(nom_c[0], nom_g[0], "Still air, nominal launch (10 m, 6 m/s level)"),
                                           (hc[k], hg[k], f"Held-out launch #{k}: h={hc[k,0]:.1f} m, v={hc[k,1]:.1f} m/s, "
                                                          f"wind {hc[k,4]:+.1f} m/s, gusts, CLa x{hc[k,5]:.2f}, CD0 x{hc[k,6]:.2f}")]):
        for n in ["baseline_hand", "random_best", "cma_best", "mapel_best"]:
            if n not in designs:
                continue
            s, tr = sim.fly(designs[n]["genes"], cond, gust, dt=DT_SOUND)
            ax.plot(tr[:, 1], tr[:, 2], color=COL[n], label=f"{LABEL[n]}: {s[0]:.1f} m")
        ax.set_title(ttl, fontsize=10, loc="left")
        ax.set_xlabel("downrange x (m)"); ax.set_ylabel("height z (m)")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "flight_paths.png"), dpi=130)
    plt.close(fig)


def fig_heldout(ev):
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in ["baseline_hand", "random_best", "cma_best", "mapel_best"]:
        c = ev["candidates"].get(n)
        if c is None:
            continue
        r = np.sort(c["heldout_ranges"])
        ax.step(r, np.arange(1, len(r) + 1) / len(r), where="post", color=COL[n],
                label=f"{LABEL[n]} (mean {c['heldout']['mean']:.1f} m, p10 {c['heldout']['p10']:.1f})")
    ax.set_xlabel("range on held-out launch (m)")
    ax.set_ylabel("fraction of 64 held-out launches")
    ax.set_title("Robustness: range distribution on unseen launches, gusts and model errors", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "heldout_ecdf.png"), dpi=130)
    plt.close(fig)


def fig_search(ck, ev):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axs[0]
    for arm, col, lab in [("random", COL["random_best"], "random search"), ("cma", COL["cma_best"], "CMA-ES")]:
        h = ck[f"{arm}_hist"]
        e = ck[f"{arm}_hist_evals"]
        ax.fill_between(e, h.min(0), h.max(0), color=col, alpha=0.18, lw=0)
        ax.plot(e, np.median(h, 0), color=col, label=f"{lab} (median, min-max of {h.shape[0]} seeds)")
    base = ev["candidates"]["baseline_hand"]["train_mean_range_m"]
    ax.axhline(base, color=COL["baseline_hand"], lw=2, ls="--", label=f"hand baseline ({base:.1f} m)")
    ax.set_xlabel("designs evaluated"); ax.set_ylabel("best training-mean range (m)")
    ax.set_title("Search progress at matched budget", fontsize=10, loc="left")
    ax.set_ylim(bottom=max(0, base - 40))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axs[1]
    fit = np.where(np.isfinite(ck["me_fit"]), ck["me_fit"], np.nan)
    im = ax.imshow(fit.T, origin="lower", aspect="auto", cmap="Blues",
                   extent=[ME_RANGE[0][0], ME_RANGE[0][1], ME_RANGE[1][0], ME_RANGE[1][1]],
                   vmin=max(0, np.nanmin(fit)), vmax=np.nanmax(fit))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("training-mean range of elite (m)")
    a = ev["archive"]
    ax.set_title(f"MAP-Elites archive: {a['cells_filled']}/{a['cells_total']} cells, "
                 f"{a['n_elites_beating_baseline_train']} elites beat the baseline", fontsize=10, loc="left")
    ax.set_xlabel("still-air flight time (s)"); ax.set_ylabel("log10 pitch-rate RMS (rad/s)  [smooth -> porpoising]")
    ax.grid(False)
    for n, mk in [("archive_floater", "o"), ("archive_smooth", "s"), ("archive_porpoise", "^"), ("mapel_best", "*")]:
        c = ev["candidates"].get(n)
        if c is None:
            continue
        t = c["nominal_time_s"]; q = math.log10(max(c["nominal_pitch_rate_rms"], 1e-6))
        ax.scatter([t], [q], marker=mk, s=70, facecolor="none", edgecolor=INK, lw=1.5)
        ax.annotate(n.replace("archive_", "").replace("_", " "), (t, q), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "search_and_archive.png"), dpi=130)
    plt.close(fig)


def fig_sanity(designs):
    """Energy vs time and dt-halving overlay; cheap-sim winner under Euler vs RK4."""
    nom_c, nom_g = sim.still_air()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for n in ["baseline_hand", "cma_best"]:
        g = designs[n]["genes"]
        s1, t1 = sim.fly(g, nom_c[0], nom_g[0], dt=DT_SOUND)
        s2, t2 = sim.fly(g, nom_c[0], nom_g[0], dt=DT_SOUND / 2, rec_every=8)
        axs[0].plot(t1[:, 0], t1[:, 5] / t1[0, 5], color=COL[n], label=LABEL[n])
        axs[1].plot(t1[:, 1], t1[:, 2], color=COL[n], label=f"{LABEL[n]}, dt=5 ms: {s1[0]:.3f} m")
        axs[1].plot(t2[:, 1], t2[:, 2], color=INK, lw=0.8, ls=":", label=f"same, dt=2.5 ms: {s2[0]:.3f} m")
    axs[0].set_xlabel("time (s)"); axs[0].set_ylabel("mechanical energy / launch energy")
    axs[0].set_title("Still air: energy only ever decreases", fontsize=10, loc="left")
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].set_xlabel("x (m)"); axs[1].set_ylabel("z (m)")
    axs[1].set_title("Halving the step: same flight path", fontsize=10, loc="left")
    axs[1].legend(frameon=False, fontsize=7.5)
    n = "cheap_sim_best"
    if n in designs:
        g = designs[n]["genes"]
        se, te = sim.fly(g, nom_c[0], nom_g[0], dt=DT_CHEAP, method=1, rec_every=1)
        sr, trk = sim.fly(g, nom_c[0], nom_g[0], dt=DT_SOUND)
        axs[2].plot(te[:, 0], te[:, 5] / te[0, 5], color=COL[n], label=f"explicit Euler dt=20 ms (range {se[0]:.1f} m)")
        axs[2].plot(trk[:, 0], trk[:, 5] / trk[0, 5], color=INK, lw=1, ls="--", label=f"RK4 dt=5 ms (range {sr[0]:.1f} m)")
        axs[2].set_title("Cheap-sim winner: Euler vs RK4 energy", fontsize=10, loc="left")
        axs[2].set_xlabel("time (s)"); axs[2].set_ylabel("energy / launch energy")
        axs[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "sanity_checks.png"), dpi=130)
    plt.close(fig)


def gif(designs, path, names, cond, gust, title, fps=20, speed=4.0):
    import imageio.v2 as imageio
    runs = []
    for n in names:
        s, tr = sim.fly(designs[n]["genes"], cond, gust, dt=DT_SOUND, rec_every=2)
        runs.append((n, s, tr))
    tend = max(tr[-1, 0] for _, _, tr in runs)
    xmax = max(tr[:, 1].max() for _, _, tr in runs) * 1.05
    xmin = min(-2.0, min(tr[:, 1].min() for _, _, tr in runs) * 1.05)
    zmax = max(tr[:, 2].max() for _, _, tr in runs) * 1.15
    frames = []
    times = np.arange(0, tend + 1.0, speed / fps)
    for t in times:
        fig, ax = plt.subplots(figsize=(8, 3.4), dpi=90)
        for n, s, tr in runs:
            i = min(np.searchsorted(tr[:, 0], t), len(tr) - 1)
            ax.plot(tr[:i + 1, 1], tr[:i + 1, 2], color=COL[n], lw=1.5)
            x, z, th = tr[i, 1], tr[i, 2], tr[i, 3]
            L = 0.035 * (xmax - xmin)       # glider glyph, exaggerated size
            ax.plot([x - L * math.cos(th), x + 0.4 * L * math.cos(th)],
                    [z - L * math.sin(th) * (zmax / (xmax - xmin)) * 6, z + 0.4 * L * math.sin(th) * (zmax / (xmax - xmin)) * 6],
                    color=COL[n], lw=4, solid_capstyle="round")
            done = tr[i, 0] >= tr[-1, 0] - 1e-9
            lab = f"{LABEL[n]}: {tr[i,1]:5.1f} m" + ("  (landed)" if done else "")
            ax.plot([], [], color=COL[n], label=lab)
        ax.set_xlim(xmin, xmax); ax.set_ylim(0, zmax)
        ax.set_xlabel("downrange (m)"); ax.set_ylabel("height (m)")
        ax.set_title(f"{title}   t = {t:4.1f} s  ({speed:.0f}x)", fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    frames += [frames[-1]] * fps
    imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()
    os.makedirs(MEDIA, exist_ok=True)
    designs, ev, ck = load()
    fig_planforms(designs)
    fig_paths(designs)
    fig_heldout(ev)
    fig_search(ck, ev)
    fig_sanity(designs)
    if not args.no_gif:
        nom_c, nom_g = sim.still_air()
        gif(designs, os.path.join(MEDIA, "flight.gif"), ["baseline_hand", "cma_best"], nom_c[0], nom_g[0],
            "Still air, nominal launch: hand baseline vs evolved")
        # the good failure: the near-tailless MAP-Elites champion tumbles in a held-out gust and
        # then glides tail-first, backwards, for 145 m (a loophole in the flat-plate post-stall model)
        hc, hg = sim.heldout_conditions(n=64)
        gif(designs, os.path.join(MEDIA, "tumble_backwards.gif"), ["cma_best", "mapel_best"], hc[10], hg[10],
            "Held-out launch #10 (8.8 m/s, headwind, gusts)", speed=3.0)
    print("media written to", MEDIA)


if __name__ == "__main__":
    main()
