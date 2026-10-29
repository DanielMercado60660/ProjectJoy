"""Render the demo media into media/.

  ../.venv/bin/python make_media.py

  media/rogues_gallery.png      nastiest designer levels vs solver_v1, with a failed trajectory
  media/nightmares.png          the designer's best level against each solver (different weaknesses)
  media/escape.gif              the curriculum solver escaping held-out + designer dungeons
  media/before_after.gif        the same designer dungeons: solver_v1 lost vs solver_cur3 escaping
  media/training_curves.png     PPO training curves (solver_v1)
  media/designer_evolution.png  designer fitness and level structure over generations
  media/curriculum.png          before/after curriculum comparison (matched compute)
  media/curriculum_rounds.png   3 curriculum rounds vs control, seed replicates, exploitability
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio

from dungeon.mazes import bfs_path
from dungeon.env import R
from dungeon.policy import load_policy
from dungeon.rollout import run_policy
from dungeon.scripted import run_batch, ALGOS, oracle_lengths
from dungeon.data import test_set

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "media")
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
         green="#008300", violet="#6250d6", red="#e34948")
WALL, FLOOR = "#2e2a26", "#efe8da"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": "#c9c6bf", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def ckpt(name):
    return os.path.join(HERE, "checkpoints", name, "weights.safetensors")


def have(path):
    return os.path.exists(os.path.join(HERE, path))


def draw_level(ax, w, s, g, traj=None, show_bfs=True, title=None, traj_color=C["red"]):
    ax.imshow(w, cmap=ListedColormap([FLOOR, WALL]), vmin=0, vmax=1, interpolation="nearest")
    if show_bfs:
        p = bfs_path(w, s[0], s[1], g[0], g[1])
        ax.plot(p[:, 1], p[:, 0], ls=(0, (1.5, 1.5)), color=C["blue"], lw=1.6, alpha=0.9)
    if traj is not None and len(traj) > 1:
        jit = np.random.default_rng(0).normal(0, 0.06, traj.shape)
        pts = (traj[:, ::-1] + jit).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], 1)
        lc = LineCollection(segs, colors=traj_color, linewidths=1.6,
                            alpha=np.linspace(0.25, 0.95, len(segs)))
        ax.add_collection(lc)
    ax.scatter([s[1]], [s[0]], s=70, color=C["aqua"], edgecolor=SURF, linewidth=1.5, zorder=5)
    ax.scatter([g[1]], [g[0]], s=140, marker="*", color=C["yellow"], edgecolor=INK, linewidth=0.8, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, color=INK)


def pick_traj(policy, L, i, K=32, want_fail=True, seed=0):
    one = {k: np.repeat(v[i:i + 1], K, axis=0) for k, v in L.items()}
    o = run_policy(policy, one, greedy=False, seed=seed, keep_traj=True)
    succ = o["success"]
    cand = np.nonzero(~succ if want_fail else succ)[0]
    j = cand[0] if len(cand) else int(np.argmax(o["steps"]))
    return o["traj"][j], int((~succ).sum()), K


def load_designer(name):
    z = np.load(os.path.join(HERE, "results", f"designer_{name}.npz"))
    return z, dict(walls=z["walls"], start=z["start"], goal=z["goal"])


GALLERY_RUNS = [("solver_v1", "v1_s0", "v1"), ("solver_v1", "v1_s1", "v1"), ("solver_ctl1", "ctl1_s2", "ctl1"),
                ("solver_cur1", "cur1_s2", "cur1"), ("solver_v1", "v1_md12_s7", "v1, exit >= 12 steps")]


def rogues_gallery(per_run=2):
    """Each designer run converges on its own motif, so show the top levels of several runs, each against
    the solver it was evolved against. Candidates (final population + archive) are re-scored with 32
    fresh rollouts and picked greedily for diversity."""
    runs = [r for r in GALLERY_RUNS if have(f"results/designer_{r[1]}.npz") and os.path.exists(ckpt(r[0]))]
    if not runs:
        return
    picks = []
    from dungeon.designer import policy_eval_fn
    for solver, run, label in runs:
        pol = load_policy(ckpt(solver))
        z, _ = load_designer(run)
        W = np.concatenate([z["walls"], z["arch_walls"]]); S = np.concatenate([z["start"], z["arch_start"]])
        G = np.concatenate([z["goal"], z["arch_goal"]]); pre = np.concatenate([z["fit"], z["arch_fit"]])
        rng = np.random.default_rng(0)
        good = np.nonzero(pre >= 0.75)[0]
        if len(good) == 0:
            good = np.argsort(-pre)[:50]
        top = rng.choice(good, min(300, len(good)), replace=False)
        C = dict(walls=W[top], start=S[top], goal=G[top])
        fresh = policy_eval_fn(pol, 32)(C, seed=99)["fitness"]
        chosen = []
        for i in np.argsort(-fresh):
            if all((C["walls"][i] != C["walls"][j]).sum() > 25 for j in chosen):
                chosen.append(i)
            if len(chosen) == per_run:
                break
        for i in chosen:
            tr, nf, K = pick_traj(pol, C, i)
            picks.append((C["walls"][i], C["start"][i], C["goal"][i], tr, nf, K, label))
    ncol = len(runs)
    fig, axes = plt.subplots(per_run, ncol, figsize=(2.7 * ncol, 3.0 * per_run + 1.1), squeeze=False)
    for k, (w, s0, g, tr, nf, K, label) in enumerate(picks):
        ax = axes[k % per_run, k // per_run]
        opt = oracle_lengths(dict(walls=w[None], start=s0[None], goal=g[None]))[0]
        draw_level(ax, w, s0, g, tr, title=f"vs {label}\nfails {nf}/{K} · BFS {opt} steps")
    for ax in axes.flat[len(picks):]:
        ax.axis("off")
    fig.suptitle("Rogues' gallery: dungeons evolved against PPO-GRU solvers (one column per designer run)\n"
                 "red = one failed 200-step attempt (fades in over time) · dotted blue = BFS shortest path · "
                 "green dot = start · star = exit", fontsize=9, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2.2)
    fig.savefig(os.path.join(MEDIA, "rogues_gallery.png"), dpi=130)
    plt.close(fig)


def nightmares():
    runs = [("wall_follow", "wall_follow_s0", "wall follower"), ("tremaux", "tremaux_s0", "Trémaux (DFS + memory)"),
            ("solver_v1", "v1_s0", "PPO-GRU solver_v1"), ("solver_cur1", "cur1_s2", "PPO-GRU after curriculum")]
    runs = [r for r in runs if have(f"results/designer_{r[1]}.npz")]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(3.0 * len(runs), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (solver, run, label) in zip(axes, runs):
        z, L = load_designer(run)
        i = int(np.argmax(z["fresh_fitness"]))
        one = {k: v[i:i + 1] for k, v in L.items()}
        if solver in ALGOS:
            rep = {k: np.repeat(v, 32, axis=0) for k, v in one.items()}
            o = run_batch(solver, rep, seed=5, keep_traj=True)
            j = int(np.argmax(o["steps"]))
            tr, nf = o["traj"][j], int((~o["success"]).sum())
        else:
            tr, nf, _ = pick_traj(load_policy(ckpt(solver)), one, 0)
        opt = oracle_lengths(one)[0]
        draw_level(ax, one["walls"][0], one["start"][0], one["goal"][0], tr,
                   title=f"vs {label}\nregret {z['fresh_fitness'][i]:.2f} · fails {nf}/32 · BFS {opt}")
    fig.suptitle("Each solver's personal nightmare: the designer's top level after evolving against it",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(MEDIA, "nightmares.png"), dpi=130)
    plt.close(fig)


def escape_gif(solver="solver_cur3"):
    if not os.path.exists(ckpt(solver)):
        solver = "solver_v1"
    if not os.path.exists(ckpt(solver)):
        return
    pol = load_policy(ckpt(solver))
    panels = []
    for fam in ("spiral", "caves"):
        L = test_set(fam)
        opt = oracle_lengths(L)
        i = int(np.argmax(opt[:60]))  # a long one
        panels.append((f"held-out {fam}", {k: v[i:i + 1] for k, v in L.items()}))
    for run in ("v1_s1", "v1_md12_s7"):
        if have(f"results/designer_{run}.npz"):
            z, L = load_designer(run)
            i = int(np.argsort(-z["fresh_fitness"])[0])
            panels.append((f"designer level ({run})", {k: v[i:i + 1] for k, v in L.items()}))
    trajs, oks = [], []
    for _, one in panels:
        rep = {k: np.repeat(v, 16, axis=0) for k, v in one.items()}
        o = run_policy(pol, rep, greedy=False, seed=11, keep_traj=True)
        cand = np.nonzero(o["success"])[0]
        j = cand[np.argmin(o["steps"][cand])] if len(cand) else 0
        trajs.append(o["traj"][j]); oks.append(bool(o["success"][j]))
    Tm = max(len(t) for t in trajs)
    frames = []
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.0), dpi=90)
    for t in list(range(0, Tm, 1)) + [Tm - 1] * 12:
        for ax, (label, one), tr, ok in zip(axes, panels, trajs, oks):
            ax.clear()
            k = min(t, len(tr) - 1)
            draw_level(ax, one["walls"][0], one["start"][0], one["goal"][0], tr[:k + 1], show_bfs=False,
                       traj_color=C["violet"])
            r, c = tr[k]
            ax.add_patch(plt.Rectangle((c - R - 0.5, r - R - 0.5), 2 * R + 1, 2 * R + 1, fill=False,
                                       ec=C["orange"], lw=1.5, ls="--"))
            ax.scatter([c], [r], s=60, color=C["orange"], edgecolor=INK, zorder=6)
            ax.set_xlim(-0.5, one["walls"].shape[2] - 0.5); ax.set_ylim(one["walls"].shape[1] - 0.5, -0.5)
            done = k == len(tr) - 1
            ax.set_title(f"{label}\nstep {k}" + ((" ESCAPED" if ok else " timeout") if done else ""),
                         fontsize=8, color=INK)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(os.path.join(MEDIA, "escape.gif"), frames, duration=0.08, loop=0)


def training_curves(name="solver_v1"):
    p = os.path.join(HERE, "logs", f"{name}.jsonl")
    if not os.path.exists(p):
        return
    recs = [json.loads(l) for l in open(p)]
    tr = [r for r in recs if "train_success" in r]
    ev = [r for r in recs if "eval" in r]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot([r["env_steps"] / 1e6 for r in tr], [r["train_success"] for r in tr], color=C["blue"], lw=2)
    ax.set_xlabel("env steps (M)"); ax.set_ylabel("success rate"); ax.set_ylim(0, 1.02)
    ax.set_title("solver_v1 training episodes (train families)", fontsize=10)
    ax.grid(alpha=0.25)
    ax = axes[1]
    fams = list(ev[0]["eval"].keys()) if ev else []
    cols = [C["blue"], C["orange"], C["aqua"], C["violet"], C["magenta"]]
    for f, col in zip(fams, cols):
        xs = [r["env_steps"] / 1e6 for r in ev]
        ys = [r["eval"][f]["success"] for r in ev]
        ls = "--" if f in ("caves", "spiral") else "-"
        ax.plot(xs, ys, color=col, lw=2, ls=ls, label=f + (" (held out)" if ls == "--" else ""))
        ax.annotate(f, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2,
                    va="center")
    ax.set_xlabel("env steps (M)"); ax.set_ylabel("test success (200 levels)"); ax.set_ylim(0, 1.02)
    ax.set_title("periodic test evaluation per maze family", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "training_curves.png"), dpi=130)
    plt.close(fig)


def designer_evolution():
    runs = [("v1_s0", "vs solver_v1 (seed 0)", C["blue"], "-"), ("v1_s1", "vs solver_v1 (seed 1)", C["blue"], "--"),
            ("v1_drift", "neutral drift (no selection)", C["violet"], ":"),
            ("ctl1_s2", "vs control fine-tune", C["orange"], "-"), ("cur1_s2", "vs curriculum fine-tune", C["aqua"], "-")]
    runs = [r for r in runs if have(f"logs/designer_{r[0]}.jsonl")]
    if not runs:
        return
    keys = [("fit_mean", "mean regret of population"), ("fail_mean", "solver failure rate"),
            ("path_len", "BFS path length"), ("dead_ends", "dead ends"), ("loops", "loops (cycle rank)"),
            ("open2x2", "open 2x2 areas")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, (k, lab) in zip(axes.flat, keys):
        for run, label, col, ls in runs:
            h = [json.loads(l) for l in open(os.path.join(HERE, "logs", f"designer_{run}.jsonl"))]
            ax.plot([r["gen"] for r in h], [r[k] for r in h], color=col, ls=ls, lw=1.8, label=label)
        ax.set_title(lab, fontsize=10); ax.set_xlabel("generation"); ax.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Evolutionary designer (population 128, 8 solver rollouts per level per generation)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(MEDIA, "designer_evolution.png"), dpi=130)
    plt.close(fig)


def curriculum_plot():
    p = os.path.join(HERE, "results", "eval.json")
    if not os.path.exists(p):
        return
    res = json.load(open(p))["results"]
    sets = [s for s in ["dfs", "braid", "rooms", "caves", "spiral", "designer_vs_v1 (held-out run)",
                        "designer_vs_ctl1", "designer_vs_cur1"] if s in res]
    solvers = [("solver_v1", "v1 (families)", C["blue"]), ("solver_ctl1", "v1 + 6M families (control)", C["orange"]),
               ("solver_cur1", "v1 + 6M families+designer (curriculum)", C["aqua"]),
               ("tremaux", "Trémaux baseline", C["violet"])]
    solvers = [s for s in solvers if s[0] in res[sets[0]]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, metric, lab in ((axes[0], "success", "success rate"), (axes[1], "spl", "SPL (success × BFS/steps)")):
        x = np.arange(len(sets))
        wdt = 0.8 / len(solvers)
        for j, (s, label, col) in enumerate(solvers):
            ys = [res[st][s][metric] for st in sets]
            ax.bar(x + (j - (len(solvers) - 1) / 2) * wdt, ys, wdt * 0.92, color=col, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([st.replace("designer_vs_", "designer vs\n").replace(" (held-out run)", "\n(held-out run)")
                            .replace("caves", "caves*").replace("spiral", "spiral*") for st in sets], fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel(lab); ax.grid(axis="y", alpha=0.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=len(solvers), bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Before / after the designer curriculum (matched 6M-step fine-tunes; * = held-out family)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(os.path.join(MEDIA, "curriculum.png"), dpi=130)
    plt.close(fig)


def rounds_plot():
    """Curriculum rounds 0-3 (cur vs matched control) from results/eval_r2.json + exploitability runs."""
    p = os.path.join(HERE, "results", "eval_r2.json")
    if not os.path.exists(p):
        return
    res = json.load(open(p))["results"]
    lines = {"curriculum": ["solver_v1", "solver_cur1", "solver_cur2", "solver_cur3"],
             "control": ["solver_v1", "solver_ctl1", "solver_ctl2", "solver_ctl3"]}
    reps = {"curriculum": ["solver_cur1_r11", "solver_cur1_r12"], "control": ["solver_ctl1_r11", "solver_ctl1_r12"]}
    cols = {"curriculum": C["aqua"], "control": C["orange"]}
    panels = [("held-out families (caves+spiral)", lambda sv: np.mean([res[f][sv]["success"] for f in ("caves", "spiral")])),
              ("designer levels vs v1 (held-out run)", lambda sv: res["designer_v1_s1"][sv]["success"]),
              ("designer levels vs ctl1 (trap motif)", lambda sv: res["designer_ctl1_s2"][sv]["success"])]
    expl = {}
    for m in ("v1", "ctl1", "cur1", "ctl2", "cur2", "ctl3", "cur3"):
        f = os.path.join(HERE, "logs", f"designer_expl_{m}_s8.jsonl")
        if os.path.exists(f):
            h = [json.loads(l) for l in open(f)]
            expl[m] = np.mean([r["fail_mean"] for r in h[:21]])  # mean failure over generations 0-20
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (title, fn) in zip(axes, panels):
        for name, svs in lines.items():
            svs = [s for s in svs if s in res[next(iter(res))]]
            ax.plot(range(len(svs)), [fn(s) for s in svs], "-o", color=cols[name], lw=2, ms=6, label=name)
            for s in reps[name]:
                if s in res[next(iter(res))]:
                    ax.plot([1], [fn(s)], "o", color=cols[name], ms=5, alpha=0.5, mfc="none")
        ax.set_title(title, fontsize=10); ax.set_xticks(range(4)); ax.set_xlabel("round (6M fine-tune steps each)")
        ax.set_ylabel("success rate"); ax.set_ylim(0, 1.03); ax.grid(alpha=0.25)
    ax = axes[3]
    for name, key in (("curriculum", "cur"), ("control", "ctl")):
        xs, ys = [0], [expl.get("v1", np.nan)]
        for r in (1, 2, 3):
            xs.append(r); ys.append(expl.get(f"{key}{r}", np.nan))
        ax.plot(xs, ys, "-o", color=cols[name], lw=2, ms=6, label=name)
    ax.set_title("exploitability: fresh designer, mean fail\nover generations 0-20 (lower = more robust)", fontsize=10)
    ax.set_xticks(range(4)); ax.set_xlabel("round"); ax.set_ylim(0, 1.03); ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Designer curriculum over 3 rounds vs matched-compute control (hollow dots = extra seeds at round 1)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(MEDIA, "curriculum_rounds.png"), dpi=130)
    plt.close(fig)


def _animate(panels, path, fps_dur=0.08, stride=1):
    """panels: list of (title, one-level dict, traj, success). Writes an animated GIF."""
    Tm = max(len(p[2]) for p in panels)
    frames = []
    fig, axes = plt.subplots(1, len(panels), figsize=(2.7 * len(panels), 3.1), dpi=90)
    axes = np.atleast_1d(axes)
    for t in list(range(0, Tm, stride)) + [Tm - 1] * 15:
        for ax, (label, one, tr, ok) in zip(axes, panels):
            ax.clear()
            k = min(t, len(tr) - 1)
            draw_level(ax, one["walls"][0], one["start"][0], one["goal"][0], tr[:k + 1], show_bfs=False,
                       traj_color=C["violet"])
            r, c = tr[k]
            ax.add_patch(plt.Rectangle((c - R - 0.5, r - R - 0.5), 2 * R + 1, 2 * R + 1, fill=False,
                                       ec=C["orange"], lw=1.5, ls="--"))
            ax.scatter([c], [r], s=60, color=C["orange"], edgecolor=INK, zorder=6)
            ax.set_xlim(-0.5, one["walls"].shape[2] - 0.5); ax.set_ylim(one["walls"].shape[1] - 0.5, -0.5)
            done = k == len(tr) - 1
            ax.set_title(f"{label}\nstep {k}" + ((" ESCAPED" if ok else " still lost") if done else ""),
                         fontsize=8, color=INK)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(path, frames, duration=fps_dur, loop=0)


def before_after_gif():
    """Same designer dungeons, before (solver_v1) and after three curriculum rounds (solver_cur3)."""
    runs = [r for r in ("v1_md12_s7", "v1_s1") if have(f"results/designer_{r}.npz")]
    if not runs or not os.path.exists(ckpt("solver_cur3")):
        return
    pols = {"v1": load_policy(ckpt("solver_v1")), "cur3": load_policy(ckpt("solver_cur3"))}
    panels = []
    for run in runs:
        z, L = load_designer(run)
        i = int(np.argsort(-z["fresh_fitness"])[0])
        one = {k: v[i:i + 1] for k, v in L.items()}
        for tag, want_fail in (("v1", True), ("cur3", False)):
            tr, nf, K = pick_traj(pols[tag], one, 0, K=16, want_fail=want_fail, seed=3)
            ok = tuple(tr[-1]) == tuple(one["goal"][0])
            name = "before: solver_v1" if tag == "v1" else "after 3 rounds: solver_cur3"
            panels.append((f"{name} ({run})\nfails {nf}/{K}", one, tr, ok))
    _animate(panels, os.path.join(MEDIA, "before_after.gif"), fps_dur=0.06, stride=2)


def main():
    os.makedirs(MEDIA, exist_ok=True)
    for fn in (training_curves, designer_evolution, curriculum_plot, rounds_plot, rogues_gallery, nightmares,
               escape_gif, before_after_gif):
        print("rendering", fn.__name__, flush=True)
        fn()


if __name__ == "__main__":
    main()
