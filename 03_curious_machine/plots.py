"""Figures from finished runs and evaluations.

  ../.venv/bin/python plots.py            # reads runs/*, writes media/*.png and runs/summary.json
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from env import INTER_NAMES  # noqa: E402

COL = {"random": "#8a8a85", "count": "#2a78d6", "rnd": "#eb6834", "rnd_ep": "#1baf7a",
       "disagree": "#d55181", "ep": "#eda100", "ngu": "#008300"}
NAME = {"random": "Random walk", "count": "PPO + hashed counts", "rnd": "PPO + RND",
        "rnd_ep": "PPO + RND x episodic count", "disagree": "PPO + ensemble disagreement",
        "ep": "PPO + episodic count only (ablation)",
        "ngu": "PPO + NGU-style (inverse-dynamics kNN)"}
AGENTS = ["random", "count", "rnd", "rnd_ep", "ep", "ngu", "disagree"]
LEARNED = AGENTS[1:]
INK, MUTED = "#222222", "#6b6b66"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbb5")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e6e6e1", lw=0.8)
    ax.set_axisbelow(True)


def load_logs(prefix):
    out = []
    for d in sorted(glob.glob(f"runs/{prefix}_s*")):
        p = os.path.join(d, "log.jsonl")
        if os.path.exists(p):
            rows = [json.loads(l) for l in open(p) if l.strip()]
            if rows:
                out.append(rows)
    return out


def load_evals(tag):
    """runs/eval_{tag}_s*.json (learned) or runs/eval_random_{env}.json."""
    return [json.load(open(p)) for p in sorted(glob.glob(f"runs/eval_{tag}*.json"))]


def training_curves(agents, env_tag, out, random_eval):
    keys = [("ep_rooms", "rooms reached (of 9)"), ("ep_interactions", "useful interactions (of 8)"),
            ("ep_coverage", "cells visited (of 233)")]
    if env_tag == "tv":
        keys.append(("ep_tv_steps", "steps with noisy TV in view (of 512)"))
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.4))
    for ax, (k, lab) in zip(axes, keys):
        style(ax)
        for ag in agents:
            logs = load_logs(f"{ag}_{env_tag}")
            if not logs:
                continue
            n = min(len(l) for l in logs)
            x = np.array([r["steps"] for r in logs[0][:n]]) / 1e6
            y = np.array([[r.get(k, np.nan) for r in l[:n]] for l in logs])
            m = np.nanmean(y, 0)
            ax.fill_between(x, np.nanmin(y, 0), np.nanmax(y, 0), color=COL[ag], alpha=0.15, lw=0)
            ax.plot(x, m, color=COL[ag], lw=2, label=f"{NAME[ag]} ({len(logs)} seeds)")
        if random_eval is not None:
            rk = k[3:]
            ax.axhline(random_eval[rk]["mean"], color=COL["random"], lw=2, ls="--", label=NAME["random"])
        ax.set_title(lab, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("training env steps (millions)", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Per-episode discovery during training (512-action episodes; band = min-max over seeds)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def interaction_timeline(evals, out, title):
    """evals: {agent: [eval dicts per seed]}. Left: mean #interactions discovered by step t.
    Right: fraction of episodes in which each mechanism was triggered."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    style(a1); style(a2)
    steps = np.arange(513)
    ags = [a for a in AGENTS if a in evals]
    w = 0.8 / len(ags)
    ends = []
    for j, ag in enumerate(ags):
        it = np.array([e["inter_t"] for ev in evals[ag] for e in ev["episodes"]])  # [eps, 8]
        disc = np.where(it >= 0, it, 10 ** 9)
        cum = (disc[:, :, None] <= steps[None, None, :]).sum(1).mean(0)
        a1.plot(steps, cum, color=COL[ag], lw=2, label=NAME[ag])
        ends.append((cum[-1], ag))
        rate = (it >= 0).mean(0)
        a2.bar(np.arange(8) + (j - (len(ags) - 1) / 2) * w, rate, w * 0.92, color=COL[ag], label=NAME[ag])
    last = -1e9  # end-of-line value labels, nudged apart so they never collide
    for v, ag in sorted(ends):
        y = max(v, last + 0.22)
        a1.text(514, y, f"{v:.2f}", color=INK, fontsize=8, va="center")
        last = y
    a1.set_xlim(0, 560)
    a1.set_xlabel("step within episode", fontsize=9, color=MUTED)
    a1.set_title("useful interactions discovered so far (mean over episodes)", fontsize=10, color=INK, loc="left")
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    a2.set_xticks(np.arange(8))
    a2.set_xticklabels([n.replace(" (", "\n(") for n in INTER_NAMES], fontsize=7.5, rotation=0)
    a2.set_ylim(0, 1.05)
    a2.set_title("fraction of episodes that triggered each mechanism (easy -> hard)", fontsize=10,
                 color=INK, loc="left")
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def load_episodes(name):
    out = []
    for d in sorted(glob.glob(f"runs/{name}_s*")):
        p = os.path.join(d, "episodes.jsonl")
        if os.path.exists(p):
            out.append([json.loads(l) for l in open(p) if l.strip()])
    return out


def lifetime_table(env_tag):
    """Discovery over the whole training budget (every episode the agent ever played)."""
    res = {}
    for ag in AGENTS:
        runs = load_episodes(f"{ag}_{env_tag}")
        if not runs:
            continue
        r = dict(n_seeds=len(runs), env_steps=int(runs[0][-1]["step"]))
        it = [np.array([e["inter_t"] for e in eps]) >= 0 for eps in runs]
        r["mechanisms_ever"] = [int(x.any(0).sum()) for x in it]
        r["max_rooms_ever"] = [int(max(e["rooms"] for e in eps)) for eps in runs]
        r["episodes_with"] = {INTER_NAMES[k]: [int(x[:, k].sum()) for x in it] for k in range(8)}
        r["n_episodes"] = [len(eps) for eps in runs]
        res[ag] = r
    return res


def lifetime_discovery(env_tag, out):
    """Left: distinct mechanisms ever triggered vs training steps. Right: share of all training
    episodes that triggered each mechanism (reliability)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    style(a1); style(a2)
    ags = [a for a in AGENTS if load_episodes(f"{a}_{env_tag}")]
    w = 0.8 / max(len(ags), 1)
    for j, ag in enumerate(ags):
        runs = load_episodes(f"{ag}_{env_tag}")
        grid = np.linspace(0, min(r[-1]["step"] for r in runs), 200)
        ys, rates = [], []
        for eps in runs:
            st = np.array([e["step"] for e in eps])
            it = np.array([e["inter_t"] for e in eps]) >= 0
            # cumulative number of episodes that got the key (hard-ish) by each step
            ys.append(np.interp(grid, st, np.cumsum(it[:, 3]), left=0))
            rates.append(it.mean(0))
        ys = np.array(ys)
        a1.plot(grid / 1e6, ys.mean(0), color=COL[ag], lw=2, label=f"{NAME[ag]} ({len(runs)} seeds)")
        rate = np.mean(rates, 0)
        a2.bar(np.arange(8) + (j - (len(ags) - 1) / 2) * w, rate, w * 0.92, color=COL[ag], label=NAME[ag])
    a1.set_xlabel("env steps (millions) - same budget for every agent", fontsize=9, color=MUTED)
    a1.set_title("cumulative episodes in which the key was picked up", fontsize=10, color=INK, loc="left")
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    a2.set_xticks(np.arange(8))
    a2.set_xticklabels([n.replace(" (", "\n(") for n in INTER_NAMES], fontsize=7.5)
    a2.set_ylim(0, 1.05)
    a2.set_title("share of all training episodes that triggered each mechanism", fontsize=10, color=INK, loc="left")
    fig.suptitle("Discovery over the whole training budget - " + ("base world" if env_tag == "base" else "noisy-TV world"),
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def summary_table(evals):
    """Mean over seeds of per-seed means (+ std across seeds)."""
    rows = {}
    for ag, evs in evals.items():
        r = {}
        for f in ["rooms", "interactions", "coverage", "unproductive", "tv_steps", "repeat_interactions"]:
            v = np.array([ev[f]["mean"] for ev in evs])
            r[f] = (float(v.mean()), float(v.std()), len(v))
        r["all9"] = float(np.mean([ev["frac_episodes_all_9_rooms"] for ev in evs]))
        rows[ag] = r
    return rows


if __name__ == "__main__":
    os.makedirs("media", exist_ok=True)
    summary = {}
    for env_tag in ["base", "tv"]:
        evals = {}
        rnd_ev = load_evals(f"random_{env_tag}")
        if rnd_ev:
            evals["random"] = rnd_ev
        for ag in LEARNED:
            ev = load_evals(f"{ag}_{env_tag}_s")
            if ev:
                evals[ag] = ev
        if len(evals) < 2:
            continue
        lifetime_discovery(env_tag, f"media/lifetime_discovery_{env_tag}.png")
        training_curves([a for a in LEARNED if a in evals], env_tag,
                        f"media/training_curves_{env_tag}.png", rnd_ev[0] if rnd_ev else None)
        t = "Interaction timeline - " + ("base world" if env_tag == "base" else "world with a noisy TV in room R10")
        interaction_timeline(evals, f"media/interaction_timeline_{env_tag}.png", t)
        summary[env_tag] = summary_table(evals)
        summary[env_tag + "_lifetime"] = lifetime_table(env_tag)
    json.dump(summary, open("runs/summary.json", "w"), indent=1)
    for env_tag, rows in summary.items():
        if env_tag.endswith("lifetime"):
            print(f"== {env_tag}")
            for ag, r in rows.items():
                print(f"{ag:9s} mech_ever={r['mechanisms_ever']} max_rooms={r['max_rooms_ever']} "
                      + " ".join(f"{k[:6]}={v}" for k, v in r["episodes_with"].items()))
            continue
        print(f"== {env_tag}")
        for ag, r in rows.items():
            print(f"{ag:9s} " + "  ".join(f"{k}={v[0]:.2f}±{v[1]:.2f}" for k, v in r.items() if k != "all9")
                  + f"  all9={r['all9']:.2f}  n={r['rooms'][2]}")
