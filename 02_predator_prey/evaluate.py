"""Held-out evaluation: fixed opponent suite, energy proxy, cross-play over archived checkpoints.

  ../.venv/bin/python evaluate.py                  # full eval (500 episodes per matchup)
  ../.venv/bin/python evaluate.py --n 32 --xn 16   # smoke size

Writes results/eval.json, results/eval_tables.md, media/crossplay.png, media/arms_race_curves.png,
media/occupancy.png. Evaluation seeds (EVAL_SEED) differ from training seeds (0..2) and from the
in-training quick eval (999). Policies sample from their action distribution unless marked argmax.
"""
import argparse, csv, json, os
import numpy as np
import mlx.core as mx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim
from policy import new_model, set_np, act, load_pair

EVAL_SEED = 12345


class Net:
    def __init__(self, params, role, name, greedy=False, seed=0):
        self.m = new_model(0)
        set_np(self.m, params)
        self.role, self.name, self.greedy = role, name, greedy
        self.rng = np.random.default_rng(seed)

    def act(self, env):
        op, oe = env.obs()
        return act(self.m, op if self.role == "p" else oe, self.rng, self.greedy)[0]


class Script:
    def __init__(self, kind, role, seed=0):
        self.s = sim.Scripted(kind, role, seed)
        self.name = kind

    def act(self, env):
        return self.s.act(env)


def play(P, E, n, seed=EVAL_SEED, track=False):
    """First episode of each of n parallel envs. Returns metrics dict (and positions if track)."""
    env = sim.Arena(n, seed)
    alive = np.ones(n, bool)
    cap = np.zeros(n, bool)
    t_end = np.full(n, sim.T_MAX)
    energy = np.zeros((n, 2))     # seconds of thrust + seconds of sprint (sprint counts double)
    path = np.zeros((n, 2))       # distance travelled
    occ = []
    for t in range(sim.T_MAX):
        prev = env.pos.copy()
        a_p, a_e = P.act(env), E.act(env)
        _, _, done, c, info = env.step(a_p, a_e)
        m = alive[:, None]
        energy += m * sim.DT * (info["thrust"].astype(float) + info["sprint"])
        moved = np.linalg.norm(info["pos"] - prev, axis=2)
        path += (m & ~done[:, None]) * moved
        if track:
            occ.append(info["pos"][alive & ~done])
        cap |= c & alive
        t_end[done & alive] = t + 1
        alive &= ~done
        if not alive.any():
            break
    dur = t_end * sim.DT
    r = {
        "capture_rate": float(cap.mean()),
        "capture_se": float(np.sqrt(cap.mean() * (1 - cap.mean()) / n)),
        "mean_capture_time_s": float(dur[cap].mean()) if cap.any() else None,
        "mean_episode_s": float(dur.mean()),
        "P_energy_per_s": float((energy[:, 0] / dur).mean()),
        "E_energy_per_s": float((energy[:, 1] / dur).mean()),
        "P_energy_per_capture": float(energy[:, 0].sum() / max(cap.sum(), 1)),
        "E_energy_per_escape_s": float(energy[:, 1].sum() / max(dur.sum(), 1e-9)),
        "P_path_per_s": float((path[:, 0] / dur).mean()),
        "E_path_per_s": float((path[:, 1] / dur).mean()),
        "n": n,
    }
    if track:
        return r, np.concatenate(occ)
    return r


def load_archive(path):
    z = np.load(path)
    snaps = {}
    for k in z.files:
        it, role, rest = k.split("/", 2)
        snaps.setdefault(int(it), {"P": {}, "E": {}})[role][rest] = z[k]
    return dict(sorted(snaps.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--xn", type=int, default=200, help="episodes per cross-play cell")
    ap.add_argument("--xsnaps", type=int, default=8)
    ap.add_argument("--results", default="results")
    ap.add_argument("--media", default="media")
    a = ap.parse_args()
    mx.set_default_device(mx.cpu)
    os.makedirs(a.results, exist_ok=True)
    os.makedirs(a.media, exist_ok=True)

    pf, ef, meta = load_pair(os.path.join(a.ckpt, "final.npz"))
    arch = load_archive(os.path.join(a.ckpt, "archive.npz"))
    its = list(arch)

    pursuers = [Script(k, "p", 1) for k in sim.PURSUER_SUITE] + [Net(pf, "p", "learned", seed=2),
                                                                  Net(pf, "p", "learned-argmax", True)]
    evaders = [Script(k, "e", 3) for k in sim.EVADER_SUITE] + [Net(ef, "e", "learned", seed=4),
                                                                Net(ef, "e", "learned-argmax", True)]
    suite = {}
    for P in pursuers:
        for E in evaders:
            suite[f"{P.name}|{E.name}"] = play(P, E, a.n)
            print(f"{P.name:>15} vs {E.name:<15} capture {suite[f'{P.name}|{E.name}']['capture_rate']:.3f}", flush=True)

    # cross-play over evenly spaced archive snapshots
    pick = sorted(set(np.linspace(0, len(its) - 1, min(a.xsnaps, len(its))).round().astype(int)))
    xs = [its[i] for i in pick]
    X = np.zeros((len(xs), len(xs)))
    for i, ip in enumerate(xs):
        for j, je in enumerate(xs):
            X[i, j] = play(Net(arch[ip]["P"], "p", f"P{ip}", seed=5), Net(arch[je]["E"], "e", f"E{je}", seed=6),
                           a.xn)["capture_rate"]
    # each snapshot vs the scripted suite (does competence transfer beyond the training rival?)
    vs_script = {}
    for it in xs:
        vs_script[it] = {
            "P_vs_" + k: play(Net(arch[it]["P"], "p", "", seed=7), Script(k, "e", 8), a.xn)["capture_rate"]
            for k in sim.EVADER_SUITE}
        vs_script[it].update({
            "E_escape_vs_" + k: 1 - play(Script(k, "p", 9), Net(arch[it]["E"], "e", "", seed=10), a.xn)["capture_rate"]
            for k in sim.PURSUER_SUITE})

    # occupancy in learned self-play (spots corner hugging / obstacle orbiting)
    _, occ_l = play(Net(pf, "p", "", seed=11), Net(ef, "e", "", seed=12), a.n, track=True)
    _, occ_s = play(Script("lead", "p", 13), Script("smartflee", "e", 14), a.n, track=True)

    def wall_frac(xy):
        return float((np.abs(xy).max(1) > sim.L - 0.35).mean())

    def corner_frac(xy):
        return float(((np.abs(xy) > sim.L - 0.5).all(1)).mean())

    occ_stats = {
        "learned_selfplay": {"E_near_wall": wall_frac(occ_l[:, 1]), "E_in_corner": corner_frac(occ_l[:, 1]),
                             "P_near_wall": wall_frac(occ_l[:, 0])},
        "lead_vs_smartflee": {"E_near_wall": wall_frac(occ_s[:, 1]), "E_in_corner": corner_frac(occ_s[:, 1])},
        "uniform_area_reference": {"near_wall": 1 - ((2 * sim.L - 0.7) / (2 * sim.L)) ** 2,
                                   "in_corner": 4 * 0.5 ** 2 / (2 * sim.L) ** 2},
    }

    out = {"final_meta": {k: (v.item() if v.ndim == 0 else v.tolist()) for k, v in meta.items()},
           "eval_seed": EVAL_SEED, "episodes_per_matchup": a.n, "suite": suite,
           "crossplay": {"snapshots": xs, "capture_rate[P_row][E_col]": X.tolist(), "episodes_per_cell": a.xn},
           "snapshots_vs_scripted": vs_script, "occupancy": occ_stats}
    json.dump(out, open(os.path.join(a.results, "eval.json"), "w"), indent=1)

    # markdown tables
    pn = [p.name for p in pursuers]
    en = [e.name for e in evaders]
    lines = [f"Capture rate (pursuer row vs evader column), {a.n} held-out episodes each, seed {EVAL_SEED}; "
             f"± is one binomial standard error\n",
             "| pursuer \\ evader | " + " | ".join(en) + " |", "|---" * (len(en) + 1) + "|"]
    for p in pn:
        lines.append(f"| {p} | " + " | ".join(
            f"{suite[f'{p}|{e}']['capture_rate']:.3f} ± {suite[f'{p}|{e}']['capture_se']:.3f}" for e in en) + " |")
    lines += ["", "Mean time to capture (s, captured episodes only; episode limit 15 s)\n",
              "| pursuer \\ evader | " + " | ".join(en) + " |", "|---" * (len(en) + 1) + "|"]
    for p in pn:
        lines.append(f"| {p} | " + " | ".join(
            "-" if suite[f'{p}|{e}']['mean_capture_time_s'] is None else f"{suite[f'{p}|{e}']['mean_capture_time_s']:.2f}"
            for e in en) + " |")
    lines += ["", "Energy proxy: pursuer / evader effort per second (thrust-seconds + sprint-seconds, per second; "
              "max 2.0)\n", "| pursuer \\ evader | " + " | ".join(en) + " |", "|---" * (len(en) + 1) + "|"]
    for p in pn:
        lines.append(f"| {p} | " + " | ".join(
            f"{suite[f'{p}|{e}']['P_energy_per_s']:.2f} / {suite[f'{p}|{e}']['E_energy_per_s']:.2f}" for e in en) + " |")
    lines += ["", "Pursuer energy per capture (effort units spent across all episodes / captures)\n",
              "| pursuer \\ evader | " + " | ".join(en) + " |", "|---" * (len(en) + 1) + "|"]
    for p in pn:
        lines.append(f"| {p} | " + " | ".join(("-" if suite[f'{p}|{e}']['capture_rate'] == 0 else f"{suite[f'{p}|{e}']['P_energy_per_capture']:.2f}") for e in en) + " |")
    lines += ["", f"Cross-play capture rate, archived pursuer snapshot (row, iteration) vs archived evader snapshot "
              f"(column), {a.xn} episodes per cell\n",
              "| P iter \\ E iter | " + " | ".join(str(x) for x in xs) + " |", "|---" * (len(xs) + 1) + "|"]
    for i, ip in enumerate(xs):
        lines.append(f"| {ip} | " + " | ".join(f"{X[i, j]:.2f}" for j in range(len(xs))) + " |")
    keys = list(next(iter(vs_script.values())).keys())
    lines += ["", f"Each snapshot vs the scripted suite ({a.xn} episodes each): pursuer capture rate, evader escape rate\n",
              "| iter | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for it in xs:
        lines.append(f"| {it} | " + " | ".join(f"{vs_script[it][k]:.2f}" for k in keys) + " |")
    lines += ["", "Occupancy (fraction of ticks)\n", "```", json.dumps(occ_stats, indent=1), "```"]
    open(os.path.join(a.results, "eval_tables.md"), "w").write("\n".join(lines) + "\n")

    # figures
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(X, vmin=0, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(xs)), xs); ax.set_yticks(range(len(xs)), xs)
    ax.set_xlabel("evader snapshot (iteration)"); ax.set_ylabel("pursuer snapshot (iteration)")
    for i in range(len(xs)):
        for j in range(len(xs)):
            ax.text(j, i, f"{X[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(X[i, j] - 0.5) > 0.3 else "black")
    fig.colorbar(im, label="capture rate")
    ax.set_title("Cross-play: capture rate of archived pursuers vs archived evaders")
    fig.tight_layout(); fig.savefig(os.path.join(a.media, "crossplay.png"), dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3))
    for k, (occ, title) in enumerate(((occ_l, "learned P vs learned E"), (occ_s, "lead (scripted) vs smartflee (scripted)"))):
        h, xe, ye = np.histogram2d(occ[:, 1, 0], occ[:, 1, 1], bins=40, range=[[-sim.L, sim.L]] * 2)
        axs[k].imshow(np.log1p(h.T), origin="lower", extent=[-sim.L, sim.L, -sim.L, sim.L], cmap="Blues")
        for ox, oy, r in sim.OBSTACLES:
            axs[k].add_patch(plt.Circle((ox, oy), r, color="0.4", alpha=0.6))
        axs[k].set_title(f"evader occupancy: {title}", fontsize=9); axs[k].set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(a.media, "occupancy.png"), dpi=120); plt.close(fig)

    ev = os.path.join(a.results, "train_eval.csv")
    lg = os.path.join(a.results, "train_log.csv")
    if os.path.exists(ev) and os.path.exists(lg):
        e = list(csv.DictReader(open(ev)))
        g = list(csv.DictReader(open(lg)))
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        x = [int(r["iter"]) for r in e]
        axs[0].plot(x, [float(r["P_vs_flee"]) for r in e], label="learned P vs flee: capture")
        axs[0].plot(x, [float(r["P_vs_smartflee"]) for r in e], label="learned P vs smartflee: capture")
        axs[0].plot(x, [1 - float(r["E_vs_greedy"]) for r in e], label="learned E vs greedy: escape")
        axs[0].plot(x, [1 - float(r["E_vs_lead"]) for r in e], label="learned E vs lead: escape")
        axs[0].set_xlabel("iteration"); axs[0].set_ylabel("rate (128 held-out episodes)")
        axs[0].set_title("Against the scripted suite (never trained on)"); axs[0].legend(fontsize=7); axs[0].set_ylim(-0.02, 1.02)
        xg = [int(r["iter"]) for r in g]
        axs[1].plot(xg, [float(r["cap_A_selfplay"]) for r in g], label="current P vs current E", lw=0.8)
        axs[1].plot(xg, [float(r["cap_B_P_vs_archE"]) for r in g], label="current P vs archived E", lw=0.8)
        axs[1].plot(xg, [float(r["cap_C_archP_vs_E"]) for r in g], label="archived P vs current E", lw=0.8)
        axs[1].set_xlabel("iteration"); axs[1].set_ylabel("capture rate (finished episodes)")
        axs[1].set_title("Training batches"); axs[1].legend(fontsize=7); axs[1].set_ylim(-0.02, 1.02)
        fig.tight_layout(); fig.savefig(os.path.join(a.media, "arms_race_curves.png"), dpi=120); plt.close(fig)
    print("wrote", os.path.join(a.results, "eval.json"))


if __name__ == "__main__":
    main()
