"""Is the bonus fooled by the TV? Feed each trained agent's intrinsic-reward module the *same*
random-walk trajectories in the noisy-TV world and compare its raw bonus on steps where the screen
is in view vs. steps where it is not. Ratio > 1 means the TV looks more interesting than the rest
of the house. (Uses each agent's own trained nets / count tables / normalisers; nothing is updated
except the per-episode memories, which start empty each episode as during training.)

  ../.venv/bin/python diagnose_tv.py            # writes runs/tv_diagnosis.json, media/tv_bonus_ratio.png
"""
import glob
import json

import numpy as np
import mlx.core as mx

import env as E
from train import Explorer, norm_fn
from agents import EpisodicKNN


def random_trajectories(n=64, T=512, seed=777):
    env = E.VecRooms(n, T, noisy_tv=True, seed=seed)
    rng = np.random.default_rng(seed)
    obs, acts, nxt, seen = [], [], [], []
    tvc = env.tv_cells
    o = env.obs.copy()
    for t in range(T):
        a = rng.integers(0, E.N_ACTIONS, n)
        o2, done, _ = env.step(a)
        y, x = env.pos[:, 0], env.pos[:, 1]
        s = np.zeros(n, bool)
        for ty, tx in tvc:
            s |= (np.abs(ty - y) <= E.HALF) & (np.abs(tx - x) <= E.HALF)
        if done.any():  # positions already reset; the step's view is unknown -> drop these
            s[done] = False
        obs.append(o); acts.append(a); nxt.append(o2); seen.append(s)
        o = o2
    return np.array(obs), np.array(acts), np.array(nxt), np.array(seen)


def bonus(ex, O, A, ON):
    """Raw per-step intrinsic signal along trajectories [T, n, ...]. For rnd_ep / ngu this is the
    *episodic* component only: their lifelong RND multiplier is confounded by where each agent
    happened to spend its training time, so it is not a clean test of TV sensitivity."""
    T, n, D = ON.shape
    ag = ex.cfg["agent"]
    nrm = norm_fn(ex.obs_rms)
    flat_on = ON.reshape(-1, D)
    if ag in ("rnd", "rnd_ep", "ngu"):
        err = np.concatenate([np.array(ex.aux.errors(nrm(mx.array(flat_on[i:i + 8192]))))
                              for i in range(0, len(flat_on), 8192)]).reshape(T, n)
    if ag == "rnd":
        return err
    if ag == "count":
        return 1.0 / np.sqrt(ex.counter.table[ex.counter.hash(flat_on)] + 1.0).reshape(T, n)
    if ag == "disagree":
        fo = O.reshape(-1, D)
        fa = A.reshape(-1)
        return np.concatenate([np.array(ex.aux.errors(nrm(mx.array(fo[i:i + 8192])), mx.array(fa[i:i + 8192]), None))
                               for i in range(0, len(fo), 8192)]).reshape(T, n)
    # episodic agents: fresh memory per episode (one episode per column here)
    ep = np.zeros((T, n))
    if ag in ("rnd_ep", "ep"):
        hs = ex.hasher.hash64(flat_on).reshape(T, n)
        mem = [dict() for _ in range(n)]
        for t in range(T):
            for i in range(n):
                c = mem[i].get(int(hs[t, i]), 0) + 1
                mem[i][int(hs[t, i])] = c
                ep[t, i] = 1.0 / np.sqrt(c)
    elif ag == "ngu":
        knn = EpisodicKNN(n, T + 1)
        knn.dm2 = ex.knn.dm2
        for t in range(T):
            e = np.array(ex.aux.embed(nrm(mx.array(ON[t]))))
            ep[t] = knn.reward_and_add(e, np.zeros(n, bool))
    return ep  # episodic component only (the lifelong RND multiplier is reported separately as "rnd")


if __name__ == "__main__":
    O, A, ON, S = random_trajectories()
    res = {}
    for ag in ["count", "rnd", "rnd_ep", "ep", "ngu", "disagree"]:
        rows = []
        for ck in sorted(glob.glob(f"runs/{ag}_tv_s*/ckpt.pkl")):
            ex, _ = Explorer.load(ck)
            b = bonus(ex, O, A, ON)
            valid = np.ones_like(S)
            valid[0] = False  # first step of each episode is novel for every episodic agent
            on, off = b[S & valid].mean(), b[~S & valid].mean()
            rows.append(dict(ckpt=ck, tv_in_view=float(on), tv_out_of_view=float(off), ratio=float(on / off)))
            print(f"{ag:9s} {ck:32s} bonus TV-in-view {on:.4g}  elsewhere {off:.4g}  ratio {on / off:.2f}")
        if rows:
            res[ag] = rows
    json.dump(res, open("runs/tv_diagnosis.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plots import COL, NAME, style, INK, MUTED
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    style(ax)
    ags = list(res)
    for j, ag in enumerate(ags):
        r = [x["ratio"] for x in res[ag]]
        ax.bar(j, np.mean(r), 0.6, color=COL[ag])
        ax.scatter([j] * len(r), r, color=INK, s=10, zorder=3)
        ax.text(j, max(r) * 1.04 + 0.05, f"{np.mean(r):.2f}x", ha="center", fontsize=8, color=INK)
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(len(ags)))
    short = {"count": "hashed\ncounts", "rnd": "RND", "rnd_ep": "RND x\nepisodic count", "ep": "episodic count\nonly",
             "ngu": "NGU-style\n(inv.-dyn. kNN)", "disagree": "ensemble\ndisagreement"}
    ax.set_xticklabels([short[a] for a in ags], fontsize=8)
    ax.set_ylim(0, 1.18 * max(max(x["ratio"] for x in res[a]) for a in ags) + 0.1)
    ax.set_ylabel("bonus on TV-in-view steps / elsewhere", fontsize=8.5, color=MUTED)
    ax.set_title("Bonus on TV-in-view steps / elsewhere (same random-walk states; episodic part for\n"
                 "RND x episodic and NGU-style; dots = seeds)",
                 fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig("media/tv_bonus_ratio.png", dpi=130)
