"""Render the detective animation: an active (EIG) detective and a random prober investigate the
SAME hidden world side by side. Each push is animated on the table, then the camera readings,
the posterior over mu_k (exact grid, plus the amortized MLX network reloaded from the
checkpoint) and the EIG map of the next possible pushes are shown.

    ../.venv/bin/python make_demo.py --mu 0.23 --pushes 8 --ckpt checkpoints/sbi
Outputs media/detective.gif and media/detective.mp4 (+ media/detective_final.png).
"""
import argparse

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from inference import GRID, GridPosterior, eig
from physics import DESIGNS, F_GRID, NT, T_GRID, TIMES, VIEW, observe, world1_positions
from sbi import MAXN, featurize, load, posterior_on_grid

C = {"active": "#2a78d6", "random": "#eb6834", "net": "#6250d6"}
INK, INK2, GRIDC, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def play(mu, pushes, seed, model):
    """Run both detectives; return per-push history used for drawing."""
    hist = {}
    for si, strat in enumerate(["active", "random"]):
        rng = np.random.default_rng([seed, si])
        post = GridPosterior()
        H = []
        feats, designs, Ys, Cs = [], [], [], []
        for k in range(pushes):
            scores = eig(post.particles(64), DESIGNS, M=2, rng=np.random.default_rng([seed, k]))
            d = DESIGNS[int(np.argmax(scores))] if strat == "active" else DESIGNS[rng.integers(len(DESIGNS))]
            x = world1_positions(d[0], d[1], mu)
            y, c = observe(x, rng)
            post.update(d[0], d[1], y, c)
            designs.append(d); Ys.append(y); Cs.append(c)
            f = featurize(np.array(designs), np.array(Ys), np.array(Cs))
            fpad = np.zeros((1, MAXN, f.shape[-1]), np.float32); fpad[0, :len(f)] = f
            m = np.zeros((1, MAXN), np.float32); m[0, :len(f)] = 1
            w_net = posterior_on_grid(model, fpad, m, GRID)[0] if model is not None else None
            H.append(dict(d=d, x=x, y=y, c=c, scores=scores.reshape(len(F_GRID), len(T_GRID)), w=post.w.copy(),
                          mean=post.mean(), sd=post.sd(), w_net=w_net, pred=post.predict([d[0]], [d[1]])[0]))
        hist[strat] = H
    return hist


def draw(fig, axes, hist, k, sub, mu, pushes):
    for col, strat in enumerate(["active", "random"]):
        h = hist[strat][k]
        a_tab, a_cam, a_post, a_eig = axes[:, col]
        for a in axes[:, col]:
            a.cla()
        # --- table strip
        ti = min(sub, NT - 1)
        t_now = TIMES[ti]
        xb = h["x"][ti]
        a_tab.set_xlim(-0.15, 2.0); a_tab.set_ylim(0, 1); a_tab.axis("off")
        a_tab.add_patch(Rectangle((-0.15, 0.19), 2.15, 0.12, color="#b8b6ae"))
        a_tab.add_patch(Rectangle((-0.15, 0.31), VIEW + 0.15, 0.62, color=C[strat], alpha=0.07))
        a_tab.text(VIEW + 0.02, 0.85, "edge of camera view", fontsize=7, color=INK2)
        a_tab.axvline(VIEW, ymin=0.31, ymax=0.93, color=INK2, ls=":", lw=1)
        a_tab.add_patch(Rectangle((min(xb, 1.95) - 0.06, 0.31), 0.12, 0.2, color=INK))
        if t_now <= h["d"][1] and h["x"][-1] > 0:
            a_tab.annotate("", xy=(min(xb, 1.95) - 0.06, 0.41), xytext=(min(xb, 1.95) - 0.06 - 0.04 * h["d"][0] / 3, 0.41),
                           arrowprops=dict(arrowstyle="-|>", color=C[strat], lw=2))
        title = "ACTIVE detective (max expected information gain)" if strat == "active" else "RANDOM prober"
        a_tab.set_title(f"{title}\npush {k+1}/{pushes}: F = {h['d'][0]:.0f} N for T = {h['d'][1]:.1f} s    t = {t_now:.1f} s",
                        fontsize=9, color=C[strat], loc="left")
        if h["x"][-1] == 0 and sub > 3:
            a_tab.text(0.1, 0.62, "didn't budge (static friction won)", fontsize=8, color=INK2)
        # --- camera readings
        show = np.arange(NT) <= ti
        obs = show & ~h["c"]
        a_cam.plot(TIMES, np.minimum(h["pred"], 1.6), color=C[strat], lw=1.5, alpha=0.6, label="posterior prediction")
        a_cam.scatter(TIMES[obs], h["y"][obs], s=12, color=INK, zorder=3, label="camera reading")
        cz = show & h["c"]
        a_cam.scatter(TIMES[cz], np.full(cz.sum(), VIEW + 0.08), marker="^", s=16, color=INK2, label="out of view")
        a_cam.axhline(VIEW, color=INK2, ls=":", lw=1)
        a_cam.set_ylim(-0.25, 1.3); a_cam.set_xlim(0, 2.05); a_cam.set_xlabel("time (s)", fontsize=8); a_cam.set_ylabel("position (m)", fontsize=8)
        a_cam.legend(fontsize=6.5, frameon=False, loc="upper left")
        # --- posterior (revealed after the push has played out)
        hp = hist[strat][k if sub >= NT - 1 else k - 1] if (k > 0 or sub >= NT - 1) else None
        if hp is not None:
            a_post.fill_between(GRID, hp["w"] / hp["w"].max(), color=C[strat], alpha=0.35, lw=0)
            a_post.plot(GRID, hp["w"] / hp["w"].max(), color=C[strat], lw=1.5, label="exact posterior")
            if hp["w_net"] is not None:
                a_post.plot(GRID, hp["w_net"] / hp["w_net"].max(), color=C["net"], lw=1.2, ls="--", label="MLX amortized net")
            a_post.text(0.98, 0.9, f"μ_k = {hp['mean']:.3f} ± {hp['sd']:.3f}", transform=a_post.transAxes, ha="right", fontsize=9)
            if hp["sd"] < 0.03:   # zoom inset so a sharp posterior is still readable
                ins = a_post.inset_axes([0.58, 0.18, 0.4, 0.55])
                half = max(6 * hp["sd"], 0.01)
                sel = np.abs(GRID - hp["mean"]) < half * 1.2
                ins.fill_between(GRID[sel], (hp["w"] / hp["w"].max())[sel], color=C[strat], alpha=0.35, lw=0)
                if hp["w_net"] is not None:
                    ins.plot(GRID[sel], (hp["w_net"] / hp["w_net"].max())[sel], color=C["net"], lw=1, ls="--")
                ins.axvline(mu, color=INK, lw=1, ls=":")
                ins.set_xlim(hp["mean"] - half, hp["mean"] + half); ins.set_yticks([]); ins.tick_params(labelsize=6)
                ins.set_title("zoom", fontsize=6, loc="left", pad=1)
        else:
            a_post.fill_between(GRID, np.ones_like(GRID), color=C[strat], alpha=0.15, lw=0)
            a_post.text(0.98, 0.9, "prior: anything in 0.05–0.8", transform=a_post.transAxes, ha="right", fontsize=9)
        a_post.axvline(mu, color=INK, lw=1, ls=":")
        a_post.text(mu, 1.08, " true μ_k (hidden)", fontsize=7, color=INK2)
        a_post.set_xlim(0.05, 0.8); a_post.set_ylim(0, 1.2); a_post.set_yticks([]); a_post.set_xlabel("kinetic friction μ_k", fontsize=8)
        a_post.legend(fontsize=6.5, frameon=False, loc="upper left")
        # --- EIG map that was used to choose this push
        im = a_eig.imshow(h["scores"].T, origin="lower", aspect="auto", cmap="Blues",
                          extent=[F_GRID[0] - 0.5, F_GRID[-1] + 0.5, T_GRID[0] - 0.05, T_GRID[-1] + 0.05], vmin=0)
        a_eig.scatter([h["d"][0]], [h["d"][1]], marker="*", s=160, color=C[strat], edgecolor=INK, lw=0.8, zorder=3)
        a_eig.set_xlabel("push force F (N)", fontsize=8); a_eig.set_ylabel("duration T (s)", fontsize=8)
        a_eig.set_title(f"expected info gain of each push before push {k+1} (★ = chosen)", fontsize=8, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.23)
    ap.add_argument("--pushes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--ckpt", default="checkpoints/sbi")
    ap.add_argument("--out", default="media/detective")
    a = ap.parse_args()
    model, _ = load(a.ckpt)
    hist = play(a.mu, a.pushes, a.seed, model)
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), gridspec_kw=dict(height_ratios=[0.55, 1.2, 1, 1.1]))
    frames = []
    subs = list(range(0, NT, 2)) + [NT - 1] * 4
    for k in range(a.pushes):
        for sub in subs:
            draw(fig, axes, hist, k, sub, a.mu, a.pushes)
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    fig.savefig(a.out + "_final.png", dpi=110)
    imageio.mimsave(a.out + ".gif", frames[::1], duration=0.12, loop=0)
    imageio.mimsave(a.out + ".mp4", frames, fps=8, macro_block_size=1)
    for s in ["active", "random"]:
        print(s, [f"{h['mean']:.3f}±{h['sd']:.3f}" for h in hist[s]])
    print("saved", a.out + ".gif/.mp4", len(frames), "frames")


if __name__ == "__main__":
    main()
