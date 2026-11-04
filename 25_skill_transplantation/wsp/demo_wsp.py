"""WSP demo: before/after clip + retained-return bar chart (runner: Claude).

Clip rows (training seed 0 unless noted, held-out reset 31000, columns = task label fwd / bwd):
  specialists switched by label (ceiling) | naive average (before) | Astra one-shot (r6wsp_os_b) |
  Astra budget-50 (r6wsp_b50_b) | a one-shot failure episode (r6wsp_os_a, seed/reset picked as its worst held-out fwd episode)

  ../.venv/bin/python wsp/demo_wsp.py   -> wsp/media/wsp_demo.{mp4,gif}, wsp_demo_final.png, wsp_retained.png
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from common import load_params  # noqa: E402
import merge as M  # noqa: E402
from common import make_env, np_mean, policy_input, task_reward  # noqa: E402


def run_tile(p, norm, task, reset_seed, every, steps, bank=tuple(range(31000, 31020))):
    """Faithful clip of a held-out episode. The rollout is chaotic: even the float rounding of a batched vs
    single-row numpy forward pass changes the trajectory, so we roll out the WHOLE held-out bank batched
    exactly as heldout.py / common.evaluate do, record qpos/qvel of the wanted reset, then replay those states
    kinematically in a rendering env."""
    import mujoco
    k = list(bank).index(reset_seed)
    envs = [make_env() for _ in bank]
    obs = np.stack([e.reset(seed=int(s))[0] for e, s in zip(envs, bank)])
    ids = np.full(len(bank), 0 if task == "fwd" else 1)
    states, ret = [], 0.0
    for t in range(steps):
        a = np_mean(p, policy_input(norm, obs, ids))
        for i, e in enumerate(envs):
            o, _, _, _, info = e.step(a[i])
            obs[i] = o
            if i == k:
                ret += task_reward(task, info["x_velocity"], a[i])
                xk = info["x_position"]
        if t % every == 0:
            d = envs[k].unwrapped.data
            states.append((d.qpos.copy(), d.qvel.copy(), ret, xk))
    for e in envs:
        e.close()
    renv = make_env(render=True)
    renv.reset(seed=reset_seed)
    m, d = renv.unwrapped.model, renv.unwrapped.data
    scale = 3000.0 / m.geom_size[0][0]
    m.geom_size[0][:2] = 3000.0
    if m.geom_matid[0] >= 0:
        m.mat_texrepeat[m.geom_matid[0]] *= scale
    frames = []
    for qp, qv, r, x in states:
        d.qpos[:], d.qvel[:] = qp, qv
        mujoco.mj_forward(m, d)
        frames.append((renv.render(), r, x))
    renv.close()
    return frames, ret

AK = ["a1.weight", "a1.bias", "a2.weight", "a2.bias", "a3.weight", "a3.bias"]


def attempt(tag, seed):
    z = np.load(os.path.join(HERE, "attempts", tag, f"merged_s{seed}.npz"))
    return {k: z[k].astype(np.float32) for k in AK}


def specs(seed):
    d = os.path.join(PROJ, "checkpoints", f"s{seed}")
    _, norm = load_params(os.path.join(d, "base.npz"))
    return load_params(os.path.join(d, "pre_fwd.npz"))[0], load_params(os.path.join(d, "pre_bwd.npz"))[0], norm


def main():
    H = json.load(open(os.path.join(HERE, "results", "heldout.json")))
    bank = H["heldout_bank"]
    # worst held-out fwd episode of the one-shot attempt with the most failures
    worst = min(((s, i, r) for s in H["results"] for i, r in enumerate(H["results"][s]["r6wsp_os_a"]["heldout"]["fwd"])),
                key=lambda t: t[2])
    ws, wi = int(worst[0]), bank[worst[1]]
    A, B, norm = specs(0)
    Aw, Bw, normw = specs(ws)
    rows = [("specialists, switched\nby label (ceiling)", {"fwd": A, "bwd": B}, norm, 31000),
            ("naive weight average\n(before)", M.interpolate(A, B, 0.5), norm, 31000),
            ("Astra one-shot\n(os_b, no rollouts)", attempt("r6wsp_os_b", 0), norm, 31000),
            ("Astra budget-50\n(b50_b)", attempt("r6wsp_b50_b", 0), norm, 31000),
            (f"one-shot failure\n(os_a, seed {ws}, reset {wi})", attempt("r6wsp_os_a", ws), normw, wi)]
    W, Ht, head, left = 320, 240, 28, 170
    tiles, finals = {}, {}
    for name, p, nrm, rs in rows:
        for task in ("fwd", "bwd"):
            q = p[task] if isinstance(p, dict) and task in p else p
            tiles[(name, task)], finals[(name, task)] = run_tile(q, nrm, task, rs, 4, 1000)
            print(f"{name!r} {task}: {finals[(name, task)]:.0f}", flush=True)
    frames = []
    n = len(tiles[(rows[0][0], "fwd")])
    for i in range(n):
        cv = Image.new("RGB", (left + 2 * W, head + Ht * len(rows)), (252, 252, 251))
        dr = ImageDraw.Draw(cv)
        for j, task in enumerate(("fwd", "bwd")):
            dr.text((left + j * W + 8, 8), f"task label = {task} ({'run +x' if task == 'fwd' else 'run -x'})", fill=(11, 11, 11))
        for r, (name, *_r) in enumerate(rows):
            y = head + r * Ht
            dr.text((6, y + Ht // 2 - 12), name, fill=(11, 11, 11))
            for j, task in enumerate(("fwd", "bwd")):
                img, ret, x = tiles[(name, task)][i]
                cv.paste(Image.fromarray(img), (left + j * W, y))
                dr.rectangle([left + j * W, y, left + j * W + 150, y + 30], fill=(252, 252, 251))
                dr.text((left + j * W + 4, y + 2), f"return {ret:7.0f}\nx = {x:6.1f} m", fill=(11, 11, 11))
        frames.append(np.array(cv))
    import imageio
    out = os.path.join(HERE, "media")
    os.makedirs(out, exist_ok=True)
    imageio.mimwrite(os.path.join(out, "wsp_demo.mp4"), frames, fps=25, quality=7, macro_block_size=2)
    Image.fromarray(frames[-1]).save(os.path.join(out, "wsp_demo_final.png"))
    small = [np.array(Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2))) for f in frames[::3]]
    imageio.mimwrite(os.path.join(out, "wsp_demo.gif"), small, duration=0.12, loop=0)
    json.dump({f"{k[0]} | {k[1]}": v for k, v in finals.items()}, open(os.path.join(out, "wsp_demo_returns.json"), "w"), indent=1)

    # bar chart of held-out mean retained
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = json.load(open(os.path.join(HERE, "results", "table.json")))["heldout"]
    show = [("average", "weight average", "#b9b9b4"), ("ties_ext", "best arithmetic merge (DARE/TIES ext.)", "#b9b9b4"),
            ("theta0", "θ0 (shared start)", "#b9b9b4"), ("fwd_spec_both", "fwd specialist, both labels", "#b9b9b4"),
            ("ppo_ft_avg_73728", "PPO from average, 74k steps", "#8fb0d8"),
            ("ppo_ft_avg_299008", "PPO from average, 299k steps", "#8fb0d8"),
            ("synth_distill", "SGD distill, synthetic inputs, 0 steps", "#8fb0d8"),
            ("ppo_ft_avg_1M", "PPO from average, 1M steps (earlier round)", "#8fb0d8"),
            ("ablate_route_only", "routing skeleton, no fitting (runner)", "#e3b28c"),
            ("ablate_route_refit_out", "routing skeleton + output refit (runner)", "#e3b28c"),
            ("r6wsp_os_a", "Astra one-shot A (0 steps)", "#d9622b"), ("r6wsp_os_b", "Astra one-shot B (0 steps)", "#d9622b"),
            ("r6wsp_b10_a", "Astra budget-10 A (60k steps)", "#d9622b"), ("r6wsp_b10_b", "Astra budget-10 B (60k steps)", "#d9622b"),
            ("r6wsp_b50_a", "Astra budget-50 A (300k steps)", "#d9622b"), ("r6wsp_b50_b", "Astra budget-50 B (300k steps)", "#d9622b"),
            ("dagger_73728", "DAgger, 74k steps", "#8fb0d8"), ("dagger_200k", "DAgger, 200k steps", "#8fb0d8")]
    show = [s for s in show if s[0] in T]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(show) + 1))
    y = np.arange(len(show))[::-1]
    v = [100 * T[k]["mean_ret"] for k, _, _ in show]
    e = [100 * T[k]["mean_ret_std"] for k, _, _ in show]
    ax.barh(y, v, xerr=e, color=[c for *_, c in show], height=0.7, error_kw=dict(lw=0.8, color="#555"))
    for yy, vv, ee in zip(y, v, e):
        ax.text(min(vv + ee + 1.5, 103), yy, f"{vv:.0f}%", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _ in show], fontsize=8)
    ax.axvline(100, color="#333", lw=0.8, ls=":")
    ax.set_xlim(0, 115)
    ax.set_xlabel("held-out mean retained return (% of specialist), ± std over 4 seeds")
    ax.set_title("One network for fwd + bwd HalfCheetah: Astra-written weights vs baselines", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "wsp_retained.png"), dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
