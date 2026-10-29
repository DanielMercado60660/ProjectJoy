"""Closed-loop task: choose a launch velocity using only an imagined world, then check reality.

Task: a held-out 3-ball scene; we pick ball 0's initial velocity. The goal is
for ball 0 to be at a target point after T frames (0.8 s, so usually after one
or more bounces). Reward = -distance(ball 0 at frame T, target). Success =
distance < ball radius (the target point is under the ball).

Every planner scores the SAME 512 random candidate velocities with its own
predictor and picks the best-looking one ("imagined distance"); that velocity is
then run in the true simulator ("realized distance").
  model     : the learned Dynamics MLP
  ballistic : free fall, no collisions (cheap baseline)
  const_vel : straight lines (cheap baseline)
  random    : first candidate, no evaluation (floor)
  oracle    : the true simulator itself (ceiling for this candidate set)
"""
import argparse, json, os
import numpy as np
import sim
from model import load, imagine

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/dynamics")
ap.add_argument("--data", default="data")
ap.add_argument("--tasks", type=int, default=200)
ap.add_argument("--cands", type=int, default=512)
ap.add_argument("--T", type=int, default=40)
ap.add_argument("--out", default="results/plan.json")
ap.add_argument("--plot", default="media/plan_examples.png")
a = ap.parse_args()

rng = np.random.default_rng(2)
scenes = np.load(os.path.join(a.data, "test.npz"))["states"][: a.tasks, 0].astype(np.float64)
targets = rng.uniform(0.15, 0.85, size=(a.tasks, 2))
cands = rng.uniform(-1.5, 1.5, size=(a.cands, 2))
model, cfg = load(a.ckpt)

# every (task, candidate) initial state
X = np.repeat(scenes[:, None], a.cands, 1).copy()  # (tasks, cands, 12)
X[..., 2:4] = cands[None]
X = X.reshape(-1, sim.STATE_DIM)


def final_pos(traj):
    return traj[:, a.T, 0:2].reshape(a.tasks, a.cands, 2)


predict = {
    "model": lambda x: final_pos(imagine(model, x, a.T)),
    "ballistic": lambda x: final_pos(sim.ballistic(x, a.T)),
    "const_vel": lambda x: final_pos(sim.constant_velocity(x, a.T)),
    "walls_only": lambda x: final_pos(sim.walls_only(x, a.T)),
}
real_traj, real_flags = sim.simulate(X, a.T)
# did the LAUNCHED ball (ball 0) itself bounce? its velocity deviates from pure free fall
v0 = real_traj[:, :, 2:4]
dv = v0[:, 1:] - v0[:, :-1]
dv[..., 1] += sim.G * sim.FRAME_DT
ball0_bounces = (np.linalg.norm(dv, axis=-1) > 1e-3).sum(1).reshape(a.tasks, a.cands)
real_final = final_pos(real_traj)
real_dist = np.linalg.norm(real_final - targets[:, None], axis=-1)  # (tasks, cands)

res = dict(frac_candidates_ball0_bounced=float((ball0_bounces > 0).mean()), tasks=a.tasks, candidates=a.cands, T=a.T, success_radius=sim.R, planners={})
chosen = {}
for name, f in list(predict.items()) + [("oracle", None), ("random", None)]:
    if name == "oracle":
        imag = real_dist
    elif name == "random":
        imag = None
    else:
        imag = np.linalg.norm(f(X) - targets[:, None], axis=-1)
        imag = np.nan_to_num(imag, nan=np.inf, posinf=np.inf)  # diverged imaginations never win
    pick = np.zeros(a.tasks, int) if imag is None else imag.argmin(1)
    chosen[name] = pick
    r = real_dist[np.arange(a.tasks), pick]
    rec = dict(success=float((r < sim.R).mean()), success_2r=float((r < 2 * sim.R).mean()),
               realized_dist_mean=float(r.mean()), realized_dist_median=float(np.median(r)),
               realized_dist_per_task=[float(x) for x in r])
    if imag is not None:
        i = imag[np.arange(a.tasks), pick]
        rec.update(imagined_dist_mean=float(i.mean()), gap_mean=float((r - i).mean()),
                   imagined_success=float((i < sim.R).mean()),
                   fooled=float(((i < sim.R) & (r >= 2 * sim.R)).mean()))
        # how well does this predictor rank candidates? Spearman-ish: corr of ranks with truth
        ri = imag.argsort(1).argsort(1)
        rr = real_dist.argsort(1).argsort(1)
        rec["rank_corr_with_truth"] = float(np.mean([np.corrcoef(ri[t], rr[t])[0, 1] for t in range(a.tasks)]))
    # does the chosen launch actually involve a bounce in reality (any wall/ball event for any ball)?
    nb = ball0_bounces[np.arange(a.tasks), pick]
    rec["chosen_ball0_bounced"] = float((nb > 0).mean())
    rec["chosen_ball0_bounces_mean"] = float(nb.mean())
    res["planners"][name] = rec

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as fh:
    json.dump(res, fh, indent=2)
for k, v in res["planners"].items():
    print(f"{k:<10}", " ".join(f"{m}={x:.3f}" for m, x in v.items() if not isinstance(x, list)))

if a.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [0, 1, 2, 3, 4, 5]
    fig, axs = plt.subplots(1, len(show), figsize=(3 * len(show), 3.3))
    for ax, t in zip(axs, show):
        x0 = X.reshape(a.tasks, a.cands, -1)[t, chosen["model"][t]][None]
        im = imagine(model, x0, a.T)[0]
        re, _ = sim.simulate(x0, a.T)
        re = re[0]
        for b, c in zip(range(sim.N_BALLS), ["#2a6fdb", "#aaaaaa", "#aaaaaa"]):
            ax.plot(re[:, 4 * b], re[:, 4 * b + 1], "-", c=c if b else "k", lw=1.5 if b == 0 else 0.8)
            if b == 0:
                ax.plot(im[:, 0], im[:, 1], "--", c="#2a6fdb", lw=1.5)
        ax.add_patch(plt.Circle(targets[t], sim.R, color="#d33", alpha=0.35))
        ax.plot(*re[-1, :2], "ko", ms=4)
        ax.plot(*im[-1, :2], "o", c="#2a6fdb", ms=4)
        d_r = np.linalg.norm(re[-1, :2] - targets[t])
        d_i = np.linalg.norm(im[-1, :2] - targets[t])
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[],
               title=f"task {t}: imagined {d_i:.2f} / real {d_r:.2f}")
        ax.title.set_fontsize(8)
    fig.suptitle("Model planner: dashed = imagined path of the launched ball, black = what really happened, red = target", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.plot, dpi=110)
    print("wrote", a.plot)
