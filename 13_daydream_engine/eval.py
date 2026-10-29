"""Open-loop forecast error by horizon on held-out scenes, model vs baselines.

Rollouts start from the TRUE state at frames t0 in {0, 25, 50} of each test
trajectory and run H=50 frames (1 s) with no correction. Error is the mean
Euclidean position error per ball, in box units (box side = 1, radius 0.06).

Also: one-step error split by what happens in that transition (free flight,
wall bounce, ball-ball collision), error by horizon for windows with/without
collisions, and how often imagined balls leave the box.
"""
import argparse, json, os
import numpy as np
import sim
from model import load, imagine

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/dynamics")
ap.add_argument("--data", default="data")
ap.add_argument("--horizon", type=int, default=50)
ap.add_argument("--n", type=int, default=2000, help="test scenes used")
ap.add_argument("--out", default="results/eval.json")
ap.add_argument("--plot", default="media/horizon_error.png")
a = ap.parse_args()

d = np.load(os.path.join(a.data, "test.npz"))
S, F = d["states"][: a.n], d["flags"][: a.n]
H = a.horizon
starts = [t for t in (0, 25, 50) if t + H < S.shape[1]]
init = np.concatenate([S[:, t] for t in starts])
true = np.concatenate([S[:, t:t + H + 1] for t in starts])
flags = np.concatenate([F[:, t:t + H] for t in starts])  # (M, H, 2)
model, cfg = load(a.ckpt)

preds = {
    "model": imagine(model, init, H),
    "const_velocity": sim.constant_velocity(init, H),
    "ballistic_no_collisions": sim.ballistic(init, H),
    "walls_only_physics": sim.walls_only(init, H),
}


def pos_err(p, t):
    """(M, H+1) mean per-ball Euclidean position error."""
    B, T, D = p.shape
    pp, tt = p.reshape(B, T, -1, 4)[..., :2], t.reshape(B, T, -1, 4)[..., :2]
    return np.linalg.norm(pp - tt, axis=-1).mean(-1)


report_h = [h for h in (1, 2, 5, 10, 20, 25, 30, 40, 50) if h <= H]
res = dict(n_rollouts=int(len(init)), horizon=H, starts=starts, ckpt=a.ckpt,
           train_step=cfg["step"], units="mean per-ball position error, box side = 1",
           by_horizon={}, by_horizon_split={}, one_step={}, out_of_box={})
errs = {k: pos_err(v, true) for k, v in preds.items()}
any_col = flags.any(-1)  # (M, H)
cum_col = np.cumsum(any_col, 1)  # collisions so far
for k, e in errs.items():
    res["by_horizon"][k] = {h: float(e[:, h].mean()) for h in report_h}
    res.setdefault("by_horizon_median", {})[k] = {h: float(np.median(e[:, h])) for h in report_h}
    res.setdefault("n_nonfinite_rollouts", {})[k] = int((~np.isfinite(e)).any(1).sum())
    # split rollouts by whether any collision happened before horizon h
    res["by_horizon_split"][k] = {
        h: dict(no_collision=float(e[cum_col[:, h - 1] == 0, h].mean()) if (cum_col[:, h - 1] == 0).any() else None,
                collided=float(e[cum_col[:, h - 1] > 0, h].mean()),
                frac_collided=float((cum_col[:, h - 1] > 0).mean()),
                n_no_collision=int((cum_col[:, h - 1] == 0).sum()))
        for h in report_h}
    pp = preds[k].reshape(len(init), H + 1, -1, 4)[..., :2]
    # ball-frames whose centre is >0.02 past the wall-contact line (centre at R from a wall)
    pen = np.maximum(sim.R - pp, pp - (1 - sim.R)).max(-1)  # (M, H+1, balls) penetration
    res["out_of_box"][k] = dict(
        frac_ball_frames_pen_gt_0p02=float((pen > 0.02).mean()),
        frac_rollouts_pen_gt_0p02=float((pen.max((1, 2)) > 0.02).mean()),
        frac_rollouts_center_outside_box=float((pen.max((1, 2)) > sim.R).mean()),
        rollout_max_pen_p50_p99=[float(x) for x in np.percentile(pen.max((1, 2)), [50, 99])])

# one-step error from true states, every transition of the test set, split by event type
s0 = S[:, :-1].reshape(-1, S.shape[-1])
s1 = S[:, 1:].reshape(-1, S.shape[-1])
fw, fb = F[..., 0].reshape(-1), F[..., 1].reshape(-1)
one = {
    "model": imagine(model, s0, 1)[:, 1],
    "const_velocity": sim.constant_velocity(s0, 1)[:, 1],
    "ballistic_no_collisions": sim.ballistic(s0, 1)[:, 1],
    "walls_only_physics": sim.walls_only(s0, 1)[:, 1],
}
groups = {"free_flight": ~fw & ~fb, "wall_bounce": fw & ~fb, "ball_ball": fb}
for k, p in one.items():
    pe = pos_err(p[:, None], s1[:, None])[:, 0]
    ve = np.linalg.norm(p.reshape(-1, sim.N_BALLS, 4)[..., 2:] - s1.reshape(-1, sim.N_BALLS, 4)[..., 2:], axis=-1).mean(-1)
    res["one_step"][k] = {g: dict(pos=float(pe[m].mean()), vel=float(ve[m].mean()), n=int(m.sum()))
                          for g, m in groups.items()}

# error growth around the first bounce: align model rollouts on the frame of the first event
first = np.where(any_col.any(1), any_col.argmax(1), -1)
align = {}
for k in ("model", "ballistic_no_collisions", "walls_only_physics"):
    rel = {}
    for off in range(-5, 16):
        idx = first + 1 + off
        ok = (first >= 5) & (idx >= 0) & (idx <= H)
        if ok.sum() > 50:
            rel[off] = float(errs[k][ok, idx[ok]].mean())
    align[k] = rel
res["aligned_on_first_bounce"] = align

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    json.dump(res, f, indent=2)

print(f"rollouts {len(init)}, ckpt step {cfg['step']}")
print("horizon  " + "  ".join(f"{h:>7d}" for h in report_h))
for k, e in res["by_horizon"].items():
    print(f"{k[:22]:<22} " + "  ".join(f"{e[h]:7.4f}" for h in report_h))
print("one-step position / velocity error by event:")
for k, g in res["one_step"].items():
    print(f"  {k[:22]:<22} " + "  ".join(f"{n}: {v['pos']:.5f}/{v['vel']:.4f}" for n, v in g.items()))
print("out of box:", res["out_of_box"])

if a.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    hs = np.arange(H + 1)
    col = {"model": "#2a6fdb", "const_velocity": "#999999", "ballistic_no_collisions": "#e08a1e", "walls_only_physics": "#2a9d4b"}
    for k, e in errs.items():
        ax[0].plot(hs, e.mean(0), label=k, color=col[k])
    ax[0].axvline(cfg["args"]["unroll"], ls=":", c="k", lw=0.8)
    ax[0].text(cfg["args"]["unroll"] + 0.5, ax[0].get_ylim()[1] * 0.9, "training unroll", fontsize=8)
    ax[0].set(xlabel="horizon (frames, 20 ms each)", ylabel="mean position error (box = 1)",
              title=f"Open-loop error, {len(init)} held-out rollouts")
    ax[0].legend(fontsize=8)
    for k in ("model", "ballistic_no_collisions"):
        sp = res["by_horizon_split"][k]
        hh = [h for h in report_h if sp[h]["n_no_collision"] >= 30]
        ax[1].plot(hh, [sp[h]["no_collision"] for h in hh], "--", color=col[k], label=f"{k}: no collision yet")
        ax[1].plot(report_h, [sp[h]["collided"] for h in report_h], "-", color=col[k], label=f"{k}: after >=1 collision")
    ax[1].set(xlabel="horizon (frames)", ylabel="mean position error", title="Split by whether a bounce happened", yscale="log")
    ax[1].legend(fontsize=7)
    for k in ("model", "ballistic_no_collisions"):
        r = align[k]
        ax[2].plot(list(r.keys()), list(r.values()), "o-", ms=3, color=col[k], label=k)
    ax[2].axvline(0, c="k", lw=0.8)
    ax[2].set(xlabel="frames after first bounce in rollout", ylabel="mean position error", yscale="log",
              title="Error around the first bounce")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.plot), exist_ok=True)
    fig.savefig(a.plot, dpi=110)
    print("wrote", a.plot)
