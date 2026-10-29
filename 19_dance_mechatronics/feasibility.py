"""Reproduce the clip feasibility sweep: for each candidate clip, build it (lean solved so the inverse-
dynamics CoP sits mid-foot) and report CoP range, min normal force, friction ratio, peak joint torque.
Also plots CoP / torques over one loop of the shipped clip -> media/feasibility.png.

  ../.venv/bin/python feasibility.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from clip import Clip, DEFAULT, FOOT_LEN

SWEEP = [
    ("first draft: 100 BPM, 1-beat kicks, knee -1.45", dict(bpm=100, kick_knee=-1.45, kick_thigh=-0.25, kick_width=1.0, gain=0.5)),
    ("100 BPM, 1-beat kicks, knee -1.2", dict(bpm=100, kick_knee=-1.2, kick_width=1.0, gain=0.5)),
    ("80 BPM, 1-beat kicks, knee -1.45", dict(bpm=80, kick_knee=-1.45, kick_thigh=-0.25, kick_width=1.0, gain=0.5)),
    ("80 BPM, 1-beat kicks, knee -1.2", dict(bpm=80, kick_width=1.0, gain=0.5)),
    ("60 BPM, 1-beat kicks, knee -1.45", dict(bpm=60, kick_knee=-1.45, kick_thigh=-0.25, kick_width=1.0, gain=0.5)),
    ("100 BPM, 1.5-beat kicks, knee -1.0, dip -0.2", dict(bpm=100, kick_knee=-1.0, kick_thigh=-0.1, kick_width=1.5, dip_knee=-0.2)),
    ("SHIPPED: 90 BPM, 1.4-beat kicks, knee -1.2, dip -0.6", {}),
    ("shipped clip at 110 BPM (re-solved)", dict(bpm=110)),
]


def main():
    rows = []
    for name, kw in SWEEP:
        c = Clip(**kw)
        f = c.feasibility()
        ok = f["cop_inside_foot"] and f["friction_ok"] and f["tau_ok"] and f["Fz_min"] > 0
        rows.append(dict(name=name, params=c.p, feasible=bool(ok), cop_iter_err_last=c.cop_err_hist[-1] if c.cop_err_hist else None,
                         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}))
        print(f"{name:52s} feasible={ok!s:5s} CoP [{f['cop_rel_min']:.3f},{f['cop_rel_max']:.3f}] m  "
              f"Fz_min {f['Fz_min']:.0f} N  |Fx/Fz| {f['friction_ratio_max']:.2f}  tau_max {f['tau_max']:.0f} Nm", flush=True)
    json.dump(rows, open("results/feasibility_sweep.json", "w"), indent=1, default=float)
    # plot the shipped clip
    c = Clip()
    r = c.inverse()
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    t = c.t
    for b in range(4):
        for a in ax:
            a.axvline(b * c.beat, color="0.8", lw=0.8)
    ax[0].plot(t, r["cop_rel"], label="CoP (inverse dynamics)")
    ax[0].plot(t, c.com[:, 0] - (c.qpos[:, 0] * 0 + np.array([c._fk(c.qpos[i])[1][c.stance[i]][0] - FOOT_LEN / 2 for i in range(len(t))])),
               "--", label="COM ground projection")
    ax[0].axhspan(0, FOOT_LEN, color="g", alpha=0.08, label="foot sole")
    ax[0].set_ylabel("x rel. ankle (m)"); ax[0].legend(fontsize=7, loc="upper right")
    hs = ax[1].plot(t, r["tau"][:, :3], lw=1) + ax[1].plot(t, r["tau"][:, 3:], lw=1, ls="--")
    ax[1].axhline(100, color="r", lw=0.8); ax[1].axhline(-100, color="r", lw=0.8)
    ax[1].set_ylabel("joint torque (Nm)")
    ax[1].legend(hs, ["R hip", "R knee", "R ankle", "L hip", "L knee", "L ankle"], fontsize=6, ncol=6)
    ax[2].plot(t, c.foot_pos[:, 0, 1], label="R sole height"); ax[2].plot(t, c.foot_pos[:, 1, 1], label="L sole height")
    ax[2].plot(t, c.root_z - c.root_z.max(), label="pelvis drop")
    ax[2].set_ylabel("m"); ax[2].set_xlabel("time in loop (s); grey = beats 1-4"); ax[2].legend(fontsize=7)
    ax[0].set_title(f"Reference clip feasibility ({c.p['bpm']:.0f} BPM, Walker2d)")
    plt.tight_layout()
    plt.savefig("media/feasibility.png", dpi=110)


if __name__ == "__main__":
    main()
