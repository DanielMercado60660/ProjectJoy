"""media/r8_nav_strip.png: the first servo and the first bot episode of the zero-shot set (no picking), primary
condition (normal camera, G1 goal from the same episode's crossing, 0.25 s context). Per row: six query frames
evenly spaced over the approach plus the goal image; on each frame, ticks at the top mark the bearing of the true
plate (green), ViNT's first waypoint (orange) and NoMaD's first waypoint (purple, sample 0) as horizon columns
(arrows at the edge when outside the 45 deg view); the inset is a top-down view (robot at the origin facing up)
of ViNT's 5 and NoMaD's 8 predicted waypoints in their own step units, with the true plate direction as a green
ray. Right: the distance heads over the approach vs the true remaining time in 4 Hz steps; dotted: ViNT with
the wrong-room goal.

    ../.venv/bin/python r8_nav/visual.py      (after r8_nav/zeroshot.py; reads runs/r8_nav/preds_*.npz)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from r8_nav import zeroshot as Z  # noqa: E402

OUT = HERE / "media" / "r8_nav_strip.png"
HFOV = np.radians(45.0)


def preds(cond, ei):
    z = np.load(Z.PRED_DIR / f"preds_{cond.replace('|', '_')}.npz")
    return {k.split("_", 1)[1]: z[k] for k in z.files if k.split("_", 1)[0] == str(ei)}


def column(theta, width=160):
    f = (width / 2) / np.tan(HFOV / 2)
    return width / 2 - f * np.tan(theta)


def main():
    rep = json.loads(Z.OUT.read_text())
    idx = rep["episode_index"]
    rows = [next(r for r in idx if r["expert"] == x) for x in ("servo", "bot")]
    fig = plt.figure(figsize=(17, 5.6), dpi=110)
    gs = fig.add_gridspec(2, 9, width_ratios=[1] * 7 + [0.15, 2.1], hspace=0.5, wspace=0.06)
    for r, row in enumerate(rows):
        ei = row["i"]
        z = np.load(Z.EPI / f"{row['expert']}_{row['config']}.npz")
        e = {k: z[k] for k in z.files if k != "meta"}
        e["meta"] = json.loads(str(z["meta"]))
        tr = Z.truth(e)
        pv, pn = preds("vint|normal|G1|0.25s", ei), preds("nomad|normal|G1|0.25s", ei)
        pw = preds("vint|normal|wrong|0.25s", ei)
        q = pv["q"]
        pick = np.linspace(0, len(q) - 1, 6).round().astype(int)
        for j, qi in enumerate(pick):
            ax = fig.add_subplot(gs[r, j])
            fi = q[qi]
            ax.imshow(e["normal"][fi])
            th = {"plate": tr["bearing_plate"][fi], "vint": np.arctan2(pv["wp"][qi, 0, 1], pv["wp"][qi, 0, 0]),
                  "nomad": np.arctan2(pn["wp"][qi, 0, 1], pn["wp"][qi, 0, 0])}
            for k, col, y0 in (("plate", "#1a9850", 0), ("vint", "#f46d43", 10), ("nomad", "#762a83", 20)):
                u = column(th[k]) if abs(th[k]) < np.pi / 2 else (-1e9 if th[k] > 0 else 1e9)
                if 0 <= u <= 159:
                    ax.plot([u, u], [y0, y0 + 14], color=col, lw=3)
                else:
                    ax.annotate("", xy=(2 if u < 0 else 157, y0 + 7), xytext=(18 if u < 0 else 141, y0 + 7),
                                arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
            ax.set_title(f"-{tr['remaining'][fi]:.1f}s  {tr['dist'][fi]:.2f}m", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ins = ax.inset_axes([0.62, 0.0, 0.38, 0.5])
            ins.set_facecolor((1, 1, 1, 0.8))
            for s in pn["samples"][qi][1:]:
                ins.plot(-s[:, 1], s[:, 0], color="#762a83", alpha=0.25, lw=0.8)
            ins.plot(-pn["wp"][qi][:, 1], pn["wp"][qi][:, 0], color="#762a83", lw=1.4)
            ins.plot(-pv["wp"][qi][:, 1], pv["wp"][qi][:, 0], color="#f46d43", lw=1.6, marker=".", ms=3)
            lim = max(2.0, float(np.abs(np.concatenate([pv["wp"][qi][:, :2].ravel(), pn["samples"][qi].ravel()])).max()))
            b = tr["bearing_plate"][fi]
            ins.plot([0, -np.sin(b) * lim], [0, np.cos(b) * lim], color="#1a9850", lw=1.6)
            ins.set_xlim(-lim, lim)
            ins.set_ylim(-lim * 0.3, lim)
            ins.set_xticks([])
            ins.set_yticks([])
        ax = fig.add_subplot(gs[r, 6])
        ax.imshow(e["normal"][e["meta"]["g1_index"]])
        ax.set_title("goal G1 (on plate)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[r, 8])
        t = -tr["remaining"][q]
        ax.plot(t, tr["remaining"][q] * 4, "k--", lw=1.2, label="true remaining (4 Hz steps)")
        ax.plot(t, pv["dist"], color="#f46d43", lw=1.5, label="ViNT distance")
        ax.plot(t, pn["dist"], color="#762a83", lw=1.5, label="NoMaD distance")
        ax.plot(t, pw["dist"], color="#f46d43", lw=1.0, ls=":", label="ViNT, wrong-room goal")
        for qi in pick:
            ax.axvline(-tr["remaining"][q[qi]], color="0.85", lw=0.8, zorder=0)
        ax.set_ylim(0, max(25, float(np.max(pv["dist"])) + 2))
        ax.set_xlabel("time to goal frame (s)", fontsize=8)
        ax.set_title(f"{row['expert']} {row['config']}: distance heads", fontsize=9)
        ax.tick_params(labelsize=7)
        if r == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("R8 Gate C: published ViNT / NoMaD, zero-shot on the duck's head camera (primary condition). "
                 "Ticks: green = true plate bearing, orange = ViNT, purple = NoMaD (first waypoint)", fontsize=10)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
