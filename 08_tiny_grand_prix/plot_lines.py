"""Racing lines on held-out tracks: one row per controller, one column per track,
line colored by speed (single-hue ramp, light = slow, dark = fast). A red x marks a crash.

    ../.venv/bin/python plot_lines.py --controllers es ppo ray pursuit --out media/racing_lines.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from racing.controllers import LABELS, get_controller  # noqa: E402
from racing.evaluate import run_lap  # noqa: E402
from racing.track import make_tracks, pack  # noqa: E402

SURF, INK2 = "#fcfcfb", "#52514e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", default=["es", "ppo", "ray", "pursuit"])
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--tracks", nargs="+", type=int, default=[0, 2, 5, 7])
    ap.add_argument("--grip", type=float, default=1.0)
    ap.add_argument("--mode", default="strict")
    ap.add_argument("--out", default="media/racing_lines.png")
    a = ap.parse_args()
    bank = pack(make_tracks(a.split, max(a.tracks) + 1))
    R, C = len(a.controllers), len(a.tracks)
    fig, axs = plt.subplots(R, C, figsize=(3.2 * C, 3.3 * R), facecolor=SURF, squeeze=False)
    cmap = plt.get_cmap("Blues")
    for r, name in enumerate(a.controllers):
        res = run_lap(get_controller(name, bank), bank, grip=a.grip, mode=a.mode, record=True, track_ids=a.tracks)
        for c, tid in enumerate(a.tracks):
            ax = axs[r, c]
            t = bank.tracks[tid]
            for w in (t.left, t.right):
                ax.plot(*np.vstack([w, w[:1]]).T, color="#9b9a95", lw=0.8)
            tr = res["traj"][:, c]
            lt = res["lap_time"][c]
            n = int(round(lt / 0.1)) if np.isfinite(lt) else len(tr)
            n = min(n, len(tr))
            pts = tr[:n, :2]
            segs = np.stack([pts[:-1], pts[1:]], 1)
            lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(-10, 40), linewidths=1.6)
            lc.set_array(tr[: n - 1, 3])
            ax.add_collection(lc)
            if res["collisions"][c] > 0 and a.mode == "strict":
                ax.plot(*pts[-1], "x", color="#e34948", ms=10, mew=2.5)
            label = f"{lt:.1f} s" if np.isfinite(lt) else f"DNF ({res['progress_frac'][c]:.0%})"
            ax.set_title(f"track {tid}: {label}", fontsize=9, color=INK2)
            ax.set_aspect("equal")
            ax.axis("off")
        axs[r, 0].text(-0.08, 0.5, LABELS.get(name, name), transform=axs[r, 0].transAxes, rotation=90,
                       va="center", ha="right", fontsize=10, color=INK2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-10, 40))
    cb = fig.colorbar(sm, ax=axs, shrink=0.5, pad=0.01)
    cb.set_label("speed (m/s)", color=INK2)
    cb.ax.set_ylim(0, 40)
    fig.suptitle(f"Racing lines on {a.split} tracks (grip {a.grip}, {a.mode} rules), darker = faster",
                 color="#0b0b0b", x=0.02, ha="left")
    fig.savefig(a.out, dpi=110, facecolor=SURF, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
