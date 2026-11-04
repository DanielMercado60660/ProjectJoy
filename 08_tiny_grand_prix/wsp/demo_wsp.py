"""Before/after demo on fresh held-out tracks: reuses render.py's drawing, but with wsp policies
(.npz) and an explicit seed range (render.py itself builds tracks 0..max(tid)).

    python wsp/demo_wsp.py --start 90000 --tracks 3 17 --policy before=../checkpoints/es_soft_best.npz \
        after=wsp/edit_a1/best.npz --named ray --mode strict --out wsp/media/hack_before_after.gif
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_common import ROOT, WeightsController, load_weights  # noqa: E402
sys.path.insert(0, ROOT)
import render  # noqa: E402
from racing.controllers import get_controller  # noqa: E402
from racing.evaluate import run_lap  # noqa: E402
from racing.track import make_tracks, pack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--tracks", nargs="+", type=int, required=True, help="offsets from --start")
    ap.add_argument("--policy", nargs="*", default=[], help="name=path.npz")
    ap.add_argument("--named", nargs="*", default=[])
    ap.add_argument("--grip", type=float, default=1.0)
    ap.add_argument("--mode", default="strict")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=int, default=320)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()
    tracks = [make_tracks(a.split, 1, start=a.start + t)[0] for t in a.tracks]
    bank = pack(tracks)
    ctrls = {}
    for spec in a.policy:
        n, p = spec.split("=", 1)
        ctrls[n] = WeightsController(load_weights(p), name=n)
    for n in a.named:
        ctrls[n] = get_controller(n, bank)
    palette = [(255, 120, 20), (40, 200, 120), (250, 190, 30), (40, 120, 230), (220, 70, 200)]
    for i, n in enumerate(ctrls):
        render.COLORS.setdefault(n, palette[i % len(palette)])

    def simulate(names, split, tids, grip, mode="strict"):
        return bank, {n: run_lap(ctrls[n], bank, grip=grip, mode=mode, record=True, track_ids=tids) for n in names}

    render.simulate = simulate
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    runs = render.render(list(ctrls), f"{a.split}+{a.start}", list(range(len(tracks))), a.grip, a.out,
                         a.cell, a.fps, a.every, None, a.title, a.mode)
    for n, r in runs.items():
        print(n, "finished", r["finished"].tolist(), "lap", r["lap_time"].round(1).tolist())


if __name__ == "__main__":
    main()
