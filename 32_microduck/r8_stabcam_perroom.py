"""POST HOC (not in protocol.json; written after the scripted results): per-room scripted numbers with the head
camera's displacement over the scored window, to show where blind walks ran into walls (the duck stalls, its gait
loses rhythm and every stream's wobble rises). -> results/r8/stabcam/scripted_per_room.json

    ../.venv/bin/python r8_stabcam_perroom.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_stabcam as S  # noqa: E402
import r8_stabcam_analyze as A  # noqa: E402
from r8_stabcam_run import load_npz  # noqa: E402

EXPECTED = 0.44 * 7.0          # M0's forward ratio x the 7 s window: metres per commanded m/s


def main():
    rows = []
    lo, hi = S.WINDOW
    for p in sorted((S.DATA / "runs").glob("train-*_off.npz")):
        r = load_npz(p)
        m = r["meta"]
        if m["vx"] <= 0:
            continue
        o = A.scripted_run(r)
        ev = r["events"]
        L = ev[ev[:, 1] == 0, 0]
        gaps = np.diff(L[(L >= lo) & (L < hi)])
        P = r["pos"][lo:hi:25, :2]                       # every 0.5 s: path length, robust to turning
        disp = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        rows.append({"config": m["config"], "vx": m["vx"], "wz": m["wz"], "moved_m": round(disp, 3),
                     "moved_frac_of_expected": round(disp / (EXPECTED * m["vx"]), 2),
                     "touchdown_gap_steps_std": round(float(gaps.std()), 2) if len(gaps) > 2 else None,
                     "wobble": {k: np.round(v, 3).tolist() for k, v in o["wobble"].items()},
                     "pl_head": None if not o.get("pl_wobble") else o["pl_wobble"]["head"]})
    free = [x for x in rows if x["moved_frac_of_expected"] >= 0.6]
    stuck = [x for x in rows if x["moved_frac_of_expected"] < 0.6]

    def avg(xs, s):
        return np.mean([x["wobble"][s] for x in xs], 0).round(3).tolist() if xs else None
    summary = {g: {"runs": len(xs), **{s: avg(xs, s) for s in ("head", "trunk", "oracle", "eis", "eis_lagc")},
                   "pl_head": np.mean([x["pl_head"] for x in xs if x["pl_head"]], 0).round(3).tolist() if xs else None,
                   "touchdown_gap_std": float(np.mean([x["touchdown_gap_steps_std"] for x in xs if x["touchdown_gap_steps_std"] is not None]))}
               for g, xs in (("moved_ge_60pct", free), ("moved_lt_60pct", stuck))}
    out = {"note": "POST HOC split (not pre-registered): walking runs (reflex off) by the head's path length over the "
                   "scored 7 s window (positions every 0.5 s) vs 0.44 x commanded speed x 7 s; under 60% = the blind "
                   "walk ran into something",
           "summary": summary, "runs": rows}
    (S.OUT / "scripted_per_room.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
