"""M0 start-threshold test: each command once from standing and once after 3 s of walking at 0.3.

Shows the gait's start threshold (small commands don't start a gait from rest but work once moving).
Steady state = mean over the last 4.5 s of a 6 s segment. Writes results/m0/from_rest.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from duck import BodyConfig, Duck  # noqa: E402
from m0_gate import FROZEN_BODY  # noqa: E402

CMDS = [(0.1, 0, 0), (0.15, 0, 0), (0.2, 0, 0), (0.3, 0, 0), (0.4, 0, 0), (-0.1, 0, 0), (-0.2, 0, 0),
        (-0.3, 0, 0), (0, 0.15, 0), (0, -0.15, 0), (0, 0.3, 0), (0, 0, 0.3), (0, 0, -0.5), (0, 0, 1.0)]


def main():
    out = {}
    for name, body in (("plain", {"scene": "allcollisions"}), ("chosen", FROZEN_BODY)):
        duck = Duck(BodyConfig(**body))
        rows = []
        for cmd in CMDS:
            res = {}
            for start in ("rest", "after_walk"):
                duck.reset()
                if start == "after_walk":
                    duck.set_command(0.3, 0, 0)
                    for _ in range(150):
                        duck.step()
                duck.set_command(*cmd)
                vs = []
                for k in range(300):
                    duck.step()
                    if k >= 75:
                        vs.append(duck.planar_velocity())
                ax = int(np.argmax(np.abs(cmd)))
                res[start] = round(float(np.mean(vs, 0)[ax]), 4)
            rows.append({"cmd": cmd, **res})
            print(f"{name:6s} {str(cmd):18s} rest {res['rest']:+.3f} after_walk {res['after_walk']:+.3f}")
        out[name] = {"body": body, "rows": rows}
    (HERE / "results" / "m0" / "from_rest.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
