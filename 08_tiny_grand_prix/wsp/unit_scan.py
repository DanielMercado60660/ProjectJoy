"""Exp 1 reference: mean-ablate every hidden unit of a policy one at a time (64 runs) on the fresh
held-out tracks; shows which single units, if any, carry the wall-riding.
    python wsp/unit_scan.py --policy es_soft --start 90000 --n 200
"""
import argparse, json, os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "4"); os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_common import ROOT, WSP, load_weights  # noqa: E402
from ablate import run, unit_means  # noqa: E402
from racing.track import make_tracks, pack  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default="es_soft"); ap.add_argument("--start", type=int, required=True)
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
d = load_weights(os.path.join(ROOT, "checkpoints", f"{a.policy}_best.npz"))
means = unit_means(d)
bank = pack(make_tracks("heldout", a.n, start=a.start))
out = {"none": run(d, {"h1": [], "h2": []}, means, bank)}
for l in ("h1", "h2"):
    for j in range(32):
        out[f"{l}:{j}"] = run(d, {l: [j]}, means, bank)
json.dump(out, open(os.path.join(WSP, "results", f"unit_scan_{a.policy}.json"), "w"), indent=1)
rows = sorted(((k, v) for k, v in out.items() if k != "none"), key=lambda kv: kv[1]["soft_wall_steps"])
print("none", out["none"])
for k, v in rows[:8] + rows[-4:]:
    print(k, {kk: round(vv, 3) if vv == vv else vv for kk, vv in v.items()})
rows = sorted(((k, v) for k, v in out.items() if k != "none"), key=lambda kv: -kv[1]["strict_progress"])
print("by strict progress:", [(k, round(v["strict_progress"], 3), v["strict_completion"]) for k, v in rows[:6]])
