"""Exp 1 follow-up: test Astra's *mechanism* claim (the wall-rider lacks speed restraint: higher
pedal bias, braking paths weakened) behaviourally and with two one-number interventions, on the
fresh held-out tracks.
    python wsp/mechanism_check.py --start 90000 --n 200
"""
import argparse, json, os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "4"); os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_common import ROOT, WSP, WeightsController, load_weights  # noqa: E402
from racing.evaluate import run_lap, summarize  # noqa: E402
from racing.track import make_tracks, pack  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--start", type=int, required=True); ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
bank = pack(make_tracks("heldout", a.n, start=a.start))
soft, clean = (load_weights(os.path.join(ROOT, "checkpoints", f"{n}_best.npz")) for n in ("es_soft", "es"))


def stats(d, name):
    c = WeightsController(d); acts = []
    def f(o, env=None):
        x = c(o); acts.append(np.clip(x, -1, 1)); return x
    r = run_lap(f, bank, mode="soft", record=True)
    A = np.concatenate(acts)  # includes finished cars' idle steps; fine for a rough profile
    ss = summarize(r); st = summarize(run_lap(c, bank, mode="strict"))
    return dict(name=name, pedal_mean=float(A[:, 1].mean()), frac_full_throttle=float((A[:, 1] > 0.9).mean()),
                frac_braking=float((A[:, 1] < 0).mean()), mean_abs_steer=float(np.abs(A[:, 0]).mean()),
                soft_wall_steps=ss["collisions_per_lap"], soft_lap=ss["mean_lap_time"], strict_completion=st["completion"],
                strict_progress=st["mean_progress"])


out = [stats(soft, "es_soft"), stats(clean, "es")]
d = {k: v.copy() for k, v in soft.items()}; d["b3"][1] = clean["b3"][1]
out.append(stats(d, "es_soft with es's pedal bias (b3[1] 0.68->0.32)"))
for bias in (0.0, -0.5, -1.0):
    d = {k: v.copy() for k, v in soft.items()}; d["b3"][1] = bias
    out.append(stats(d, f"es_soft with pedal bias {bias}"))
for r in out:
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
json.dump(out, open(os.path.join(WSP, "results", "mechanism_check.json"), "w"), indent=1)
