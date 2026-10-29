"""Build media/demo.html: a single self-contained page (no network) that runs the
simulator and the trained policies live in JS on held-out and hard tracks.

    ../.venv/bin/python make_demo_html.py
"""
import json
import os

import numpy as np

from racing.baselines import DEFAULT_RAY_PARAMS
from racing.controllers import CK, LABELS
from racing.policy import load_policy
from racing.track import make_track

HERE = os.path.dirname(os.path.abspath(__file__))


def track_json(split, i):
    t = make_track(i, split)
    return dict(label=f"{'held-out' if split == 'heldout' else 'hard (OOD)'} #{i}",
                center=np.round(t.center, 3).tolist(), tangent=np.round(t.tangent, 5).tolist(),
                hw=float(t.half_width), length=float(t.length))


def main(out="media/demo.html", n_heldout=20, n_hard=10):
    tracks = [track_json("heldout", i) for i in range(n_heldout)] + [track_json("hard", i) for i in range(n_hard)]
    policies = {}
    for k in ("es", "ppo", "es_dr", "ppo_dr"):
        p = os.path.join(CK, f"{k}_best.npz")
        if os.path.exists(p):
            policies[k] = [float(x) for x in load_policy(p)[0]]
    rp = os.path.join(CK, "ray_follower.json")
    ray = json.load(open(rp))["params"] if os.path.exists(rp) else dict(DEFAULT_RAY_PARAMS)
    rdp = os.path.join(CK, "ray_follower_dr.json")
    ray_dr = json.load(open(rdp))["params"] if os.path.exists(rdp) else None
    ev = None
    note = ""
    ep = os.path.join(HERE, "results", "eval.json")
    if os.path.exists(ep):
        rows = json.load(open(ep))
        ev = [dict(label=LABELS.get(r["controller"], r["controller"]), grip=r["grip"], completion=r["completion"],
                   mean_lap_time=r["mean_lap_time"] if r["mean_lap_time"] == r["mean_lap_time"] else None)
              for r in rows if r["split"] == "heldout" and r["mode"] == "strict"
              and r["controller"] in ("es", "ppo", "es_dr", "ppo_dr", "ray", "ray_dr", "pursuit") and r["grip"] in (1.0, 0.7, 0.5)]
        note = "Lap time = mean over the laps that controller finished. Pure pursuit is privileged (sees the centerline)."
    data = dict(tracks=tracks, policies=policies, ray_params=ray, ray_dr_params=ray_dr, eval=ev, eval_note=note)
    html = open(os.path.join(HERE, "web", "demo_template.html")).read()
    html = html.replace("/*__SIM_JS__*/", open(os.path.join(HERE, "web", "sim.js")).read())
    html = html.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")).replace("NaN", "null"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(html)
    print(f"wrote {out} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2])
