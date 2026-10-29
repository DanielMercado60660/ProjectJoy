"""Reviewer check: is the scripted 'lead' pursuer a fair baseline?

Tries cheap variants of the scripted pursuer against the scripted evaders and the learned evader on the
held-out eval seed (12345, 500 episodes, same play() as evaluate.py). Variants:
  sprint_r   sprint only when the evader is within this distance (lead uses 1.2)
  avoid      steer around obstacles (tangent slide when an obstacle blocks the target direction)
  cutoff     aim at a point on the centre side of the evader's predicted position (herds it to walls)
  horizon    cap on the lead-time prediction (lead uses 1.0 s)
Tuning was done on seed 777 (not the held-out seed); the chosen variant is then reported on seed 12345.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sim
import evaluate as ev
from policy import load_pair


class Lead2:
    def __init__(self, sprint_r=1.2, avoid=False, cutoff=0.0, horizon=1.0, name="lead2"):
        self.sprint_r, self.avoid, self.cutoff, self.horizon, self.name = sprint_r, avoid, cutoff, horizon, name

    def act(self, env):
        pm, po = env.pos[:, 0], env.pos[:, 1]
        rel = po - pm
        dist = np.linalg.norm(rel, axis=1)
        tl = np.clip(dist / sim.P_SPRINT, 0, self.horizon)[:, None]
        tgt = po + env.vel[:, 1] * tl
        if self.cutoff:
            c = -tgt / np.maximum(np.linalg.norm(tgt, axis=1, keepdims=True), 1e-9)   # toward centre
            tgt = tgt + self.cutoff * np.minimum(dist, 1.0)[:, None] * c
        d = tgt - pm
        if self.avoid:
            u = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            for ox, oy, r in sim.OBSTACLES:
                ro = np.array([ox, oy]) - pm
                proj = (ro * u).sum(1)
                perp = ro - proj[:, None] * u
                pd = np.linalg.norm(perp, axis=1)
                dob = np.linalg.norm(ro, axis=1)
                block = (proj > 0) & (proj < np.linalg.norm(d, axis=1)) & (pd < r + 0.15) & (dob < r + 0.8)
                side = -perp / np.maximum(pd, 1e-9)[:, None]          # away from obstacle centre line
                side[pd < 1e-6] = np.stack([-u[pd < 1e-6, 1], u[pd < 1e-6, 0]], 1)
                d = np.where(block[:, None], u + 1.5 * side, d)
        return sim.dir_to_action(d, dist < self.sprint_r)


def run(seed, n, evaders, variants):
    out = {}
    for v in variants:
        for E in evaders:
            r = ev.play(v, E, n, seed=seed)
            out[f"{v.name}|{E.name}"] = {"capture_rate": r["capture_rate"], "capture_se": r["capture_se"],
                                         "P_energy_per_capture": r["P_energy_per_capture"]}
    return out


if __name__ == "__main__":
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mode = sys.argv[1] if len(sys.argv) > 1 else "tune"
    sf = ev.Script("smartflee", "e", 3)
    if mode == "tune":       # seed 777, smartflee only
        grid = []
        for sr in (1.2, 0.8, 0.5, 0.35):
            for avoid in (False, True):
                for cut in (0.0, 0.3, 0.6):
                    for hz in (1.0, 0.5):
                        grid.append(Lead2(sr, avoid, cut, hz, f"sr{sr}_av{int(avoid)}_cut{cut}_hz{hz}"))
        res = run(777, 300, [sf], grid)
        for k, v in sorted(res.items(), key=lambda kv: -kv[1]["capture_rate"])[:12]:
            print(f"{k:40s} {v['capture_rate']:.3f}")
        print("baseline lead:", ev.play(ev.Script("lead", "p", 1), sf, 300, seed=777)["capture_rate"])
    else:                    # held-out: python review/stronger_pursuers.py heldout sr avoid cut hz
        sr, avoid, cut, hz = float(sys.argv[2]), bool(int(sys.argv[3])), float(sys.argv[4]), float(sys.argv[5])
        variants = [ev.Script("lead", "p", 1), Lead2(sr, False, 0.0, 1.0, f"lead_sprint{sr}"),
                    Lead2(sr, avoid, cut, hz, "lead_tuned")]
        variants[0].name = "lead (original)"
        pf, ef, _ = load_pair(os.path.join(here, "checkpoints", "final.npz"))
        evs = [ev.Script("flee", "e", 3), sf, ev.Net(ef, "e", "learned", seed=4)]
        res = run(ev.EVAL_SEED, 500, evs, variants)
        res["_config"] = {"sprint_r": sr, "avoid": avoid, "cutoff": cut, "horizon": hz,
                          "tuned_on_seed": 777, "eval_seed": ev.EVAL_SEED, "n": 500}
        for k, v in res.items():
            if not k.startswith("_"):
                print(f"{k:40s} {v['capture_rate']:.3f} ± {v['capture_se']:.3f}  energy/capture {v['P_energy_per_capture']:.1f}")
        json.dump(res, open(os.path.join(here, "review", "stronger_pursuers.json"), "w"), indent=1)
