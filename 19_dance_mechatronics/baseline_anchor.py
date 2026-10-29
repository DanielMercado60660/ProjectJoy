"""Review addition: the foot-relative baseline plus two bounded absolute-position anchors.

`baseline_rel` (BaselineController with com_rel=True) fixes the original baseline's runaway (balance on
the absolute COM error, gain -1783 Nm/m, tips the body after ~1 m of drift) but has nothing that
pulls the body back to its spot, so it wanders. This keeps the foot-relative balance loop and adds:

  1. a bounded COM-target shift: the foot-relative COM error gets an offset
     clip(k_anchor * (root_x - root_x_ref), -delta, delta), so the body leans back toward home by at
     most `delta` (inside the 20 cm sole) and can never wind up into a runaway;
  2. swing-foot placement: while a foot is in swing (reference contact off) its hip target is offset by
     k_sw * clip(root_x - root_x_ref, -0.1, 0.1) rad, so the kick lands nearer the reference spot.

Both use only the root-x error, which the PPO policy also observes. No integrators. The tuner picks the
signs. Tuned by tune_anchor.py with the same protocol and training seeds as tune_baseline.py.
"""
import numpy as np
from dance_env import BaselineController


class AnchorBaselineController(BaselineController):
    def __init__(self, params=None):
        p = dict(k_anchor=0.0, delta=0.02, k_sw=0.0)
        p.update(params or {})
        p["com_rel"] = True
        super().__init__(p)

    def __call__(self, env):
        p = self.p
        rn = env.ref(env.t)
        ex = env.d.qpos[0] - rn["qpos"][0]                   # root drift from the reference spot (m)
        # 1. bounded COM-target shift, applied through the same stance-ankle gain as the balance term:
        #    k_com * (e_rel - off) = k_com * e_rel - k_com * off
        target, u = super().__call__(env)
        off = float(np.clip(p["k_anchor"] * ex, -p["delta"], p["delta"]))
        stance = rn["contact"]
        for s in range(2):
            if stance[s]:
                w = 1.0 / max(1, stance.sum())
                u[3 * s + 2] -= w * p["k_com"] * off
            else:
                # 2. swing-foot placement on the swing hip
                target[3 * s + 0] += p["k_sw"] * float(np.clip(ex, -0.1, 0.1))
        return target, u


def make(params):
    return AnchorBaselineController(params)
