"""Reviewer re-check of the corner-stall diagnostic (learned P vs naive flee, 200 episodes, seed 12345)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
import sim, evaluate as ev
from policy import load_pair
mx.set_default_device(mx.cpu)
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pf, ef, _ = load_pair(os.path.join(here, "checkpoints", "final.npz"))
n = 200
P, E = ev.Net(pf, "p", "learned", seed=2), ev.Script("flee", "e", 3)
env = sim.Arena(n, ev.EVAL_SEED)
alive = np.ones(n, bool); cap = np.zeros(n, bool)
spd = []
for t in range(sim.T_MAX):
    a_p, a_e = P.act(env), E.act(env)
    _, _, done, c, info = env.step(a_p, a_e)
    cap |= c & alive; alive &= ~done
    spd.append(np.linalg.norm(env.vel[:, 0], axis=1))
fail = ~cap
# envs that timed out were auto-reset at t=300; use state just before reset is lost, so re-run 299 ticks
env = sim.Arena(n, ev.EVAL_SEED); P = ev.Net(pf, "p", "learned", seed=2); E = ev.Script("flee", "e", 3)
for t in range(sim.T_MAX - 1):
    env.step(P.act(env), E.act(env))
d = env.dist(); epos = env.pos[:, 1]
spd = np.array(spd)[-61:-1]
corner = (np.abs(epos) > 1.8).all(1)
r = {"n": n, "fail_frac": float(fail.mean()), "fail_final_dist_mean": float(d[fail].mean()),
     "fail_P_speed_last3s": float(spd[:, fail].mean()), "fail_E_in_corner_frac": float(corner[fail].mean())}
print(r)
json.dump(r, open(os.path.join(here, "review", "flee_stall_check.json"), "w"), indent=1)
