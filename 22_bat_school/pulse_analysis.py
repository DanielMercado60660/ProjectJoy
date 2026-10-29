"""When does the learned bat choose to pulse? Logs every decision on held-out 6x6 mazes.

  ../.venv/bin/python pulse_analysis.py --ckpt checkpoints/bat   # -> results/pulse_analysis.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from bat_env import DC, DR, BatEnv, MazePool
from evaluate import HELDOUT, LearnedAgent
from model import load

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/bat")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--out", default="results/pulse_analysis.json")
a = ap.parse_args()
model, _ = load(a.ckpt)
pool = MazePool([6], a.n, seed=HELDOUT[6])
env = BatEnv(pool, a.n, seed=0, fixed_order=True)
ag = LearnedAgent(model, a.n, seed=1)
obs = env.reset()
first = env.ep_id.copy()
alive = np.ones(a.n, bool)
since = np.zeros(a.n, int)       # steps since the last echo was heard
rows = []
while alive.any():
    m, r, c, hd = env.m.copy(), env.r.copy(), env.c.copy(), env.hd.copy()
    mv, pu = ag.act(obs)
    pp = ag.last_pulse_prob.copy()
    ok = pool.open[m, r, c, hd]
    moved = (mv == 0) & ok
    r2, c2 = r + moved * DR[hd], c + moved * DC[hd]
    hd2 = np.where(mv == 1, (hd + 3) % 4, np.where(mv == 2, (hd + 1) % 4, hd))
    deg = pool.open[m, r2, c2].sum(-1)                         # open sides of the cell it pulses in
    front_open = pool.open[m, r2, c2, hd2]
    for i in np.nonzero(alive)[0]:
        rows.append(dict(pulse=bool(pu[i]), p=float(pp[i]), move=int(mv[i]), bumped=bool((mv[i] == 0) & ~ok[i]),
                         deg=int(deg[i]), front_open=bool(front_open[i]), since=int(since[i])))
    obs, rew, d, info = env.step(mv, pu)
    ag.on_done(d)
    since = np.where(pu, 0, since + 1)
    for i in np.nonzero(d)[0]:
        if info["ep_id"][i] == first[i]:
            alive[i] = False


def rate(sel):
    s = [x["pulse"] for x in rows if sel(x)]
    return dict(n=len(s), pulse_rate=float(np.mean(s)) if s else None)


out = {
    "all": rate(lambda x: True),
    "after_forward_move": rate(lambda x: x["move"] == 0 and not x["bumped"]),
    "after_turn": rate(lambda x: x["move"] != 0),
    "arriving_in_dead_end(deg1)": rate(lambda x: x["move"] == 0 and not x["bumped"] and x["deg"] == 1),
    "arriving_in_corridor(deg2)": rate(lambda x: x["move"] == 0 and not x["bumped"] and x["deg"] == 2),
    "arriving_at_junction(deg3+)": rate(lambda x: x["move"] == 0 and not x["bumped"] and x["deg"] >= 3),
    "front_open_after_action": rate(lambda x: x["front_open"]),
    "front_blocked_after_action": rate(lambda x: not x["front_open"]),
}
by_since = defaultdict(list)
for x in rows:
    by_since[min(x["since"], 4)].append(x["pulse"])
out["by_steps_since_last_echo(capped at 4)"] = {k: dict(n=len(v), pulse_rate=float(np.mean(v))) for k, v in sorted(by_since.items())}
json.dump(out, open(a.out, "w"), indent=2)
for k, v in out.items():
    print(k, v)
