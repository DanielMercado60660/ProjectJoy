"""Tune each scripted opponent by grid search, on TRAINING seeds only (seeds 100+), before the learner trains.

Objective for a script: mean score (+1 win, -1 loss, 0 draw) against the other three scripts
(current best versions), 16 bouts per opponent per candidate, on the training ring/friction
distribution. Two coordinate-ascent rounds: round 1 against defaults, round 2 against round-1 winners.
Output: results/tuned_scripts.json (params + scores for every candidate).
"""
import itertools, json, time
import numpy as np
from sumo_env import SumoEnv, make_script, play_bout, SCRIPT_CLASSES

GRIDS = {
    "charger": dict(k=[1.0, 2.0, 3.5], speed=[0.7, 1.0], turn_slow=[0.6, 1.2]),
    "flanker": dict(standoff=[0.25, 0.4], k=[2.0, 3.5], push_dist=[0.35, 0.6], speed=[0.8, 1.0]),
    "circler": dict(orbit=[0.3, 0.5], expose=[0.3, 0.6, 0.85], k=[2.0, 3.5]),
    "counter": dict(hold=[0.15, 0.35], trigger=[0.25, 0.5], k=[2.0, 3.5], dodge_time=[4, 8]),
}
N = 16


def score(name, params, best, seed):
    env = SumoEnv(seed=seed)
    tot, n = 0, 0
    for opp in SCRIPT_CLASSES:
        if opp == name:
            continue
        for i in range(N):
            # alternate sides so start asymmetries cancel
            a, b = make_script(name, params), make_script(opp, best[opp])
            if i % 2 == 0:
                tot += play_bout(env, a, b)["outcome"]
            else:
                tot -= play_bout(env, b, a)["outcome"]
            n += 1
    return tot / n


def main():
    t0 = time.time()
    best = {k: {} for k in GRIDS}
    log = {}
    for rnd in range(2):
        new_best = {}
        for name, grid in GRIDS.items():
            keys = list(grid)
            rows = []
            for vals in itertools.product(*grid.values()):
                p = dict(zip(keys, vals))
                s = score(name, p, best, seed=100 + rnd)
                rows.append((s, p))
            rows.sort(key=lambda r: -r[0])
            new_best[name] = rows[0][1]
            log[f"round{rnd}_{name}"] = [dict(score=s, params=p) for s, p in rows]
            default = score(name, {}, best, seed=100 + rnd)
            log[f"round{rnd}_{name}_default"] = default
            print(f"round {rnd} {name}: best {rows[0][0]:+.3f} {rows[0][1]}  default {default:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
        best = new_best
    json.dump(dict(best=best, log=log, n_per_opp=N, seeds=[100, 101], wall_s=time.time() - t0),
              open("results/tuned_scripts.json", "w"), indent=1)
    print("saved results/tuned_scripts.json", best)


if __name__ == "__main__":
    main()
