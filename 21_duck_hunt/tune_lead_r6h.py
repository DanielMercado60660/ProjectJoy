"""Give the scripted lead-pursuit hunter a tuning budget (R6 hunter fairness check).
Random search + local refinement of the Kalman lead hunter's 6 parameters (filter order, process
noise, fire tolerance, min observations, max position std, lead-horizon multiplier) on TRAINING
courses against the evaders the learned hunter's stage 1 trained against (straight, jink, dodge).
Objective: mean hit rate. -> checkpoints/r6h_lead_tuned.json, results/r6h_lead_tuning.json

  ../bin/gpu_run ../.venv/bin/python tune_lead_r6h.py
"""
import json, time
import numpy as np
import hunter as H

R = dict(ORDER=(1, 2), Q=(10.0, 20000.0), TOL=(0.05, 0.6), MIN_OBS=(2, 10), STD_MAX=(0.15, 1.0), H_MULT=(0.6, 1.6))


def sample(rng):
    p = {k: float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if k == "Q" else float(rng.uniform(lo, hi))
         for k, (lo, hi) in R.items()}
    p["ORDER"] = int(rng.integers(1, 3)); p["MIN_OBS"] = int(round(p["MIN_OBS"]))
    return p


def perturb(rng, p, s):
    q = dict(p)
    for k, (lo, hi) in R.items():
        if k == "ORDER":
            q[k] = p[k] if rng.random() > 0.1 else 3 - p[k]
        elif k == "Q":
            q[k] = float(np.clip(p[k] * np.exp(rng.normal() * 3 * s), lo, hi))
        else:
            q[k] = float(np.clip(p[k] + rng.normal() * s * (hi - lo), lo, hi))
    q["MIN_OBS"] = int(round(q["MIN_OBS"]))
    return q


def score(p, C):
    rng = np.random.default_rng(0)
    cls = H.lead_class(p)
    return float(np.mean([H.play(C, H.make_agent(e, len(C), rng), cls(len(C), rng), seed=3)["hit"].mean()
                          for e in ["straight", "jink", "dodge"]]))


def main(n_random=600, n_local=400):
    rng = np.random.default_rng(0)
    C = H.train_courses(256); Ca = C[:128]
    t0 = time.time()
    default = {k: getattr(H.Lead, k) for k in H.LEAD_PARAMS}
    best = (score(default, Ca), default)
    for _ in range(n_random):
        p = sample(rng); s = score(p, Ca)
        if s > best[0]:
            best = (s, p)
    cur, cur_s = best[1], score(best[1], C)
    for i in range(n_local):
        p = perturb(rng, cur, 0.15 * (1 - i / n_local) + 0.02); s = score(p, C)
        if s > cur_s:
            cur, cur_s = p, s
    out = dict(params=cur, train_hit_rate=cur_s, default_train_hit_rate=score(default, C),
               n_configs=1 + n_random + n_local, wall_s=time.time() - t0, ranges=R)
    print(json.dumps(out, indent=1))
    json.dump(cur, open("checkpoints/r6h_lead_tuned.json", "w"), indent=1)
    json.dump(out, open("results/r6h_lead_tuning.json", "w"), indent=1)


if __name__ == "__main__":
    main()
