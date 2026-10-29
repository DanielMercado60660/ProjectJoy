"""Cheap baselines and their tuning (CEM / grid search on training seeds only).

  zero     : action 0 = every joint at mid-range (what gym does during reset). "hold still".
  posture  : best constant action, CEM-tuned (20 params). "tuned hold still".
  random   : uniform random actions sigma*U[-1,1] around 0; sigma grid-searched.
  gait     : scripted closed-loop controller: a(t) = c + A*sin(s*2*pi*f*t*dt + phi), where
             s = sign of the yaw direction that reduces the axis error (from the observed egg
             orientation and goal, same information the policy sees); switches to the constant
             posture c once error < 0.8*thr and resumes the gait if error > thr. CEM over
             c, A, phi, f (61 params), c initialised from the tuned posture.

Tuning objective (all, on training seeds): success - drop - 0.5*over_budget (end-of-step force).
Usage: ../bin/gpu_run ../.venv/bin/python baselines.py --out checkpoints/baselines.json
"""
import argparse
import json
import time

import numpy as np

from egg_env import FORCE_BUDGET, N_SUB, THR, EggVecEnv, quat_to_zaxis, run_episodes

DT = 0.002 * N_SUB


class Zero:
    def __call__(self, obs, priv, env):
        return np.zeros((env.n, 20))


class Posture:
    def __init__(self, c):
        self.c = np.clip(np.asarray(c), -1, 1)

    def __call__(self, obs, priv, env):
        return np.tile(self.c, (env.n, 1))


class Random:
    def __init__(self, sigma, seed=0):
        self.sigma, self.rng = sigma, np.random.default_rng(seed)

    def __call__(self, obs, priv, env):
        return self.sigma * self.rng.uniform(-1, 1, (env.n, 20))


def yaw_sign(env, i):
    """+1 if rotating the egg about +z reduces the (symmetric) axis error, else -1."""
    a = quat_to_zaxis(env.datas[i].qpos[env.oq + 3:env.oq + 7])
    g = env.goal_axis[i] * (1.0 if a @ env.goal_axis[i] >= 0 else -1.0)
    return 1.0 if np.cross(a, g)[2] >= 0 else -1.0


class Gait:
    def __init__(self, c, A, phi, f):
        self.c, self.A, self.phi, self.f = np.clip(c, -1, 1), np.abs(A), np.asarray(phi), float(f)

    def reset(self, env):
        self.active = np.ones(env.n, bool)
        self.clock = np.zeros(env.n)

    def __call__(self, obs, priv, env):
        if not hasattr(self, "active") or len(self.active) != env.n:
            self.reset(env)
        out = np.empty((env.n, 20))
        for i in range(env.n):
            e = env.err[i]
            if e < 0.8 * THR:
                self.active[i] = False
            elif e > THR:
                self.active[i] = True
            if env.t[i] == 0:
                self.clock[i] = 0.0
            if self.active[i]:
                s = yaw_sign(env, i)
                self.clock[i] += s * DT
                out[i] = self.c + self.A * np.sin(2 * np.pi * self.f * self.clock[i] + self.phi)
            else:
                out[i] = self.c
        return out

    def to_dict(self):
        return {"c": self.c.tolist(), "A": self.A.tolist(), "phi": self.phi.tolist(), "f": self.f}


OBJ = {"kind": "default", "margin": FORCE_BUDGET}


def score(env, pol, seed, n):
    env.rngs = [np.random.default_rng([seed, i]) for i in range(n)]
    r = run_episodes(env, pol, n, budget=OBJ["margin"])
    if OBJ["kind"] == "within_budget":  # force-aware retune: success with end-of-step force < margin
        s = np.mean(r["success"] & ~r["over_budget"]) - r["dropped"].mean()
    else:
        s = r["success"].mean() - r["dropped"].mean() - 0.5 * r["over_budget"].mean()
    return s, r


def cem(env, make, mu, sd, lo, hi, iters, pop, n_eps, seed, log, name):
    rng = np.random.default_rng(seed)
    best = (-1e9, mu.copy())
    for it in range(iters):
        X = np.clip(mu + sd * rng.standard_normal((pop, len(mu))), lo, hi)
        X[0] = mu  # keep the current mean in the population
        S = []
        for k in range(pop):
            s, _ = score(env, make(X[k]), 1000 + it, n_eps)  # common random numbers per iteration
            S.append(s)
        S = np.array(S)
        el = X[np.argsort(-S)[: max(2, pop // 4)]]
        mu, sd = el.mean(0), np.maximum(el.std(0), 0.02)
        if S.max() > best[0]:
            best = (float(S.max()), X[np.argmax(S)].copy())
        log(f"{name} it {it}: best {S.max():.3f} mean {S.mean():.3f} (mean-param score {S[0]:.3f})")
    # re-score the best and the final mean on fresh training seeds, keep the better one
    sb = np.mean([score(env, make(best[1]), 2000 + k, n_eps)[0] for k in range(2)])
    sm = np.mean([score(env, make(mu), 2000 + k, n_eps)[0] for k in range(2)])
    log(f"{name} final: best-sample {sb:.3f}, cem-mean {sm:.3f}")
    return (best[1], sb) if sb >= sm else (mu, sm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/baselines.json")
    ap.add_argument("--n_eps", type=int, default=48)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--theta_max", type=float, default=0.4)
    ap.add_argument("--log", default="logs/baselines_tune.log")
    ap.add_argument("--force_aware", default=None,
                    help="path to an existing baselines.json: re-tune only the gait (warm start) with "
                         "objective success-within-budget (end-of-step force < --margin) - drop")
    ap.add_argument("--margin", type=float, default=7.0)
    a = ap.parse_args()
    logf = open(a.log, "a")

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    t0 = time.time()
    env = EggVecEnv(a.n_eps, seed=0, auto_reset=False, theta_max=a.theta_max)
    if a.force_aware:
        OBJ.update(kind="within_budget", margin=a.margin)
        prev = json.load(open(a.force_aware))
        g0 = prev["gait"]
        lo = np.r_[-np.ones(20), np.zeros(20), -np.pi * np.ones(20), 0.25]
        hi = np.r_[np.ones(20), 0.6 * np.ones(20), np.pi * np.ones(20), 4.0]
        mu = np.r_[g0["c"], g0["A"], g0["phi"], g0["f"]]
        sd = np.r_[0.1 * np.ones(20), 0.05 * np.ones(20), 0.5 * np.ones(20), 0.3]
        mk = lambda x: Gait(x[:20], x[20:40], x[40:60], x[60])  # noqa: E731
        log(f"force-aware: warm-start gait scores {score(env, mk(mu), 999, a.n_eps)[0]:.3f}")
        g, sg = cem(env, mk, mu, sd, lo, hi, a.iters, a.pop, a.n_eps, 3, log, "gait-forceaware")
        out = {"theta_max": a.theta_max, "objective": f"mean(success & end-of-step force < {a.margin} N) - drop",
               "gait": {**mk(g).to_dict(), "score": float(sg)}, "tune_wall_s": time.time() - t0,
               "tune_env_steps": int((a.iters * a.pop + 5) * a.n_eps * env.T)}
        json.dump(out, open(a.out, "w"), indent=1)
        log(f"saved {a.out} in {time.time() - t0:.0f}s")
        return
    out = {"theta_max": a.theta_max, "objective": "success - drop - 0.5*over_budget, training seeds"}
    s, _ = score(env, Zero(), 999, a.n_eps)
    log(f"zero: {s:.3f}")
    out["zero_score"] = float(s)
    # random: sigma grid
    grid = [0.02, 0.05, 0.1, 0.2, 0.4, 1.0]
    rs = {sg: score(env, Random(sg, 0), 999, a.n_eps)[0] for sg in grid}
    log(f"random grid {rs}")
    out["random"] = {"sigma": max(rs, key=rs.get), "grid": {str(k): float(v) for k, v in rs.items()}}
    # posture CEM
    c, sc = cem(env, Posture, np.zeros(20), 0.3 * np.ones(20), -1, 1, max(4, a.iters - 2), a.pop,
                a.n_eps, 1, log, "posture")
    out["posture"] = {"c": c.tolist(), "score": float(sc)}
    # gait CEM: params = c(20), A(20), phi(20), f(1)
    lo = np.r_[-np.ones(20), np.zeros(20), -np.pi * np.ones(20), 0.25]
    hi = np.r_[np.ones(20), 0.6 * np.ones(20), np.pi * np.ones(20), 4.0]
    mu = np.r_[c, 0.15 * np.ones(20), np.zeros(20), 1.5]
    sd = np.r_[0.1 * np.ones(20), 0.1 * np.ones(20), 1.5 * np.ones(20), 0.8]
    mk = lambda x: Gait(x[:20], x[20:40], x[40:60], x[60])  # noqa: E731
    g, sg = cem(env, mk, mu, sd, lo, hi, a.iters, a.pop, a.n_eps, 2, log, "gait")
    out["gait"] = {**mk(g).to_dict(), "score": float(sg)}
    out["tune_wall_s"] = time.time() - t0
    n_runs = 1 + len(grid) + (max(4, a.iters - 2) + a.iters) * a.pop + 8  # approx episodes/n_eps
    out["tune_env_steps"] = int(n_runs * a.n_eps * env.T)
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"saved {a.out} in {time.time() - t0:.0f}s, ~{out['tune_env_steps']} env steps")


if __name__ == "__main__":
    main()
