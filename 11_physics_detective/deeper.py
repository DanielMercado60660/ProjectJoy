"""Go deeper: two hidden numbers at once.

World 2 "two frictions": mu_k AND mu_s are both hidden (mu_s = r * mu_k, r unknown in
[1, 1.8]). Sliding pushes only reveal mu_k; whether the block breaks free at all is the only
evidence about mu_s. Can EIG find experiments that separate the two?

World 3 "uncalibrated pusher": the block's mass m and the pusher's gain k are hidden (friction
known). Every push trajectory depends only on k/m, so (k, m) is NOT identifiable from pushes:
the posterior is a ridge. We report the identifiable combination k/m and the ridge, and show
that one different kind of experiment (weighing the block on a spring scale) breaks the tie.

    ../.venv/bin/python deeper.py world2 --worlds 200 --steps 12
    ../.venv/bin/python deeper.py world3
"""
import argparse
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp

from inference import eig
from physics import NT, SIGMA, loglik_frames, observe, positions

# ---------------------------------------------------------------- generic 2-D grid posterior
class Grid2D:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.A, self.B = np.meshgrid(a, b, indexing="ij")
        self.logp = np.zeros(self.A.shape)

    def add(self, ll):
        self.logp = self.logp + ll
        self.logp -= self.logp.max()

    @property
    def w(self):
        w = np.exp(self.logp)
        return w / w.sum()

    def particles(self, K, rng):
        """Systematic resampling of grid cells + uniform jitter inside the cell."""
        w = self.w.ravel()
        c = np.cumsum(w)
        u = (rng.random() + np.arange(K)) / K
        idx = np.minimum(np.searchsorted(c, u), len(w) - 1)
        ia, ib = np.unravel_index(idx, self.A.shape)
        da, db = np.gradient(self.a).mean(), np.gradient(self.b).mean()
        pa = self.a[ia] + (rng.random(K) - 0.5) * da
        pb = self.b[ib] + (rng.random(K) - 0.5) * db
        return np.clip(pa, self.a[0], self.a[-1]), np.clip(pb, self.b[0], self.b[-1])


def marg_stats(vals, w, levels=(0.5, 0.9)):
    """Mean and central intervals of a 1-D marginal given values on the grid and weights."""
    order = np.argsort(vals.ravel())
    v, ww = vals.ravel()[order], w.ravel()[order]
    c = np.cumsum(ww); c /= c[-1]
    out = {"mean": float((v * ww).sum())}
    for lv in levels:
        out[f"ci{int(lv*100)}"] = (float(np.interp((1 - lv) / 2, c, v)), float(np.interp((1 + lv) / 2, c, v)))
    return out


# ---------------------------------------------------------------- world 2: two frictions
MUK = np.linspace(0.05, 0.8, 1500)   # fine: active posteriors get ~0.001 wide
RAT = np.linspace(1.0, 1.8, 200)
F2 = np.round(np.arange(0.5, 15.01, 0.25), 3)
T2 = np.array([0.1, 0.2, 0.4, 0.7, 1.0])
DESIGNS2 = np.array([(f, t) for f in F2 for t in T2])


def sim2(F, T, mu_k, r):
    return positions(F, T, mu_k, r * np.asarray(mu_k))


def world2_one(args):
    w, steps, seed = args
    rng_w = np.random.default_rng([seed, w, 77])
    mu_k, r = rng_w.uniform(0.05, 0.8), rng_w.uniform(1.0, 1.8)
    mu_s = mu_k * r
    out = {"mu_k": mu_k, "mu_s": mu_s}
    for si, strat in enumerate(["active", "random"]):
        rng = np.random.default_rng([seed, w, si, 77])
        post = Grid2D(MUK, RAT)
        X_grid = None
        rec = {k: [] for k in ["muk_mean", "mus_mean", "muk_lo90", "muk_hi90", "mus_lo90", "mus_hi90",
                               "muk_lo50", "muk_hi50", "mus_lo50", "mus_hi50", "design", "moved"]}
        for n in range(steps):
            if strat == "active":
                p = post.particles(96, rng)
                d = DESIGNS2[int(np.argmax(eig(p, DESIGNS2, M=2, rng=rng, sim=sim2)))]
            else:
                d = DESIGNS2[rng.integers(len(DESIGNS2))]
            x = sim2(d[0], d[1], mu_k, r)
            y, c = observe(x, rng)
            post.add(loglik_frames(y, c, sim2(d[0], d[1], post.A, post.B)))
            w_ = post.w
            mk = marg_stats(post.A, w_); ms = marg_stats(post.A * post.B, w_)
            rec["muk_mean"].append(mk["mean"]); rec["mus_mean"].append(ms["mean"])
            for lv in (50, 90):
                rec[f"muk_lo{lv}"].append(mk[f"ci{lv}"][0]); rec[f"muk_hi{lv}"].append(mk[f"ci{lv}"][1])
                rec[f"mus_lo{lv}"].append(ms[f"ci{lv}"][0]); rec[f"mus_hi{lv}"].append(ms[f"ci{lv}"][1])
            rec["design"].append(d); rec["moved"].append(bool(x[-1] > 0))
        out[strat] = {k: np.array(v) for k, v in rec.items()}
    return out


def run_world2(worlds, steps, seed, outp, procs=4):
    t0 = time.time()
    jobs = [(w, steps, seed) for w in range(worlds)]
    with Pool(procs) as p:
        recs = p.map(world2_one, jobs, chunksize=2)
    out = {"mu_k": np.array([r["mu_k"] for r in recs]), "mu_s": np.array([r["mu_s"] for r in recs])}
    for s in ["active", "random"]:
        for k in recs[0][s]:
            out[f"{s}_{k}"] = np.stack([r[s][k] for r in recs])
    np.savez_compressed(outp, **out)
    print(f"world2: saved {outp} in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------- world 3: uncalibrated pusher
MU3_K, MU3_S = 0.3, 0.36            # known friction in world 3
LOGK = np.linspace(np.log(0.5), np.log(2.0), 160)
LOGM = np.linspace(np.log(0.5), np.log(3.0), 160)
SCALE_SIGMA = 0.05                  # spring-scale reading noise (kg)
DESIGNS3 = np.array([(f, t) for f in np.linspace(1, 15, 15) for t in np.linspace(0.1, 1.0, 10)])


def sim3(F, T, logk, logm):
    return positions(F, T, MU3_K, MU3_S, m=np.exp(logm), k=np.exp(logk))


def eig_scale(particles, rng, M=4):
    """EIG of one spring-scale weighing (reading = m + N(0, SCALE_SIGMA^2))."""
    lk, lm = particles
    m = np.exp(lm)
    K = len(m)
    y = m[:, None] + SCALE_SIGMA * rng.standard_normal((K, M))
    L = -0.5 * ((y[:, :, None] - m[None, None, :]) / SCALE_SIGMA) ** 2           # (K,M,K)
    own = np.einsum("imi->im", L)
    return float((own - logsumexp(L, axis=2) + np.log(K)).mean())


def run_world3(outp, seed=3):
    rng = np.random.default_rng(seed)
    k_true, m_true = 1.3, 1.8
    res = {"k_true": k_true, "m_true": m_true, "logk": LOGK, "logm": LOGM}
    # (a) six random pushes -> ridge
    post = Grid2D(LOGK, LOGM)
    data = []
    while len(data) < 6:
        d = DESIGNS3[rng.integers(len(DESIGNS3))]
        x = sim3(d[0], d[1], np.log(k_true), np.log(m_true))
        y, c = observe(x, rng)
        data.append((d, y, c))
        post.add(loglik_frames(y, c, sim3(d[0], d[1], post.A, post.B)))
    res["post_pushes"] = post.w.copy()
    ratio = np.exp(post.A - post.B)
    res["ratio_stats"] = marg_stats(ratio, post.w)
    res["m_stats_pushes"] = marg_stats(np.exp(post.B), post.w)
    res["k_stats_pushes"] = marg_stats(np.exp(post.A), post.w)
    # prior 90% interval of m for comparison (log-uniform prior)
    pw = np.ones_like(post.w) / post.w.size
    res["m_stats_prior"] = marg_stats(np.exp(post.B), pw)
    # (b) conventional least squares on (k, m) from 20 random starts: lands anywhere on the ridge
    def resid(th):
        return np.concatenate([np.where(c, 0.0, np.nan_to_num(y) - sim3(d[0], d[1], th[0], th[1])) for d, y, c in data])
    fits = []
    for s in range(20):
        th0 = np.array([rng.uniform(LOGK[0], LOGK[-1]), rng.uniform(LOGM[0], LOGM[-1])])
        f = least_squares(resid, th0, bounds=([LOGK[0], LOGM[0]], [LOGK[-1], LOGM[-1]]))
        J = f.jac
        cond = np.linalg.cond(J.T @ J)
        fits.append([f.x[0], f.x[1], np.sum(f.fun ** 2), cond])
    res["lsq_fits"] = np.array(fits)
    # (c) one spring-scale weighing breaks the ridge
    post2 = Grid2D(LOGK, LOGM); post2.logp = post.logp.copy()
    reading = m_true + SCALE_SIGMA * rng.standard_normal()
    post2.add(-0.5 * ((reading - np.exp(post2.B)) / SCALE_SIGMA) ** 2)
    res["post_scale"] = post2.w.copy()
    res["m_stats_scale"] = marg_stats(np.exp(post2.B), post2.w)
    res["k_stats_scale"] = marg_stats(np.exp(post2.A), post2.w)
    # (d) an EIG detective that may push OR weigh: which does it pick, step by step?
    post3 = Grid2D(LOGK, LOGM)
    choice_log = []
    for n in range(6):
        p = post3.particles(128, rng)
        e_push = eig(p, DESIGNS3, M=2, rng=rng, sim=sim3)
        e_scale = eig_scale(p, rng)
        if e_scale > e_push.max():
            reading = m_true + SCALE_SIGMA * rng.standard_normal()
            post3.add(-0.5 * ((reading - np.exp(post3.B)) / SCALE_SIGMA) ** 2)
            choice_log.append(("weigh", float(e_scale), float(e_push.max())))
        else:
            d = DESIGNS3[int(np.argmax(e_push))]
            y, c = observe(sim3(d[0], d[1], np.log(k_true), np.log(m_true)), rng)
            post3.add(loglik_frames(y, c, sim3(d[0], d[1], post3.A, post3.B)))
            choice_log.append((f"push F={d[0]:.0f}N T={d[1]:.1f}s", float(e_scale), float(e_push.max())))
    res["eig_choices"] = choice_log
    res["m_stats_active"] = marg_stats(np.exp(post3.B), post3.w)
    res["k_stats_active"] = marg_stats(np.exp(post3.A), post3.w)
    np.save(outp, res, allow_pickle=True)
    print("world3 saved", outp)
    for k in ["ratio_stats", "m_stats_prior", "m_stats_pushes", "k_stats_pushes", "m_stats_scale", "k_stats_scale",
              "m_stats_active", "k_stats_active"]:
        print(k, res[k])
    print("true k/m", k_true / m_true)
    print("LSQ fits (k, m, sse, cond(JtJ)):")
    for f in res["lsq_fits"][:8]:
        print(f"  k={np.exp(f[0]):.3f} m={np.exp(f[1]):.3f} k/m={np.exp(f[0]-f[1]):.4f} sse={f[2]:.4f} cond={f[3]:.2e}")
    print("EIG detective choices:", choice_log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["world2", "world3"])
    ap.add_argument("--worlds", type=int, default=200)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    if a.which == "world2":
        run_world2(a.worlds, a.steps, a.seed, "results/world2.npz", a.procs)
    else:
        run_world3("results/world3.npy")
