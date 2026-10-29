"""PPO (MLX) with an asymmetric critic for the narrow egg z-rotation task.

Actor input : egg_env obs (188): hand joint pos/vel, egg vel/pos/quat/axis, goal quat/axis,
              axis cross/dot/error, log(1+touch) for the 92 touch sensors, previous action.
Critic input: actor obs + privileged (6): friction scale, mass scale, true contact force on the
              egg / budget, time fraction t/T, signed target delta, initial error.
Action      : Gaussian around a tanh-bounded mean, absolute joint targets in [-1,1] (gym semantics).
Rollouts run the actor in numpy on CPU; log-probs and values used by the update are recomputed
by the MLX model on the batch before the first gradient step (so the ratio starts at exactly 1).

  ../bin/gpu_run ../.venv/bin/python train.py --run run1 --minutes 43
  ../bin/gpu_run ../.venv/bin/python train.py --run run2 --resume checkpoints/run1/latest.npz --minutes 43
"""
import argparse
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from egg_env import EggVecEnv


class RunningNorm:
    def __init__(self, dim):
        self.mean, self.var, self.count = np.zeros(dim), np.ones(dim), 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        d, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -5, 5).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, i, o, h=256, out_scale=1.0):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(i, h), nn.Linear(h, h), nn.Linear(h, o)
        self.l3.weight = self.l3.weight * out_scale
        self.l3.bias = mx.zeros_like(self.l3.bias)

    def __call__(self, x):
        return self.l3(nn.elu(self.l2(nn.elu(self.l1(x)))))


class AC(nn.Module):
    def __init__(self, od, pd, ad=20, h=256, log_std=-1.0):
        super().__init__()
        self.actor = MLP(od, ad, h, out_scale=0.01)
        self.critic = MLP(od + pd, 1, h)
        self.log_std = mx.full((ad,), log_std)

    def dist(self, o):
        return mx.tanh(self.actor(o)), mx.clip(self.log_std, -3.0, 0.0)

    def value(self, o, p):
        return self.critic(mx.concatenate([o, p], axis=-1)).squeeze(-1)


def logp(a, mu, ls):
    return mx.sum(-0.5 * ((a - mu) / mx.exp(ls)) ** 2 - ls - 0.5 * np.log(2 * np.pi), axis=-1)


def np_actor(model):
    """Snapshot actor weights to numpy for CPU rollouts."""
    W = [(np.array(l.weight), np.array(l.bias)) for l in (model.actor.l1, model.actor.l2, model.actor.l3)]
    ls = np.clip(np.array(model.log_std), -3, 0)

    def elu(x):
        return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))

    def f(o):
        h = elu(o @ W[0][0].T + W[0][1])
        h = elu(h @ W[1][0].T + W[1][1])
        return np.tanh(h @ W[2][0].T + W[2][1]), ls
    return f


def save_ckpt(path, model, opt, norm_o, norm_p, state, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrs = {f"model/{k}": v for k, v in tree_flatten(model.parameters())}
    arrs.update({f"opt/{k}": v for k, v in tree_flatten(opt.state)})
    arrs.update({"norm_o/mean": mx.array(norm_o.mean), "norm_o/var": mx.array(norm_o.var),
                 "norm_p/mean": mx.array(norm_p.mean), "norm_p/var": mx.array(norm_p.var)})
    tmp = path + ".tmp.npz"
    mx.savez(tmp, **arrs)
    os.replace(tmp, path)
    meta = {"state": state, "cfg": cfg, "norm_o_count": norm_o.count, "norm_p_count": norm_p.count}
    json.dump(meta, open(path.replace(".npz", ".json"), "w"), indent=1, default=str)


def load_ckpt(path, model, opt=None, norm_o=None, norm_p=None):
    arrs = mx.load(path)
    meta = json.load(open(path.replace(".npz", ".json")))
    model.update(tree_unflatten([(k[6:], v) for k, v in arrs.items() if k.startswith("model/")]))
    if opt is not None:
        opt.state = tree_unflatten([(k[4:], v) for k, v in arrs.items() if k.startswith("opt/")])
    if norm_o is not None:
        norm_o.mean, norm_o.var = np.array(arrs["norm_o/mean"]), np.array(arrs["norm_o/var"])
        norm_p.mean, norm_p.var = np.array(arrs["norm_p/mean"]), np.array(arrs["norm_p/var"])
        norm_o.count, norm_p.count = meta["norm_o_count"], meta["norm_p_count"]
    return meta


def load_policy(path):
    """For eval/demo: returns policy(obs, priv, env) -> deterministic (mean) action, and meta."""
    meta = json.load(open(path.replace(".npz", ".json")))
    cfg = meta["cfg"]
    od, pd = cfg["obs_dim"], cfg["priv_dim"]
    model = AC(od, pd, h=cfg["hidden"])
    no, npv = RunningNorm(od), RunningNorm(pd)
    load_ckpt(path, model, None, no, npv)
    f = np_actor(model)

    def pol(obs, priv, env):
        return f(no(obs))[0]
    pol.meta = meta
    pol.std = float(np.exp(np.clip(np.array(model.log_std), -3, 0)).mean())
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run1")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--minutes", type=float, default=43)
    ap.add_argument("--max_iters", type=int, default=10 ** 9)
    ap.add_argument("--n_envs", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--target_kl", type=float, default=0.03)
    ap.add_argument("--rew_scale", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--theta_max", type=float, default=0.4)
    ap.add_argument("--curr_up", type=float, default=0.6, help="widen theta_max when success > this")
    ap.add_argument("--theta_cap", type=float, default=np.pi / 2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force_w", type=float, default=0.05, help="reward penalty per N above force_margin")
    ap.add_argument("--force_margin", type=float, default=10.0)
    ap.add_argument("--force_cap", type=float, default=0.5, help="max force penalty per step")
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    a = ap.parse_args()
    cfg = vars(a).copy()
    run_dir = f"{a.ckpt_dir}/{a.run}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(a.log_dir, exist_ok=True)
    logf = open(f"{a.log_dir}/{a.run}.jsonl", "a")

    state = {"iter": 0, "env_steps": 0, "theta_max": a.theta_max, "episodes": 0, "wall": 0.0}
    mx.random.seed(a.seed)
    env_seed = a.seed
    rng = np.random.default_rng(a.seed)
    probe = EggVecEnv(1, threads=1)
    od, pd = probe.obs_dim, probe.priv_dim
    cfg.update(obs_dim=od, priv_dim=pd)
    del probe
    model = AC(od, pd, h=a.hidden)
    opt = optim.Adam(learning_rate=a.lr)
    norm_o, norm_p = RunningNorm(od), RunningNorm(pd)
    if a.resume:
        meta = load_ckpt(a.resume, model, opt, norm_o, norm_p)
        state = meta["state"]
        rng.bit_generator.state = state.pop("np_rng")
        # fresh env RNG stream per resume (episodes in flight at save time are not restored)
        env_seed = a.seed + 1000 * (1 + state["iter"])
        print(f"resumed {a.resume} at iter {state['iter']}, steps {state['env_steps']}", flush=True)
    mx.eval(model.parameters())
    env = EggVecEnv(a.n_envs, seed=env_seed, theta_max=state["theta_max"], threads=a.threads)
    env.force_w, env.force_margin, env.force_cap = a.force_w, a.force_margin, a.force_cap
    obs, priv = env.reset()

    def loss_fn(mdl, o, p, act, lp_old, adv, ret):
        mu, ls = mdl.dist(o)
        lp = logp(act, mu, ls)
        ratio = mx.exp(lp - lp_old)
        pg = -mx.mean(mx.minimum(ratio * adv, mx.clip(ratio, 1 - a.clip, 1 + a.clip) * adv))
        v = mdl.value(o, p)
        vl = mx.mean((v - ret) ** 2)
        kl = mx.mean((ratio - 1) - (lp - lp_old))
        cf = mx.mean((mx.abs(ratio - 1) > a.clip).astype(mx.float32))
        return pg + 0.5 * vl, (pg, vl, kl, cf)

    grad_fn = nn.value_and_grad(model, loss_fn)
    t_start = time.time()
    wall0 = state["wall"]
    ep_stats = []  # (success, dropped, over_budget, return)
    ep_ret = np.zeros(a.n_envs)
    ep_over = np.zeros(a.n_envs, bool)
    H, N = a.horizon, a.n_envs
    while state["iter"] < a.max_iters:
        it_t0 = time.time()
        actor = np_actor(model)
        O = np.zeros((H, N, od), np.float32)
        P = np.zeros((H, N, pd), np.float32)
        A = np.zeros((H, N, 20), np.float32)
        R = np.zeros((H, N), np.float32)
        D = np.zeros((H, N), np.float32)
        raw_o, raw_p = [], []
        for t in range(H):
            raw_o.append(obs)
            raw_p.append(priv)
            on, pn = norm_o(obs), norm_p(priv)
            mu, ls = actor(on)
            act = (mu + np.exp(ls) * rng.standard_normal(mu.shape)).astype(np.float32)
            O[t], P[t], A[t] = on, pn, act
            obs, priv, r, done, info = env.step(act)
            R[t], D[t] = r * a.rew_scale, done
            ep_ret += r
            ep_over |= info["force"] > 10.0
            for i in np.nonzero(done)[0]:
                ep_stats.append((info["success"][i], info["dropped"][i], ep_over[i], ep_ret[i]))
                ep_ret[i], ep_over[i] = 0.0, False
        state["env_steps"] += H * N
        roll_t = time.time() - it_t0
        # recompute values/logp with MLX (one consistent numeric path for the ratio)
        Om, Pm, Am = mx.array(O.reshape(H * N, od)), mx.array(P.reshape(H * N, pd)), mx.array(A.reshape(H * N, 20))
        mu, ls = model.dist(Om)
        LP = logp(Am, mu, ls)
        V = model.value(Om, Pm)
        Vlast = model.value(mx.array(norm_o(obs)), mx.array(norm_p(priv)))
        mx.eval(LP, V, Vlast)
        Vn = np.array(V).reshape(H, N)
        vl_ = np.array(Vlast)
        adv = np.zeros((H, N), np.float32)
        last = np.zeros(N, np.float32)
        for t in reversed(range(H)):
            nv = vl_ if t == H - 1 else Vn[t + 1]
            nonterm = 1.0 - D[t]
            delta = R[t] + a.gamma * nv * nonterm - Vn[t]
            last = delta + a.gamma * a.lam * nonterm * last
            adv[t] = last
        ret = (adv + Vn).reshape(-1)
        advf = adv.reshape(-1)
        # update normalizers after computing this batch's inputs (stats used next iteration)
        norm_o.update(np.concatenate(raw_o))
        norm_p.update(np.concatenate(raw_p))
        LPn = LP
        idx_all = np.arange(H * N)
        stats = []
        stop = False
        for ep in range(a.epochs):
            rng.shuffle(idx_all)
            for mb in np.array_split(idx_all, a.minibatches):
                mbi = mx.array(mb)
                ad = advf[mb]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                (l, aux), g = grad_fn(model, Om[mbi], Pm[mbi], Am[mbi], LPn[mbi], mx.array(ad),
                                      mx.array(ret[mb]))
                g, _ = optim.clip_grad_norm(g, 1.0)
                opt.update(model, g)
                mx.eval(model.parameters(), opt.state, *aux)
                stats.append([float(x) for x in aux])
                if stats[-1][2] > 1.5 * a.target_kl:
                    stop = True
                    break
            if stop:
                break
        state["iter"] += 1
        st = np.array(stats).mean(0)
        recent = ep_stats[-256:]
        state["episodes"] += 0
        succ = float(np.mean([x[0] for x in recent])) if recent else 0.0
        rec = {"iter": state["iter"], "env_steps": state["env_steps"], "theta_max": state["theta_max"],
               "succ": succ, "drop": float(np.mean([x[1] for x in recent])) if recent else 0.0,
               "over_budget": float(np.mean([x[2] for x in recent])) if recent else 0.0,
               "ep_return": float(np.mean([x[3] for x in recent])) if recent else 0.0,
               "pg": st[0], "vloss": st[1], "approx_kl": st[2], "clip_frac": st[3],
               "std": float(np.exp(np.clip(np.array(model.log_std), -3, 0)).mean()),
               "updates": len(stats), "sps": H * N / (time.time() - it_t0), "roll_s": roll_t,
               "wall_min": (wall0 + time.time() - t_start) / 60}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        if state["iter"] % 5 == 0 or state["iter"] == 1:
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        # curriculum: widen the target range once the current one is mostly solved
        if len(ep_stats) >= 256 and succ > a.curr_up and state["theta_max"] < a.theta_cap - 1e-6:
            state["theta_max"] = float(min(a.theta_cap, state["theta_max"] + 0.2))
            env.theta_max = state["theta_max"]
            ep_stats.clear()
            print(f"curriculum: theta_max -> {state['theta_max']:.2f} at iter {state['iter']}", flush=True)
            logf.write(json.dumps({"curriculum": state["theta_max"], "iter": state["iter"]}) + "\n")
        out_of_time = (time.time() - t_start) / 60 > a.minutes
        if state["iter"] % a.ckpt_every == 0 or out_of_time or state["iter"] >= a.max_iters:
            state["wall"] = wall0 + time.time() - t_start
            s2 = dict(state, np_rng=rng.bit_generator.state)
            save_ckpt(f"{run_dir}/latest.npz", model, opt, norm_o, norm_p, s2, cfg)
        if out_of_time:
            print(f"time budget reached at iter {state['iter']}", flush=True)
            break


if __name__ == "__main__":
    main()
