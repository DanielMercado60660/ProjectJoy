"""Residual PPO (MLX) on top of the tuned CPG trot: Go1 crossing random stone layouts.

Rollouts: 4 worker processes, each stepping `--envs_per_worker` LavaTasks on random TRAINING layouts
(seeds < 10M); the policy runs in numpy on CPU for rollouts. Updates run in MLX; old log-probs and
values are recomputed with MLX before the update so the PPO ratio starts at exactly 1.
The action mean is tanh-bounded; approx-KL and clip fraction are logged.

Usage:
  ../bin/gpu_run ../.venv/bin/python ppo.py --minutes 40 --run main
  ../bin/gpu_run ../.venv/bin/python ppo.py --minutes 40 --run main --resume checkpoints/main_latest.npz
  ../.venv/bin/python ppo.py --tiny --run smoke
"""
import argparse, json, os, sys, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# observation slots that carry terrain information: trunk height above the stone (9) and the
# next-4-stones block (56:76). `--blind` zeroes them (ablation: same CPG + heading term, no stone obs).
STONE_OBS = np.r_[9, 56:76]


def mask_obs(o, blind):
    if blind:
        o = o.copy()
        o[..., STONE_OBS] = 0.0
    return o


# ============================================================================ workers
def worker(conn, n_env, seed, cpg):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import task as T, lava_env as E
    rng = np.random.default_rng(seed)
    tasks = [T.LavaTask(cpg, model=None) for _ in range(n_env)]
    level = [1.0]
    lv = np.ones(n_env)

    def new_ep(t, i=0):
        # gap curriculum: 80% of episodes at the current level, 20% at full difficulty (level 1)
        lvl = level[0] if rng.random() < 0.8 else 1.0
        lv[i] = lvl
        cfg = dict(E.LAYOUT_CFG)
        g0, g1 = E.LAYOUT_CFG["gap"]
        cfg["gap"] = (g0, g0 + lvl * (g1 - g0))
        ls = int(rng.integers(0, E.HELDOUT_SEED_BASE))      # training layouts only
        return t.reset(E.make_layout(ls, cfg), rng, init_noise=0.05)

    obs = np.stack([new_ep(t, i) for i, t in enumerate(tasks)])
    conn.send(obs)
    while True:
        msg = conn.recv()
        if msg is None:
            break
        if isinstance(msg[0], str) and msg[0] == "reseed":
            rng = np.random.default_rng(msg[1])
            obs = np.stack([new_ep(t, i) for i, t in enumerate(tasks)])
            conn.send(obs)
            continue
        acts, level[0] = msg
        rews = np.zeros(n_env, np.float32)
        dones = np.zeros(n_env, np.float32)
        infos = []
        for i, t in enumerate(tasks):
            o, r, d, info = t.step(acts[i])
            rews[i] = r
            if d:
                dones[i] = 1.0
                infos.append(dict(success=info["success"], lava=info["lava"], flip=info["flip"],
                                  frac=float(np.clip(t.env.progress() / (t.goal - t.env.x0), 0, 1)),
                                  energy=t.env.energy, dist=t.env.progress(), t=t.env.t, level=lv[i]))
                o = new_ep(t, i)
            obs[i] = o
        conn.send((obs, rews, dones, infos))


class VecEnv:
    def __init__(self, n_workers, n_env, seed, cpg):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(n_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(b, n_env, seed * 1000 + w, cpg), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.n_env = n_env
        self.blind = False
        self.obs = np.concatenate([c.recv() for c in self.conns])

    def reseed(self, seed):
        for w, c in enumerate(self.conns):
            c.send(("reseed", seed * 1000 + w))
        self.obs = np.concatenate([c.recv() for c in self.conns])

    def step(self, acts, level=1.0):
        for w, c in enumerate(self.conns):
            c.send((acts[w * self.n_env:(w + 1) * self.n_env], level))
        res = [c.recv() for c in self.conns]
        self.obs = mask_obs(np.concatenate([r[0] for r in res]), self.blind)
        infos = [i for r in res for i in r[3]]
        return self.obs, np.concatenate([r[1] for r in res]), np.concatenate([r[2] for r in res]), infos

    def close(self):
        for c in self.conns:
            c.send(None)
        for p in self.procs:
            p.join(timeout=5)


# ============================================================================ model
def build(obs_dim, act_dim, hidden, log_std0=-1.6):
    import mlx.core as mx
    import mlx.nn as nn

    class AC(nn.Module):
        def __init__(self):
            super().__init__()
            self.pi = [nn.Linear(obs_dim, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, act_dim)]
            self.v = [nn.Linear(obs_dim, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, 1)]
            self.log_std = mx.full((act_dim,), log_std0)
            # start as (almost) the pure CPG: tiny last-layer weights
            self.pi[2].weight = self.pi[2].weight * 0.01
            self.pi[2].bias = mx.zeros((act_dim,))

        def mean(self, x):
            h = nn.elu(self.pi[0](x))
            h = nn.elu(self.pi[1](h))
            return mx.tanh(self.pi[2](h))

        def value(self, x):
            h = nn.elu(self.v[0](x))
            h = nn.elu(self.v[1](h))
            return self.v[2](h)[:, 0]

    return AC()


def np_params(model):
    from mlx.utils import tree_flatten
    return {k: np.array(v) for k, v in tree_flatten(model.parameters())}


def np_mean(P, x):
    """Numpy forward of the policy mean (rollouts and evaluation)."""
    elu = lambda z: np.where(z > 0, z, np.expm1(np.minimum(z, 0)))
    h = elu(x @ P["pi.0.weight"].T + P["pi.0.bias"])
    h = elu(h @ P["pi.1.weight"].T + P["pi.1.bias"])
    return np.tanh(h @ P["pi.2.weight"].T + P["pi.2.bias"])


class RunningNorm:
    def __init__(self, dim):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -5, 5).astype(np.float32)


# ============================================================================ checkpoint
def save_ckpt(path, model, opt, norm, it, steps, cfg, extra=None):
    from mlx.utils import tree_flatten
    d = {f"p::{k}": np.array(v) for k, v in tree_flatten(model.parameters())}
    for k, v in tree_flatten(opt.state):
        d[f"o::{k}"] = np.array(v)
    d["norm_mean"], d["norm_var"], d["norm_count"] = norm.mean, norm.var, np.array(norm.count)
    d["it"], d["steps"] = np.array(it), np.array(steps)
    d["cfg"] = np.array(json.dumps(cfg))
    if extra:
        d["extra"] = np.array(json.dumps(extra))
    tmp = path + ".tmp.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)


def load_policy(path):
    """For evaluation: returns (cfg, normaliser, numpy policy params)."""
    z = np.load(path, allow_pickle=False)
    cfg = json.loads(str(z["cfg"]))
    norm = RunningNorm(len(z["norm_mean"]))
    norm.mean, norm.var, norm.count = z["norm_mean"], z["norm_var"], float(z["norm_count"])
    P = {k[3:]: z[k] for k in z.files if k.startswith("p::")}
    return cfg, norm, P


def make_policy_fn(path, mask_stones=False):
    cfg, norm, P = load_policy(path)
    blind = cfg.get("blind", False) or mask_stones
    return (lambda o: np_mean(P, norm(mask_obs(o, blind)[None]))[0]), cfg


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--max_iters", type=int, default=10**9)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--envs_per_worker", type=int, default=16)
    ap.add_argument("--n_steps", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log_std0", type=float, default=-1.6, help="initial log std (std 0.2: small noise around the CPG)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--target_kl", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpg", default="checkpoints/cpg_params.json")
    ap.add_argument("--resume", default="")
    ap.add_argument("--ckpt_every", type=int, default=20)
    ap.add_argument("--level0", type=float, default=0.0, help="initial gap-curriculum level (1 = full gaps)")
    ap.add_argument("--level_up", type=float, default=0.5, help="raise level when success at level >= this")
    ap.add_argument("--blind", action="store_true", help="ablation: zero the stone observations")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    if a.tiny:
        a.workers, a.envs_per_worker, a.n_steps, a.hidden, a.ckpt_every = 2, 2, 32, 32, 2
        a.max_iters = 3 if a.max_iters == 10**9 else a.max_iters
    os.chdir(HERE)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_unflatten
    import task as T

    cpg = json.load(open(a.cpg))["params"] if os.path.exists(a.cpg) else dict(T.DEFAULT_CPG)
    cfg = dict(vars(a), cpg_params=cpg, obs_dim=T.LavaTask.obs_dim, act_dim=T.ACT_DIM)
    obs_dim, act_dim = T.LavaTask.obs_dim, T.ACT_DIM
    mx.random.seed(a.seed)
    model = build(obs_dim, act_dim, a.hidden, a.log_std0)
    opt = optim.Adam(learning_rate=a.lr)
    opt.init(model.trainable_parameters())
    norm = RunningNorm(obs_dim)
    it, steps = 0, 0
    level = a.level0
    if a.resume:
        z = np.load(a.resume, allow_pickle=False)
        model.update(tree_unflatten([(k[3:], mx.array(z[k])) for k in z.files if k.startswith("p::")]))
        opt.state = tree_unflatten([(k[3:], mx.array(z[k])) for k in z.files if k.startswith("o::")])
        norm.mean, norm.var, norm.count = z["norm_mean"], z["norm_var"], float(z["norm_count"])
        it, steps = int(z["it"]), int(z["steps"])
        level = float(json.loads(str(z["extra"]))["level"]) if "extra" in z.files else 1.0
        cpg = json.loads(str(z["cfg"]))["cpg_params"]   # keep the CPG the run started with
        cfg["cpg_params"] = cpg
        print(f"resumed {a.resume} at iter {it}, {steps} env steps", flush=True)
    rng = np.random.default_rng(a.seed * 7919 + it)
    venv = VecEnv(a.workers, a.envs_per_worker, a.seed * 100003 + it, cpg)
    venv.blind = a.blind
    venv.obs = mask_obs(venv.obs, a.blind)
    N = a.workers * a.envs_per_worker
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{a.run}_train.csv"
    if not a.resume or not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("iter,env_steps,wall_s,episodes,success,lava,flip,frac,energy_per_m,mean_rew,approx_kl,clip_frac,"
                    "v_loss,std,fps\n")
    ckpt_path = f"checkpoints/{a.run}_latest.npz"

    def loss_fn(mdl, ob, ac, adv, ret, logp_old):
        mu = mdl.mean(ob)
        ls = mdl.log_std
        logp = (-0.5 * ((ac - mu) / mx.exp(ls)) ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum(-1)
        ratio = mx.exp(logp - logp_old)
        pg = -mx.minimum(ratio * adv, mx.clip(ratio, 1 - a.clip, 1 + a.clip) * adv).mean()
        v = mdl.value(ob)
        vl = ((v - ret) ** 2).mean()
        ent = (ls + 0.5 * np.log(2 * np.pi * np.e)).sum()
        kl = ((ratio - 1) - (logp - logp_old)).mean()
        cf = (mx.abs(ratio - 1) > a.clip).astype(mx.float32).mean()
        return pg + 0.5 * vl - a.ent * ent, (vl, kl, cf)

    grad_fn = nn.value_and_grad(model, loss_fn)

    def logp_value(ob, ac):
        mu = model.mean(ob)
        ls = model.log_std
        logp = (-0.5 * ((ac - mu) / mx.exp(ls)) ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum(-1)
        return logp, model.value(ob)

    t_start = time.time()
    ep_stats = []
    while it < a.max_iters and (time.time() - t_start) < a.minutes * 60:
        P = np_params(model)
        std = np.exp(P["log_std"])
        OB = np.zeros((a.n_steps + 1, N, obs_dim), np.float32)
        AC = np.zeros((a.n_steps, N, act_dim), np.float32)
        RW = np.zeros((a.n_steps, N), np.float32)
        DN = np.zeros((a.n_steps, N), np.float32)
        raw = np.zeros((a.n_steps, N, obs_dim), np.float32)
        obs = venv.obs
        for t in range(a.n_steps):
            raw[t] = obs
            on = norm(obs)
            OB[t] = on
            act = np_mean(P, on) + std * rng.standard_normal((N, act_dim))
            AC[t] = act
            obs, r, d, infos = venv.step(act.astype(np.float64), level)
            RW[t], DN[t] = r, d
            ep_stats += infos
        OB[-1] = norm(obs)
        norm.update(raw.reshape(-1, obs_dim))
        steps += a.n_steps * N
        # recompute old log-probs and values in MLX (same precision as the update)
        flat_ob = mx.array(OB[:-1].reshape(-1, obs_dim))
        flat_ac = mx.array(AC.reshape(-1, act_dim))
        logp_old, v_old = logp_value(flat_ob, flat_ac)
        v_last = model.value(mx.array(OB[-1]))
        mx.eval(logp_old, v_old, v_last)
        V = np.array(v_old).reshape(a.n_steps, N)
        VL = np.array(v_last)
        adv = np.zeros_like(RW)
        last = np.zeros(N, np.float32)
        for t in reversed(range(a.n_steps)):
            nv = VL if t == a.n_steps - 1 else V[t + 1]
            nonterm = 1.0 - DN[t]
            delta = RW[t] + a.gamma * nv * nonterm - V[t]
            last = delta + a.gamma * a.lam * nonterm * last
            adv[t] = last
        ret = (adv + V).reshape(-1)
        advf = adv.reshape(-1)
        advf = (advf - advf.mean()) / (advf.std() + 1e-8)
        B = a.n_steps * N
        mb = B // a.minibatches
        kls, cfs, vls = [], [], []
        stop = False
        for ep in range(a.epochs):
            perm = rng.permutation(B)
            for k in range(a.minibatches):
                idx = mx.array(perm[k * mb:(k + 1) * mb])
                (l, (vl, kl, cf)), g = grad_fn(model, flat_ob[idx], flat_ac[idx], mx.array(advf)[idx],
                                               mx.array(ret)[idx], logp_old[idx])
                g, _ = optim.clip_grad_norm(g, 1.0)
                opt.update(model, g)
                mx.eval(model.parameters(), opt.state, kl, cf, vl)
                kls.append(float(kl)); cfs.append(float(cf)); vls.append(float(vl))
            if np.mean(kls[-a.minibatches:]) > a.target_kl:
                stop = True
                break
        it += 1
        wall = time.time() - t_start
        if ep_stats:
            dist = sum(e["dist"] for e in ep_stats)
            row = [it, steps, round(wall, 1), len(ep_stats), np.mean([e["success"] for e in ep_stats]),
                   np.mean([e["lava"] for e in ep_stats]), np.mean([e["flip"] for e in ep_stats]),
                   np.mean([e["frac"] for e in ep_stats]),
                   sum(e["energy"] for e in ep_stats) / max(dist, 0.1)]
        else:
            row = [it, steps, round(wall, 1), 0, np.nan, np.nan, np.nan, np.nan, np.nan]
        row += [RW.mean(), np.mean(kls), np.mean(cfs), np.mean(vls), float(np.exp(P["log_std"]).mean()),
                a.n_steps * N / max(1e-6, (wall - (getattr(main, "_lw", 0.0))))]
        main._lw = wall
        with open(log_path, "a") as f:
            f.write(",".join(f"{v:.5g}" if isinstance(v, float) else str(v) for v in row) + "\n")
        if it % 5 == 0 or a.tiny:
            print(f"it {it} steps {steps} eps {row[3]} succ {row[4]:.3f} lava {row[5]:.3f} frac {row[7]:.3f} "
                  f"E/m {row[8]:.1f} kl {row[10]:.4f} cf {row[11]:.3f} std {row[13]:.3f} fps {row[14]:.0f} "
                  f"{'(kl stop)' if stop else ''}", flush=True)
        cur = [e for e in ep_stats if e["level"] < 1.0 and abs(e["level"] - level) < 1e-9] if level < 1 else []
        if level < 1.0 and len(cur) >= 20 and np.mean([e["success"] for e in cur]) >= a.level_up:
            level = min(1.0, level + 0.1)
            print(f"it {it}: curriculum level -> {level:.1f}", flush=True)
        with open(f"logs/{a.run}_level.csv", "a") as f:
            f.write(f"{it},{steps},{level:.2f},{np.mean([e['success'] for e in ep_stats if e['level'] >= 1.0] or [np.nan]):.4f}\n")
        ep_stats = []
        if it % a.ckpt_every == 0:
            save_ckpt(ckpt_path, model, opt, norm, it, steps, cfg, dict(level=level))
            if it % (a.ckpt_every * 10) == 0:
                save_ckpt(f"checkpoints/{a.run}_it{it}.npz", model, opt, norm, it, steps, cfg, dict(level=level))
    save_ckpt(ckpt_path, model, opt, norm, it, steps, cfg, dict(level=level))
    print(f"done: iter {it}, env steps {steps}, wall {time.time()-t_start:.0f}s -> {ckpt_path}", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
