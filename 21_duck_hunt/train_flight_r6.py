"""R6 flight: PPO (MLX, CPU device) with the options the R6 brief asks to try, plus full
save/kill/resume. Same sim, same observations and same thrust limits as train_flight.py;
flight.py is not modified.

Options
  --reward orig   the original reward (progress to the waypoint centre - 0.01/step, +2/wp, +5/-5)
  --reward time   time-optimal: -dt per step, -K on crash/timeout, plus POTENTIAL-BASED shaping
                  gamma*Phi(s') - Phi(s), Phi = -(remaining path length)/V_REF (policy-invariant,
                  so no per-step progress bias toward ring centres)
  --residual 1    action = norm_clip(PD+(state) + res_scale * policy), PD+ = pd_plus.py with the
                  tuned params; the policy also sees PD+'s action (derived from the same obs)
  --lr_sched linear   linear lr decay to lr_end over --total (the planned budget)
  --ent           entropy coefficient on the state-independent log-std

Checkpoints (checkpoints/r6/<run>/): state.pkl (weights, Adam state, step, numpy RNG, env state,
config, logs counters), best.npz (best on the 128 training selection courses, flight.score),
snap_<M>M.npz at fixed env-step milestones (for the held-out learning curve; never selected on).

  ../bin/gpu_run ../.venv/bin/python train_flight_r6.py --run t0 --reward time --steps 20000000
  ... --resume  continues from checkpoints/r6/<run>/state.pkl
"""
import argparse, json, os, pickle, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import flight as F
import pd_plus as P

mx.set_default_device(mx.cpu)
HERE = os.path.dirname(os.path.abspath(__file__))
V_REF = 10.0          # m/s, scale of the shaping potential (any value keeps the optimal policy)


# ---------------- observation / policy plumbing ----------------
def remaining_path(env):
    """Distance to the current ring edge + centre-to-centre length of the legs still to fly."""
    ar = np.arange(env.n)
    k = np.minimum(env.k, F.N_WP)
    cur = env.W[ar, k]
    d = np.maximum(np.linalg.norm(cur - env.p, axis=1) - F.WP_R, 0.0)
    legs = np.linalg.norm(np.diff(env.W[:, 1:], axis=1), axis=2)          # (n, N_WP-1)
    idx = np.arange(1, F.N_WP)[None, :]                                    # leg j: wp j -> j+1
    rest = (legs * (idx >= k[:, None])).sum(1)
    return d + rest


class Policy(nn.Module):
    def __init__(self, obs_dim, hid=128, init_log_std=-0.7, zero_last=False):
        super().__init__()
        self.pi = [nn.Linear(obs_dim, hid), nn.Linear(hid, hid), nn.Linear(hid, 2)]
        self.vf = [nn.Linear(obs_dim, hid), nn.Linear(hid, hid), nn.Linear(hid, 1)]
        self.log_std = mx.full((2,), init_log_std)
        if zero_last:
            self.pi[2].weight = mx.zeros_like(self.pi[2].weight)
            self.pi[2].bias = mx.zeros_like(self.pi[2].bias)

    def mean(self, x):
        h = mx.tanh(self.pi[0](x)); h = mx.tanh(self.pi[1](h))
        return mx.tanh(self.pi[2](h))

    def value(self, x):
        h = mx.tanh(self.vf[0](x)); h = mx.tanh(self.vf[1](h))
        return self.vf[2](h)[:, 0]


def logp(mu, log_std, a):
    return (-0.5 * ((a - mu) / mx.exp(log_std)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)


class Agent:
    """Observation builder + action mapping, shared by training and evaluation."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.residual = bool(cfg.get("residual", 0))
        self.res_scale = cfg.get("res_scale", 0.5)
        self.pdp = json.load(open(os.path.join(HERE, cfg.get("pd_plus_params", "checkpoints/pd_plus_params.json")))) \
            if self.residual else None
        self.obs_dim = F.OBS_DIM + (2 if self.residual else 0)

    def obs(self, env, o):
        if not self.residual:
            return o
        self._base = P.pd_plus_action(env, self.pdp)
        return np.concatenate([o, self._base.astype(np.float32)], 1)

    def act(self, env, a):
        """a: raw policy output (sampled or mean) -> env action (call obs() first)."""
        if not self.residual:
            return a
        return P.norm_clip((self._base + self.res_scale * a) * F.AMAX)


def make_policy_fn(model, agent):
    def f(env, o):
        x = agent.obs(env, o)
        return agent.act(env, np.array(model.mean(mx.array(x))))
    return f


def load_policy(run_dir, which="best.npz"):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    agent = Agent(cfg)
    m = Policy(agent.obs_dim, cfg["hid"])
    m.load_weights(os.path.join(run_dir, which))
    return make_policy_fn(m, agent), cfg


# ---------------- checkpointing ----------------
def tree_np(tree):
    return {k: np.array(v) for k, v in nn.utils.tree_flatten(tree) if isinstance(v, mx.array)}


def save_state(path, model, opt, env, rng, extra):
    st = dict(params=tree_np(model.parameters()), opt=tree_np(opt.state),
              env={k: getattr(env, k).copy() for k in ["W", "p", "v", "thr", "wind", "k", "t"]},
              env_rng=env.rng.bit_generator.state, rng=rng.bit_generator.state, **extra)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(st, f)
    os.replace(tmp, path)


def load_state(path, model, opt, env, rng):
    st = pickle.load(open(path, "rb"))
    model.update(nn.utils.tree_unflatten([(k, mx.array(v)) for k, v in st["params"].items()]))
    opt.init(model.trainable_parameters())
    opt.state = nn.utils.tree_unflatten([(k, mx.array(v)) for k, v in st["opt"].items()])
    for k, v in st["env"].items():
        setattr(env, k, v.copy())
    env.rng.bit_generator.state = st["env_rng"]
    rng.bit_generator.state = st["rng"]
    return st


# ---------------- training ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=20_000_000, help="stop after this many env steps (this invocation's target)")
    ap.add_argument("--total", type=int, default=0, help="planned budget for the lr schedule (default = --steps)")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--mb", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_end", type=float, default=0.0)
    ap.add_argument("--lr_sched", default="const", choices=["const", "linear"])
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--init_log_std", type=float, default=-0.7)
    ap.add_argument("--reward", default="time", choices=["orig", "time"])
    ap.add_argument("--fail_pen", type=float, default=10.0, help="time reward: penalty (s) on crash/timeout")
    ap.add_argument("--residual", type=int, default=0)
    ap.add_argument("--res_scale", type=float, default=0.5)
    ap.add_argument("--pd_plus_params", default="checkpoints/pd_plus_params.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=2_000_000, help="env steps between selection evals")
    ap.add_argument("--snap_every", type=int, default=10_000_000)
    ap.add_argument("--save_every", type=int, default=5_000_000)
    ap.add_argument("--max_wall", type=float, default=2580, help="s; save state and exit before this")
    ap.add_argument("--update_device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run_dir = os.path.join(HERE, "checkpoints", "r6", args.run)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(HERE, "results", f"r6_train_{args.run}.jsonl")
    state_path = os.path.join(run_dir, "state.pkl")

    if args.resume:
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
        cfg["steps"] = args.steps; cfg["max_wall"] = args.max_wall
    else:
        cfg = vars(args).copy()
        cfg["total"] = cfg["total"] or cfg["steps"]
        json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    cfg.setdefault("update_device", "cpu")
    c = argparse.Namespace(**cfg)
    mx.random.seed(c.seed)
    rng = np.random.default_rng(c.seed)
    agent = Agent(cfg)
    model = Policy(agent.obs_dim, c.hid, c.init_log_std, zero_last=bool(c.residual))
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=c.lr)
    env = F.FlightEnv(c.envs, seed=1000 + c.seed)
    sel_courses = F.train_courses(256)[:128]      # selection on training courses (same as PD/PD+)
    step, upd, best, next_eval, next_snap, next_save = 0, 0, -1e9, 0, c.snap_every, c.save_every
    if args.resume and os.path.exists(state_path):
        st = load_state(state_path, model, opt, env, rng)
        step, upd, best = st["step"], st["upd"], st["best"]
        next_eval, next_snap, next_save = st["next_eval"], st["next_snap"], st["next_save"]
        print(f"resumed at step {step}", flush=True)
    flog = open(log_path, "a" if args.resume else "w")

    def loss_fn(model, ob, ac, old_lp, adv, ret):
        mu = model.mean(ob)
        lp = logp(mu, model.log_std, ac)
        ratio = mx.exp(lp - old_lp)
        pg = -mx.minimum(ratio * adv, mx.clip(ratio, 1 - c.clip, 1 + c.clip) * adv).mean()
        vl = ((model.value(ob) - ret) ** 2).mean()
        ent = model.log_std.sum()
        return pg + 0.5 * vl - c.ent * ent, (vl, mx.mean(old_lp - lp),
                                             mx.mean((mx.abs(ratio - 1) > c.clip).astype(mx.float32)))

    lg = nn.value_and_grad(model, loss_fn)
    T, N = c.horizon, c.envs
    D_OBS = agent.obs_dim
    o_env = env.obs()
    ep_ret = np.zeros(N); ep_rets, ep_done, ep_t = [], [], []
    t0 = time.time()
    while step < c.steps:
        if time.time() - t0 > c.max_wall:
            print("max_wall reached", flush=True)
            break
        lr = c.lr if c.lr_sched == "const" else c.lr_end + (c.lr - c.lr_end) * max(0.0, 1 - step / c.total)
        opt.learning_rate = lr
        O = np.zeros((T, N, D_OBS), np.float32); A = np.zeros((T, N, 2), np.float32)
        R = np.zeros((T, N), np.float32); Dn = np.zeros((T, N), np.float32); V = np.zeros((T + 1, N), np.float32)
        std = np.exp(np.array(model.log_std))
        for t in range(T):
            x = agent.obs(env, o_env)
            xm = mx.array(x)
            mu, v = model.mean(xm), model.value(xm)
            mu = np.array(mu); V[t] = np.array(v)
            a = (mu + std * rng.normal(size=mu.shape)).astype(np.float32)
            O[t], A[t] = x, a
            if c.reward == "time":
                phi0 = -remaining_path(env) / V_REF
            o_env, r, d, info = env.step(agent.act(env, a))
            if c.reward == "time":
                fail = info["crash"] | info["timeout"]
                phi1 = np.where(d, 0.0, -remaining_path(env) / V_REF)   # done envs were reset: Phi(term)=0
                r = (-F.DT + c.gamma * phi1 - phi0 - c.fail_pen * fail).astype(np.float32)
            R[t], Dn[t] = r, d
            ep_ret += r
            for i in np.nonzero(d)[0]:
                ep_rets.append(ep_ret[i]); ep_done.append(float(info["complete"][i]))
                if info["complete"][i]:
                    ep_t.append(info["t"][i] * F.DT)
                ep_ret[i] = 0
        V[T] = np.array(model.value(mx.array(agent.obs(env, o_env))))
        adv = np.zeros((T, N), np.float32); last = 0
        for t in reversed(range(T)):
            nonterm = 1 - Dn[t]
            delta = R[t] + c.gamma * V[t + 1] * nonterm - V[t]
            last = delta + c.gamma * c.lam * nonterm * last
            adv[t] = last
        ret = adv + V[:T]
        with mx.stream(mx.gpu if c.update_device == "gpu" else mx.cpu):   # rollouts stay on CPU
            ob = mx.array(O.reshape(-1, D_OBS)); ac = mx.array(A.reshape(-1, 2))
            old_lp = logp(model.mean(ob), model.log_std, ac)
            advf = adv.reshape(-1); advf = (advf - advf.mean()) / (advf.std() + 1e-8)
            advm = mx.array(advf); retm = mx.array(ret.reshape(-1))
            n = T * N
            kls, cfs = [], []
            for ep in range(c.epochs):
                perm = rng.permutation(n)
                for s in range(0, n, c.mb):
                    idx = mx.array(perm[s:s + c.mb])
                    (l, (vl, kl, cf)), g = lg(model, ob[idx], ac[idx], old_lp[idx], advm[idx], retm[idx])
                    opt.update(model, g)
                    mx.eval(model.parameters(), opt.state)
                    kls.append(kl.item()); cfs.append(cf.item())
        step += n; upd += 1
        rec = dict(step=step, upd=upd, wall=time.time() - t0, lr=lr,
                   ep_ret=float(np.mean(ep_rets[-500:])) if ep_rets else None,
                   train_completion=float(np.mean(ep_done[-500:])) if ep_done else None,
                   train_time=float(np.mean(ep_t[-500:])) if ep_t else None,
                   approx_kl=float(np.mean(kls)), clip_frac=float(np.mean(cfs)),
                   vloss=float(vl.item()), log_std=np.array(model.log_std).tolist())
        ep_rets, ep_done, ep_t = ep_rets[-500:], ep_done[-500:], ep_t[-500:]
        pol = make_policy_fn(model, agent)
        if step >= next_eval or step >= c.steps:
            next_eval += c.eval_every
            r_ = F.run_episodes(sel_courses, pol, seed=11)
            sc = F.score(r_); rec["sel_score"] = sc; rec["sel"] = F.summarize(r_)
            if sc > best:
                best = sc
                model.save_weights(os.path.join(run_dir, "best.npz"))
                json.dump(dict(step=step, score=sc, summary=rec["sel"]), open(os.path.join(run_dir, "best.json"), "w"))
        if step >= next_snap:
            model.save_weights(os.path.join(run_dir, f"snap_{next_snap // 1_000_000:04d}M.npz"))
            next_snap += c.snap_every
        flog.write(json.dumps(rec) + "\n"); flog.flush()
        if upd % 10 == 0 or "sel" in rec:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("sel",)}
                  | ({"sel_time": round(rec["sel"]["mean_time_completed"], 3), "sel_comp": rec["sel"]["completion"]} if "sel" in rec else {}),
                  flush=True)
        if step >= next_save or step >= c.steps or time.time() - t0 > c.max_wall:
            next_save = step + c.save_every
            save_state(state_path, model, opt, env, rng,
                       dict(step=step, upd=upd, best=best, next_eval=next_eval, next_snap=next_snap,
                            next_save=next_save, cfg=cfg))
    save_state(state_path, model, opt, env, rng,
               dict(step=step, upd=upd, best=best, next_eval=next_eval, next_snap=next_snap,
                    next_save=next_save, cfg=cfg))
    model.save_weights(os.path.join(run_dir, "last.npz"))
    print(f"done step {step} best {best:.4f} wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
