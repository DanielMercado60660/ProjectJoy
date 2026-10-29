"""DeepMimic-style motion-imitation PPO in MLX: phase input, reference-state initialisation (random start
phase), early termination on falls, DeepMimic tracking reward. The policy outputs a residual on the
PD targets: target = q_ref(t+dt) + act_scale * a, a in [-1, 1] (act_scale in rad, see --act_scale).

Variants (--residual):
  pd    : residual on plain PD tracking (same tuned kp/kd as the baseline, no feed-forward, no balance term)
  base  : residual on the full tuned baseline (PD + feed-forward + balance)

Rollouts run in 4 worker processes with a numpy copy of the policy; the PPO update runs in MLX on the
CPU device (fp32), so rollout and update log-probs match (no TF32 ratio distortion).

  ../bin/gpu_run ../.venv/bin/python ppo.py --residual pd --minutes 40 --out checkpoints/ppo_pd
  ../bin/gpu_run ../.venv/bin/python ppo.py --residual pd --minutes 40 --out checkpoints/ppo_pd --resume
"""
import argparse, json, os, pickle, signal, time
import numpy as np
import multiprocessing as mp

HID = 256


# ---------------------------------------------------------------- numpy policy (workers + eval)
def np_forward(W, x):
    h = np.tanh(x @ W["p0w"] + W["p0b"])
    h = np.tanh(h @ W["p1w"] + W["p1b"])
    mean = np.tanh(h @ W["p2w"] + W["p2b"])
    return mean


def normalize(o, stats):
    return np.clip((o - stats["mean"]) / np.sqrt(stats["var"] + 1e-8), -10, 10).astype(np.float32)


class PolicyController:
    """Deterministic (mean-action) controller with the same interface as BaselineController."""

    def __init__(self, ck, base_ctrl):
        self.W, self.stats, self.residual = ck["weights"], ck["obs_stats"], ck["config"]["residual"]
        self.scale = ck["config"]["act_scale"]
        self.base = base_ctrl

    def gains(self):
        return self.base.gains()

    def __call__(self, env, obs=None):
        from dance_env import DT
        o = env.obs() if obs is None else obs
        a = np_forward(self.W, normalize(o[None], self.stats))[0]
        return action_to_ctrl(env, a, self.residual, self.base, self.scale)


def action_to_ctrl(env, a, residual, base, scale):
    from dance_env import DT
    if residual == "base":
        target, u = base(env)
    else:
        target, u = env.ref(env.t + DT)["qpos"][3:].copy(), np.zeros(6)
    return target + scale * np.clip(a, -1, 1), u


# ---------------------------------------------------------------- rollout worker
def worker(conn, wid, n_env, seed, base_params, max_steps, scale, push):
    import mujoco
    from dance_env import DanceEnv, BaselineController
    base = BaselineController(base_params)
    envs = [DanceEnv(seed=seed * 1000 + wid * 100 + i, max_steps=max_steps) for i in range(n_env)]
    push_rng = np.random.default_rng(seed * 13 + wid)
    for e in envs:
        e.set_gains(*base.gains())
    obs = np.stack([e.reset() for e in envs])
    ep_ret = np.zeros(n_env)
    ep_len = np.zeros(n_env, int)
    rng = np.random.default_rng(seed * 7 + wid)
    while True:
        cmd = conn.recv()
        if cmd[0] == "close":
            break
        _, W, stats, logstd, T, residual = cmd
        std = np.exp(logstd)
        B = dict(obs=np.zeros((T, n_env, obs.shape[1]), np.float32), act=np.zeros((T, n_env, 6), np.float32),
                 logp=np.zeros((T, n_env), np.float32), rew=np.zeros((T, n_env), np.float32),
                 done=np.zeros((T, n_env), np.float32), trunc=np.zeros((T, n_env), np.float32),
                 trunc_obs=np.zeros((T, n_env, obs.shape[1]), np.float32))
        finished = []
        for t in range(T):
            on = normalize(obs, stats)
            mean = np_forward(W, on)
            a = (mean + std * rng.normal(size=mean.shape)).astype(np.float32)
            logp = (-0.5 * (((a - mean) / std) ** 2) - logstd - 0.5 * np.log(2 * np.pi)).sum(1)
            B["obs"][t], B["act"][t], B["logp"][t] = obs, a, logp
            for i, e in enumerate(envs):
                mujoco.mj_subtreeVel(e.m, e.d)
                target, u = action_to_ctrl(e, a[i], residual, base, scale)
                o2, r, d, tr, _ = e.step(target, u)
                B["rew"][t, i] = r
                ep_ret[i] += r
                ep_len[i] += 1
                if d or tr:
                    B["done"][t, i] = float(d)
                    B["trunc"][t, i] = float(tr and not d)
                    if tr and not d:
                        B["trunc_obs"][t, i] = o2
                    finished.append((ep_ret[i], ep_len[i], bool(d)))
                    ep_ret[i], ep_len[i] = 0, 0
                    e.push = push_rng.uniform(0, push) if push > 0 else 0.0   # per-episode push size
                    o2 = e.reset()
                obs[i] = o2
        B["last_obs"] = obs.copy()
        B["finished"] = finished
        conn.send(B)


# ---------------------------------------------------------------- MLX model
def build_model(obs_dim, init_logstd=-1.0):
    import mlx.core as mx
    import mlx.nn as nn

    class AC(nn.Module):
        def __init__(self):
            super().__init__()
            self.p0, self.p1, self.p2 = nn.Linear(obs_dim, HID), nn.Linear(HID, HID), nn.Linear(HID, 6)
            self.v0, self.v1, self.v2 = nn.Linear(obs_dim, HID), nn.Linear(HID, HID), nn.Linear(HID, 1)
            self.logstd = mx.full((6,), init_logstd)
            self.p2.weight = self.p2.weight * 0.01
            self.p2.bias = mx.zeros((6,))

        def pi(self, x):
            h = mx.tanh(self.p0(x)); h = mx.tanh(self.p1(h))
            return mx.tanh(self.p2(h))

        def v(self, x):
            h = mx.tanh(self.v0(x)); h = mx.tanh(self.v1(h))
            return self.v2(h)[:, 0]
    return AC()


def np_weights(model):
    import numpy as np
    W = {}
    for name in ["p0", "p1", "p2"]:
        layer = getattr(model, name)
        W[name + "w"] = np.array(layer.weight).T.copy()
        W[name + "b"] = np.array(layer.bias).copy()
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual", default="pd", choices=["pd", "base"])
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--max_updates", type=int, default=10 ** 9)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--envs_per_worker", type=int, default=8)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--target_kl", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--act_scale", type=float, default=0.1)
    ap.add_argument("--init_logstd", type=float, default=-1.0)
    ap.add_argument("--push", type=float, default=0.0, help="training pushes: per-episode size U(0, push) N")
    ap.add_argument("--baseline", default="results/baseline_tuned.json")
    ap.add_argument("--out", default="checkpoints/ppo_pd")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init_from", default=None, help="start from another checkpoint's weights/optimizer/obs stats")
    ap.add_argument("--save_every", type=int, default=10)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    from dance_env import OBS_DIM
    mx.set_default_device(mx.cpu)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    base_params = json.load(open(args.baseline))["params"] if os.path.exists(args.baseline) else {}
    model = build_model(OBS_DIM, args.init_logstd)
    opt = optim.Adam(learning_rate=args.lr)
    stats = dict(mean=np.zeros(OBS_DIM), var=np.ones(OBS_DIM), count=1e-4)
    history, update, env_steps, start_minutes = [], 0, 0, 0.0
    ck_path = args.out + ".pkl"
    src = ck_path if (args.resume and os.path.exists(ck_path)) else args.init_from
    if src:
        ck = pickle.load(open(src, "rb"))
        model.update(tree_unflatten([(k, mx.array(v)) for k, v in ck["model"].items()]))
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten([(k, mx.array(v)) for k, v in ck["opt"].items()])
        stats, history, update, env_steps = ck["obs_stats"], ck["history"], ck["update"], ck["env_steps"]
        old_scale = ck["config"]["act_scale"]
        if not args.resume and abs(old_scale - args.act_scale) > 1e-9:
            # same residual in radians with a larger action range: tanh(x) ~ x for the small residuals here,
            # so shrink the output layer and the std by old/new (approximately the same policy, more authority)
            k = old_scale / args.act_scale
            model.p2.weight = model.p2.weight * k
            model.p2.bias = model.p2.bias * k
            model.logstd = model.logstd + float(np.log(k))
            opt = optim.Adam(learning_rate=args.lr)
            opt.init(model.trainable_parameters())
            print(f"rescaled policy output from act_scale {old_scale} to {args.act_scale} (k={k:.3f}); fresh Adam", flush=True)
        rng = np.random.default_rng()
        rng.bit_generator.state = ck["rng"]
        base_params = ck["base_params"]
        print(f"resumed at update {update} ({env_steps} env steps) from {src}", flush=True)
    cfg = dict(vars(args))
    if src:
        cfg["resumed_from_config"] = ck["config"]

    def save():
        ck = dict(model={k: np.array(v) for k, v in tree_flatten(model.parameters())},
                  opt={k: np.array(v) for k, v in tree_flatten(opt.state)},
                  weights=np_weights(model), logstd=np.array(model.logstd), obs_stats=stats,
                  history=history, update=update, env_steps=env_steps, rng=rng.bit_generator.state,
                  config=cfg, base_params=base_params)
        tmp = ck_path + ".tmp"
        pickle.dump(ck, open(tmp, "wb"))
        os.replace(tmp, ck_path)

    # workers (seeded by update count so a resumed run gets fresh, reproducible streams)
    ctx = mp.get_context("spawn")
    pipes, procs = [], []
    for w in range(args.workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(b, w, args.envs_per_worker, args.seed * 100003 + update, base_params,
                                             args.max_steps, args.act_scale, args.push), daemon=True)
        p.start()
        pipes.append(a); procs.append(p)

    def loss_fn(model, obs, act, logp_old, adv, ret):
        mean = model.pi(obs)
        logstd = model.logstd
        logp = (-0.5 * ((act - mean) / mx.exp(logstd)) ** 2 - logstd - 0.5 * np.log(2 * np.pi)).sum(1)
        ratio = mx.exp(logp - logp_old)
        pg = -mx.minimum(ratio * adv, mx.clip(ratio, 1 - args.clip, 1 + args.clip) * adv).mean()
        v = model.v(obs)
        vl = ((v - ret) ** 2).mean()
        return pg + 0.5 * vl, (pg, vl, (logp_old - logp).mean(), (mx.abs(ratio - 1) > args.clip).mean())

    grad_fn = nn.value_and_grad(model, loss_fn)
    t_start = time.time()
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *a: stop.update(flag=True))
    while update < args.max_updates and (time.time() - t_start) / 60 < args.minutes and not stop["flag"]:
        t0 = time.time()
        W = np_weights(model)
        logstd = np.array(model.logstd)
        for pp in pipes:
            pp.send(("rollout", W, stats, logstd, args.T, args.residual))
        Bs = [pp.recv() for pp in pipes]
        t_roll = time.time() - t0
        cat = lambda k: np.concatenate([B[k] for B in Bs], axis=1)
        obs, act, logp, rew, done, trunc, tobs = (cat(k) for k in ["obs", "act", "logp", "rew", "done", "trunc", "trunc_obs"])
        last = np.concatenate([B["last_obs"] for B in Bs], 0)
        finished = [f for B in Bs for f in B["finished"]]
        T, N = rew.shape
        # update obs normalisation with this batch (before computing values, as the workers used old stats)
        flat = obs.reshape(-1, OBS_DIM).astype(np.float64)
        bm, bv, bc = flat.mean(0), flat.var(0), len(flat)
        delta = bm - stats["mean"]; tot = stats["count"] + bc
        new_mean = stats["mean"] + delta * bc / tot
        m2 = stats["var"] * stats["count"] + bv * bc + delta ** 2 * stats["count"] * bc / tot
        old_stats = dict(stats)
        # values with the stats the policy acted under
        nrm = lambda x: normalize(x, old_stats)
        V = np.array(model.v(mx.array(nrm(obs.reshape(-1, OBS_DIM))))).reshape(T, N)
        Vt = np.array(model.v(mx.array(nrm(tobs.reshape(-1, OBS_DIM))))).reshape(T, N)
        Vl = np.array(model.v(mx.array(nrm(last))))
        adv = np.zeros((T, N), np.float32)
        g = np.zeros(N)
        for t in reversed(range(T)):
            nv = Vl if t == T - 1 else V[t + 1]
            end = np.maximum(done[t], trunc[t])
            boot = np.where(trunc[t] > 0, Vt[t], nv) * (1 - done[t])
            dlt = rew[t] + args.gamma * boot - V[t]
            g = dlt + args.gamma * args.lam * (1 - end) * g
            adv[t] = g
        ret = adv + V
        # PPO epochs
        on = nrm(obs.reshape(-1, OBS_DIM))
        A = act.reshape(-1, 6); LP = logp.reshape(-1); AD = adv.reshape(-1); R = ret.reshape(-1)
        AD = (AD - AD.mean()) / (AD.std() + 1e-8)
        n = len(AD)
        kls, cfs, pgs, vls = [], [], [], []
        for ep in range(args.epochs):
            perm = rng.permutation(n)
            ep_kl = []
            for s in range(0, n, args.minibatch):
                idx = perm[s:s + args.minibatch]
                (l, (pg, vl, kl, cf)), grads = grad_fn(model, mx.array(on[idx]), mx.array(A[idx]), mx.array(LP[idx]),
                                                       mx.array(AD[idx]), mx.array(R[idx]))
                grads, _ = optim.clip_grad_norm(grads, 0.5)
                opt.update(model, grads)
                mx.eval(model.parameters(), opt.state)
                ep_kl.append(float(kl)); cfs.append(float(cf)); pgs.append(float(pg)); vls.append(float(vl))
            kls.append(np.mean(ep_kl))
            if np.mean(ep_kl) > args.target_kl:
                break
        model.logstd = mx.clip(model.logstd, -4.0, 0.0)
        stats = dict(mean=new_mean, var=m2 / tot, count=tot)
        update += 1
        env_steps += T * N
        rets = [f[0] for f in finished]; lens = [f[1] for f in finished]; falls = [f[2] for f in finished]
        h = dict(update=update, env_steps=env_steps, mean_rew=float(rew.mean()),
                 ep_ret=float(np.mean(rets)) if rets else None, ep_len=float(np.mean(lens)) if lens else None,
                 fall_frac=float(np.mean(falls)) if falls else None, approx_kl=float(kls[-1]),
                 clip_frac=float(np.mean(cfs)), v_loss=float(np.mean(vls)), epochs=len(kls),
                 std=float(np.exp(np.array(model.logstd)).mean()), t_roll=t_roll, t_iter=time.time() - t0,
                 wall_min=(time.time() - t_start) / 60)
        history.append(h)
        if update % 5 == 0 or update == 1:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()}), flush=True)
        if update % args.save_every == 0:
            save()
    save()
    for pp in pipes:
        pp.send(("close",))
    for p in procs:
        p.join(timeout=5)
    print(f"saved {ck_path} at update {update}, {env_steps} env steps", flush=True)


if __name__ == "__main__":
    main()
