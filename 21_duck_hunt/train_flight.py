"""PPO (MLX) for the waypoint course. Runs on MLX CPU device so rollouts and updates share
the same fp32 arithmetic (no TF32 ratio distortion). Action mean is tanh-squashed.

  ../bin/gpu_run ../.venv/bin/python train_flight.py --steps 6000000
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import flight as F

mx.set_default_device(mx.cpu)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=F.OBS_DIM, hid=128):
        super().__init__()
        self.pi = [nn.Linear(obs_dim, hid), nn.Linear(hid, hid), nn.Linear(hid, 2)]
        self.vf = [nn.Linear(obs_dim, hid), nn.Linear(hid, hid), nn.Linear(hid, 1)]
        self.log_std = mx.full((2,), -0.7)

    def mean(self, x):
        h = mx.tanh(self.pi[0](x)); h = mx.tanh(self.pi[1](h))
        return mx.tanh(self.pi[2](h))

    def value(self, x):
        h = mx.tanh(self.vf[0](x)); h = mx.tanh(self.vf[1](h))
        return self.vf[2](h)[:, 0]


def logp(mu, log_std, a):
    return (-0.5 * ((a - mu) / mx.exp(log_std)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)


def policy_fn(model):
    def f(env, obs):
        return np.array(model.mean(mx.array(obs)))
    return f


def save(model, opt, step, cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = dict(nn.utils.tree_flatten(model.parameters()))
    mx.savez(path + ".npz", **flat)
    json.dump(dict(step=step, cfg=cfg), open(path + ".json", "w"))
    if opt is not None:
        oflat = {k: v for k, v in nn.utils.tree_flatten(opt.state) if isinstance(v, mx.array)}
        mx.savez(path + "_opt.npz", **oflat)


def load(path, hid=128):
    m = ActorCritic(hid=hid)
    m.load_weights(path + ".npz")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6_000_000)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/flight_ppo")
    ap.add_argument("--log", default="results/flight_train_log.jsonl")
    ap.add_argument("--eval_every", type=int, default=10)
    args = ap.parse_args()
    cfg = vars(args)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = ActorCritic()
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=args.lr)
    env = F.FlightEnv(args.envs, seed=1000 + args.seed)
    val_courses = F.train_courses(256)[:128]   # model selection on training courses (same as PD)
    gamma, lam, clip, epochs, mb = 0.99, 0.95, 0.2, 4, 4096

    def loss_fn(model, ob, ac, old_lp, adv, ret):
        mu = model.mean(ob)
        lp = logp(mu, model.log_std, ac)
        ratio = mx.exp(lp - old_lp)
        pg = -mx.minimum(ratio * adv, mx.clip(ratio, 1 - clip, 1 + clip) * adv).mean()
        v = model.value(ob)
        vl = ((v - ret) ** 2).mean()
        ent = model.log_std.sum()
        return pg + 0.5 * vl - 0.0 * ent, (pg, vl, mx.mean(old_lp - lp), mx.mean((mx.abs(ratio - 1) > clip).astype(mx.float32)))

    lg = nn.value_and_grad(model, loss_fn)
    obs = env.obs()
    T, N = args.horizon, args.envs
    step, upd = 0, 0
    best = -1e9
    t0 = time.time()
    flog = open(args.log, "w")
    ep_ret = np.zeros(N); ep_rets = []; ep_done = []
    while step < args.steps:
        O = np.zeros((T, N, F.OBS_DIM), np.float32); A = np.zeros((T, N, 2), np.float32)
        R = np.zeros((T, N), np.float32); D = np.zeros((T, N), np.float32); V = np.zeros((T + 1, N), np.float32)
        std = np.exp(np.array(model.log_std))
        for t in range(T):
            o = mx.array(obs)
            mu, v = model.mean(o), model.value(o)
            mu = np.array(mu); V[t] = np.array(v)
            a = (mu + std * rng.normal(size=mu.shape)).astype(np.float32)
            O[t], A[t] = obs, a
            obs, r, d, info = env.step(a)
            R[t], D[t] = r, d
            ep_ret += r
            for i in np.nonzero(d)[0]:
                ep_rets.append(ep_ret[i]); ep_done.append(float(info["complete"][i])); ep_ret[i] = 0
        V[T] = np.array(model.value(mx.array(obs)))
        adv = np.zeros((T, N), np.float32); last = 0
        for t in reversed(range(T)):
            nonterm = 1 - D[t]
            delta = R[t] + gamma * V[t + 1] * nonterm - V[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        ret = adv + V[:T]
        ob = mx.array(O.reshape(-1, F.OBS_DIM)); ac = mx.array(A.reshape(-1, 2))
        old_lp = logp(model.mean(ob), model.log_std, ac)   # recomputed with the update graph
        advf = adv.reshape(-1); advf = (advf - advf.mean()) / (advf.std() + 1e-8)
        advm = mx.array(advf); retm = mx.array(ret.reshape(-1))
        n = T * N
        kls, cfs = [], []
        for ep in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, mb):
                idx = mx.array(perm[s:s + mb])
                (l, (pg, vl, kl, cf)), g = lg(model, ob[idx], ac[idx], old_lp[idx], advm[idx], retm[idx])
                opt.update(model, g)
                mx.eval(model.parameters(), opt.state)
                kls.append(kl.item()); cfs.append(cf.item())
        step += n; upd += 1
        rec = dict(step=step, upd=upd, wall=time.time() - t0,
                   ep_ret=float(np.mean(ep_rets[-200:])) if ep_rets else None,
                   train_completion=float(np.mean(ep_done[-200:])) if ep_done else None,
                   approx_kl=float(np.mean(kls)), clip_frac=float(np.mean(cfs)),
                   vloss=float(vl.item()), log_std=np.array(model.log_std).tolist())
        if upd % args.eval_every == 0 or step >= args.steps:
            r = F.run_episodes(val_courses, policy_fn(model), seed=11)
            sc = F.score(r); rec["val_score"] = sc; rec["val"] = F.summarize(r)
            if sc > best:
                best = sc
                save(model, opt, step, cfg, args.out)
            save(model, opt, step, cfg, args.out + "_last")
        flog.write(json.dumps(rec) + "\n"); flog.flush()
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "val"}, flush=True)
    print("best val score", best, "wall", time.time() - t0)


if __name__ == "__main__":
    main()
