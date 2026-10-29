"""PPO (MLX) for the learner robot vs the scripted goalie.

    ../.venv/bin/python ppo.py --minutes 18 --out checkpoints/ppo.npz

Stops on a wall-clock budget. Curriculum: ball-start spread ramps linearly from "near the opponent
goal" to "anywhere" over the first half of the budget (20% of resets are always "anywhere").
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from env import Arena, OBS_DIM, OUT_GOAL, OUT_CONCEDED, OUT_OWN_GOAL, OUT_TIMEOUT
import scripted


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=64):
        super().__init__()
        self.l1 = nn.Linear(d_in, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, d_out)

    def __call__(self, x):
        return self.l3(mx.tanh(self.l2(mx.tanh(self.l1(x)))))


class ActorCritic(nn.Module):
    def __init__(self, hidden=64, tanh_mean=False):
        super().__init__()
        self.pi = MLP(OBS_DIM, 2, hidden)
        self.v = MLP(OBS_DIM, 1, hidden)
        self.log_std = mx.zeros((2,)) - 0.5
        self.tanh_mean = tanh_mean   # run 4: keep the Gaussian mean inside the [-1, 1] action box

    def __call__(self, x):
        mu = self.pi(x)
        if self.tanh_mean:
            mu = mx.tanh(mu)
        return mu, self.v(x)[:, 0]


def logprob(mean, log_std, a):
    return mx.sum(-0.5 * ((a - mean) / mx.exp(log_std)) ** 2 - log_std - 0.5 * np.log(2 * np.pi), axis=-1)


class RunningNorm:
    def __init__(self, d):
        self.mean = np.zeros(d)
        self.var = np.ones(d)
        self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)


def save_ckpt(path, model, norm, cfg, extra):
    flat = dict(tree_flatten(model.parameters()))
    flat["_norm_mean"] = mx.array(norm.mean.astype(np.float32))
    flat["_norm_var"] = mx.array(norm.var.astype(np.float32))
    tmp = path + ".tmp.npz"
    mx.savez(tmp, **flat)
    os.replace(tmp, path)
    with open(path.replace(".npz", ".json"), "w") as f:
        json.dump(dict(cfg=cfg, **extra), f, indent=1)


def load_ckpt(path):
    flat = dict(mx.load(path))
    cfg = json.load(open(path.replace(".npz", ".json")))["cfg"]
    norm = RunningNorm(OBS_DIM)
    norm.mean = np.array(flat.pop("_norm_mean"))
    norm.var = np.array(flat.pop("_norm_var"))
    model = ActorCritic(cfg["hidden"], bool(cfg.get("tanh_mean", 0)))
    model.update(tree_unflatten(list(flat.items())))
    mx.eval(model.parameters())
    return model, norm, cfg


def make_policy(model, norm):
    """Deterministic policy for eval/demo: mean action."""
    def act(arena):
        mean, _ = model(mx.array(norm(arena.obs())))
        return np.array(mean)
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=18.0)
    ap.add_argument("--max-updates", type=int, default=10 ** 9)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu-rollout", type=int, default=1)
    ap.add_argument("--lr-anneal", type=int, default=0, help="1: linear lr decay to 0 over the budget")
    ap.add_argument("--keep-best", type=int, default=0,
                    help="1: also save <out>_best.npz at the best training goal rate once curriculum=1")
    ap.add_argument("--tanh-mean", type=int, default=0, help="1: squash the action mean with tanh")
    ap.add_argument("--relp", type=int, default=0,
                    help="1: recompute old log-probs on the training device (CPU fp32 vs GPU TF32 mismatch)")
    ap.add_argument("--touch-bonus", default="first", choices=["first", "every"])
    ap.add_argument("--out", default="checkpoints/ppo.npz")
    ap.add_argument("--log", default="logs/train.jsonl")
    args = ap.parse_args()
    cfg = vars(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    env = Arena(args.envs, seed=args.seed, opponent="goalie")
    env.curriculum = 0.0
    env.touch_bonus = args.touch_bonus
    env.reset(np.ones(args.envs, dtype=bool))
    model = ActorCritic(args.hidden, bool(args.tanh_mean))
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=args.lr)
    norm = RunningNorm(OBS_DIM)

    def loss_fn(m, obs, act, old_lp, adv, ret):
        mean, v = m(obs)
        lp = logprob(mean, m.log_std, act)
        ratio = mx.exp(lp - old_lp)
        pg = -mx.mean(mx.minimum(ratio * adv, mx.clip(ratio, 1 - args.clip, 1 + args.clip) * adv))
        vl = mx.mean((v - ret) ** 2)
        ent = mx.sum(m.log_std) + 0.5 * 2 * (1 + np.log(2 * np.pi))
        return pg + 0.5 * vl - args.ent * ent, (pg, vl, mx.mean((ratio - 1) - mx.log(ratio)))

    grad_fn = nn.value_and_grad(model, loss_fn)

    N, T = args.envs, args.horizon
    log_f = open(args.log, "w")
    t_start = time.time()
    budget = args.minutes * 60
    recent = []      # outcomes of recent episodes
    total_steps = 0
    best_goal = -1.0
    upd = 0
    rng = np.random.default_rng(args.seed + 1)
    while upd < args.max_updates and time.time() - t_start < budget:
        frac = (time.time() - t_start) / budget
        env.curriculum = float(min(1.0, frac / 0.5))
        if args.lr_anneal:
            opt.learning_rate = args.lr * max(0.0, 1.0 - frac)
        obs_b = np.zeros((T, N, OBS_DIM), np.float32)
        act_b = np.zeros((T, N, 2), np.float32)
        lp_b = np.zeros((T, N), np.float32)
        val_b = np.zeros((T, N), np.float32)
        rew_b = np.zeros((T, N), np.float32)
        done_b = np.zeros((T, N), np.float32)
        t_roll = time.time()
        dev = mx.cpu if args.cpu_rollout else mx.gpu
        for t in range(T):
            raw = env.obs()
            norm.update(raw)
            o = norm(raw)
            with mx.stream(dev):
                mean, v = model(mx.array(o))
                a = mean + mx.exp(model.log_std) * mx.random.normal(mean.shape)
                lp = logprob(mean, model.log_std, a)
                mx.eval(a, v, lp)
            a = np.array(a)
            r, d, tr, outc, info = env.step(a, scripted.goalie(env))
            if tr.any():   # bootstrap through timeouts (truncation, not termination)
                with mx.stream(dev):
                    vb = np.array(model.v(mx.array(norm(env.obs()[tr])))[:, 0])
                r = r.copy()
                r[tr] += args.gamma * vb
            obs_b[t], act_b[t], lp_b[t], val_b[t], rew_b[t] = o, a, np.array(lp), np.array(v), r
            done_b[t] = (d | tr)
            if info is not None:
                recent.extend(info["outcome"].tolist())
                env.reset(d | tr)
        t_roll = time.time() - t_roll
        with mx.stream(dev):
            last_v = np.array(model(mx.array(norm(env.obs())))[1])
        # GAE
        adv = np.zeros((T, N), np.float32)
        g = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val_b[t + 1]
            nonterm = 1.0 - done_b[t]
            delta = rew_b[t] + args.gamma * nv * nonterm - val_b[t]
            g = delta + args.gamma * args.lam * nonterm * g
            adv[t] = g
        ret = adv + val_b
        B = T * N
        fo, fa, flp = obs_b.reshape(B, -1), act_b.reshape(B, -1), lp_b.reshape(B)
        fadv, fret = adv.reshape(B), ret.reshape(B)
        if args.relp:
            lps = []
            for s in range(0, B, 8192):
                mu_, _ = model(mx.array(fo[s:s + 8192]))
                lps.append(np.array(logprob(mu_, model.log_std, mx.array(fa[s:s + 8192]))))
            flp = np.concatenate(lps).astype(np.float32)
        t_train = time.time()
        stats = []
        for ep in range(args.epochs):
            perm = rng.permutation(B)
            for s in range(0, B, args.minibatch):
                idx = perm[s:s + args.minibatch]
                ad = fadv[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                (l, aux), grads = grad_fn(model, mx.array(fo[idx]), mx.array(fa[idx]), mx.array(flp[idx]),
                                          mx.array(ad), mx.array(fret[idx]))
                grads, _ = optim.clip_grad_norm(grads, 0.5)
                opt.update(model, grads)
                mx.eval(model.parameters(), opt.state, l, *aux)
                stats.append([float(x) for x in aux])
        t_train = time.time() - t_train
        total_steps += B
        upd += 1
        recent = recent[-400:]
        ro = np.array(recent) if recent else np.zeros(1)
        rec = dict(update=upd, steps=total_steps, wall=round(time.time() - t_start, 1),
                   curriculum=round(env.curriculum, 3),
                   goal=float((ro == OUT_GOAL).mean()), conceded=float((ro == OUT_CONCEDED).mean()),
                   own_goal=float((ro == OUT_OWN_GOAL).mean()), timeout=float((ro == OUT_TIMEOUT).mean()),
                   pg=float(np.mean([s[0] for s in stats])), vloss=float(np.mean([s[1] for s in stats])),
                   kl=float(np.mean([s[2] for s in stats])), std=float(np.exp(np.array(model.log_std)).mean()),
                   mean_rew=float(rew_b.mean()), t_roll=round(t_roll, 3), t_train=round(t_train, 3))
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if upd % 10 == 0 or upd == 1:
            print(json.dumps(rec), flush=True)
        if args.keep_best and env.curriculum >= 1.0 and len(recent) >= 400 and rec["goal"] > best_goal:
            best_goal = rec["goal"]
            save_ckpt(args.out.replace(".npz", "_best.npz"), model, norm, cfg,
                      dict(update=upd, steps=total_steps, best_train_goal=best_goal, last=rec))
        if upd % 25 == 0:
            save_ckpt(args.out, model, norm, cfg, dict(update=upd, steps=total_steps, last=rec))
    save_ckpt(args.out, model, norm, cfg, dict(update=upd, steps=total_steps,
                                              wall=time.time() - t_start, last=rec))
    print("saved", args.out, "updates", upd, "steps", total_steps, flush=True)


if __name__ == "__main__":
    main()
