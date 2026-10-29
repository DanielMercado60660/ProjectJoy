"""PPO with one shared signal policy for all four intersections (MLX).

Reward per intersection, per 10-s decision (local, with a little of the neighbours mixed in):
    r_i = -(mean vehicles waiting at its stop lines)/10 - BETA * mean sum_lanes (front-vehicle wait / 60 s)^2
The squared front-wait term is what makes starving the quiet street expensive: one car waiting
4 minutes costs as much as 16 cars waiting 1 minute.

Usage: ../.venv/bin/python train.py --minutes 15 --out checkpoints/ppo
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from sim import Grid, demand, T_STEPS, K, N_INT, run_episode
from policy import ActorCritic, save_checkpoint, greedy_controller

N_DEC = T_STEPS // K


def make_rates(B, rng):
    return np.stack([demand("train", rng) for _ in range(B)])


def validate(model, n=16, seed=5000):
    """Validation on training-distribution demand with fixed seeds (never the held-out test seeds)."""
    rng = np.random.default_rng(seed)
    env = run_episode(greedy_controller(model), make_rates(n, rng), seed + 1)
    m = env.metrics()
    return dict(mean_wait=float(m["mean_wait"].mean()), max_wait=float(m["max_wait"].mean()),
                quiet_max_wait=float(m["quiet_max_wait"].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--max-iters", type=int, default=100000)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--mix", type=float, default=0.25)
    ap.add_argument("--rscale", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/ppo")
    ap.add_argument("--log", default="runs/train_log.jsonl")
    a = ap.parse_args()

    np_rng = np.random.default_rng(a.seed)
    mx.random.seed(a.seed)
    cfg = dict(vars(a), obs_dim=Grid.OBS_DIM)
    model = ActorCritic(Grid.OBS_DIM, a.hidden)
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=a.lr)

    def loss_fn(m, o, act, old_lp, adv, ret):
        logits, v = m(o)
        lp_all = logits - mx.logsumexp(logits, -1, keepdims=True)
        lp = mx.take_along_axis(lp_all, act[:, None], -1)[:, 0]
        ratio = mx.exp(lp - old_lp)
        pg = -mx.mean(mx.minimum(ratio * adv, mx.clip(ratio, 1 - a.clip, 1 + a.clip) * adv))
        vl = mx.mean((v - ret) ** 2)
        ent = -mx.mean(mx.sum(mx.exp(lp_all) * lp_all, -1))
        return pg + 0.5 * vl - a.ent * ent, (pg, vl, ent)

    vg = nn.value_and_grad(model, loss_fn)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    logf = open(a.log, "w")
    t_start = time.time()
    best = None
    B, D = a.envs, Grid.OBS_DIM
    it = 0
    while it < a.max_iters and (time.time() - t_start) < a.minutes * 60:
        it += 1
        rates = make_rates(B, np_rng)
        env = Grid(B).reset(rates, int(np_rng.integers(1 << 30)))
        O = np.zeros((N_DEC, B, N_INT, D), np.float32)
        A = np.zeros((N_DEC, B, N_INT), np.int64)
        LP = np.zeros((N_DEC, B, N_INT), np.float32)
        V = np.zeros((N_DEC + 1, B, N_INT), np.float32)
        Rw = np.zeros((N_DEC, B, N_INT), np.float32)
        for s in range(N_DEC):
            o = env.obs()
            logits, v = model(mx.array(o))
            lp_all = np.array(logits - mx.logsumexp(logits, -1, keepdims=True))
            p = np.exp(lp_all)
            act = (np_rng.random((B, N_INT)) < p[..., 1]).astype(np.int64)
            O[s], A[s], V[s] = o, act, np.array(v)
            LP[s] = np.take_along_axis(lp_all, act[..., None], -1)[..., 0]
            q, sq = env.step_decision(act)
            r_loc = a.rscale * (-(q / K) / 10.0 - a.beta * (sq / K))
            Rw[s] = (1 - a.mix) * r_loc + a.mix * r_loc.mean(1, keepdims=True)
        _, v_last = model(mx.array(env.obs()))
        V[N_DEC] = np.array(v_last)          # episode is truncated at 1 h: bootstrap
        # GAE
        adv = np.zeros_like(Rw)
        last = 0.0
        for s in reversed(range(N_DEC)):
            delta = Rw[s] + a.gamma * V[s + 1] - V[s]
            last = delta + a.gamma * a.lam * last
            adv[s] = last
        ret = adv + V[:N_DEC]
        n = N_DEC * B * N_INT
        o_f = O.reshape(n, D); a_f = A.reshape(n); lp_f = LP.reshape(n)
        adv_f = adv.reshape(n); ret_f = ret.reshape(n)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        stats = []
        for _ in range(a.epochs):
            perm = np_rng.permutation(n)
            for k in range(0, n, a.minibatch):
                idx = perm[k:k + a.minibatch]
                (l, aux), g = vg(model, mx.array(o_f[idx]), mx.array(a_f[idx]), mx.array(lp_f[idx]),
                                 mx.array(adv_f[idx]), mx.array(ret_f[idx]))
                opt.update(model, g)
                mx.eval(model.parameters(), opt.state)
                stats.append([float(x) for x in aux])
        m = env.metrics()
        rec = dict(it=it, t=round(time.time() - t_start, 1), ep_reward=float(Rw.sum(0).mean()),
                   train_mean_wait=float(m["mean_wait"].mean()), train_max_wait=float(m["max_wait"].mean()),
                   train_quiet_max=float(m["quiet_max_wait"].mean()),
                   switches=float(m["switches_per_int"].mean()),
                   pg=float(np.mean([s[0] for s in stats])), vloss=float(np.mean([s[1] for s in stats])),
                   entropy=float(np.mean([s[2] for s in stats])))
        if it % a.val_every == 0 or it == 1:
            val = validate(model)
            score = val["mean_wait"] + 0.1 * val["max_wait"]
            rec.update({f"val_{k}": v for k, v in val.items()}, val_score=score)
            if best is None or score < best:
                best = score
                save_checkpoint(a.out + "_best", model, opt, cfg, it, dict(val=val, val_score=score))
        save_checkpoint(a.out + "_last", model, opt, cfg, it)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        print(json.dumps(rec), flush=True)
    print(f"done: {it} iterations in {time.time() - t_start:.0f}s, best val score {best}")


if __name__ == "__main__":
    main()
