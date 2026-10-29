"""PPO + GRU training for the bat. Wall-clock capped; checkpoints every few iterations.

  ../.venv/bin/python train.py --smoke                      # < 1 min
  ../bin/gpu_run ../.venv/bin/python train.py --minutes 18  # main run
"""
import argparse
import json
import os
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from bat_env import OBS_DIM, BatEnv, MazePool
from model import BatPolicy, save

TRAIN_SEED = 1000        # training maze pool; held-out pools use other seeds (checked in tests)


def log_softmax(x):
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=18.0)
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pulse_cost", type=float, default=0.01)
    ap.add_argument("--pool", type=int, default=3000)
    ap.add_argument("--sizes", type=str, default="4,5,6")
    ap.add_argument("--ent", type=float, default=0.02)
    ap.add_argument("--collision_cost", type=float, default=0.05)
    ap.add_argument("--new_cell_bonus", type=float, default=0.1)
    ap.add_argument("--goal_reward", type=float, default=2.0)
    ap.add_argument("--out", type=str, default="checkpoints/bat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.minutes, a.envs, a.T, a.pool, a.hidden = 0.6, 32, 32, 60, 64
        a.out = "checkpoints/smoke"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.random.seed(a.seed); mx.random.seed(a.seed)
    sizes = [int(s) for s in a.sizes.split(",")]
    t0 = time.time()
    pool = MazePool(sizes, a.pool, seed=TRAIN_SEED)
    print(f"pool of {a.pool} mazes built in {time.time()-t0:.1f}s", flush=True)
    env = BatEnv(pool, a.envs, seed=a.seed, pulse_cost=a.pulse_cost, collision_cost=a.collision_cost,
                 new_cell_bonus=a.new_cell_bonus, goal_reward=a.goal_reward)
    model = BatPolicy(OBS_DIM, a.hidden)
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=a.lr)
    cfg = dict(vars(a), obs_dim=OBS_DIM, train_seed=TRAIN_SEED)

    def loss_fn(model, obs, h0, starts, mv, pu, old_lp, adv, ret):
        lm, lp, v = model.sequence(obs, h0, starts)
        lm, lp = log_softmax(lm), log_softmax(lp)
        lpm = mx.take_along_axis(lm, mv[..., None], -1)[..., 0]
        lpp = mx.take_along_axis(lp, pu[..., None], -1)[..., 0]
        logp = lpm + lpp
        ratio = mx.exp(logp - old_lp)
        pg = -mx.minimum(ratio * adv, mx.clip(ratio, 0.8, 1.2) * adv).mean()
        vl = 0.5 * ((v - ret) ** 2).mean()
        ent = -(mx.exp(lm) * lm).sum(-1).mean() - (mx.exp(lp) * lp).sum(-1).mean()
        return pg + 0.5 * vl - a.ent * ent, (pg, vl, ent)

    vg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(obs, h0, starts, mv, pu, old_lp, adv, ret):
        (loss, aux), g = vg(model, obs, h0, starts, mv, pu, old_lp, adv, ret)
        g, _ = optim.clip_grad_norm(g, 0.5)
        opt.update(model, g)
        return loss, aux[0], aux[1], aux[2]

    # NB: compile must take the model state as an input, or the weights get baked in as constants
    @partial(mx.compile, inputs=[model.state])
    def act_fn(obs, h):
        return model.step(obs, h)

    N, T = a.envs, a.T
    obs = env.reset()
    h = mx.zeros((N, a.hidden))
    start = np.ones(N, np.float32)
    log = open(a.out + "_log.jsonl", "w")
    ep_stats = []
    it = 0
    deadline = t0 + a.minutes * 60
    rng = np.random.default_rng(a.seed)
    while it < a.iters and time.time() < deadline:
        it += 1
        B_obs = np.zeros((T, N, OBS_DIM), np.float32); B_st = np.zeros((T, N), np.float32)
        B_mv = np.zeros((T, N), np.int32); B_pu = np.zeros((T, N), np.int32)
        B_lp = np.zeros((T, N), np.float32); B_v = np.zeros((T, N), np.float32)
        B_r = np.zeros((T, N), np.float32); B_d = np.zeros((T, N), np.float32)
        h0 = h
        for t in range(T):
            B_obs[t] = obs; B_st[t] = start
            hm = h * mx.array(1 - start)[:, None]
            lm, lp, v, h = act_fn(mx.array(obs), hm)
            pm = np.array(mx.softmax(lm, -1)); pp = np.array(mx.softmax(lp, -1)); vv = np.array(v)
            mv = (rng.random((N, 1)) > np.cumsum(pm, -1)).sum(-1).clip(0, 2)
            pu = (rng.random(N) < pp[:, 1]).astype(int)
            B_lp[t] = np.log(pm[np.arange(N), mv] + 1e-8) + np.log(pp[np.arange(N), pu] + 1e-8)
            B_mv[t], B_pu[t], B_v[t] = mv, pu, vv
            obs, r, d, info = env.step(mv, pu)
            B_r[t], B_d[t] = r, d
            start = d.astype(np.float32)
            for i in np.nonzero(d)[0]:
                ep_stats.append((info["success"][i], info["t"][i], info["pulses"][i], info["collisions"][i]))
        _, _, last_v, _ = act_fn(mx.array(obs), h * mx.array(1 - start)[:, None])
        last_v = np.array(last_v)
        adv = np.zeros((T, N), np.float32); gae = 0
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else B_v[t + 1]
            delta = B_r[t] + 0.99 * nv * (1 - B_d[t]) - B_v[t]
            gae = delta + 0.99 * 0.95 * (1 - B_d[t]) * gae
            adv[t] = gae
        ret = adv + B_v
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        h0n = np.array(h0)
        stats = []
        for _ in range(a.epochs):
            perm = rng.permutation(N)
            for chunk in np.array_split(perm, a.minibatches):
                out = train_step(mx.array(B_obs[:, chunk]), mx.array(h0n[chunk]), mx.array(B_st[:, chunk]),
                                 mx.array(B_mv[:, chunk]), mx.array(B_pu[:, chunk]), mx.array(B_lp[:, chunk]),
                                 mx.array(advn[:, chunk]), mx.array(ret[:, chunk]))
                mx.eval(out)
                stats.append([float(x) for x in out])
        if it % 5 == 0 or a.smoke:
            es = np.array(ep_stats[-400:]) if ep_stats else np.zeros((1, 4))
            rec = dict(it=it, steps=it * N * T, sec=round(time.time() - t0, 1),
                       loss=float(np.mean([s[0] for s in stats])), vloss=float(np.mean([s[2] for s in stats])),
                       ent=float(np.mean([s[3] for s in stats])), success=float(es[:, 0].mean()),
                       ep_len=float(es[:, 1].mean()), pulses_per_step=float((es[:, 2] / np.maximum(es[:, 1], 1)).mean()),
                       coll_per_step=float((es[:, 3] / np.maximum(es[:, 1], 1)).mean()), mean_reward=float(B_r.mean()))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
        if it % 25 == 0:
            save(model, a.out, dict(cfg, iters_done=it, steps_done=it * N * T))
    save(model, a.out, dict(cfg, iters_done=it, steps_done=it * N * T, wall_sec=time.time() - t0))
    print(f"done: {it} iters, {it*N*T} env steps, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
