"""PPO + GRU navigator in MLX on vectorised numpy mazes.

Examples
  ../.venv/bin/python train_solver.py --name solver_v1 --steps 30000000
  ../.venv/bin/python train_solver.py --name solver_tiny --steps 40000 --envs 32   # smoke
  # fine-tune on a mix of family levels and designer levels:
  ../.venv/bin/python train_solver.py --name solver_cur --init checkpoints/solver_v1 \
        --extra results/designer_archive.npz --extra-prob 0.5 --steps 10000000
Resume: rerun the same command; it continues from checkpoints/<name>/ if present.
"""
import argparse, json, os, time
from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from dungeon.env import VecMaze, OBS_DIM, MAX_STEPS
from dungeon.policy import RecurrentPolicy
from dungeon.data import train_pool, test_set, pool_sampler, mixed_sampler
from dungeon.mazes import ALL_FAMILIES, TRAIN_FAMILIES
from dungeon.rollout import run_policy
from dungeon.scripted import oracle_lengths, summarize

HERE = os.path.dirname(os.path.abspath(__file__))


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="solver_v1")
    p.add_argument("--steps", type=int, default=30_000_000)
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default=None, help="checkpoint dir to start weights from")
    p.add_argument("--extra", default=None, help="npz of extra levels (e.g. designer archive)")
    p.add_argument("--extra-prob", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=100, help="updates between evals/checkpoints")
    p.add_argument("--eval-n", type=int, default=200)
    p.add_argument("--stop-after", type=int, default=0, help="debug: exit after this many updates")
    p.add_argument("--pool-n", type=int, default=20000)
    p.add_argument("--memoryless", action="store_true", help="ablation: zero the GRU state every step")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-minutes", type=float, default=0, help="save and exit after this wall time")
    p.add_argument("--novelty", type=float, default=0.01, help="episodic first-visit bonus (shaping)")
    return p.parse_args(argv)


def save_ckpt(d, model, opt, meta):
    os.makedirs(d, exist_ok=True)
    model.save_weights(os.path.join(d, "weights.safetensors"))
    mx.save_safetensors(os.path.join(d, "opt.safetensors"), dict(tree_flatten(opt.state)))
    with open(os.path.join(d, "meta.json.tmp"), "w") as f:
        json.dump(meta, f)
    os.replace(os.path.join(d, "meta.json.tmp"), os.path.join(d, "meta.json"))


def evaluate(model, n, seed=123, families=ALL_FAMILIES):
    res = {}
    for fam in families:
        L = test_set(fam)
        L = {k: v[:n] for k, v in L.items()}
        o = run_policy(model, L, greedy=False, seed=seed)
        res[fam] = summarize(o["success"], o["steps"], oracle_lengths(L))
    return res


def main(argv=None):
    a = get_args(argv)
    ckdir = os.path.join(HERE, "checkpoints", a.name)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    logf = open(os.path.join(HERE, "logs", f"{a.name}.jsonl"), "a")

    mx.random.seed(a.seed)
    model = RecurrentPolicy(a.hidden, a.memoryless)
    os.makedirs(ckdir, exist_ok=True)
    with open(os.path.join(ckdir, "policy.json"), "w") as f:
        json.dump(dict(hidden=a.hidden, memoryless=a.memoryless), f)
    opt = optim.Adam(learning_rate=a.lr)
    meta = dict(update=0, env_steps=0, config=vars(a))
    rng = np.random.default_rng(a.seed)

    resumed = os.path.exists(os.path.join(ckdir, "meta.json"))
    if resumed:
        with open(os.path.join(ckdir, "meta.json")) as f:
            meta = json.load(f)
        model.load_weights(os.path.join(ckdir, "weights.safetensors"))
        opt.init(model.trainable_parameters())
        st = mx.load(os.path.join(ckdir, "opt.safetensors"))
        opt.state = tree_unflatten(list(st.items()))
        rng.bit_generator.state = meta["np_rng"]
        mx.random.seed(meta["mx_seed"])
        print(f"resumed {a.name} at update {meta['update']} ({meta['env_steps']} steps)", flush=True)
    elif a.init:
        model.load_weights(os.path.join(a.init, "weights.safetensors"))
        print(f"initialised weights from {a.init}", flush=True)
    mx.eval(model.parameters())

    pool = train_pool(a.pool_n)
    sampler = pool_sampler(pool)
    if a.extra:
        z = np.load(a.extra)
        extra = {k: z[k] for k in ("walls", "start", "goal")}
        sampler = mixed_sampler([sampler, pool_sampler(extra)], [1 - a.extra_prob, a.extra_prob])
        print(f"mixing {len(extra['walls'])} extra levels with p={a.extra_prob}", flush=True)

    env = VecMaze(a.envs, sampler, seed=int(rng.integers(1 << 31)), novelty=a.novelty)
    N, L = a.envs, a.rollout
    obs = env.reset()
    h = model.initial_state(N)
    starts = np.ones(N, np.float32)

    def _act(obs, h, key):
        logits, value, h2 = model.step(obs, h)
        act = mx.random.categorical(logits, key=key)
        logp = mx.take_along_axis(nn.log_softmax(logits), act[:, None], axis=-1)[:, 0]
        return act, logp, value, h2

    act_fn = mx.compile(_act, inputs=model.state)  # params are inputs, not baked constants

    def loss_fn(model, obs, h0, st, act, old_logp, adv, ret):
        logits, values = model.unroll(obs, h0, st)
        logp_all = nn.log_softmax(logits)
        logp = mx.take_along_axis(logp_all, act[..., None], axis=-1)[..., 0]
        ratio = mx.exp(logp - old_logp)
        pg = -mx.minimum(ratio * adv, mx.clip(ratio, 1 - a.clip, 1 + a.clip) * adv).mean()
        vloss = ((values - ret) ** 2).mean()
        ent = -(mx.exp(logp_all) * logp_all).sum(-1).mean()
        loss = pg + a.vf * vloss - a.ent * ent
        approx_kl = ((ratio - 1) - (logp - old_logp)).mean()
        return loss, (pg, vloss, ent, approx_kl)

    lg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(obs, h0, st, act, old_logp, adv, ret):
        (loss, aux), grads = lg(model, obs, h0, st, act, old_logp, adv, ret)
        grads, gn = optim.clip_grad_norm(grads, a.max_grad)
        opt.update(model, grads)
        return loss, aux[0], aux[1], aux[2], aux[3], gn

    key = mx.random.key(int(rng.integers(1 << 31)))
    ep_succ, ep_steps, rew_hist = [], [], []
    t0 = time.time()
    total_updates = a.steps // (N * L)
    while meta["update"] < total_updates:
        B_obs = np.zeros((L, N, OBS_DIM), np.float32)
        B_st = np.zeros((L, N), np.float32)
        B_a = np.zeros((L, N), np.int64)
        B_lp = np.zeros((L, N), np.float32)
        B_v = np.zeros((L, N), np.float32)
        B_r = np.zeros((L, N), np.float32)
        B_d = np.zeros((L, N), np.float32)
        h0 = h
        for t in range(L):
            B_obs[t], B_st[t] = obs, starts
            h = h * mx.array(1.0 - starts)[:, None]
            key, sub = mx.random.split(key)
            act, logp, v, h = act_fn(mx.array(obs), h, sub)
            mx.eval(act, logp, v, h)
            act_np = np.array(act)
            obs, r, done, info = env.step(act_np)
            B_a[t], B_lp[t], B_v[t], B_r[t], B_d[t] = act_np, np.array(logp), np.array(v), r, done
            starts = done.astype(np.float32)
            if done.any():
                ep_succ.extend(info["success"][done].tolist())
                ep_steps.extend(info["steps"][done].tolist())
        h_boot = h * mx.array(1.0 - starts)[:, None]
        _, v_last, _ = model.step(mx.array(obs), h_boot)
        v_last = np.array(v_last)
        adv = np.zeros((L, N), np.float32)
        last = np.zeros(N, np.float32)
        for t in reversed(range(L)):
            nv = v_last if t == L - 1 else B_v[t + 1]
            nonterm = 1.0 - B_d[t]
            delta = B_r[t] + a.gamma * nv * nonterm - B_v[t]
            last = delta + a.gamma * a.lam * nonterm * last
            adv[t] = last
        ret = adv + B_v
        rew_hist.append(float(B_r.mean()))
        mb = N // a.minibatches
        stats = []
        h0_np = np.array(h0)
        for _ in range(a.epochs):
            perm = rng.permutation(N)
            for k in range(a.minibatches):
                ix = perm[k * mb:(k + 1) * mb]
                ad = adv[:, ix]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                out = train_step(mx.array(B_obs[:, ix]), mx.array(h0_np[ix]), mx.array(B_st[:, ix]),
                                 mx.array(B_a[:, ix]), mx.array(B_lp[:, ix]), mx.array(ad), mx.array(ret[:, ix]))
                mx.eval(state, *out)
                stats.append([float(x) for x in out])
        meta["update"] += 1
        meta["env_steps"] += N * L
        if meta["update"] % a.log_every == 0:
            s = np.mean(stats, 0)
            rec = dict(update=meta["update"], env_steps=meta["env_steps"],
                       train_success=float(np.mean(ep_succ)) if ep_succ else 0.0,
                       train_ep_len=float(np.mean(ep_steps)) if ep_steps else 0.0,
                       rew_per_step=float(np.mean(rew_hist)),
                       loss=s[0], pg=s[1], vloss=s[2], entropy=s[3], kl=s[4], grad_norm=s[5],
                       sps=meta["env_steps"] / max(1e-9, time.time() - t0) if not resumed else None,
                       wall=time.time() - t0)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            ep_succ, ep_steps, rew_hist = [], [], []
        if meta["update"] % a.eval_every == 0 or meta["update"] == total_updates:
            ev = evaluate(model, a.eval_n)
            rec = dict(update=meta["update"], env_steps=meta["env_steps"], eval=ev)
            print("EVAL", json.dumps({f: (round(v["success"], 3), round(v["spl"], 3)) for f, v in ev.items()}), flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            meta["np_rng"] = rng.bit_generator.state
            meta["mx_seed"] = int(rng.integers(1 << 31))
            save_ckpt(ckdir, model, opt, meta)
        if (a.stop_after and meta["update"] >= a.stop_after) or \
                (a.max_minutes and time.time() - t0 > 60 * a.max_minutes):
            meta["np_rng"] = rng.bit_generator.state
            meta["mx_seed"] = int(rng.integers(1 << 31))
            save_ckpt(ckdir, model, opt, meta)
            print("stop-after reached", flush=True)
            return model
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return model


if __name__ == "__main__":
    main()
