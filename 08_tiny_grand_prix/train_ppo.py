"""PPO in MLX with vectorized numba cars.

Actor: the same 13-32-32-2 tanh MLP as ES (its output is the Gaussian mean, with a
learned state-independent log-std); critic: separate 13-64-64-1 MLP.
Rollouts run in numpy (the actor weights are frozen during a rollout, so a numpy
forward pass is cheaper than a GPU round trip); the update runs in MLX.

Reward per step = signed progress (m) / 10; touching a wall ends the episode with
-10 (= the ES fitness's -100 m). Episodes last at most 600 steps (60 s) and are
bootstrapped at the time limit. Every env starts at rest at a random point of a
random training track and is re-drawn on reset.

    ../bin/gpu_run ../.venv/bin/python train_ppo.py --minutes 40 --tag ppo
    ../.venv/bin/python train_ppo.py --resume checkpoints/ppo_state.npz ...
"""
import argparse
import json
import os
import time

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from racing.policy import forward, shapes, save_policy
from racing.rollout import episodes, score
from racing.sim import ACT_DIM, OBS_DIM, VecRace
from racing.track import make_tracks, pack

VAL_START = 5000
REWARD_SCALE = 0.1
CRASH_R = -10.0


def init_params(rng):
    p = {}
    for name, dims in (("a", (OBS_DIM, 32, 32, ACT_DIM)), ("c", (OBS_DIM, 64, 64, 1))):
        for i, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:])):
            gain = 0.01 if (name == "a" and i == 2) else 1.0
            p[f"{name}W{i}"] = mx.array((rng.standard_normal((n_in, n_out)) * gain / np.sqrt(n_in)).astype(np.float32))
            p[f"{name}b{i}"] = mx.zeros((n_out,))
    p["log_std"] = mx.full((ACT_DIM,), -0.5)
    p["ab2"] = mx.array(np.array([0.0, 0.3], np.float32))  # small initial throttle, like ES
    return p


def actor_mean(p, x):
    h = mx.tanh(x @ p["aW0"] + p["ab0"])
    h = mx.tanh(h @ p["aW1"] + p["ab1"])
    return mx.tanh(h @ p["aW2"] + p["ab2"])


def critic(p, x):
    h = mx.tanh(x @ p["cW0"] + p["cb0"])
    h = mx.tanh(h @ p["cW1"] + p["cb1"])
    return (h @ p["cW2"] + p["cb2"])[:, 0]


def actor_theta(p):
    """Flatten actor into the shared policy format (racing/policy.py)."""
    names = ["aW0", "ab0", "aW1", "ab1", "aW2", "ab2"]
    flat = np.concatenate([np.array(p[n]).ravel() for n in names]).astype(np.float32)
    assert flat.size == sum(int(np.prod(s)) for s in shapes())
    return flat


def gauss_logp(a, mean, log_std):
    return (-0.5 * ((a - mean) / mx.exp(log_std)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)


def loss_fn(p, obs, act, old_logp, adv, ret, clip, vcoef, ecoef):
    mean = actor_mean(p, obs)
    logp = gauss_logp(act, mean, p["log_std"])
    ratio = mx.exp(logp - old_logp)
    pg = -mx.minimum(ratio * adv, mx.clip(ratio, 1 - clip, 1 + clip) * adv).mean()
    v = critic(p, obs)
    vl = ((v - ret) ** 2).mean()
    ent = (p["log_std"] + 0.5 * np.log(2 * np.pi * np.e)).sum()
    return pg + vcoef * vl - ecoef * ent, (pg, vl, ent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--max_updates", type=int, default=100000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vcoef", type=float, default=0.5)
    ap.add_argument("--ecoef", type=float, default=0.0)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--n_train_tracks", type=int, default=256)
    ap.add_argument("--n_val", type=int, default=128)
    ap.add_argument("--grip_lo", type=float, default=1.0)
    ap.add_argument("--grip_hi", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="ppo")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--snapshots", action="store_true",
                    help="also keep the actor weights at every validation step in checkpoints/<tag>_snapshots.npz "
                         "(for re-selecting the checkpoint offline, e.g. select_dr.py); draws no random numbers")
    a = ap.parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    mx.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    bank = pack(make_tracks("train", a.n_train_tracks))
    val_bank = pack(make_tracks("train", a.n_val, start=VAL_START))
    val_starts = np.random.default_rng(12345).random(a.n_val)

    p = init_params(rng)
    opt = optim.Adam(learning_rate=a.lr)
    opt.init(p)
    update, env_steps, best_val = 0, 0, -1e9
    if a.resume:
        st = dict(np.load(a.resume, allow_pickle=False))
        p = {k[2:]: mx.array(v) for k, v in st.items() if k.startswith("p/")}
        ost = {k[2:]: mx.array(v) for k, v in st.items() if k.startswith("o/")}
        opt.state = tree_unflatten(list(ost.items()))
        update, env_steps, best_val = int(st["update"]), int(st["env_steps"]), float(st["best_val"])
        rng.bit_generator.state = json.loads(str(st["rng"]))
        print(f"resumed from {a.resume} at update {update}", flush=True)

    E, T = a.envs, a.horizon
    env = VecRace(bank, E)

    def draw(mask):
        n = int(mask.sum())
        env.reset(rng.integers(0, a.n_train_tracks, E), start_frac=rng.random(E),
                  grip=rng.uniform(a.grip_lo, a.grip_hi, E), mask=mask)
        _ = n

    draw(np.ones(E, bool))
    obs = env.observe()
    vg = mx.value_and_grad(loss_fn)
    cfg = vars(a)
    ep_ret = np.zeros(E)
    ep_len = np.zeros(E, int)
    finished_returns, finished_crash = [], []
    snap_path = f"checkpoints/{a.tag}_snapshots.npz"
    snaps = dict(update=[], env_steps=[], theta=[])
    if a.snapshots and a.resume and os.path.exists(snap_path):
        snaps = {k: list(v) for k, v in np.load(snap_path).items()}
    t0 = time.time()
    log = open(f"logs/{a.tag}_history.jsonl", "a")
    while update < a.max_updates and (time.time() - t0) / 60 < a.minutes:
        theta = actor_theta(p)
        log_std = np.array(p["log_std"])
        std = np.exp(log_std)
        B_obs = np.zeros((T, E, OBS_DIM), np.float32)
        B_act = np.zeros((T, E, ACT_DIM), np.float32)
        B_logp = np.zeros((T, E), np.float32)
        B_rew = np.zeros((T, E), np.float32)
        B_done = np.zeros((T, E), np.float32)
        trunc_obs, trunc_idx = [], []
        for t in range(T):
            mean = forward(theta, obs)
            act = mean + std * rng.standard_normal((E, ACT_DIM)).astype(np.float32)
            B_obs[t] = obs
            B_act[t] = act
            B_logp[t] = (-0.5 * ((act - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
            obs, dprog, done = env.step(act)
            r = dprog * REWARD_SCALE + CRASH_R * env.crashed
            ep_ret += r
            ep_len += 1
            trunc = (~done) & (env.t >= a.max_steps)
            if trunc.any():  # bootstrap value of the final state at the time limit
                trunc_obs.append(obs[trunc].copy())
                trunc_idx.append((t, np.where(trunc)[0]))
            end = done | trunc
            B_rew[t] = r
            B_done[t] = end
            if end.any():
                finished_returns += list(ep_ret[end])
                finished_crash += list(env.crashed[end])
                ep_ret[end] = 0
                ep_len[end] = 0
                draw(end)
                obs = env.observe()
        env_steps += E * T
        if trunc_obs:
            tv = np.array(critic(p, mx.array(np.concatenate(trunc_obs))))
            o = 0
            for (t, idx) in trunc_idx:
                B_rew[t, idx] += a.gamma * tv[o:o + len(idx)]
                o += len(idx)
        vals = np.array(critic(p, mx.array(B_obs.reshape(-1, OBS_DIM)))).reshape(T, E)
        last_v = np.array(critic(p, mx.array(obs.astype(np.float32))))
        adv = np.zeros((T, E), np.float32)
        gae = np.zeros(E, np.float32)
        for t in range(T - 1, -1, -1):
            nv = last_v if t == T - 1 else vals[t + 1]
            nonterm = 1.0 - B_done[t]
            delta = B_rew[t] + a.gamma * nv * nonterm - vals[t]
            gae = delta + a.gamma * a.lam * nonterm * gae
            adv[t] = gae
        ret = adv + vals
        N = T * E
        fo, fa, fl = B_obs.reshape(N, -1), B_act.reshape(N, -1), B_logp.reshape(N)
        fadv, fret = adv.reshape(N), ret.reshape(N)
        for _ in range(a.epochs):
            perm = rng.permutation(N)
            for s in range(0, N, a.minibatch):
                ix = perm[s:s + a.minibatch]
                ad = fadv[ix]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                (l, aux), g = vg(p, mx.array(fo[ix]), mx.array(fa[ix]), mx.array(fl[ix]), mx.array(ad),
                                 mx.array(fret[ix]), a.clip, a.vcoef, a.ecoef)
                gn = mx.sqrt(sum((v * v).sum() for _, v in tree_flatten(g)))
                scale = mx.minimum(1.0, 0.5 / (gn + 1e-6))
                g = {k: v * scale for k, v in g.items()}
                opt.update(p, g)
                mx.eval(p, opt.state)
        update += 1
        rec = dict(update=update, env_steps=env_steps, minutes=(time.time() - t0) / 60,
                   ep_return=float(np.mean(finished_returns[-200:])) if finished_returns else None,
                   ep_crash=float(np.mean(finished_crash[-200:])) if finished_crash else None,
                   pg=float(aux[0]), vloss=float(aux[1]), std=[float(x) for x in np.exp(np.array(p["log_std"]))])
        if update % 10 == 0:
            th = actor_theta(p)
            prog, cr = episodes(lambda o: forward(th, o), val_bank, np.arange(a.n_val), val_starts, a.max_steps)
            rec.update(val=float(score(prog, cr).mean()), val_crash=float(cr.mean()))
            if rec["val"] > best_val:
                best_val = rec["val"]
                save_policy(f"checkpoints/{a.tag}_best.npz", th,
                            dict(method="ppo", update=update, env_steps=env_steps, val=best_val, cfg=cfg))
            if a.snapshots:
                snaps["update"].append(update); snaps["env_steps"].append(env_steps); snaps["theta"].append(th)
                np.savez(snap_path, **{k: np.array(v) for k, v in snaps.items()})
            save = {f"p/{k}": np.array(v) for k, v in p.items()}
            save.update({f"o/{k}": np.array(v) for k, v in tree_flatten(opt.state)})
            np.savez(f"checkpoints/{a.tag}_state.npz", update=update, env_steps=env_steps, best_val=best_val,
                     rng=json.dumps(rng.bit_generator.state), cfg=json.dumps(cfg), **save)
            print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
    save_policy(f"checkpoints/{a.tag}_final.npz", actor_theta(p),
                dict(method="ppo", update=update, env_steps=env_steps, cfg=cfg))


if __name__ == "__main__":
    main()
