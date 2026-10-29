"""Neuroevolution: OpenAI-style evolution strategies over the MLP weights.

Each generation samples `half` Gaussian directions (antithetic pairs, so 2*half
members), drives every member on the same K training episodes (common random
numbers: same tracks, same start points), ranks the scores and takes an Adam
step along the rank-weighted direction. Score = ES fitness from racing/rollout.py.

Model selection uses a separate *validation* set (training-distribution tracks
never trained on). The held-out test tracks are only touched by eval.py.

    ../bin/gpu_run ../.venv/bin/python train_es.py --gens 400 --tag es
    ../.venv/bin/python train_es.py --resume checkpoints/es_state.npz ...
"""
import argparse
import json
import os
import time

import numpy as np

from racing.policy import forward, forward_pop, init_theta, n_params, save_policy
from racing.rollout import score
from racing.sim import OBS_DIM, VecRace
from racing.track import make_tracks, pack

VAL_START = 5000  # validation tracks: train-split seeds 5000.. (not in the training set)


def run_pop(bank, thetas, tids, starts, grips, steps, soft_walls=False):
    M, K = thetas.shape[0], len(tids)
    env = VecRace(bank, M * K, soft_walls=soft_walls)
    obs = env.reset(np.tile(tids, M), start_frac=np.tile(starts, M), grip=np.tile(grips, M))
    for _ in range(steps):
        act = forward_pop(thetas, obs.reshape(M, K, OBS_DIM)).reshape(M * K, 2)
        obs, _, done = env.step(act)
        if done.all():
            break
    s = score(env.prog, env.crashed).reshape(M, K)
    return s, env.crashed.reshape(M, K), int(env.t.sum())


def centered_ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=400)
    ap.add_argument("--half", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--sigma", type=float, default=0.04)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.005)
    ap.add_argument("--n_train_tracks", type=int, default=256)
    ap.add_argument("--n_val", type=int, default=128)
    ap.add_argument("--grip_lo", type=float, default=1.0)
    ap.add_argument("--grip_hi", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="es")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max_minutes", type=float, default=40.0)
    ap.add_argument("--soft_walls", action="store_true",
                    help="reward-hacking demo: walls push back instead of ending the episode")
    ap.add_argument("--snapshots", action="store_true",
                    help="also keep theta at every validation step in checkpoints/<tag>_snapshots.npz "
                         "(for re-selecting the checkpoint offline, e.g. select_dr.py); draws no random numbers")
    a = ap.parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    state_path = f"checkpoints/{a.tag}_state.npz"
    best_path = f"checkpoints/{a.tag}_best.npz"
    log_path = f"logs/{a.tag}_history.jsonl"

    bank = pack(make_tracks("train", a.n_train_tracks))
    val_bank = pack(make_tracks("train", a.n_val, start=VAL_START))
    vr = np.random.default_rng(12345)
    val_starts = vr.random(a.n_val)

    rng = np.random.default_rng(a.seed)
    P = n_params()
    theta = init_theta(rng)
    theta[-1] = 0.3  # small initial throttle bias so early cars move at all
    m = np.zeros(P)
    v = np.zeros(P)
    gen0, env_steps, best_val = 0, 0, -1e9
    if a.resume:
        st = np.load(a.resume, allow_pickle=False)
        theta, m, v = st["theta"], st["m"], st["v"]
        gen0, env_steps, best_val = int(st["gen"]), int(st["env_steps"]), float(st["best_val"])
        rng.bit_generator.state = json.loads(str(st["rng"]))
        print(f"resumed from {a.resume} at gen {gen0}", flush=True)
    cfg = vars(a)
    snap_path = f"checkpoints/{a.tag}_snapshots.npz"
    snaps = dict(gen=[], env_steps=[], theta=[])
    if a.snapshots and a.resume and os.path.exists(snap_path):
        snaps = {k: list(v) for k, v in np.load(snap_path).items()}
    t0 = time.time()
    b1, b2 = 0.9, 0.999
    for gen in range(gen0, a.gens):
        eps = rng.standard_normal((a.half, P)).astype(np.float32)
        thetas = np.concatenate([theta + a.sigma * eps, theta - a.sigma * eps, theta[None]], 0)
        tids = rng.integers(0, a.n_train_tracks, a.episodes)
        starts = rng.random(a.episodes)
        grips = rng.uniform(a.grip_lo, a.grip_hi, a.episodes)
        s, crashed, n = run_pop(bank, thetas, tids, starts, grips, a.steps, a.soft_walls)
        env_steps += n
        f = s.mean(1)
        ranks = centered_ranks(f[:-1])
        w = ranks[: a.half] - ranks[a.half: 2 * a.half]
        g = -(w @ eps) / (2 * a.half * a.sigma) + a.wd * theta  # gradient of loss
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        t = gen + 1
        theta = (theta - a.lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + 1e-8)).astype(np.float32)
        rec = dict(gen=gen, env_steps=env_steps, pop_mean=float(f[:-1].mean()), pop_max=float(f[:-1].max()),
                   center=float(f[-1]), center_crash=float(crashed[-1].mean()), minutes=(time.time() - t0) / 60)
        if gen % 10 == 9 or gen == a.gens - 1:
            vs, vc, _ = run_pop(val_bank, theta[None], np.arange(a.n_val), val_starts,
                                np.ones(a.n_val), a.steps, a.soft_walls)
            rec.update(val=float(vs.mean()), val_crash=float(vc.mean()))
            if rec["val"] > best_val:
                best_val = rec["val"]
                save_policy(best_path, theta, dict(method="es", gen=gen, env_steps=env_steps, val=best_val, cfg=cfg))
            if a.snapshots:
                snaps["gen"].append(gen); snaps["env_steps"].append(env_steps); snaps["theta"].append(theta.copy())
                np.savez(snap_path, **{k: np.array(v) for k, v in snaps.items()})
            np.savez(state_path, theta=theta, m=m, v=v, gen=gen + 1, env_steps=env_steps, best_val=best_val,
                     rng=json.dumps(rng.bit_generator.state), cfg=json.dumps(cfg))
            print(json.dumps(rec), flush=True)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if (time.time() - t0) / 60 > a.max_minutes:
            print("time budget reached", flush=True)
            break
    save_policy(f"checkpoints/{a.tag}_final.npz", theta, dict(method="es", gen=gen, env_steps=env_steps, cfg=cfg))


if __name__ == "__main__":
    main()
