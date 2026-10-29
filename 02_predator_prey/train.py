"""Competitive self-play PPO for one pursuer vs one evader, with an opponent archive.

Every iteration runs three batches of matches for one full episode length (300 ticks):
  A: current pursuer vs current evader   (both learn from it)
  B: current pursuer vs K archived evaders (pursuer learns)
  C: K archived pursuers vs current evader (evader learns)
Archived opponents are drawn uniformly from every snapshot so far (snapshot every --snap-every
iterations, starting with the random init), so old strategies stay in circulation.
Scripted opponents are never used for training: they are the held-out evaluation suite.

  ../.venv/bin/python train.py --minutes 18            # main run (wrap in ../bin/gpu_run)
  ../.venv/bin/python train.py --max-iters 3 --group 16 --out /tmp/x   # smoke
"""
import argparse, csv, json, os, signal, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import sim
from policy import new_model, get_np, set_np, act, save_pair

p = argparse.ArgumentParser()
p.add_argument("--minutes", type=float, default=18.0)
p.add_argument("--max-iters", type=int, default=100000)
p.add_argument("--group", type=int, default=64, help="envs per batch (A, B, C)")
p.add_argument("--k-arch", type=int, default=2, help="archived opponents per iteration per side")
p.add_argument("--snap-every", type=int, default=10)
p.add_argument("--eval-every", type=int, default=10)
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--mb", type=int, default=4096)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--ent", type=float, default=0.01)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--lam", type=float, default=0.95)
p.add_argument("--clip", type=float, default=0.2)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
p.add_argument("--out", default="checkpoints")
p.add_argument("--log", default="results")
args = p.parse_args()

mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
os.makedirs(args.out, exist_ok=True)
os.makedirs(args.log, exist_ok=True)
rng = np.random.default_rng(args.seed)
G, T, K = args.group, sim.T_MAX, args.k_arch
assert G % K == 0

P, E = new_model(args.seed * 2 + 1), new_model(args.seed * 2 + 2)
oppP = [new_model(100 + i) for i in range(K)]
oppE = [new_model(200 + i) for i in range(K)]
optP, optE = optim.Adam(learning_rate=args.lr), optim.Adam(learning_rate=args.lr)
archive = [(0, get_np(P), get_np(E))]
envs = {k: sim.Arena(G, seed=args.seed * 10 + i) for i, k in enumerate("ABC")}


def loss_fn(model, obs, a, lp_old, adv, ret):
    lg, v = model(obs)
    lp_all = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    lp = mx.take_along_axis(lp_all, a[:, None], axis=1)[:, 0]
    ratio = mx.exp(lp - lp_old)
    pg = -mx.mean(mx.minimum(ratio * adv, mx.clip(ratio, 1 - args.clip, 1 + args.clip) * adv))
    vf = mx.mean((v - ret) ** 2)
    ent = -mx.mean(mx.sum(mx.exp(lp_all) * lp_all, axis=-1))
    kl = mx.mean(lp_old - lp)
    return pg + 0.5 * vf - args.ent * ent, (pg, vf, ent, kl)


def make_step(model, opt):
    vg = nn.value_and_grad(model, loss_fn)

    def step(obs, a, lp_old, adv, ret):
        (loss, aux), g = vg(model, obs, a, lp_old, adv, ret)
        g, _ = optim.clip_grad_norm(g, 0.5)
        opt.update(model, g)
        return aux
    return step


stepP, stepE = make_step(P, optP), make_step(E, optE)


def gae(r, v, done, v_last):
    """r, v, done: (T, n). Returns advantages and returns."""
    adv = np.zeros_like(r)
    last = np.zeros(r.shape[1], np.float32)
    for t in range(len(r) - 1, -1, -1):
        nv = v_last if t == len(r) - 1 else v[t + 1]
        nonterm = 1.0 - done[t]
        delta = r[t] + args.gamma * nv * nonterm - v[t]
        last = delta + args.gamma * args.lam * nonterm * last
        adv[t] = last
    return adv, adv + v


def ppo_update(step, data):
    obs, a, lp, adv, ret = data
    n = len(obs)
    stats = []
    for _ in range(args.epochs):
        perm = rng.permutation(n)
        for s in range(0, n, args.mb):
            i = perm[s:s + args.mb]
            ad = adv[i]
            ad = (ad - ad.mean()) / (ad.std() + 1e-8)
            aux = step(mx.array(obs[i]), mx.array(a[i]), mx.array(lp[i]), mx.array(ad), mx.array(ret[i]))
            mx.eval(P.parameters(), E.parameters(), optP.state, optE.state)
            stats.append([float(x) for x in aux])
    return np.mean(stats, 0)


def rollout():
    """Run all three batches for T ticks. Returns per-learner data and episode stats."""
    for e in envs.values():
        e.reset(np.arange(G))
    bufP = {k: [] for k in ("obs", "a", "lp", "v", "r", "d")}
    bufE = {k: [] for k in ("obs", "a", "lp", "v", "r", "d")}
    caps = {k: 0 for k in "ABC"}
    eps = {k: 0 for k in "ABC"}
    effort = np.zeros(2)
    blk = G // K
    for t in range(T):
        oA_p, oA_e = envs["A"].obs()
        oB_p, oB_e = envs["B"].obs()
        oC_p, oC_e = envs["C"].obs()
        # learners: pursuer on A+B, evader on A+C
        ap, lpp, vp = act(P, np.concatenate([oA_p, oB_p]), rng)
        ae, lpe, ve = act(E, np.concatenate([oA_e, oC_e]), rng)
        aB_e = np.concatenate([act(oppE[k], oB_e[k * blk:(k + 1) * blk], rng)[0] for k in range(K)])
        aC_p = np.concatenate([act(oppP[k], oC_p[k * blk:(k + 1) * blk], rng)[0] for k in range(K)])
        rA_p, rA_e, dA, cA, iA = envs["A"].step(ap[:G], ae[:G])
        rB_p, _, dB, cB, iB = envs["B"].step(ap[G:], aB_e)
        _, rC_e, dC, cC, iC = envs["C"].step(aC_p, ae[G:])
        for k, d, c in (("A", dA, cA), ("B", dB, cB), ("C", dC, cC)):
            caps[k] += int(c.sum()); eps[k] += int(d.sum())
        effort += iA["thrust"].mean(0) + iA["sprint"].mean(0)
        bufP["obs"].append(np.concatenate([oA_p, oB_p])); bufP["a"].append(ap); bufP["lp"].append(lpp)
        bufP["v"].append(vp); bufP["r"].append(np.concatenate([rA_p, rB_p])); bufP["d"].append(np.concatenate([dA, dB]))
        bufE["obs"].append(np.concatenate([oA_e, oC_e])); bufE["a"].append(ae); bufE["lp"].append(lpe)
        bufE["v"].append(ve); bufE["r"].append(np.concatenate([rA_e, rC_e])); bufE["d"].append(np.concatenate([dA, dC]))
    oA_p, oA_e = envs["A"].obs(); oB_p, _ = envs["B"].obs(); _, oC_e = envs["C"].obs()
    vlastP = act(P, np.concatenate([oA_p, oB_p]), rng)[2]
    vlastE = act(E, np.concatenate([oA_e, oC_e]), rng)[2]
    out = []
    for buf, vl in ((bufP, vlastP), (bufE, vlastE)):
        arr = {k: np.stack(v) for k, v in buf.items()}
        adv, ret = gae(arr["r"], arr["v"], arr["d"].astype(np.float32), vl)
        out.append((arr["obs"].reshape(-1, sim.OBS_DIM), arr["a"].reshape(-1).astype(np.int32),
                    arr["lp"].reshape(-1), adv.reshape(-1).astype(np.float32), ret.reshape(-1).astype(np.float32)))
    rates = {k: caps[k] / max(eps[k], 1) for k in "ABC"}
    return out, rates, effort / T


def quick_eval(n=128, seed=999):
    """Held-out check vs scripted suite (sampled actions). Returns capture rates."""
    res = {}
    for role, kinds in (("p", ["flee", "smartflee"]), ("e", ["greedy", "lead"])):
        for kind in kinds:
            env = sim.Arena(n, seed)
            S = sim.Scripted(kind, "e" if role == "p" else "p", seed)
            erng = np.random.default_rng(seed)
            alive = np.ones(n, bool); cap = np.zeros(n, bool)
            for t in range(T):
                op, oe = env.obs()
                if role == "p":
                    a_p, a_e = act(P, op, erng)[0], S.act(env)
                else:
                    a_p, a_e = S.act(env), act(E, oe, erng)[0]
                _, _, done, c, _ = env.step(a_p, a_e)
                cap |= c & alive; alive &= ~done
            res[f"{'P' if role == 'p' else 'E'}_vs_{kind}"] = float(cap.mean())
    return res


cfg = vars(args)
json.dump(cfg, open(os.path.join(args.log, "train_config.json"), "w"), indent=1)
logf = open(os.path.join(args.log, "train_log.csv"), "w", newline="")
log = csv.writer(logf)
log.writerow(["iter", "wall_s", "samples", "cap_A_selfplay", "cap_B_P_vs_archE", "cap_C_archP_vs_E",
              "P_effort", "E_effort", "P_ent", "E_ent", "P_kl", "E_kl", "P_vf", "E_vf"])
evf = open(os.path.join(args.log, "train_eval.csv"), "w", newline="")
evw = csv.writer(evf)
evw.writerow(["iter", "wall_s", "P_vs_flee", "P_vs_smartflee", "E_vs_greedy", "E_vs_lead"])

def save_all(final=False):
    snaps = archive if archive[-1][0] == it else archive + [(it, get_np(P), get_np(E))]
    save_pair(os.path.join(args.out, "final.npz"), get_np(P), get_np(E),
              {"iter": it, "samples": samples, "wall_s": time.time() - t0, "config": json.dumps(cfg),
               "complete": final})
    arc = {}
    for i, pp, ee in snaps:
        arc.update({f"{i}/P/{k}": v for k, v in pp.items()})
        arc.update({f"{i}/E/{k}": v for k, v in ee.items()})
    np.savez(os.path.join(args.out, "archive.npz"), **arc)
    return snaps



stop = []
for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(sig, lambda *_: stop.append(1))

t0 = time.time()
samples = 0
it = 0
while it < args.max_iters and time.time() - t0 < args.minutes * 60 and not stop:
    it += 1
    picks = rng.integers(0, len(archive), (2, K))
    for k in range(K):
        set_np(oppE[k], archive[picks[0, k]][2])
        set_np(oppP[k], archive[picks[1, k]][1])
    (dP, dE), rates, effort = rollout()
    sP = ppo_update(stepP, dP)
    sE = ppo_update(stepE, dE)
    samples += len(dP[0]) + len(dE[0])
    wall = time.time() - t0
    log.writerow([it, round(wall, 1), samples, round(rates["A"], 3), round(rates["B"], 3), round(rates["C"], 3),
                  round(effort[0], 3), round(effort[1], 3), round(sP[2], 3), round(sE[2], 3),
                  round(sP[3], 4), round(sE[3], 4), round(sP[1], 4), round(sE[1], 4)])
    logf.flush()
    if it % args.snap_every == 0:
        archive.append((it, get_np(P), get_np(E)))
        save_all()          # crash-safe: the first main run was killed externally at iter 230 with no checkpoint
    if it % args.eval_every == 0 or it == 1:
        q = quick_eval()
        evw.writerow([it, round(wall, 1)] + [round(q[k], 3) for k in
                     ("P_vs_flee", "P_vs_smartflee", "E_vs_greedy", "E_vs_lead")])
        evf.flush()
        print(f"it {it:4d} {wall:6.0f}s  selfplay cap {rates['A']:.2f}  P-vs-arch {rates['B']:.2f}  "
              f"arch-vs-E {rates['C']:.2f}  ent {sP[2]:.2f}/{sE[2]:.2f}  eval {q}", flush=True)


archive = save_all(final=True)
print(f"done: {it} iters, {samples} samples, {time.time() - t0:.0f}s, archive {len(archive)} snapshots" + (" (stopped by signal)" if stop else ""))
