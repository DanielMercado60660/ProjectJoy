"""Round 6: PPO attacker trained against a POPULATION of opponents in slot 1 (optionally with a learned defender).

    ../bin/gpu_run ../.venv/bin/python r6_train.py --seed 0 --total-steps 160e6 --minutes 40 --out checkpoints/r6_s0

Population (per-episode draw, probabilities --mix over r6_common.KINDS):
  scripted legacy goalie / keeper / chaser (dev-visible in round 6), randomized families goalie_r / keeper_r /
  chaser_r / marker / wall (r6_opponents.py; speed x U[0.7,1], reaction delay 0-6 steps, action noise),
  self_arch / self_cur: a frozen past / the current snapshot of the attacker itself driving slot 1 through the
    mirrored observation (it plays as a striker: it attacks the learner's goal),
  def_arch / def_cur (--learn-def 1): a second PPO learner that defends slot 1 (zero-sum goal reward plus its
    own shaping), trained only in def_cur envs against the current attacker; its snapshots form a second archive.
Never in training: the three opponents in r6_opponents.HELDOUT (see STATUS.md, R6 specs).

Schedule is STEP-based (round-3 lesson: a wall-clock schedule got compressed on a loaded machine):
curriculum ramps over the first --curr-frac of --total-steps, lr anneals linearly to 0 at --total-steps.
--minutes is only a safety cap: on hitting it the run checkpoints and exits; --resume continues it.
Checkpoint (<out>.pkl) = weights, Adam state, obs normalisers, archives, the pickled env (positions + RNG),
trainer RNGs, update/step counters, config. MLX noise is re-seeded from (seed, update) every update.
"""
import argparse, json, os, pickle, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from env import W, OUT_GOAL, OUT_CONCEDED, OUT_OWN_GOAL, OUT_TIMEOUT
from r6_common import (R6Arena, AC, OBS6, KINDS, K, RunningNorm, logprob, snapshot, np_policy, pool_actions,
                       save_pickle, tree_np, tree_mx)

# Final round-6 mix (frozen 18:51). Pilots 1 and 3 used PILOT_MIX; the goalie weight was raised because the exact
# full-speed goalie was the weakest dev opponent (pilot 1: 28%, pilot 3: 12.7%).
PILOT_MIX = dict(goalie=0.08, keeper=0.06, chaser=0.06, goalie_r=0.12, keeper_r=0.12, chaser_r=0.10,
                 marker=0.08, wall=0.06, self_arch=0.20, self_cur=0.12)
DEFAULT_MIX = dict(goalie=0.14, keeper=0.05, chaser=0.05, goalie_r=0.14, keeper_r=0.10, chaser_r=0.08,
                   marker=0.06, wall=0.04, self_arch=0.22, self_cur=0.12)
DEF_MIX = dict(goalie=0.06, keeper=0.05, chaser=0.05, goalie_r=0.09, keeper_r=0.09, chaser_r=0.08,
               marker=0.06, wall=0.04, self_arch=0.12, self_cur=0.06, def_arch=0.14, def_cur=0.16)


class Learner:
    """One PPO agent: model + Adam + obs normaliser + archive of frozen snapshots."""

    def __init__(self, hidden, lr, clip, ent):
        self.model = AC(hidden)
        mx.eval(self.model.parameters())
        self.opt = optim.Adam(learning_rate=lr)
        self.norm = RunningNorm(OBS6)
        self.archive = []

        def loss_fn(m, obs, act, old_lp, adv, ret):
            mean, v = m(obs)
            lp = logprob(mean, m.log_std, act)
            ratio = mx.exp(lp - old_lp)
            pg = -mx.mean(mx.minimum(ratio * adv, mx.clip(ratio, 1 - clip, 1 + clip) * adv))
            vl = mx.mean((v - ret) ** 2)
            return pg + 0.5 * vl - ent * mx.sum(m.log_std), (pg, vl, mx.mean((ratio - 1) - mx.log(ratio)))

        self.grad_fn = nn.value_and_grad(self.model, loss_fn)

    def state(self):
        return dict(params=dict(tree_flatten(tree_np(self.model.parameters()))), opt_state=tree_np(self.opt.state),
                    norm=self.norm, archive=self.archive)

    def load(self, S):
        self.model.update(tree_unflatten(list(tree_mx(S["params"]).items())))
        self.opt.init(self.model.trainable_parameters())
        self.opt.state = tree_mx(S["opt_state"])
        self.norm, self.archive = S["norm"], S["archive"]
        mx.eval(self.model.parameters())

    def act(self, raw, update_norm=True):
        if update_norm:
            self.norm.update(raw)
        o = self.norm(raw)
        with mx.stream(mx.cpu):
            mean, v = self.model(mx.array(o))
            a = mean + mx.exp(self.model.log_std) * mx.random.normal(mean.shape)
            mx.eval(a, v, mean)
        return o, np.array(a), np.array(v), np.array(mean)

    def value(self, raw):
        with mx.stream(mx.cpu):
            return np.array(self.model.v(mx.array(self.norm(raw)))[:, 0])

    def update(self, fo, fa, fadv, fret, epochs, minibatch, rng):
        B = len(fo)
        lps = []                                           # old log-probs recomputed on the training device
        for s in range(0, B, 16384):
            mu_, _ = self.model(mx.array(fo[s:s + 16384]))
            lps.append(np.array(logprob(mu_, self.model.log_std, mx.array(fa[s:s + 16384]))))
        flp = np.concatenate(lps).astype(np.float32)
        stats = []
        for ep in range(epochs):
            perm = rng.permutation(B)
            for s in range(0, B, minibatch):
                idx = perm[s:s + minibatch]
                ad = fadv[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                (l, aux), grads = self.grad_fn(self.model, mx.array(fo[idx]), mx.array(fa[idx]), mx.array(flp[idx]),
                                               mx.array(ad), mx.array(fret[idx]))
                grads, _ = optim.clip_grad_norm(grads, 0.5)
                self.opt.update(self.model, grads)
                mx.eval(self.model.parameters(), self.opt.state, l, *aux)
                stats.append([float(x) for x in aux])
        return np.mean(stats, 0) if stats else np.zeros(3)


def gae(rew, val, done, last_v, gamma, lam):
    T, N = rew.shape
    adv = np.zeros((T, N), np.float32)
    g = np.zeros(N, np.float32)
    for t in reversed(range(T)):
        nv = last_v if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
    return adv, adv + val


def def_potential(env):
    """Defender shaping: stay near the ball, keep the ball far from its own (+x) goal."""
    return -0.5 * np.linalg.norm(env.ball - env.pos[:, 1], axis=1) + 1.0 * np.linalg.norm(env.ball - np.array([W, 0.0]), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=float, default=160e6)
    ap.add_argument("--minutes", type=float, default=40.0, help="wall-clock safety cap (checkpoint + exit)")
    ap.add_argument("--max-updates", type=int, default=10 ** 9)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--curr-frac", type=float, default=0.25)
    ap.add_argument("--snap-every", type=float, default=8e6, help="env steps between archive snapshots")
    ap.add_argument("--arch-max", type=int, default=24)
    ap.add_argument("--learn-def", type=int, default=0, help="1: also train a defender in slot 1 (def_* kinds)")
    ap.add_argument("--mix", default=None, help="json {kind: prob}; default DEFAULT_MIX or DEF_MIX")
    ap.add_argument("--ckpt-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/r6_s0")
    ap.add_argument("--log", default=None)
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--init", default=None, help="warm start: attacker weights + obs normaliser + archive from an r6 "
                    "checkpoint (.pkl); fresh optimizer, env and schedule (post-hoc fine-tunes only)")
    ap.add_argument("--nominal-frac", type=float, default=None, help="override r6_common.NOMINAL_FRAC")
    args = ap.parse_args()
    if args.nominal_frac is not None:
        import r6_common
        r6_common.NOMINAL_FRAC = args.nominal_frac
    out = args.out
    log_path = args.log or out.replace("checkpoints/", "logs/") + ".jsonl"
    for p in (out, log_path):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    arch_dir = out + "_arch"
    os.makedirs(arch_dir, exist_ok=True)
    mixd = json.loads(args.mix) if args.mix else (DEF_MIX if args.learn_def else DEFAULT_MIX)
    if not args.learn_def:
        assert not any(mixd.get(k, 0) for k in ("def_arch", "def_cur")), "def_* kinds need --learn-def 1"
    mix = np.array([mixd.get(k, 0.0) for k in KINDS])

    mx.random.seed(args.seed)                              # weight init is seeded too
    att = Learner(args.hidden, args.lr, args.clip, args.ent)
    dfn = Learner(args.hidden, args.lr, args.clip, args.ent) if args.learn_def else None
    if args.resume and os.path.exists(out + ".pkl"):
        S = pickle.load(open(out + ".pkl", "rb"))
        cfg = S["cfg"]
        assert cfg["seed"] == args.seed and cfg["total_steps"] == args.total_steps, "resume config mismatch"
        assert cfg.get("learn_def", 0) == args.learn_def, "resume config mismatch (learn_def)"
        cfg["minutes"] = args.minutes                       # the wall cap may differ per launch
        att.load(S["att"])
        if dfn is not None:
            dfn.load(S["dfn"])
        env, rng, opp_rng = S["env"], S["rng"], S["opp_rng"]
        upd, steps, next_snap = S["upd"], S["steps"], S["next_snap"]
        recent = S["recent"]
        launches = S.get("launches", 1) + 1
        print(f"resumed from {out}.pkl at update {upd}, steps {steps}", flush=True)
        log_f = open(log_path, "a")
    else:
        cfg = vars(args)
        cfg["mix_used"] = mixd
        np.random.seed(args.seed)
        env = R6Arena(args.envs, seed=args.seed, mix=mix)
        env.curriculum = 0.0
        env.reset(np.ones(args.envs, dtype=bool))
        rng = np.random.default_rng(args.seed + 1)
        opp_rng = np.random.default_rng(args.seed + 2)
        upd, steps, next_snap = 0, 0, args.snap_every
        recent = []
        launches = 1
        log_f = open(log_path, "w")
        if args.init:
            I = pickle.load(open(args.init, "rb"))
            att.model.update(tree_unflatten(list(tree_mx(I["att"]["params"]).items())))
            mx.eval(att.model.parameters())
            att.norm, att.archive = I["att"]["norm"], list(I["att"]["archive"])
            print(f"init from {args.init} (update {I['upd']}, steps {I['steps']}), archive {len(att.archive)}", flush=True)
    env.n_arch = len(att.archive)
    env.n_darch = len(dfn.archive) if dfn else 0

    def save_state(extra=None):
        S = dict(cfg=cfg, att=att.state(), dfn=dfn.state() if dfn else None, env=env, rng=rng, opp_rng=opp_rng,
                 upd=upd, steps=steps, next_snap=next_snap, recent=recent, launches=launches,
                 snap=snapshot(att.model, att.norm), dsnap=snapshot(dfn.model, dfn.norm) if dfn else None,
                 extra=extra)
        save_pickle(out + ".pkl", S)

    N, T = args.envs, args.horizon
    total = float(args.total_steps)
    t_start = time.time()
    rec = {}
    while upd < args.max_updates and steps < total:
        if time.time() - t_start > args.minutes * 60:
            save_state(dict(stopped="wall_cap"))
            print(f"WALL CAP hit at update {upd}, steps {steps}; checkpointed, resume with --resume 1", flush=True)
            return
        mx.random.seed(args.seed * 1000003 + upd)
        frac = steps / total
        env.curriculum = float(min(1.0, frac / args.curr_frac)) if args.curr_frac > 0 else 1.0
        lr_now = args.lr * max(0.0, 1.0 - frac)
        att.opt.learning_rate = lr_now
        if dfn:
            dfn.opt.learning_rate = lr_now
        B_ = dict(obs=np.zeros((T, N, OBS6), np.float32), act=np.zeros((T, N, 2), np.float32),
                  val=np.zeros((T, N), np.float32), rew=np.zeros((T, N), np.float32))
        D_ = {k: np.zeros_like(v) for k, v in B_.items()}
        D_["mask"] = np.zeros((T, N), bool)
        done_b = np.zeros((T, N), np.float32)
        rev_b = np.zeros(T)
        cur_snap = snapshot(att.model, att.norm)
        cur_fn = lambda x: np_policy(cur_snap, x)
        t_roll = time.time()
        for t in range(T):
            o, a, v, mean_np = att.act(env.obs_r6(0))
            rev_b[t] = ((mean_np[:, 0] + mean_np[:, 1]) / 2 < -0.1).mean()
            a1 = pool_actions(env, att.archive, cur_fn, opp_rng, dfn.archive if dfn else ())
            kind_now = env.kind.copy()
            if dfn:
                dm = kind_now == K["def_cur"]
                raw1 = env.obs_r6(1)
                if dm.any():
                    dfn.norm.update(raw1[dm])
                od, ad, vd, _ = dfn.act(raw1, update_norm=False)
                a1[dm] = np.clip(ad[dm], -1, 1)
                phi_d = def_potential(env)
            r, d, tr, outc, info = env.step(a, a1)
            if tr.any():
                r = r.copy()
                r[tr] += args.gamma * att.value(env.obs_r6(0)[tr])
            B_["obs"][t], B_["act"][t], B_["val"][t], B_["rew"][t] = o, a, v, r
            done_b[t] = d | tr
            if dfn:
                rd = 10.0 * ((outc == OUT_CONCEDED) | (outc == OUT_OWN_GOAL)) - 10.0 * (outc == OUT_GOAL) \
                     + def_potential(env) - phi_d
                if tr.any():
                    rd[tr] += args.gamma * dfn.value(env.obs_r6(1)[tr])
                D_["obs"][t], D_["act"][t], D_["val"][t], D_["rew"][t], D_["mask"][t] = od, ad, vd, rd, dm
            fin = d | tr
            if fin.any():
                recent.extend(zip(kind_now[fin].tolist(), outc[fin].tolist()))
                env.reset(fin)
        t_roll = time.time() - t_roll
        t_train = time.time()
        adv, ret = gae(B_["rew"], B_["val"], done_b, att.value(env.obs_r6(0)), args.gamma, args.lam)
        B = T * N
        st = att.update(B_["obs"].reshape(B, -1), B_["act"].reshape(B, -1), adv.reshape(B), ret.reshape(B),
                        args.epochs, args.minibatch, rng)
        dst, n_def = np.zeros(3), 0
        if dfn:
            advd, retd = gae(D_["rew"], D_["val"], done_b, dfn.value(env.obs_r6(1)), args.gamma, args.lam)
            m = D_["mask"].reshape(B)
            n_def = int(m.sum())
            if n_def >= 1024:
                mb = max(1024, n_def // 4)
                dst = dfn.update(D_["obs"].reshape(B, -1)[m], D_["act"].reshape(B, -1)[m], advd.reshape(B)[m],
                                 retd.reshape(B)[m], args.epochs, mb, rng)
        t_train = time.time() - t_train
        steps += B
        upd += 1
        recent = recent[-3000:]
        kk = np.array([k for k, _ in recent]) if recent else np.zeros(0, int)
        oo = np.array([o for _, o in recent]) if recent else np.zeros(0, int)
        per = {}
        for name in KINDS:
            m = kk == K[name]
            if m.sum() >= 20:
                per[name] = [round(float((oo[m] == OUT_GOAL).mean()), 3),
                             round(float(((oo[m] == OUT_CONCEDED) | (oo[m] == OUT_OWN_GOAL)).mean()), 3),
                             round(float((oo[m] == OUT_TIMEOUT).mean()), 3)]
        rec = dict(update=upd, steps=steps, wall=round(time.time() - t_start, 1), launch=launches,
                   curriculum=round(env.curriculum, 3), lr=lr_now,
                   goal=float((oo == OUT_GOAL).mean()) if len(oo) else 0.0,
                   conc_own=float(((oo == OUT_CONCEDED) | (oo == OUT_OWN_GOAL)).mean()) if len(oo) else 0.0,
                   timeout=float((oo == OUT_TIMEOUT).mean()) if len(oo) else 0.0,
                   per_kind=per, rev=round(float(rev_b.mean()), 3), n_arch=len(att.archive),
                   pg=float(st[0]), vloss=float(st[1]), kl=float(st[2]),
                   std=float(np.exp(np.array(att.model.log_std)).mean()), mean_rew=float(B_["rew"].mean()),
                   t_roll=round(t_roll, 2), t_train=round(t_train, 2))
        if dfn:
            rec.update(def_n=n_def, def_pg=float(dst[0]), def_vloss=float(dst[1]), def_kl=float(dst[2]),
                       def_std=float(np.exp(np.array(dfn.model.log_std)).mean()), n_darch=len(dfn.archive))
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if upd % 10 == 0 or upd == 1:
            print(json.dumps(rec), flush=True)
        if steps >= next_snap:
            tag = f"{steps // 1_000_000:04d}M"
            for L, pre in ((att, "snap"), (dfn, "dsnap")):
                if L is None:
                    continue
                sn = snapshot(L.model, L.norm)
                L.archive.append(sn)
                if len(L.archive) > args.arch_max:
                    L.archive.pop(0)
                save_pickle(os.path.join(arch_dir, f"{pre}_{tag}.pkl"), dict(snap=sn, steps=steps, upd=upd))
            env.n_arch = len(att.archive)
            env.n_darch = len(dfn.archive) if dfn else 0
            next_snap += args.snap_every
        if upd % args.ckpt_every == 0:
            save_state()
    save_state(dict(stopped="done", wall=time.time() - t_start, last=rec))
    print("saved", out + ".pkl", "updates", upd, "steps", steps, flush=True)


if __name__ == "__main__":
    main()
