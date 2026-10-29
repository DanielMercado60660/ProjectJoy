"""Self-play PPO (MLX) for RoboSumo with a historical-opponent archive.

Rollouts run on CPU in numpy (MuJoCo stepping, small MLP forward). Before each update the old
log-probs are recomputed with the MLX network, so the PPO ratio starts at exactly 1 on the device
that does the update (lesson from 20: CPU-fp32 vs GPU-TF32 mismatch). Action mean is tanh-bounded.

Opponent mix per bout: latest self, a uniformly sampled archive snapshot, or one of the three
TRAINING scripts (tuned params from results/tuned_scripts.json). The held-out CounterPusher is never used.

  ../bin/gpu_run ../.venv/bin/python ppo.py --minutes 18 --out checkpoints/main
"""
import argparse, json, os, pickle, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from sumo_env import SumoEnv, OBS_DIM, make_script, TRAIN_SCRIPTS, Idle, RandomPolicy

HID = 64


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1, self.a2, self.a3 = nn.Linear(OBS_DIM, HID), nn.Linear(HID, HID), nn.Linear(HID, 2)
        self.c1, self.c2, self.c3 = nn.Linear(OBS_DIM, HID), nn.Linear(HID, HID), nn.Linear(HID, 1)
        self.log_std = mx.full((2,), -0.5)

    def mean(self, x):
        return mx.tanh(self.a3(mx.tanh(self.a2(mx.tanh(self.a1(x))))))

    def value(self, x):
        return self.c3(mx.tanh(self.c2(mx.tanh(self.c1(x)))))[:, 0]

    def log_std_c(self):
        return mx.clip(self.log_std, -2.5, 0.0)


def logp_gauss(a, mu, log_std):
    return (-0.5 * ((a - mu) / mx.exp(log_std)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)


def to_numpy_params(model):
    keys = ["a1", "a2", "a3", "c1", "c2", "c3"]
    p = {k: (np.array(getattr(model, k).weight), np.array(getattr(model, k).bias)) for k in keys}
    p["log_std"] = np.array(model.log_std_c())
    return p


class NumpyPolicy:
    """CPU forward of a frozen snapshot. deterministic=True uses the mean action."""

    def __init__(self, p, deterministic=True, seed=0, name="ppo"):
        self.p, self.det, self.rng, self.name = p, deterministic, np.random.default_rng(seed), name

    def mean(self, o):
        h = o
        for k in ("a1", "a2"):
            W, b = self.p[k]
            h = np.tanh(h @ W.T + b)
        W, b = self.p["a3"]
        return np.tanh(h @ W.T + b)

    def value(self, o):
        h = o
        for k in ("c1", "c2"):
            W, b = self.p[k]
            h = np.tanh(h @ W.T + b)
        W, b = self.p["c3"]
        return (h @ W.T + b)[..., 0]

    def __call__(self, o):
        mu = self.mean(o)
        if self.det:
            return mu
        return mu + np.exp(self.p["log_std"]) * self.rng.standard_normal(mu.shape)


def load_tuned():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tuned_scripts.json")
    if os.path.exists(path):
        return json.load(open(path))["best"]
    return {}


APPROACH = 0.0


def shaping(o):
    # potential: own edge margin - opponent edge margin (metres), optionally minus APPROACH * distance
    return o[5] - o[13] - APPROACH * o[8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=18)
    ap.add_argument("--max_updates", type=int, default=10 ** 9)
    ap.add_argument("--n_env", type=int, default=16)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mb", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--shape", type=float, default=1.0)
    ap.add_argument("--approach", type=float, default=0.0, help="potential weight on -distance to opponent")
    ap.add_argument("--draw_pen", type=float, default=0.0, help="learner reward for a timeout draw is -draw_pen")
    ap.add_argument("--p_self", type=float, default=0.3)
    ap.add_argument("--p_archive", type=float, default=0.35)
    ap.add_argument("--p_easy", type=float, default=0.0, help="prob. of an idle/random warm-up opponent")
    ap.add_argument("--snap_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/main")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    global APPROACH
    APPROACH = args.approach
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    mx.random.seed(args.seed)
    model = ActorCritic()
    opt = optim.Adam(learning_rate=args.lr)
    rng = np.random.default_rng(args.seed)
    archive, update, steps_done, history, elapsed_prev = [], 0, 0, [], 0.0
    if args.resume and os.path.exists(args.out + ".pkl"):
        ck = pickle.load(open(args.out + ".pkl", "rb"))
        model.update(nn.utils.tree_unflatten([(k, mx.array(v)) for k, v in ck["model"].items()]))
        opt.state = nn.utils.tree_unflatten([(k, mx.array(v)) for k, v in ck["opt"].items()])
        archive, update, steps_done, history = ck["archive"], ck["update"], ck["steps"], ck["history"]
        elapsed_prev = ck["elapsed"]
        rng.bit_generator.state = ck["rng"]
        print(f"resumed at update {update}, steps {steps_done}", flush=True)
    mx.eval(model.parameters())
    tuned = load_tuned()

    envs = [SumoEnv(seed=args.seed * 1000 + i) for i in range(args.n_env)]
    obs_a = np.zeros((args.n_env, OBS_DIM), np.float32)
    obs_b = np.zeros((args.n_env, OBS_DIM), np.float32)
    opp = [None] * args.n_env
    opp_kind = [""] * args.n_env

    def new_opponent(i, cur_p):
        u = rng.random()
        if u < args.p_self or (u < args.p_self + args.p_archive and not archive):
            return NumpyPolicy(cur_p, deterministic=False, seed=int(rng.integers(1 << 30))), "self"
        if u < args.p_self + args.p_archive:
            j = int(rng.integers(len(archive)))
            return NumpyPolicy(archive[j]["params"], deterministic=False, seed=int(rng.integers(1 << 30))), "archive"
        if u < args.p_self + args.p_archive + args.p_easy:
            if rng.random() < 0.5:
                return Idle(), "idle"
            return RandomPolicy(int(rng.integers(1 << 30))), "random"
        name = TRAIN_SCRIPTS[int(rng.integers(len(TRAIN_SCRIPTS)))]
        return make_script(name, tuned.get(name)), name

    cur_p = to_numpy_params(model)
    for i, e in enumerate(envs):
        obs_a[i], obs_b[i] = e.reset()
        opp[i], opp_kind[i] = new_opponent(i, cur_p)

    def loss_fn(m, ob, ac, old_lp, adv, ret):
        mu = m.mean(ob)
        ls = m.log_std_c()
        lp = logp_gauss(ac, mu, ls)
        ratio = mx.exp(lp - old_lp)
        pg = -mx.minimum(ratio * adv, mx.clip(ratio, 0.8, 1.2) * adv).mean()
        v = m.value(ob)
        vl = ((v - ret) ** 2).mean()
        ent = (ls + 0.5 * np.log(2 * np.pi * np.e)).sum()
        return pg + 0.5 * vl - args.ent * ent, (pg, vl, ((ratio - 1) - mx.log(ratio)).mean(),
                                                 (mx.abs(ratio - 1) > 0.2).mean())

    grad_fn = nn.value_and_grad(model, loss_fn)
    ep_stats = []   # (opp_kind, outcome, tipped_self, tipped_opp)
    t_start = time.time()
    log_f = open(args.out + "_log.jsonl", "a")
    gamma, lam = 0.99, 0.95
    while True:
        elapsed = elapsed_prev + time.time() - t_start
        if elapsed > args.minutes * 60 or update >= args.max_updates:
            break
        cur_p = to_numpy_params(model)
        pol = NumpyPolicy(cur_p, deterministic=False, seed=int(rng.integers(1 << 30)))
        B_o = np.zeros((args.T, args.n_env, OBS_DIM), np.float32)
        B_a = np.zeros((args.T, args.n_env, 2), np.float32)
        B_r = np.zeros((args.T, args.n_env), np.float32)
        B_d = np.zeros((args.T, args.n_env), np.float32)
        B_v = np.zeros((args.T, args.n_env), np.float32)
        for t in range(args.T):
            act = pol(obs_a).astype(np.float32)
            B_o[t], B_a[t] = obs_a, act
            B_v[t] = pol.value(obs_a)
            for i, e in enumerate(envs):
                phi0 = shaping(obs_a[i])
                (oa, ob), out, info = e.step(act[i], opp[i](obs_b[i]))
                r = args.shape * (gamma * shaping(oa) - phi0)
                if out is not None:
                    r = float(out) - args.shape * phi0   # terminal: potential to 0
                    if info.get("timeout"):
                        r -= args.draw_pen
                    ep_stats.append((opp_kind[i], out, bool(info["tip_a"]), bool(info["tip_b"])))
                    oa, ob = e.reset()
                    opp[i], opp_kind[i] = new_opponent(i, cur_p)
                    B_d[t, i] = 1
                B_r[t, i] = r
                obs_a[i], obs_b[i] = oa, ob
        last_v = pol.value(obs_a)
        adv = np.zeros_like(B_r)
        g = np.zeros(args.n_env, np.float32)
        for t in reversed(range(args.T)):
            nv = last_v if t == args.T - 1 else B_v[t + 1]
            delta = B_r[t] + gamma * nv * (1 - B_d[t]) - B_v[t]
            g = delta + gamma * lam * (1 - B_d[t]) * g
            adv[t] = g
        ret = adv + B_v
        N = args.T * args.n_env
        ob_all = mx.array(B_o.reshape(N, -1))
        ac_all = mx.array(B_a.reshape(N, -1))
        old_lp_all = logp_gauss(ac_all, model.mean(ob_all), model.log_std_c())   # recompute on MLX
        adv_f = adv.reshape(N)
        adv_all = mx.array((adv_f - adv_f.mean()) / (adv_f.std() + 1e-8))
        ret_all = mx.array(ret.reshape(N))
        kls, cfs, pgs, vls = [], [], [], []
        for ep in range(args.epochs):
            perm = rng.permutation(N)
            for s in range(0, N, args.mb):
                idx = mx.array(perm[s:s + args.mb])
                (l, (pg, vl, kl, cf)), grads = grad_fn(model, ob_all[idx], ac_all[idx], old_lp_all[idx],
                                                     adv_all[idx], ret_all[idx])
                grads, _ = optim.clip_grad_norm(grads, 0.5)
                opt.update(model, grads)
                mx.eval(model.parameters(), opt.state, kl, cf)
                kls.append(kl.item()); cfs.append(cf.item()); pgs.append(pg.item()); vls.append(vl.item())
        update += 1
        steps_done += N
        if update % args.snap_every == 0:
            archive.append(dict(update=update, steps=steps_done, params=to_numpy_params(model)))
        recent = ep_stats[-300:]
        def wr(kind):
            o = [x[1] for x in recent if x[0] == kind]
            return ((round(float(np.mean([v == 1 for v in o])), 3), round(float(np.mean([v == -1 for v in o])), 3), len(o))
                    if o else (None, None, 0))   # (win rate, loss rate, n); draw = rest
        rec = dict(update=update, steps=steps_done, elapsed=round(elapsed, 1), kl=float(np.mean(kls)),
                   clipfrac=float(np.mean(cfs)), vloss=float(np.mean(vls)),
                   log_std=[float(x) for x in np.array(model.log_std_c())], archive=len(archive),
                   win_self=wr("self"), win_archive=wr("archive"),
                   **{f"win_{n}": wr(n) for n in TRAIN_SCRIPTS + ["idle", "random"]},
                   tip_self=float(np.mean([x[2] for x in recent])) if recent else None,
                   sps=round(steps_done / max(1e-6, elapsed), 1))
        history.append(rec)
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if update % 10 == 0:
            print(json.dumps(rec), flush=True)
        if update % 25 == 0:
            save(args, model, opt, archive, update, steps_done, history, elapsed, rng)
    save(args, model, opt, archive, update, steps_done, history, elapsed_prev + time.time() - t_start, rng)
    print("done", update, steps_done, flush=True)


def save(args, model, opt, archive, update, steps, history, elapsed, rng):
    ck = dict(model={k: np.array(v) for k, v in nn.utils.tree_flatten(model.parameters())},
              opt={k: np.array(v) for k, v in nn.utils.tree_flatten(opt.state)},
              archive=archive, update=update, steps=steps, history=history, elapsed=elapsed,
              rng=rng.bit_generator.state, config=vars(args), final_params=to_numpy_params(model))
    tmp = args.out + ".pkl.tmp"
    pickle.dump(ck, open(tmp, "wb"))
    os.replace(tmp, args.out + ".pkl")


if __name__ == "__main__":
    main()
