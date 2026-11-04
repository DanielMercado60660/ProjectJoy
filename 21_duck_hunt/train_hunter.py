"""Hunter game training (R6): PPO (MLX, CPU) for the duck or the hunter against an opponent pool,
and population self-play with an archive of past opponents. Design: HUNTER_DESIGN.md.

  # stage 1: learned vs scripted
  ../bin/gpu_run ../.venv/bin/python train_hunter.py stage1 --steps 12000000
  # stage 2: population self-play (resumable at generation boundaries)
  ../bin/gpu_run ../.venv/bin/python train_hunter.py selfplay --gens 8 --steps 5000000 [--resume]

Outputs: checkpoints/r6h/<name>.npz (frozen snapshots), checkpoints/r6h/selfplay_state.pkl,
results/r6h_train_<name>.jsonl, results/r6h_selfplay_archive.json.
"""
import argparse, json, os, pickle, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import flight as F
import hunter as H
from train_flight_r6 import logp, tree_np, remaining_path

mx.set_default_device(mx.cpu)
HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, "checkpoints", "r6h")
UPDATE_GPU = os.environ.get("R6H_UPDATE", "gpu") == "gpu"
DUCK_RES = 1.0          # learned duck = PD+ + 1.0 * policy (full authority, starts as PD+)


class Pi(nn.Module):
    def __init__(self, obs_dim, act_dim, hid=128, init_log_std=-0.7, zero_last=False):
        super().__init__()
        self.pi = [nn.Linear(obs_dim, hid), nn.Linear(hid, hid), nn.Linear(hid, act_dim)]
        self.vf = [nn.Linear(obs_dim, hid), nn.Linear(hid, hid), nn.Linear(hid, 1)]
        self.log_std = mx.full((act_dim,), init_log_std)
        if zero_last:
            self.pi[2].weight = mx.zeros_like(self.pi[2].weight); self.pi[2].bias = mx.zeros_like(self.pi[2].bias)

    def mean(self, x):
        h = mx.tanh(self.pi[0](x)); h = mx.tanh(self.pi[1](h))
        return mx.tanh(self.pi[2](h))

    def value(self, x):
        h = mx.tanh(self.vf[0](x)); h = mx.tanh(self.vf[1](h))
        return self.vf[2](h)[:, 0]


def new_learner(role, seed):
    mx.random.seed(seed)
    if role == "duck":
        m = Pi(H.DUCK_OBS, 2, init_log_std=-1.2, zero_last=True)
    else:
        m = Pi(H.HUNT_OBS, 3, init_log_std=-0.5)
    mx.eval(m.parameters())
    return m, optim.Adam(learning_rate=3e-4)


def spec_name(s):
    return s if isinstance(s, str) else s["name"]


class Pool:
    """Opponents for every env slot; weights: 'latest' gets p_latest, the rest share the remainder."""

    def __init__(self, specs, n, rng, p_latest=0.5):
        self.specs = specs
        self.agents = [H.make_agent(s, n, rng) for s in specs]
        k = len(specs)
        w = np.full(k, 1.0 / k)
        if k > 1 and p_latest is not None:
            w = np.full(k, (1 - p_latest) / (k - 1)); w[-1] = p_latest
        self.w, self.rng, self.n = w, rng, n
        self.o = rng.choice(k, n, p=w)

    def reset(self, idx):
        for a in self.agents:
            a.reset(idx)
        self.o[idx] = self.rng.choice(len(self.agents), len(idx), p=self.w)

    def act(self, env, dim):
        out = np.zeros((self.n, dim))
        for j, a in enumerate(self.agents):
            rows = self.o == j
            aj = a.act(env)                 # every member acts on all rows (keeps filter state current)
            out[rows] = aj[rows]
        return out


def train_role(role, model, opt, pool_specs, steps, name, seed, envs=256, horizon=128, mb=8192, epochs=4,
               gamma=0.99, lam=0.95, clip=0.2, lr=3e-4, p_latest=0.5, log_every=10):
    rng = np.random.default_rng(seed)
    env = H.DuelEnv(envs, seed=10_000 + seed)
    pool = Pool(pool_specs, envs, rng, p_latest)
    act_dim = 2 if role == "duck" else 3
    obs_dim = H.DUCK_OBS if role == "duck" else H.HUNT_OBS
    opp_dim = 3 if role == "duck" else 2

    def obs():
        return env.duck_obs() if role == "duck" else env.hunter_obs()

    def phi():
        if role == "duck":
            return -0.1 * remaining_path(env) / 10.0
        return -0.3 * np.minimum(np.linalg.norm(env.c - env.p, axis=1), 10.0) / 10.0

    def loss_fn(model, ob, ac, old_lp, adv, ret):
        lp = logp(model.mean(ob), model.log_std, ac)
        ratio = mx.exp(lp - old_lp)
        pg = -mx.minimum(ratio * adv, mx.clip(ratio, 1 - clip, 1 + clip) * adv).mean()
        vl = ((model.value(ob) - ret) ** 2).mean()
        return pg + 0.5 * vl, (vl, mx.mean(old_lp - lp))

    lg = nn.value_and_grad(model, loss_fn)
    opt.learning_rate = lr
    flog = open(os.path.join(HERE, "results", f"r6h_train_{name}.jsonl"), "w")
    T, N = horizon, envs
    step, upd, t0 = 0, 0, time.time()
    outc = []
    while step < steps:
        O = np.zeros((T, N, obs_dim), np.float32); A = np.zeros((T, N, act_dim), np.float32)
        R = np.zeros((T, N), np.float32); Dn = np.zeros((T, N), np.float32); V = np.zeros((T + 1, N), np.float32)
        std = np.exp(np.array(model.log_std))
        for t in range(T):
            x = obs(); xm = mx.array(x)
            mu = np.array(model.mean(xm)); V[t] = np.array(model.value(xm))
            a = (mu + std * rng.normal(size=mu.shape)).astype(np.float32)
            O[t], A[t] = x, a
            ph0 = phi()
            opp = pool.act(env, opp_dim)
            if role == "duck":
                done, info = env.step(H.duck_action(env, a, DUCK_RES), opp)
            else:
                done, info = env.step(opp, a)
            ph1 = np.where(done, 0.0, phi())
            if role == "duck":
                r = 1.0 * info["complete"] - 1.0 * info["hit"] - 1.0 * (info["crash"] | info["timeout"])
            else:
                r = 1.0 * info["hit"] - 1.0 * info["complete"] - 0.02 * info["fired"]
            R[t] = r + gamma * ph1 - ph0; Dn[t] = done
            idx = np.nonzero(done)[0]
            for i in idx:
                outc.append((info["hit"][i], info["complete"][i], info["crash"][i] | info["timeout"][i]))
            env.reset(idx); pool.reset(idx)
        V[T] = np.array(model.value(mx.array(obs())))
        adv = np.zeros((T, N), np.float32); last = 0
        for t in reversed(range(T)):
            nt = 1 - Dn[t]
            last = R[t] + gamma * V[t + 1] * nt - V[t] + gamma * lam * nt * last
            adv[t] = last
        ret = adv + V[:T]
        with mx.stream(mx.gpu if UPDATE_GPU else mx.cpu):   # rollouts on CPU, PPO update on GPU
            ob = mx.array(O.reshape(-1, obs_dim)); ac = mx.array(A.reshape(-1, act_dim))
            old_lp = logp(model.mean(ob), model.log_std, ac)
            advf = adv.reshape(-1); advf = (advf - advf.mean()) / (advf.std() + 1e-8)
            advm, retm = mx.array(advf), mx.array(ret.reshape(-1))
            kls = []
            for _ in range(epochs):
                perm = rng.permutation(T * N)
                for s in range(0, T * N, mb):
                    ii = mx.array(perm[s:s + mb])
                    (l, (vl, kl)), g = lg(model, ob[ii], ac[ii], old_lp[ii], advm[ii], retm[ii])
                    opt.update(model, g); mx.eval(model.parameters(), opt.state)
                    kls.append(kl.item())
        step += T * N; upd += 1
        oc = np.array(outc[-1000:], float) if outc else np.zeros((1, 3))
        rec = dict(step=step, wall=time.time() - t0, hit=float(oc[:, 0].mean()), complete=float(oc[:, 1].mean()),
                   fail=float(oc[:, 2].mean()), kl=float(np.mean(kls)), vloss=float(vl.item()),
                   log_std=np.array(model.log_std).tolist())
        outc = outc[-1000:]
        flog.write(json.dumps(rec) + "\n"); flog.flush()
        if upd % log_every == 0 or step >= steps:
            print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    os.makedirs(CK, exist_ok=True)
    path = os.path.join(CK, f"{name}.npz")
    model.save_weights(path)
    return dict(kind="net", path=path, role=role, name=name, res_scale=DUCK_RES if role == "duck" else None)


def opt_np(opt):
    return tree_np(opt.state)


def opt_load(opt, model, st):
    opt.init(model.trainable_parameters())
    opt.state = nn.utils.tree_unflatten([(k, mx.array(v)) for k, v in st.items()])


def stage1(args):
    os.makedirs(CK, exist_ok=True)
    out = {}
    m, o = new_learner("duck", 1)
    out["duck_s"] = train_role("duck", m, o, ["naive", "lead"], args.steps, "duck_s", 1, p_latest=None)
    m, o = new_learner("hunter", 2)
    out["hunter_s"] = train_role("hunter", m, o, ["straight", "jink", "dodge"], args.steps, "hunter_s", 2, p_latest=None)
    json.dump(out, open(os.path.join(HERE, "results", "r6h_stage1.json"), "w"), indent=1)


def selfplay(args):
    """Generations: duck trains vs hunter archive, then hunter trains vs duck archive; each new
    snapshot joins its archive. The newest opponent is sampled with p=0.5, the rest uniformly."""
    os.makedirs(CK, exist_ok=True)
    sp = os.path.join(CK, "selfplay_state.pkl")
    if args.resume and os.path.exists(sp):
        st = pickle.load(open(sp, "rb"))
        print("resuming after generation", st["gen"], flush=True)
    else:
        s1 = json.load(open(os.path.join(HERE, "results", "r6h_stage1.json")))
        st = dict(gen=0, arch_h=["naive", "lead", s1["hunter_s"]], arch_d=["straight", "jink", "dodge", s1["duck_s"]],
                  duck_w=s1["duck_s"]["path"], hunt_w=s1["hunter_s"]["path"], duck_opt=None, hunt_opt=None)
    md, od = new_learner("duck", 11); mh, oh = new_learner("hunter", 12)
    md.load_weights(st["duck_w"]); mh.load_weights(st["hunt_w"])
    if st["duck_opt"] is not None:
        opt_load(od, md, st["duck_opt"]); opt_load(oh, mh, st["hunt_opt"])
    t_start = time.time()
    for g in range(st["gen"] + 1, args.gens + 1):
        if time.time() - t_start > args.max_wall:
            print(f"max_wall reached before generation {g}; rerun with --resume", flush=True)
            break
        t0 = time.time()
        d = train_role("duck", md, od, st["arch_h"], args.steps, f"duck_g{g}", 100 + g)
        h = train_role("hunter", mh, oh, st["arch_d"] + [d], args.steps, f"hunter_g{g}", 200 + g)
        st["arch_d"].append(d); st["arch_h"].append(h)
        st.update(gen=g, duck_w=d["path"], hunt_w=h["path"], duck_opt=opt_np(od), hunt_opt=opt_np(oh))
        with open(sp + ".tmp", "wb") as f:
            pickle.dump(st, f)
        os.replace(sp + ".tmp", sp)
        json.dump(dict(gen=g, arch_d=st["arch_d"], arch_h=st["arch_h"]),
                  open(os.path.join(HERE, "results", "r6h_selfplay_archive.json"), "w"), indent=1)
        print(f"generation {g} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stage1", "selfplay"])
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_wall", type=float, default=2300, help="s; no new generation starts after this")
    args = ap.parse_args()
    {"stage1": stage1, "selfplay": selfplay}[args.mode](args)
