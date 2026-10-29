"""Evolution-strategies colony: OpenAI-style ES over the *same* actor network as PPO.

Every population member is a whole colony (all its ants share the member's weights).
Fitness = the colony's summed PPO reward: delivered + 0.3*picked - energy - deaths (the team
term only redistributes credit inside a colony, so it is a no-op here),
summed over WPM worlds per member, all members on the same seeds each generation
(common random numbers). Antithetic sampling, centred-rank fitness shaping, Adam.

  ../bin/gpu_run ../.venv/bin/python es.py --run es1 --minutes 40
"""
import argparse, json, os, pickle, time
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import sim
from policy import Policy, save_weights, load_weights

p = argparse.ArgumentParser()
p.add_argument("--run", default="es")
p.add_argument("--pop", type=int, default=48, help="must be even (antithetic pairs)")
p.add_argument("--wpm", type=int, default=2, help="worlds per member")
p.add_argument("--N", type=int, default=64)
p.add_argument("--T_EP", type=int, default=600)
p.add_argument("--hid", type=int, default=128)
p.add_argument("--sigma", type=float, default=0.03)
p.add_argument("--lr", type=float, default=0.02)
p.add_argument("--minutes", type=float, default=40)
p.add_argument("--max_gens", type=int, default=10 ** 9)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--resume", action="store_true")
args = p.parse_args()

CK = os.path.join("checkpoints", args.run)
os.makedirs(CK, exist_ok=True)
LOG = open(os.path.join("logs", f"{args.run}.jsonl"), "a")
NAMES = ["a1.weight", "a1.bias", "a2.weight", "a2.bias", "move.weight", "move.bias", "dep.weight", "dep.bias"]

mx.random.seed(args.seed)
model = Policy(hid=args.hid)
params = dict(tree_flatten(model.parameters()))
shapes = [params[n].shape for n in NAMES]
sizes = [int(np.prod(s)) for s in shapes]
theta = np.concatenate([np.array(params[n]).ravel() for n in NAMES]).astype(np.float32)
m_adam, v_adam = np.zeros_like(theta), np.zeros_like(theta)
gen, seed_ctr, t_used = 0, 3_000_000 + 100_000 * args.seed, 0.0
rng = np.random.default_rng(args.seed)

if args.resume and os.path.exists(os.path.join(CK, "state.pkl")):
    st = pickle.load(open(os.path.join(CK, "state.pkl"), "rb"))
    theta, m_adam, v_adam, gen, seed_ctr, t_used = st["theta"], st["m"], st["v"], st["gen"], st["seed_ctr"], st["t_used"]
    rng.bit_generator.state = st["rng"]
    print(f"resumed from generation {gen}", flush=True)


def unflat(th):  # th: [P, D] -> list of per-member arrays
    out, o = [], 0
    for s, n in zip(shapes, sizes):
        out.append(th[:, o:o + n].reshape((th.shape[0],) + s))
        o += n
    return out


def save_ckpt():
    ws = unflat(theta[None])
    model.update(tree_unflatten([(n, mx.array(w[0])) for n, w in zip(NAMES, ws)]))
    save_weights(model, os.path.join(CK, "model.safetensors"))
    json.dump(vars(args), open(os.path.join(CK, "config.json"), "w"), indent=1)
    st = {"theta": theta, "m": m_adam, "v": v_adam, "gen": gen, "seed_ctr": seed_ctr, "t_used": t_used,
          "rng": rng.bit_generator.state, "config": vars(args)}
    pickle.dump(st, open(os.path.join(CK, "state.pkl.tmp"), "wb"))
    os.replace(os.path.join(CK, "state.pkl.tmp"), os.path.join(CK, "state.pkl"))


def batched_probs(ws, obs):
    """ws: per-member weights (mx arrays with leading P); obs [P, M, D]"""
    W1, b1, W2, b2, Wm, bm, Wd, bd = ws
    h = mx.tanh(obs @ W1.transpose(0, 2, 1) + b1[:, None])
    h = mx.tanh(h @ W2.transpose(0, 2, 1) + b2[:, None])
    pm = mx.softmax(h @ Wm.transpose(0, 2, 1) + bm[:, None], -1)
    pd = mx.softmax(h @ Wd.transpose(0, 2, 1) + bd[:, None], -1)
    return pm, pd


def evaluate(th, seeds):
    P = th.shape[0]
    allseeds = np.tile(seeds, P)  # member p gets worlds p*wpm .. p*wpm+wpm-1, same seeds for all members
    w = sim.make_worlds(allseeds, args.N, "train")
    ws = [mx.array(a) for a in unflat(th)]
    for _ in range(args.T_EP):
        obs = sim.observe(w).reshape(P, args.wpm * args.N, -1)
        pm, pd = batched_probs(ws, mx.array(obs))
        pm = np.array(pm).reshape(w.B, w.N, -1)
        pd = np.array(pd).reshape(w.B, w.N, -1)
        m, d = sim.sample_actions(w, pm, pd)
        sim.step(w, m, d)
    deliv = w.delivered.reshape(P, args.wpm).sum(1)
    spent = w.spent.reshape(P, args.wpm).sum(1)
    picked = w.picked.reshape(P, args.wpm).sum(1)
    deaths = w.deaths.reshape(P, args.wpm).sum(1)
    return deliv + 0.3 * picked - spent - deaths, deliv / args.wpm, spent / args.wpm


def centered_rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


t0 = time.time() - t_used
while gen < args.max_gens and (time.time() - t0) / 60 < args.minutes:
    half = args.pop // 2
    eps = rng.standard_normal((half, theta.size)).astype(np.float32)
    eps = np.concatenate([eps, -eps])
    seeds = np.arange(seed_ctr, seed_ctr + args.wpm); seed_ctr += args.wpm
    fit, deliv, spent = evaluate(theta[None] + args.sigma * eps, seeds)
    g = (centered_rank(fit)[:, None] * eps).sum(0) / (args.pop * args.sigma)
    g = -g + 0.005 * theta  # minimise -fitness with a little weight decay
    gen += 1
    m_adam = 0.9 * m_adam + 0.1 * g
    v_adam = 0.999 * v_adam + 0.001 * g * g
    theta = theta - args.lr * (m_adam / (1 - 0.9 ** gen)) / (np.sqrt(v_adam / (1 - 0.999 ** gen)) + 1e-8)
    t_used = time.time() - t0
    rec = {"gen": gen, "min": round(t_used / 60, 2), "fit_mean": float(fit.mean()), "fit_max": float(fit.max()),
           "deliv_mean": float(deliv.mean()), "deliv_max": float(deliv.max()), "spent_mean": float(spent.mean())}
    print(json.dumps(rec), flush=True)
    LOG.write(json.dumps(rec) + "\n"); LOG.flush()
    if gen % 5 == 0:
        save_ckpt()
save_ckpt()
print("done", gen, "generations", flush=True)
