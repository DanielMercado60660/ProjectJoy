"""Shared-policy PPO colony (MLX).

Every ant in every world is one sample of the same policy. Per-ant reward:
    + 1.0  food delivered to the nest by this ant
    + 0.3  food picked up by this ant            (shaping: a local contact event, no route info)
    + TEAM * (colony deliveries this step) / N   (shared credit so laying trail for others pays)
    - ENERGY_W * energy spent by this ant this step
    - 1.0  if the ant starves
Episodes are T_EP steps; all worlds reset together with fresh procedural layouts.

Checkpoints (weights + Adam state + update count + seed counter + numpy RNG + config) are
written at episode boundaries; --resume continues from the latest one.

  ../bin/gpu_run ../.venv/bin/python train.py --run ppo1 --minutes 40
"""
import argparse, json, os, pickle, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import sim
import handwritten as hw
from policy import Policy, probs, save_weights, load_weights, mask_obs

p = argparse.ArgumentParser()
p.add_argument("--run", default="ppo")
p.add_argument("--B", type=int, default=32)
p.add_argument("--N", type=int, default=64)
p.add_argument("--T", type=int, default=100)
p.add_argument("--T_EP", type=int, default=1000)
p.add_argument("--hid", type=int, default=128)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--mb", type=int, default=8192)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--lam", type=float, default=0.95)
p.add_argument("--clip", type=float, default=0.2)
p.add_argument("--ent", type=float, default=0.01)
p.add_argument("--vf", type=float, default=0.5)
p.add_argument("--team", type=float, default=0.5)
p.add_argument("--pick", type=float, default=0.3)
p.add_argument("--energy_w", type=float, default=1.0)
p.add_argument("--death", type=float, default=1.0)
p.add_argument("--minutes", type=float, default=40)
p.add_argument("--max_updates", type=int, default=10 ** 9)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--resume", action="store_true")
p.add_argument("--no_compass", action="store_true", help="blind the path-integration compass (obs zeroed)")
p.add_argument("--eval_every", type=int, default=5, help="episodes")
args = p.parse_args()

CK = os.path.join("checkpoints", args.run)
os.makedirs(CK, exist_ok=True)
LOG = open(os.path.join("logs", f"{args.run}.jsonl"), "a")
assert args.T_EP % args.T == 0

mx.random.seed(args.seed)
model = Policy(hid=args.hid)
mx.eval(model.parameters())
opt = optim.Adam(learning_rate=args.lr)
nprng = np.random.default_rng(args.seed)
update, seed_ctr, episode = 0, 1_000_000 + 100_000 * args.seed, 0
t_used, best_val = 0.0, -1.0

if args.resume and os.path.exists(os.path.join(CK, "state.pkl")):
    load_weights(model, os.path.join(CK, "model.safetensors"))
    with open(os.path.join(CK, "state.pkl"), "rb") as f:
        st = pickle.load(f)
    opt.init(model.trainable_parameters())
    opt.state = tree_unflatten([(k, mx.array(v)) for k, v in st["opt"]])
    update, seed_ctr, episode, t_used = st["update"], st["seed_ctr"], st["episode"], st["t_used"]
    best_val = st.get("best_val", -1.0)
    nprng.bit_generator.state = st["nprng"]
    print(f"resumed from update {update} episode {episode}", flush=True)


def save_ckpt():
    save_weights(model, os.path.join(CK, "model.safetensors"))
    st = {"opt": [(k, np.array(v)) for k, v in tree_flatten(opt.state)], "update": update,
          "seed_ctr": seed_ctr, "episode": episode, "t_used": t_used, "nprng": nprng.bit_generator.state,
          "best_val": best_val,
          "config": vars(args)}
    with open(os.path.join(CK, "state.pkl.tmp"), "wb") as f:
        pickle.dump(st, f)
    os.replace(os.path.join(CK, "state.pkl.tmp"), os.path.join(CK, "state.pkl"))
    with open(os.path.join(CK, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)


def critic_in(obs, world):
    B, N, _ = obs.shape
    extra = np.empty((B, N, 2), np.float32)
    extra[..., 0] = (1.0 - world.t / args.T_EP)[:, None]
    extra[..., 1] = np.log1p(world.store / N)[:, None]
    return np.concatenate([obs, extra], -1)


def loss_fn(model, obs, cin, am, ad, logp_old, adv, ret):
    lm, ld = model.actor(obs)
    lpm = nn.log_softmax(lm, -1)
    lpd = nn.log_softmax(ld, -1)
    logp = mx.take_along_axis(lpm, am[:, None], -1)[:, 0] + mx.take_along_axis(lpd, ad[:, None], -1)[:, 0]
    ratio = mx.exp(logp - logp_old)
    pg = -mx.minimum(ratio * adv, mx.clip(ratio, 1 - args.clip, 1 + args.clip) * adv).mean()
    v = model.value(cin)
    vl = ((v - ret) ** 2).mean()
    ent = -(mx.exp(lpm) * lpm).sum(-1).mean() - (mx.exp(lpd) * lpd).sum(-1).mean()
    return pg + args.vf * vl - args.ent * ent, (pg, vl, ent, ((ratio - 1) - mx.log(ratio)).mean())


loss_and_grad = nn.value_and_grad(model, loss_fn)


def train_step(batch):
    (loss, aux), g = loss_and_grad(model, *batch)
    g, _ = optim.clip_grad_norm(g, 0.5)
    opt.update(model, g)
    mx.eval(model.parameters(), opt.state, loss, *aux)
    return [float(a) for a in aux]


def new_world():
    global seed_ctr
    seeds = np.arange(seed_ctr, seed_ctr + args.B)
    seed_ctr += args.B
    return sim.make_worlds(seeds, args.N, "train")


VAL_SEEDS = np.arange(900, 916)  # validation worlds for checkpoint selection (disjoint from train 1e6+ and test 0-31)


def quick_eval():
    w = sim.make_worlds(VAL_SEEDS, args.N, "train")
    for _ in range(args.T_EP):
        pm, pd = probs(model, mask_obs(sim.observe(w), args.no_compass))
        m, d = sim.sample_actions(w, pm, pd)
        sim.step(w, m, d)
    return float(w.delivered.mean()), float(w.alive.mean()), float(w.delivered.sum() / w.spent.sum())


B, N, T = args.B, args.N, args.T
D = sim.OBS_DIM
buf_obs = np.zeros((T, B, N, D), np.float32)
buf_cin = np.zeros((T, B, N, D + 2), np.float32)
buf_am = np.zeros((T, B, N), np.int32)
buf_ad = np.zeros((T, B, N), np.int32)
buf_lp = np.zeros((T, B, N), np.float32)
buf_v = np.zeros((T, B, N), np.float32)
buf_r = np.zeros((T, B, N), np.float32)
buf_done = np.zeros((T, B, N), np.float32)
buf_valid = np.zeros((T, B, N), np.float32)

world = new_world()
t0 = time.time() - t_used
ep_stats = []
while update < args.max_updates and (time.time() - t0) / 60 < args.minutes:
    tr0 = time.time()
    for t in range(T):
        obs = mask_obs(sim.observe(world), args.no_compass)
        cin = critic_in(obs, world)
        lm, ld = model.actor(mx.array(obs))
        v = model.value(mx.array(cin))
        pm, pd = mx.softmax(lm, -1), mx.softmax(ld, -1)
        pm, pd, v = np.array(pm), np.array(pd), np.array(v)
        valid = world.alive.astype(np.float32)
        m, d = sim.sample_actions(world, pm, pd)
        ev = sim.step(world, m, d)
        team = ev.deliver.sum(1, keepdims=True) / N
        r = ev.deliver + args.pick * ev.pick + args.team * team - args.energy_w * ev.cost - args.death * ev.die
        buf_obs[t], buf_cin[t], buf_am[t], buf_ad[t], buf_v[t] = obs, cin, m, d, v
        buf_lp[t] = np.log(np.take_along_axis(pm, m[..., None], -1)[..., 0] + 1e-8) + \
            np.log(np.take_along_axis(pd, d[..., None], -1)[..., 0] + 1e-8)
        buf_r[t] = r * valid
        buf_valid[t] = valid
        buf_done[t] = ev.die
    ep_end = world.t[0] >= args.T_EP or world.alive.mean() < 0.05  # stop early once colonies are dead
    if ep_end:
        buf_done[-1] = 1.0
        last_v = np.zeros((B, N), np.float32)
    else:
        obs = mask_obs(sim.observe(world), args.no_compass)
        last_v = np.array(model.value(mx.array(critic_in(obs, world)))) * world.alive
    # GAE
    adv = np.zeros((T, B, N), np.float32)
    nxt_adv, nxt_v = np.zeros((B, N), np.float32), last_v
    for t in reversed(range(T)):
        nonterm = 1.0 - buf_done[t]
        delta = buf_r[t] + args.gamma * nxt_v * nonterm - buf_v[t]
        nxt_adv = delta + args.gamma * args.lam * nonterm * nxt_adv
        adv[t] = nxt_adv
        nxt_v = buf_v[t]
    ret = adv + buf_v
    tr = time.time() - tr0
    # flatten valid samples
    idx = np.flatnonzero(buf_valid.reshape(-1) > 0)
    f = lambda a: a.reshape((-1,) + a.shape[3:])[idx]
    O, C, AM, AD, LP, A, R = map(f, (buf_obs, buf_cin, buf_am, buf_ad, buf_lp, adv, ret))
    A = (A - A.mean()) / (A.std() + 1e-8)
    tt0 = time.time()
    stats = []
    for _ in range(args.epochs):
        perm = nprng.permutation(len(idx))
        for s in range(0, len(perm) - args.mb + 1, args.mb):
            j = perm[s:s + args.mb]
            stats.append(train_step(tuple(mx.array(x[j]) for x in (O, C, AM, AD, LP, A, R))))
    update += 1
    ttr = time.time() - tt0
    if ep_end:
        episode += 1
        st = np.mean(stats, 0) if stats else np.zeros(4)
        rec = {"update": update, "episode": episode, "ep_len": int(world.t[0]), "min": round((time.time() - t0) / 60, 2),
               "deliv": float(world.delivered.mean()), "alive": float(world.alive.mean()),
               "eff": float(world.delivered.sum() / max(world.spent.sum(), 1e-9)),
               "pg": st[0], "vl": st[1], "ent": st[2], "kl": st[3], "t_roll": round(tr, 2), "t_train": round(ttr, 2)}
        if episode % args.eval_every == 0:
            ev_d, ev_a, ev_e = quick_eval()
            rec.update(eval_deliv=ev_d, eval_alive=ev_a, eval_eff=ev_e)
            if ev_d > best_val:  # keep the best policy on the validation worlds (guards against late collapse)
                best_val = ev_d
                save_weights(model, os.path.join(CK, "model_best.safetensors"))
                rec["new_best"] = True
        print(json.dumps(rec), flush=True)
        LOG.write(json.dumps(rec) + "\n"); LOG.flush()
        t_used = time.time() - t0
        save_ckpt()
        if episode % 25 == 0:
            save_weights(model, os.path.join(CK, f"model_ep{episode:04d}.safetensors"))
        world = new_world()
t_used = time.time() - t0
print("done", update, "updates", round(t_used / 60, 1), "min", flush=True)
