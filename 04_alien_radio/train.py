"""Train populations of sender/receiver pairs (or receiver-only baselines) on the Alien Radio game.

A run trains P independent agent pairs at once (population axis, see alien.py). Member p is
"seed p" in the write-up: own init, own data batches, own noise.

Methods
  gumbel        sender + receiver trained end-to-end through straight-through Gumbel-softmax
  reinforce     sender trained with REINFORCE (reward = receiver log-prob of the target, EMA
                baseline, entropy bonus); receiver trained with cross-entropy on the sampled message
  nocomm        receiver only, the message is always all-zeros              (chance-level baseline)
  compositional receiver only, fixed hand-designed code: symbol j = attribute j  (upper reference)
  holistic      receiver only, fixed random one-to-one code per member (no structure)

Examples
  ../.venv/bin/python train.py --stage color --method gumbel
  ../bin/gpu_run ../.venv/bin/python train.py --stage full --method gumbel reinforce nocomm compositional holistic
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from alien import World, Game, make_episodes, reshuffle, compositional_code, holistic_code

HERE = os.path.dirname(os.path.abspath(__file__))

STAGES = {
    # first satisfying version: colours only, one symbol from a vocabulary of eight
    # (tau=2 / REINFORCE entropy 0.3 chosen by the sweep in sweep_color.sh: fewest seeds stuck)
    "color": dict(dims=(8,), L=1, V=8, K=5, steps=3000, tau=2.0, entropy_coef=0.3),
    # color x shape x location, 20% of attribute combinations held out
    "full": dict(dims=(8, 4, 4), L=3, V=8, K=5, steps=15000),
}


def default_cfg(stage, method, seed=0, **over):
    cfg = dict(stage=stage, method=method, seed=seed, P=5, batch=256, lr=3e-4, hidden=128,
               tau=1.0, entropy_coef=0.1, gumbel_entropy_coef=0.1, baseline_decay=0.99,
               cluster_frac=0.5, n_train_episodes=100_000, n_eval_episodes=4000,
               eval_every=250, ckpt_every=2500, data_seed=1234, tag="", entropy_final=None)
    cfg.update(STAGES[stage])
    cfg.update({k: v for k, v in over.items() if v is not None})
    cfg["dims"] = list(cfg["dims"])
    return cfg


def run_name(cfg):
    return f"{cfg['stage']}_{cfg['method']}{cfg['tag']}_s{cfg['seed']}"


def onehot(sym, V):
    return (sym[..., None] == mx.arange(V)).astype(mx.float32)


def per_member_ce(scores, tpos):
    return nn.losses.cross_entropy(scores, tpos)            # (P, B)


# ----------------------------------------------------------------------------- losses
def make_loss(cfg):
    method, V = cfg["method"], cfg["V"]

    def entropy(logits):  # mean per-position entropy in nats, (P, B)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -(mx.exp(logp) * logp).sum(-1).mean(-1), logp

    def loss_gumbel(model, tfeat, cfeat, tpos, fixed_msg, baseline, ent_coef):
        logits = model.sender(tfeat)                                    # (P, B, L, V)
        g = -mx.log(-mx.log(mx.random.uniform(shape=logits.shape, low=1e-10, high=1.0)))
        y_soft = mx.softmax((logits + g) / cfg["tau"], axis=-1)
        y_hard = onehot(mx.argmax(y_soft, axis=-1), V)
        msg = mx.stop_gradient(y_hard - y_soft) + y_soft              # straight-through
        ce = per_member_ce(model.receiver(msg, cfeat), tpos)
        ent, _ = entropy(logits)
        loss = (ce.mean(1) - ent_coef * ent.mean(1)).sum()
        return loss, (ce, ent)

    def loss_reinforce(model, tfeat, cfeat, tpos, fixed_msg, baseline, ent_coef):
        logits = model.sender(tfeat)
        ent, logp_all = entropy(logits)
        sym = mx.random.categorical(logits)                           # (P, B, L)
        msg = mx.stop_gradient(onehot(sym, V))                        # no gradient through the channel
        logp = (logp_all * msg).sum(axis=(-1, -2))                    # (P, B)
        ce = per_member_ce(model.receiver(msg, cfeat), tpos)
        adv = mx.stop_gradient(-ce) - baseline[:, None]
        loss = (ce.mean(1) - (adv * logp).mean(1) - ent_coef * ent.mean(1)).sum()
        return loss, (ce, ent)

    def loss_fixed(model, tfeat, cfeat, tpos, fixed_msg, baseline, ent_coef):
        ce = per_member_ce(model.receiver(fixed_msg, cfeat), tpos)
        return ce.mean(1).sum(), (ce, mx.zeros_like(ce))

    return {"gumbel": loss_gumbel, "reinforce": loss_reinforce}.get(method, loss_fixed)


def entropy_coef(cfg, step):
    """Entropy-bonus weight at this step; optional linear decay to cfg['entropy_final']."""
    c0 = cfg["gumbel_entropy_coef"] if cfg["method"] == "gumbel" else cfg["entropy_coef"]
    if cfg.get("entropy_final") is None:
        return c0
    return c0 + (cfg["entropy_final"] - c0) * min(1.0, step / cfg["steps"])


def fixed_table(cfg, world):
    """(P, N, L) message table for the receiver-only baselines, else None."""
    m, P, L, V = cfg["method"], cfg["P"], cfg["L"], cfg["V"]
    if m == "compositional":
        return np.stack([compositional_code(world, L, V)] * P)
    if m == "holistic":
        return np.stack([holistic_code(world, L, V, seed=1000 * cfg["seed"] + p) for p in range(P)])
    if m == "nocomm":
        return np.zeros((P, world.n, L), dtype=np.int32)
    return None


# ----------------------------------------------------------------------------- eval helpers
def sender_table(model, world, cfg):
    """Deterministic (argmax) message for every object, (P, N, L). Each member's 'language'."""
    tab = fixed_table(cfg, world)
    if tab is not None:
        return tab
    x = mx.broadcast_to(mx.array(world.feats)[None], (model.P, *world.feats.shape))
    return np.array(mx.argmax(model.sender(x), axis=-1)).astype(np.int32)


def receiver_choices(receiver, world, msgs, cand, V, chunk=2000):
    """msgs: (P, E, L) symbols shown to each member's receiver; cand: (E, K) ids. -> (P, E) choices."""
    P = msgs.shape[0]
    out = []
    for i in range(0, cand.shape[0], chunk):
        m = onehot(mx.array(msgs[:, i:i + chunk]), V)
        c = mx.array(world.feats[cand[i:i + chunk]])
        c = mx.broadcast_to(c[None], (P, *c.shape))
        out.append(np.array(mx.argmax(receiver(m, c), axis=-1)))
    return np.concatenate(out, axis=1)


def receiver_accuracy(receiver, world, msgs, cand, tpos, V):
    return (receiver_choices(receiver, world, msgs, cand, V) == tpos[None]).mean(1)   # (P,)


def evaluate(model, world, cfg, eval_sets, table=None):
    if table is None:
        table = sender_table(model, world, cfg)
    out = {}
    for name, (cand, tpos) in eval_sets.items():
        tgt = cand[np.arange(len(cand)), tpos]
        out[name] = receiver_accuracy(model.receiver, world, table[:, tgt], cand, tpos, cfg["V"])
    return out, table


def get_eval_sets(world, cfg, offset=0):
    n, K = cfg["n_eval_episodes"], cfg["K"]
    sets = {"train_acc": make_episodes(world, n, K, "train", seed=cfg["data_seed"] + 7 + offset,
                                       cluster_frac=cfg["cluster_frac"])}
    if len(world.test_ids):
        sets["heldout_acc"] = make_episodes(world, n, K, "test", seed=cfg["data_seed"] + 11 + offset,
                                            cluster_frac=cfg["cluster_frac"])
    return sets


# ----------------------------------------------------------------------------- checkpointing
def save_ckpt(path, model, opt, cfg, step, baseline, log):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mx.save_safetensors(path + ".safetensors", dict(tree_flatten(model.parameters())))
    mx.save_safetensors(path + ".opt.safetensors", dict(tree_flatten(opt.state)))
    with open(path + ".json", "w") as f:
        json.dump({"cfg": cfg, "step": step, "baseline": baseline, "log": log}, f)


def load_model(path):
    """Rebuild World + Game (all P members) from a checkpoint path prefix (no extension)."""
    with open(path + ".json") as f:
        meta = json.load(f)
    cfg = meta["cfg"]
    world = World(dims=cfg["dims"])
    model = Game(cfg["P"], world.feats.shape[1], cfg["L"], cfg["V"], cfg["hidden"])
    model.load_weights(path + ".safetensors")
    mx.eval(model.parameters())
    return model, world, cfg, meta


# ----------------------------------------------------------------------------- training
def train(cfg, ckpt_dir=os.path.join(HERE, "checkpoints"), resume=True, stop_at=None, verbose=True):
    """Train one population. Batches and noise are a pure function of (seed, step): resume is exact."""
    world = World(dims=cfg["dims"])
    F, P, B, V, L = world.feats.shape[1], cfg["P"], cfg["batch"], cfg["V"], cfg["L"]
    mx.random.seed(cfg["seed"])
    model = Game(P, F, L, V, cfg["hidden"])
    opt = optim.Adam(learning_rate=cfg["lr"])
    path = os.path.join(ckpt_dir, run_name(cfg))
    step, baseline, log = 0, [0.0] * P, []
    if resume and os.path.exists(path + ".json"):
        with open(path + ".json") as f:
            meta = json.load(f)
        model.load_weights(path + ".safetensors")
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(path + ".opt.safetensors").items()))
        step, baseline, log = meta["step"], meta["baseline"], meta["log"]
        if verbose:
            print(f"[resume] {run_name(cfg)} from step {step}", flush=True)
    mx.eval(model.parameters())

    bank_c, bank_t = make_episodes(world, cfg["n_train_episodes"], cfg["K"], "train",
                                   seed=cfg["data_seed"], cluster_frac=cfg["cluster_frac"])
    eval_sets = get_eval_sets(world, cfg)
    table = fixed_table(cfg, world)
    vg = nn.value_and_grad(model, make_loss(cfg))
    feats = mx.array(world.feats)
    baseline = np.array(baseline, dtype=np.float32)

    def step_fn(tfeat, cfeat, tpos, fixed_msg, base, ent_coef):
        (loss, (ce, ent)), grads = vg(model, tfeat, cfeat, tpos, fixed_msg, base, ent_coef)
        opt.update(model, grads)
        return loss, ce, ent

    state = [model.state, opt.state, mx.random.state]
    step_c = mx.compile(step_fn, inputs=state, outputs=state)

    t0 = time.time()
    end = cfg["steps"] if stop_at is None else min(stop_at, cfg["steps"])
    while step < end:
        rng = np.random.default_rng([cfg["seed"], step])
        mx.random.seed(cfg["seed"] * 1_000_003 + step)
        idx = rng.integers(0, len(bank_c), P * B)
        cand, tpos = reshuffle(bank_c[idx], bank_t[idx], rng)
        tgt = cand[np.arange(P * B), tpos]
        cand, tpos, tgt = cand.reshape(P, B, -1), tpos.reshape(P, B), tgt.reshape(P, B)
        if table is not None:
            fixed = onehot(mx.array(table[np.arange(P)[:, None], tgt]), V)
        else:
            fixed = mx.zeros((P, B, L, V))
        loss, ce, ent = step_c(feats[mx.array(tgt)], feats[mx.array(cand)], mx.array(tpos), fixed,
                               mx.array(baseline), mx.array(entropy_coef(cfg, step), dtype=mx.float32))
        mx.eval(state, loss, ce, ent)
        d = cfg["baseline_decay"]
        baseline = d * baseline + (1 - d) * (-np.array(ce.mean(1)))
        step += 1
        if step % cfg["eval_every"] == 0 or step == end:
            accs, _ = evaluate(model, world, cfg, eval_sets, table)
            rec = {"step": step, "ce": float(ce.mean()), "entropy": float(ent.mean()),
                   **{k: float(v.mean()) for k, v in accs.items()},
                   **{k + "_members": [round(float(x), 4) for x in v] for k, v in accs.items()},
                   "time": round(time.time() - t0, 1)}
            log.append(rec)
            if verbose:
                short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}
                print(json.dumps(short), flush=True)
        if step % cfg["ckpt_every"] == 0 or step == end:
            save_ckpt(path, model, opt, cfg, step, baseline.tolist(), log)
    return model, world, log, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="color", choices=list(STAGES))
    ap.add_argument("--method", nargs="+", default=["gumbel"])
    ap.add_argument("--seed", type=int, default=0, help="population seed (members are seeds 0..P-1 within it)")
    ap.add_argument("--P", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--L", type=int)
    ap.add_argument("--V", type=int)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--K", type=int, help="number of candidates the receiver sees")
    ap.add_argument("--cluster_frac", type=float, help="fraction of near-miss (cluster) episodes")
    ap.add_argument("--entropy_coef", type=float)
    ap.add_argument("--gumbel_entropy_coef", type=float)
    ap.add_argument("--entropy_final", type=float, help="linearly decay the entropy bonus to this value")
    ap.add_argument("--tag", default=None, help="suffix for the run name (for variants)")
    ap.add_argument("--stop_at", type=int, help="stop early (for resume tests)")
    ap.add_argument("--ckpt_dir", default=os.path.join(HERE, "checkpoints"))
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    for method in args.method:
        cfg = default_cfg(args.stage, method, args.seed, P=args.P, steps=args.steps, lr=args.lr, L=args.L,
                          V=args.V, tau=args.tau, entropy_coef=args.entropy_coef,
                          gumbel_entropy_coef=args.gumbel_entropy_coef, tag=args.tag,
                          entropy_final=args.entropy_final, K=args.K, cluster_frac=args.cluster_frac)
        t = time.time()
        _, _, log, path = train(cfg, ckpt_dir=args.ckpt_dir, resume=not args.fresh, stop_at=args.stop_at)
        last = log[-1] if log else {}
        print(f"[done] {run_name(cfg)} train={last.get('train_acc_members')} "
              f"heldout={last.get('heldout_acc_members')} in {time.time() - t:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
