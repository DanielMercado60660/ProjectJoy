"""AlphaZero-style training loop for Connect Four.

Each generation:
  1. self-play `games_per_gen` games with the current net + batched PUCT MCTS
     (`concurrent` games searched in lock-step so the GPU sees big batches),
  2. push (position, visit-distribution, outcome) into a replay buffer,
  3. train for a number of minibatch steps (random horizontal-flip augmentation),
  4. save a generational checkpoint (weights) + a full resumable state.

Usage:
  ../bin/gpu_run ../.venv/bin/python train.py --run main --minutes 44        # fresh or resume
  ../.venv/bin/python train.py --run smoke --tiny --gens 2                     # smoke test
"""
import argparse
import json
import os
import pickle
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

import c4
from c4 import W
from mcts import BatchMCTS
from net import PVNet, make_evaluator, n_params, save_model

HERE = os.path.dirname(os.path.abspath(__file__))


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--gens", type=int, default=10_000, help="stop after this many generations total")
    ap.add_argument("--minutes", type=float, default=44.0, help="wall-clock budget for this invocation")
    ap.add_argument("--games-per-gen", type=int, default=512)
    ap.add_argument("--concurrent", type=int, default=512)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--dir-alpha", type=float, default=1.0)
    ap.add_argument("--dir-eps", type=float, default=0.25)
    ap.add_argument("--temp-moves", type=int, default=10, help="sample moves ~ visits for this many plies")
    ap.add_argument("--random-plies", type=int, default=0,
                    help="start each self-play game with k ~ U{0..N} uniformly random plies (not recorded)")
    ap.add_argument("--buffer", type=int, default=250_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--sample-ratio", type=float, default=4.0, help="train samples per new position")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--no-flip", action="store_true", help="disable symmetry augmentation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=5, help="inline eval vs greedy/ab4/uct1000 every K gens")
    ap.add_argument("--eval-opps", default="greedy,ab4,uct1000")
    ap.add_argument("--tiny", action="store_true", help="tiny sizes for a smoke test")
    a = ap.parse_args(argv)
    if a.tiny:
        a.games_per_gen, a.concurrent, a.sims = 16, 16, 16
        a.channels, a.blocks, a.batch, a.buffer = 16, 1, 64, 5000
    return a


# ----------------------------------------------------------------------------- self-play
def random_start(rng, max_plies):
    """k ~ U{0..max_plies} uniformly random legal plies (never a winning one)."""
    pos, mask = np.int64(0), np.int64(0)
    for _ in range(int(rng.integers(0, max_plies + 1))):
        legal = [c for c in range(W) if c4.can_play(mask, c) and not c4.winning_move(pos, mask, c)]
        pos, mask = c4.play(pos, mask, int(rng.choice(legal)))
    return pos, mask


def self_play(evaluate, mcts, n_games, concurrent, temp_moves, rng, random_plies=0):
    """Returns arrays (pos, mask, pi, z) for every position played, plus stats."""
    G = min(concurrent, n_games)
    pos = np.zeros(G, np.int64)
    mask = np.zeros(G, np.int64)
    for g in range(G):
        pos[g], mask[g] = random_start(rng, random_plies)
    ply = np.array([c4.popcount(m) for m in mask], np.int64)
    active = np.ones(G, bool)
    hist = [[] for _ in range(G)]
    started = G
    out_pos, out_mask, out_pi, out_z = [], [], [], []
    results = []   # +1 first player won, -1 second, 0 draw
    lengths = []
    while active.any():
        act = np.nonzero(active)[0]
        visits, _, _ = mcts.search(pos[act], mask[act], evaluate, add_noise=True)
        for k, g in enumerate(act):
            v = visits[k].astype(np.float64)
            pi = v / v.sum()
            hist[g].append((pos[g], mask[g], pi.astype(np.float32)))
            if ply[g] < temp_moves:
                a = rng.choice(W, p=pi)
            else:
                best = np.flatnonzero(v == v.max())
                a = rng.choice(best)
            won = c4.winning_move(pos[g], mask[g], a)
            pos[g], mask[g] = c4.play(pos[g], mask[g], a)
            ply[g] += 1
            if won or c4.is_full(mask[g]):
                # z from the perspective of the player to move at each recorded position
                n = len(hist[g])
                for i, (p, m, pi_i) in enumerate(hist[g]):
                    if won:
                        z = 1.0 if (n - 1 - i) % 2 == 0 else -1.0
                    else:
                        z = 0.0
                    out_pos.append(p); out_mask.append(m); out_pi.append(pi_i); out_z.append(z)
                results.append(0 if not won else (1 if (ply[g] - 1) % 2 == 0 else -1))
                lengths.append(n)
                hist[g] = []
                if started < n_games:
                    pos[g], mask[g] = random_start(rng, random_plies)
                    ply[g] = c4.popcount(mask[g])
                    started += 1
                else:
                    active[g] = False
    stats = dict(games=len(results), positions=len(out_z),
                 first_win=float(np.mean(np.array(results) == 1)),
                 draws=float(np.mean(np.array(results) == 0)),
                 avg_len=float(np.mean(lengths)))
    return (np.array(out_pos, np.int64), np.array(out_mask, np.int64),
            np.stack(out_pi), np.array(out_z, np.float32)), stats


# ----------------------------------------------------------------------------- replay
class Replay:
    def __init__(self, cap):
        self.cap = cap
        self.pos = np.zeros(cap, np.int64)
        self.mask = np.zeros(cap, np.int64)
        self.pi = np.zeros((cap, W), np.float32)
        self.z = np.zeros(cap, np.float32)
        self.n = 0
        self.ptr = 0

    def add(self, pos, mask, pi, z):
        for i in range(len(z)):
            j = self.ptr
            self.pos[j], self.mask[j], self.pi[j], self.z[j] = pos[i], mask[i], pi[i], z[i]
            self.ptr = (j + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def sample(self, b, rng, flip=True):
        idx = rng.integers(0, self.n, b)
        x = c4.features_batch(self.pos[idx], self.mask[idx])
        pi = self.pi[idx].copy()
        z = self.z[idx]
        if flip:
            f = rng.random(b) < 0.5
            x[f] = x[f][:, :, ::-1, :]
            pi[f] = pi[f][:, ::-1]
        return x, pi, z

    def state(self):
        return dict(pos=self.pos[:self.n] if self.n < self.cap else self.pos, mask=self.mask[:self.n] if self.n < self.cap else self.mask,
                    pi=self.pi[:self.n] if self.n < self.cap else self.pi, z=self.z[:self.n] if self.n < self.cap else self.z,
                    n=self.n, ptr=self.ptr)

    def load(self, d):
        n = int(d["n"])
        self.pos[:n], self.mask[:n], self.pi[:n], self.z[:n] = d["pos"][:n], d["mask"][:n], d["pi"][:n], d["z"][:n]
        self.n, self.ptr = n, int(d["ptr"])


# ----------------------------------------------------------------------------- training
def loss_fn(model, x, pi, z):
    logits, v = model(x)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    pl = -mx.mean(mx.sum(pi * logp, axis=-1))
    vl = mx.mean(mx.square(v - z))
    return pl + vl, (pl, vl)


def train_steps(model, opt, replay, steps, batch, rng, flip):
    model.train()
    vg = nn.value_and_grad(model, loss_fn)
    pls, vls = [], []
    for _ in range(steps):
        x, pi, z = replay.sample(batch, rng, flip)
        (loss, (pl, vl)), grads = vg(model, mx.array(x), mx.array(pi), mx.array(z))
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, pl, vl)
        pls.append(pl.item()); vls.append(vl.item())
    model.eval()
    return float(np.mean(pls)) if pls else float("nan"), float(np.mean(vls)) if vls else float("nan")


# ----------------------------------------------------------------------------- checkpoints
def run_dir(run):
    return os.path.join(HERE, "checkpoints", run)


def save_state(d, model, opt, replay, gen, rng, args, history):
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, "state_tmp")
    os.makedirs(tmp, exist_ok=True)
    model.save_weights(os.path.join(tmp, "model.safetensors"))
    mx.save_safetensors(os.path.join(tmp, "opt.safetensors"), dict(tree_flatten(opt.state)))
    np.savez(os.path.join(tmp, "replay.npz"), **replay.state())
    with open(os.path.join(tmp, "meta.pkl"), "wb") as f:
        pickle.dump(dict(gen=gen, rng=rng.bit_generator.state, args=vars(args), history=history), f)
    final = os.path.join(d, "state")
    if os.path.exists(final):
        import shutil
        shutil.rmtree(final)
    os.rename(tmp, final)


def load_state(d, model, opt, replay):
    s = os.path.join(d, "state")
    model.load_weights(os.path.join(s, "model.safetensors"))
    ost = mx.load(os.path.join(s, "opt.safetensors"))
    opt.state = tree_unflatten(list(ost.items()))
    replay.load(dict(np.load(os.path.join(s, "replay.npz"))))
    with open(os.path.join(s, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return meta


def inline_eval(model, args):
    """Current net (+100-sim MCTS, no noise) vs a few cheap fixed opponents, 98 games each
    (49 two-ply openings x both colours)."""
    import arena
    ev = make_evaluator(model)
    sims = 16 if args.tiny else 100
    agent = arena.MCTSAgent("net", ev, sims, max_games=128)
    out = {}
    t = time.time()
    for opp in args.eval_opps.split(","):
        r = arena.play_match(agent, arena.make_agent(opp), openings=arena.ALL_2PLY[:4] if args.tiny else None)
        out[opp] = round(r["score"], 3)
    out["eval_s"] = round(time.time() - t, 1)
    return out


def main(argv=None):
    args = get_args(argv)
    t_start = time.time()
    d = run_dir(args.run)
    os.makedirs(d, exist_ok=True)
    log_path = os.path.join(HERE, "logs", f"{args.run}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    mx.random.seed(args.seed)
    model = PVNet(args.channels, args.blocks)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.wd)
    opt.init(model.trainable_parameters())
    replay = Replay(args.buffer)
    rng = np.random.default_rng(args.seed)
    gen, history = 0, []
    if os.path.exists(os.path.join(d, "state")):
        meta = load_state(d, model, opt, replay)
        gen, history = meta["gen"], meta["history"]
        rng.bit_generator.state = meta["rng"]
        opt.learning_rate = args.lr
        print(f"resumed {args.run} at gen {gen} (replay {replay.n})", flush=True)
    else:
        save_model(model, os.path.join(d, "gen_000"), extra=dict(gen=0))
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=1)
    print(f"params {n_params(model):,}  args {vars(args)}", flush=True)

    mcts = BatchMCTS(min(args.concurrent, args.games_per_gen), args.sims, c_puct=args.c_puct,
                     dir_alpha=args.dir_alpha, dir_eps=args.dir_eps)
    gen_times = []
    while gen < args.gens:
        elapsed = time.time() - t_start
        est = np.mean(gen_times[-3:]) if gen_times else 0
        if elapsed + est > args.minutes * 60:
            print(f"time budget reached after {elapsed/60:.1f} min", flush=True)
            break
        tg = time.time()
        c4.seed_numba(args.seed * 100003 + gen)
        evaluate = make_evaluator(model)
        mcts.n_evals = mcts.n_batches = 0
        t0 = time.time()
        (p, m, pi, z), st = self_play(evaluate, mcts, args.games_per_gen, args.concurrent,
                                      args.temp_moves, rng, args.random_plies)
        t_sp = time.time() - t0
        replay.add(p, m, pi, z)
        steps = int(np.ceil(len(z) * args.sample_ratio / args.batch))
        t0 = time.time()
        pl, vl = train_steps(model, opt, replay, steps, args.batch, rng, not args.no_flip)
        t_tr = time.time() - t0
        gen += 1
        save_model(model, os.path.join(d, f"gen_{gen:03d}"), extra=dict(gen=gen))
        rec = dict(gen=gen, t_selfplay=round(t_sp, 2), t_train=round(t_tr, 2), steps=steps,
                   evals_per_s=round(mcts.n_evals / t_sp), avg_batch=round(mcts.n_evals / max(1, mcts.n_batches), 1),
                   policy_loss=round(pl, 4), value_loss=round(vl, 4), replay=replay.n,
                   wall_min=round((time.time() - t_start) / 60, 2), **st)
        history.append(rec)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if args.eval_every and gen % args.eval_every == 0:
            rec["eval"] = inline_eval(model, args)
            print("eval", json.dumps(rec["eval"]), flush=True)
            with open(os.path.join(HERE, "logs", f"{args.run}_eval.jsonl"), "a") as f:
                f.write(json.dumps(dict(gen=gen, **rec["eval"])) + "\n")
        save_state(d, model, opt, replay, gen, rng, args, history)
        gen_times.append(time.time() - tg)
    print("done", flush=True)


if __name__ == "__main__":
    main()
