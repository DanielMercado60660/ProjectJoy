"""Train the drum VAE (MLX) with checkpoint/resume.

  ../.venv/bin/python train.py --out checkpoints/vae_cond              # full run
  ../.venv/bin/python train.py --out checkpoints/vae_cond --resume     # continue from last checkpoint
  ../.venv/bin/python train.py --no_cond --out checkpoints/vae_uncond  # unconditional ablation

Checkpoint dir holds: vae.safetensors (weights), vae_config.json, optimizer.safetensors,
state.json (epoch, step, numpy RNG state, args, loss history), and best/ = weights at the epoch with the
lowest validation negative ELBO of the hit grid (checked every 10 epochs). Evaluation uses best/.
"""
import argparse
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

import data
from model import VAE, ARVAE, recon_terms, to_input, one_hot, save_vae


def loss_fn(model, x, s, beta, key, w_vel, w_off, free_bits=0.0):
    bce, vmse, omse, kl = recon_terms(model, x, s, key)
    kl_term = mx.maximum(kl.mean(), free_bits)  # batch-level free bits
    return (bce + w_vel * vmse + w_off * omse).mean() + beta * kl_term


def evaluate(model, X, S, w_vel, w_off, batch=1024):
    tot = np.zeros(4)
    for i in range(0, len(X), batch):
        x, s = mx.array(X[i:i + batch]), mx.array(S[i:i + batch])
        terms = recon_terms(model, x, s if model.cond else None, mx.random.key(0))
        tot += np.array([float(t.sum()) for t in terms])
    return dict(zip(["bce", "vel_mse", "off_mse", "kl"], (tot / len(X)).tolist()))


def save_ckpt(out, model, opt, state):
    save_vae(model, out, {"trained_epochs": state["epoch"]})
    mx.save_safetensors(os.path.join(out, "optimizer.safetensors"), dict(tree_flatten(opt.state)))
    with open(os.path.join(out, "state.json"), "w") as f:
        json.dump(state, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/drums.npz")
    ap.add_argument("--out", default="checkpoints/vae_cond")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=50, help="epochs of linear KL warm-up")
    ap.add_argument("--w_vel", type=float, default=4.0)
    ap.add_argument("--w_off", type=float, default=4.0)
    ap.add_argument("--no_cond", action="store_true")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "ar"], help="ar = autoregressive GRU decoder")
    ap.add_argument("--rnn", type=int, default=256)
    ap.add_argument("--free_bits", type=float, default=0.0, help="total KL nats not penalised (fights posterior collapse)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_train", type=int, default=0, help="subsample train set (smoke tests)")
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop_after", type=int, default=0, help="exit after this many epochs (resume test)")
    a = ap.parse_args()

    d = data.load(a.data)
    tr, va = data.subset(d, "train"), data.subset(d, "val")
    if a.max_train:
        tr = {k: v[:a.max_train] for k, v in tr.items()}
    Xtr, Str = to_input(tr["hits"], tr["vel"], tr["off"]), one_hot(tr["style"])
    Xva, Sva = to_input(va["hits"], va["vel"], va["off"]), one_hot(va["style"])

    mx.random.seed(a.seed)
    if a.arch == "ar":
        model = ARVAE(a.latent, a.hidden, cond=not a.no_cond, rnn=a.rnn)
    else:
        model = VAE(a.latent, a.hidden, cond=not a.no_cond)
    opt = optim.Adam(learning_rate=a.lr)
    rng = np.random.default_rng(a.seed)
    state = {"epoch": 0, "step": 0, "args": vars(a), "history": []}
    if a.resume and os.path.exists(os.path.join(a.out, "state.json")):
        with open(os.path.join(a.out, "state.json")) as f:
            state = json.load(f)
        model.load_weights(os.path.join(a.out, "vae.safetensors"))
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(os.path.join(a.out, "optimizer.safetensors")).items()))
        rng.bit_generator.state = state["rng"]
        print(f"resumed from epoch {state['epoch']} step {state['step']}")
    mx.eval(model.parameters())

    lg = nn.value_and_grad(model, loss_fn)

    t0 = time.time()
    n = len(Xtr)
    while state["epoch"] < a.epochs:
        ep = state["epoch"]
        beta = a.beta * min(1.0, (ep + 1) / max(1, a.warmup))
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            x, s = mx.array(Xtr[idx]), mx.array(Str[idx])
            key = mx.random.key(a.seed * 1_000_003 + state["step"])
            loss, grads = lg(model, x, s if model.cond else None, beta, key, a.w_vel, a.w_off, a.free_bits)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            tot += float(loss) * len(idx)
            state["step"] += 1
        state["epoch"] += 1
        if state["epoch"] % 10 == 0 or state["epoch"] == a.epochs:
            model.eval()
            v = evaluate(model, Xva, Sva, a.w_vel, a.w_off)
            model.train()
            rec = {"epoch": state["epoch"], "step": state["step"], "train_loss": tot / n, "beta": beta, **{f"val_{k}": x for k, x in v.items()}}
            state["history"].append(rec)
            elbo = v["bce"] + v["kl"]  # hit-grid negative ELBO on val, comparable across beta
            if elbo < state.get("best_val_elbo", float("inf")):
                state["best_val_elbo"], state["best_epoch"] = elbo, state["epoch"]
                save_vae(model, os.path.join(a.out, "best"), {"trained_epochs": state["epoch"], "val_neg_elbo": elbo})
            print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rec.items()}),
                  f"{time.time() - t0:.0f}s", flush=True)
        if state["epoch"] % a.ckpt_every == 0 or state["epoch"] == a.epochs:
            state["rng"] = rng.bit_generator.state
            save_ckpt(a.out, model, opt, state)
        if a.stop_after and state["epoch"] >= a.stop_after:
            state["rng"] = rng.bit_generator.state
            save_ckpt(a.out, model, opt, state)
            print(f"stopping early at epoch {state['epoch']} (--stop_after)")
            return
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
