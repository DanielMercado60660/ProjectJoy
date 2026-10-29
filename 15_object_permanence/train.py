"""Train the CNN+GRU Gaussian tracker with Gaussian NLL on rendered-on-the-fly frames.

Usage:
  ../.venv/bin/python train.py --variant base --steps 12000 --out checkpoints/base
  ../.venv/bin/python train.py --variant hard --steps 12000 --out checkpoints/hard
Resumes automatically from <out>/last.npz if present.
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

import sim
from model import POS_SCALE, Tracker, gaussian_nll

H = W = sim.H
_YY = mx.array(np.mgrid[0:H, 0:W][0].astype(np.float32) + 0.5)
_XX = mx.array(np.mgrid[0:H, 0:W][1].astype(np.float32) + 0.5)
_XC = mx.array(np.arange(W, dtype=np.float32) + 0.5)


def render_mx(pos, present, occ, colors, noise=0.02):
    """GPU twin of sim.render (same math). Inputs are mx arrays."""
    B, T, K = pos.shape[:3]
    img = mx.broadcast_to(mx.array(sim.BG), (B, T, H, W, 3))
    for k in range(K):
        dx = _XX - pos[:, :, k, 0, None, None]
        dy = _YY - pos[:, :, k, 1, None, None]
        a = mx.clip(sim.R + 0.5 - mx.sqrt(dx * dx + dy * dy), 0, 1)
        a = a * present[:, None, k, None, None]
        img = img * (1 - a[..., None]) + a[..., None] * colors[:, None, None, None, k]
    for m in range(occ.shape[1]):
        x0, x1 = occ[:, m, 0:1], occ[:, m, 1:2]
        cover = mx.clip(mx.minimum(_XC + 0.5, x1) - mx.maximum(_XC - 0.5, x0), 0, 1)
        cover = cover * (x0 >= 0)
        cover = cover[:, None, None, :, None]
        img = img * (1 - cover) + cover * 0.45
    if noise > 0:
        img = img + noise * mx.random.normal(img.shape)
    return mx.clip(img, 0, 1)


def augment(pos, occ, present, colors, rng):
    """Random horizontal / vertical mirror of whole sequences (physics is mirror-symmetric).
    For identical-looking balls, slot identity is "k-th ball from the left at frame 0", so after a
    horizontal flip the slots are re-sorted to keep that convention (an earlier run without this
    re-sort trained on contradictory labels - see RESULTS.md)."""
    pos, occ = pos.copy(), occ.copy()
    fx = rng.random(len(pos)) < 0.5
    fy = rng.random(len(pos)) < 0.5
    pos[fx, ..., 0] = W - pos[fx, ..., 0]
    pos[fy, ..., 1] = H - pos[fy, ..., 1]
    o = occ[fx]
    valid = o[..., 0] >= 0
    flipped = np.stack([W - o[..., 1], W - o[..., 0]], -1)
    o = np.where(valid[..., None], flipped, o)
    occ[fx] = o
    same = np.all(np.isclose(colors[:, 0], colors[:, 1]), -1)
    for i in np.where(fx & same)[0]:
        nb = int(present[i].sum())
        order = np.argsort(pos[i, 0, :nb, 0])
        pos[i, :, :nb] = pos[i, :, order].transpose(1, 0, 2)
    return pos, occ


def save_ckpt(path, model, opt, step, rng, cfg, hist):
    tmp = path + ".tmp"
    w = dict(tree_flatten(model.parameters()))
    mx.savez(tmp + ".weights.npz", **w)
    os.replace(tmp + ".weights.npz", path + ".weights.npz")
    state = dict(step=step, rng=rng.bit_generator.state, cfg=cfg, hist=hist,
                 opt={k: np.array(v) for k, v in tree_flatten(opt.state)})
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path + ".state.pkl")


def load_model(path_prefix, cfg=None):
    if cfg is None:
        with open(path_prefix + ".state.pkl", "rb") as f:
            cfg = pickle.load(f)["cfg"]
    m = Tracker(K=sim.K, enc_dim=cfg["enc_dim"], hid=cfg["hid"], layers=cfg["layers"])
    m.load_weights(path_prefix + ".weights.npz")
    mx.eval(m.parameters())
    return m, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base", choices=["base", "hard"])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--enc_dim", type=int, default=512)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--n_train", type=int, default=0, help="use first n train seqs (0=all)")
    ap.add_argument("--train_split", default="", help="override training split (smoke test uses val)")
    ap.add_argument("--out", default="checkpoints/base")
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, default=0.5, help="beta-NLL exponent (0 = plain NLL)")
    ap.add_argument("--aux_frac", type=float, default=0.5, help="MSE warm-up decays to 0 over this fraction")
    ap.add_argument("--stop_after", type=int, default=0, help="exit after this many steps (resume test)")
    ap.add_argument("--max_minutes", type=float, default=0, help="wall-clock guard: save final and stop")
    args = ap.parse_args()
    cfg = vars(args).copy()
    os.makedirs(args.out, exist_ok=True)
    pre = "" if args.variant == "base" else "hard_"
    tr = sim.load_split(args.train_split or (pre + "train"))
    va = sim.load_split(pre + "val")
    ntr = args.n_train or len(tr["pos"])

    mx.random.seed(args.seed)
    model = Tracker(K=sim.K, enc_dim=args.enc_dim, hid=args.hid, layers=args.layers)
    mx.eval(model.parameters())
    sched = optim.join_schedules([optim.linear_schedule(1e-5, args.lr, 300),
                                  optim.cosine_decay(args.lr, max(args.steps - 300, 1), args.lr * 0.05)], [300])
    opt = optim.AdamW(learning_rate=sched, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    step, hist = 0, []
    last = os.path.join(args.out, "last")
    if os.path.exists(last + ".state.pkl"):
        with open(last + ".state.pkl", "rb") as f:
            st = pickle.load(f)
        model.load_weights(last + ".weights.npz")
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten([(k, mx.array(v)) for k, v in st["opt"].items()])
        step, hist = st["step"], st["hist"]
        rng.bit_generator.state = st["rng"]
        mx.random.seed(args.seed + step)
        print(f"resumed from step {step}", flush=True)
    nparams = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"params={nparams/1e6:.2f}M train={ntr} cfg={cfg}", flush=True)

    def loss_fn(m, frames, tgt, mask, aux_w):
        # beta-NLL (Seitzer et al. 2022): weight each element's NLL by stop_grad(sigma^(2*beta)).
        # Plain NLL got stuck in a local optimum where 2 of 3 slots "give up" (huge sigma, mean
        # ignored) - see RESULTS.md.  Plus a mean-squared-error warm-up term that decays to 0.
        out = m(frames)
        nll = gaussian_nll(out, tgt)
        if args.beta > 0:
            ls = mx.clip(out[..., 2:4], -5.5, 1.0)
            nll = nll * mx.stop_gradient(mx.exp(args.beta * (ls[..., 0] + ls[..., 1])))
        mse = ((out[..., :2] - tgt) ** 2).sum(-1) * 100.0   # normalized units -> (3.2 px)^2 units
        # normalize per slot so the often-absent slot 2 gets the same gradient share as slots 0/1
        per_slot = ((nll + aux_w * mse) * mask).sum((0, 1)) / mx.maximum(mask.sum((0, 1)), 1.0)
        return per_slot.mean()

    lg = nn.value_and_grad(model, loss_fn)

    def batch(d, idx, aug):
        pos, occ = d["pos"][idx], d["occ"][idx]
        if aug:
            pos, occ = augment(pos, occ, d["present"][idx], d["colors"][idx], rng)
        pres = d["present"][idx]
        frames = render_mx(mx.array(pos), mx.array(pres.astype(np.float32)), mx.array(occ),
                           mx.array(d["colors"][idx]))
        tgt = mx.array((pos - 32.0) / POS_SCALE)
        mask = mx.array(np.broadcast_to(pres[:, None, :], pos.shape[:3]).astype(np.float32))
        return frames, tgt, mask

    vidx = np.arange(256)
    t0 = time.time()
    run_loss = None
    t_start = time.time()
    while step < args.steps:
        if args.max_minutes and time.time() - t_start > args.max_minutes * 60:
            print(f"max_minutes reached at step {step}; saving final", flush=True)
            break
        idx = rng.integers(0, ntr, args.batch)
        frames, tgt, mask = batch(tr, idx, True)
        aux_w = max(0.0, 1.0 - step / max(1, args.aux_frac * args.steps)) if args.aux_frac > 0 else 0.0
        loss, g = lg(model, frames, tgt, mask, aux_w)
        g, gn = optim.clip_grad_norm(g, 5.0)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state, loss)
        step += 1
        lv = loss.item()
        run_loss = lv if run_loss is None else 0.98 * run_loss + 0.02 * lv
        if step % 100 == 0:
            print(f"step {step} loss {run_loss:.3f} gn {gn.item():.2f} lr {opt.learning_rate.item():.2e} "
                  f"{(time.time()-t0)/100*1000:.0f}ms/step", flush=True)
            t0 = time.time()
        if step % args.ckpt_every == 0 or step == args.steps:
            vl, verr, vslot = [], [], []
            for i in range(0, len(vidx), 64):
                f, tg, mk = batch(va, vidx[i:i + 64], False)
                out = model(f)
                vl.append(((gaussian_nll(out, tg) * mk).sum() / mk.sum()).item())
                err = mx.sqrt(((out[..., :2] - tg) ** 2).sum(-1)) * POS_SCALE
                verr.append(((err * mk).sum() / mk.sum()).item())
                vslot.append(np.array((err * mk).sum((0, 1)) / mx.maximum(mk.sum((0, 1)), 1)))
            vs = np.mean(vslot, 0)
            hist.append(dict(step=step, train_loss=run_loss, val_nll=float(np.mean(vl)),
                             val_err_px=float(np.mean(verr)), val_err_slot=vs.tolist()))
            print(f"[val] step {step} nll {np.mean(vl):.3f} err {np.mean(verr):.2f}px "
                  f"per-slot {np.round(vs, 2).tolist()}", flush=True)
            save_ckpt(last, model, opt, step, rng, cfg, hist)
            with open(os.path.join(args.out, "history.json"), "w") as f:
                json.dump(hist, f, indent=1)
            if args.stop_after and step >= args.stop_after:
                print("stop_after reached", flush=True)
                return
    save_ckpt(os.path.join(args.out, "final"), model, opt, step, rng, cfg, hist)
    print("done", flush=True)


if __name__ == "__main__":
    main()
