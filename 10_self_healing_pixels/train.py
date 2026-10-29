"""Train a neural cellular automaton to grow (and optionally regenerate) a target image.

Modes (Distill 2020 "experiments"):
  grow     - every batch starts from the seed (no pool)                      [Exp. 1]
  persist  - sample pool, no damage  ("growth-only" baseline for this project) [Exp. 2]
  regen    - sample pool + circular damage on the 3 best samples per batch     [Exp. 3]

Example:
  ../bin/gpu_run ../.venv/bin/python train.py --mode regen --steps 8000 --out checkpoints/lizard_regen
Resume:   same command + --resume
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map

import nca
import damage


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="lizard")
    p.add_argument("--mode", choices=["grow", "persist", "regen"], default="regen")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--pool", type=int, default=1024)
    p.add_argument("--n_damage", type=int, default=3)
    p.add_argument("--min_iter", type=int, default=64)
    p.add_argument("--max_iter", type=int, default=96)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr_decay_at", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=nca.TARGET_SIZE)
    p.add_argument("--pad", type=int, default=nca.TARGET_PAD)
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop_after", type=int, default=0, help="debug: exit after this many steps (for resume tests)")
    return p.parse_args()


def per_sample_loss_np(x, target):
    return ((x[..., :4] - target) ** 2).mean(axis=(1, 2, 3))


def save_ckpt(out, model, opt, step, pool, rng, cfg, history):
    tmp = os.path.join(out, "ckpt_tmp")
    os.makedirs(tmp, exist_ok=True)
    model.save_weights(os.path.join(tmp, "model.safetensors"))
    mx.save_safetensors(os.path.join(tmp, "opt.safetensors"), dict(tree_flatten(opt.state)))
    if pool is not None:
        np.save(os.path.join(tmp, "pool.npy"), pool)
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump({"step": step, "config": cfg, "np_rng": rng.bit_generator.state, "history": history}, f)
    final = os.path.join(out, "ckpt")
    if os.path.exists(final):
        import shutil
        shutil.rmtree(final)
    os.replace(tmp, final)


def main():
    a = parse()
    cfg = vars(a).copy()
    os.makedirs(a.out, exist_ok=True)
    target_np = nca.load_target(a.target, a.size, a.pad)
    H, W = target_np.shape[:2]
    target = mx.array(target_np)

    model = nca.NCA()
    opt = optim.Adam(learning_rate=a.lr)
    rng = np.random.default_rng(a.seed)
    mx.random.seed(a.seed)
    seed_state = nca.make_seed(H, W)[0]
    pool = None if a.mode == "grow" else np.repeat(seed_state[None], a.pool, 0)
    start, history = 0, []

    ck = os.path.join(a.out, "ckpt")
    if a.resume and os.path.exists(ck):
        model.load_weights(os.path.join(ck, "model.safetensors"))
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(os.path.join(ck, "opt.safetensors")).items()))
        meta = json.load(open(os.path.join(ck, "meta.json")))
        start, history = meta["step"], meta["history"]
        rng.bit_generator.state = meta["np_rng"]
        mx.random.seed(a.seed + start)  # MLX key re-derived from (seed, step) on resume
        if pool is not None:
            pool = np.load(os.path.join(ck, "pool.npy"))
        print(f"resumed from step {start}", flush=True)

    def loss_fn(model, x, n):
        x = model.rollout(x, n)
        l = ((x[..., :4] - target) ** 2).mean(axis=(1, 2, 3))
        return l.mean(), x

    vg = nn.value_and_grad(model, loss_fn)
    t0 = time.time()
    for step in range(start, a.steps):
        opt.learning_rate = a.lr if step < a.lr_decay_at else a.lr * 0.1
        if pool is None:
            idx = None
            x0 = np.repeat(seed_state[None], a.batch, 0)
        else:
            idx = rng.choice(a.pool, a.batch, replace=False)
            x0 = pool[idx]
            order = np.argsort(-per_sample_loss_np(x0, target_np))  # worst first
            idx, x0 = idx[order], x0[order]
            x0[0] = seed_state  # replace worst sample with a fresh seed
            if a.mode == "regen":
                for j in range(a.batch - a.n_damage, a.batch):  # damage the best ones
                    x0[j] = damage.apply(x0[j], damage.circle(H, W, rng))
        n = int(rng.integers(a.min_iter, a.max_iter + 1))
        (loss, xf), grads = vg(model, mx.array(x0), n)
        grads = tree_map(lambda g: g / (mx.linalg.norm(g) + 1e-8), grads)  # per-variable grad normalisation
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss, xf)
        if pool is not None:
            pool[idx] = np.array(xf)
        lv = float(loss)
        history.append([step + 1, lv])
        if not np.isfinite(lv):
            print(f"step {step+1}: non-finite loss, stopping", flush=True)
            break
        if (step + 1) % a.log_every == 0 or step == start:
            el = time.time() - t0
            print(f"step {step+1:6d} loss {lv:.5f} log10 {np.log10(lv):.3f} n={n} lr={opt.learning_rate.item():.1e} "
                  f"{el:.0f}s ({el/(step+1-start):.3f}s/it)", flush=True)
        if (step + 1) % a.ckpt_every == 0 or step + 1 == a.steps:
            save_ckpt(a.out, model, opt, step + 1, pool, rng, cfg, history)
        if a.stop_after and step + 1 - start >= a.stop_after:
            save_ckpt(a.out, model, opt, step + 1, pool, rng, cfg, history)
            print(f"stop_after reached at step {step+1}", flush=True)
            return
    model.save_weights(os.path.join(a.out, "model.safetensors"))
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    np.savetxt(os.path.join(a.out, "loss.csv"), np.array(history), delimiter=",", header="step,loss")
    print(f"done in {time.time()-t0:.0f}s, final loss {history[-1][1]:.5f}", flush=True)


if __name__ == "__main__":
    main()
