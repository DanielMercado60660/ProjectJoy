"""Train the learned heuristic (MLX U-Net) on exact cost-to-go labels.

  ../bin/gpu_run ../.venv/bin/python train.py --steps 6000            # fresh
  ../bin/gpu_run ../.venv/bin/python train.py --steps 6000 --resume   # continue from ckpt

Checkpoint (checkpoints/hnet.*): weights (safetensors), optimizer state, step, RNG state,
config (json). Saved every --ckpt_every steps and at the end.
"""
import argparse, json, os, pickle, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from terra import learn

CK = "checkpoints"


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def make_batch(D, idx, rng, augment):
    st = D["static"][D["map_index"][idx]].astype(np.float32)  # B,6,H,W
    B, _, H, W = st.shape
    gf = np.stack([learn.goal_features(H, W, *D["goal"][i]) for i in idx])
    x = np.concatenate([st, gf], axis=1)
    y = D["target"][idx].astype(np.float32)
    m = D["mask"][idx].astype(np.float32)
    if augment:  # dihedral symmetries leave the cost definition invariant
        k = rng.integers(4)
        x, y, m = np.rot90(x, k, (2, 3)), np.rot90(y, k, (1, 2)), np.rot90(m, k, (1, 2))
        if rng.random() < 0.5:
            x, y, m = x[..., ::-1], y[..., ::-1], m[..., ::-1]
    x = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
    return mx.array(x), mx.array(np.ascontiguousarray(y)), mx.array(np.ascontiguousarray(m))


def loss_fn(model, x, y, m):
    p = model(x)
    err = (p - y) ** 2
    return (err * m).sum() / mx.maximum(m.sum(), 1.0)


def evaluate(model, D, bs=32):
    n = len(D["map_index"])
    tot, cnt, over = 0.0, 0.0, 0.0
    for i in range(0, n, bs):
        idx = np.arange(i, min(n, i + bs))
        x, y, m = make_batch(D, idx, None, False)
        p = mx.maximum(model(x), 0.0)
        e = mx.abs(p - y) * m
        tot += float(e.sum()); cnt += float(m.sum())
        over += float(((p > y + 0.01) * m).sum())
    return tot / cnt, over / cnt


def save(model, opt, step, rng, cfg, tag="hnet"):
    os.makedirs(CK, exist_ok=True)
    model.save_weights(f"{CK}/{tag}.safetensors")
    with open(f"{CK}/{tag}_state.pkl", "wb") as f:
        pickle.dump({"opt": {k: np.array(v) for k, v in tree_flatten(opt.state)},
                     "step": step, "rng": rng.bit_generator.state}, f)
    with open(f"{CK}/{tag}.json", "w") as f:
        json.dump(cfg, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--data", default="data")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--tag", default="hnet")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = vars(a).copy()

    mx.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tr, va = load(f"{a.data}/train.npz"), load(f"{a.data}/val.npz")
    print(f"train samples {len(tr['map_index'])}, val {len(va['map_index'])}", flush=True)

    model = learn.HNet(base=a.base)
    sched = optim.cosine_decay(a.lr, a.steps, end=a.lr * 0.02)
    opt = optim.AdamW(learning_rate=sched, weight_decay=1e-4)
    step = 0
    if a.resume and os.path.exists(f"{CK}/{a.tag}_state.pkl"):
        model.load_weights(f"{CK}/{a.tag}.safetensors")
        with open(f"{CK}/{a.tag}_state.pkl", "rb") as f:
            st = pickle.load(f)
        opt.state = tree_unflatten([(k, mx.array(v)) for k, v in st["opt"].items()])
        step = st["step"]
        rng.bit_generator.state = st["rng"]
        print(f"resumed at step {step}", flush=True)
    mx.eval(model.parameters())
    nparams = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"params {nparams}", flush=True)
    cfg["params"] = int(nparams)

    lg = nn.value_and_grad(model, loss_fn)

    @mx.compile
    def _noop(x):
        return x

    n = len(tr["map_index"])
    t0 = time.time()
    run = 0.0
    while step < a.steps:
        idx = rng.integers(0, n, a.bs)
        x, y, m = make_batch(tr, idx, rng, True)
        loss, g = lg(model, x, y, m)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state, loss)
        step += 1
        run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
        if step % 100 == 0:
            print(f"step {step} loss {run:.5f} {time.time() - t0:.0f}s", flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            mae, over = evaluate(model, va)
            print(f"  val |logratio err| {mae:.4f}  frac overestimate {over:.3f}", flush=True)
            cfg.update(step=step, val_mae=mae, val_over=over)
            save(model, opt, step, rng, cfg, a.tag)
    print("done", flush=True)


if __name__ == "__main__":
    main()
