"""Train the tiny U-Net on training-family scenes. Validation uses seen families only (val split);
the held-out families are never touched here."""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import model as M


def load_split(path):
    d = np.load(path)
    return d["X"], d["Y"]


def augment(xb, yb, rng):
    """Flips / rot90 (geometry) and channel permutation + per-channel gain (colour)."""
    k = rng.integers(4)
    xb, yb = np.rot90(xb, k, axes=(1, 2)), np.rot90(yb, k, axes=(1, 2))
    if rng.random() < 0.5:
        xb, yb = xb[:, :, ::-1], yb[:, :, ::-1]
    xb = xb[..., rng.permutation(3)]
    gain = rng.uniform(0.8, 1.2, size=(xb.shape[0], 1, 1, 3)).astype(np.float32)
    return np.clip(xb * gain, 0, 1), yb


def loss_fn(model, x, y):
    logits = model(x)
    bce = nn.losses.binary_cross_entropy(logits, y, with_logits=True, reduction="mean")
    p = mx.sigmoid(logits)
    inter = (p * y).sum(axis=(1, 2))
    dice = 1 - (2 * inter + 1) / (p.sum(axis=(1, 2)) + y.sum(axis=(1, 2)) + 1)
    return bce + dice.mean()


def predict(model, X, bs=64):
    out = []
    for i in range(0, len(X), bs):
        x = mx.array(X[i:i + bs].astype(np.float32) / 255)
        out.append(np.array(mx.sigmoid(model(x))))
    return np.concatenate(out)


def iou(prob, Y, thr=0.5):
    p, g = prob > thr, Y > 0
    inter = (p & g).sum((1, 2))
    union = (p | g).sum((1, 2))
    return float(np.mean(np.where(union > 0, inter / np.maximum(union, 1), 1.0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--max_minutes", type=float, default=19.0, help="hard wall-clock stop")
    ap.add_argument("--n_train", type=int, default=0, help="use only first n train scenes (0=all)")
    ap.add_argument("--out", default="checkpoints/unet")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    mx.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    Xtr, Ytr = load_split(os.path.join(a.data, "train.npz"))
    if a.n_train:
        Xtr, Ytr = Xtr[:a.n_train], Ytr[:a.n_train]
    Xva, Yva = load_split(os.path.join(a.data, "val.npz"))
    net = M.UNet(width=a.width, depth=a.depth)
    print(f"params {M.n_params(net):,}  train scenes {len(Xtr)}  val {len(Xva)}", flush=True)
    sched = optim.join_schedules([optim.linear_schedule(1e-5, a.lr, 200),
                                  optim.cosine_decay(a.lr, max(1, a.steps - 200), 1e-5)], [200])
    opt = optim.AdamW(learning_rate=sched, weight_decay=1e-4)
    step_fn = nn.value_and_grad(net, loss_fn)

    def train_step(x, y):
        loss, g = step_fn(net, x, y)
        opt.update(net, g)
        return loss

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    log = []
    t0, best = time.time(), -1
    run_loss = None
    for step in range(1, a.steps + 1):
        idx = rng.integers(0, len(Xtr), a.bs)
        xb, yb = augment(Xtr[idx].astype(np.float32) / 255, (Ytr[idx] > 0).astype(np.float32), rng)
        loss = train_step(mx.array(np.ascontiguousarray(xb)), mx.array(np.ascontiguousarray(yb)))
        mx.eval(net.parameters(), opt.state, loss)
        run_loss = float(loss) if run_loss is None else 0.98 * run_loss + 0.02 * float(loss)
        elapsed = (time.time() - t0) / 60
        last = step == a.steps or elapsed > a.max_minutes
        if step % a.eval_every == 0 or last or step == 50:
            net.eval()
            v = iou(predict(net, Xva), Yva)
            net.train()
            log.append(dict(step=step, loss=run_loss, val_iou=v, minutes=elapsed))
            print(f"step {step:5d} loss {run_loss:.4f} val_iou {v:.4f} {elapsed:.1f} min", flush=True)
            if v > best:  # selection on seen-family val only
                best = v
                M.save(net, a.out)
        if last:
            break
    M.save(net, a.out + "_last")
    with open(a.out + "_log.json", "w") as f:
        json.dump(dict(args=vars(a), params=M.n_params(net), log=log, best_val_iou=best,
                       steps_done=step, minutes=(time.time() - t0) / 60), f, indent=1)
    print("done; best val IoU", best)
