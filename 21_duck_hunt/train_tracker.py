"""Small MLX CNN duck locator: 80x80 RGB frame -> 20x20 heatmap logits + sub-cell offsets.
Single frame in, (x, y) out. Trained only on the 'train' scene condition; model selection on
'val' scenes (same distribution, different seeds). No colour augmentation (only h-flips), so
held-out lighting/backgrounds/distractors are a real test.

  ../bin/gpu_run ../.venv/bin/python train_tracker.py --scenes 500 --steps 6000
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import scene as S

CELL = 4
G = S.H // CELL  # 20


class DuckNet(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.c1 = nn.Conv2d(3, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c * 3 // 2, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(c * 3 // 2, 2 * c, 3, stride=2, padding=1)
        self.c4 = nn.Conv2d(2 * c, 2 * c, 3, padding=1)
        self.c5 = nn.Conv2d(2 * c, 2 * c, 3, padding=1)
        self.head = nn.Conv2d(2 * c, 3, 1)

    def __call__(self, x):  # x: (B,80,80,3) float in [-0.5,0.5]
        h = nn.relu(self.c1(x)); h = nn.relu(self.c2(h)); h = nn.relu(self.c3(h))
        h = nn.relu(self.c4(h)) + h; h = nn.relu(self.c5(h)) + h
        return self.head(h)  # (B,20,20,3): logit, dx, dy


def prep(frames):
    return mx.array(frames.astype(np.float32) / 255.0 - 0.5)


def decode(out):
    """out (B,20,20,3) -> xy (B,2) in output pixels, confidence (B,)"""
    out = np.array(out)
    B = out.shape[0]
    lg = out[..., 0].reshape(B, -1)
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    i = p.argmax(1)
    gy, gx = i // G, i % G
    off = out.reshape(B, -1, 3)[np.arange(B), i, 1:]
    x = (gx + 0.5 + off[:, 0]) * CELL - 0.5
    y = (gy + 0.5 + off[:, 1]) * CELL - 0.5
    return np.stack([x, y], 1), p.max(1)


def predict(model, frames, bs=256):
    xy, cf = [], []
    for s in range(0, len(frames), bs):
        a, b = decode(model(prep(frames[s:s + bs]))); xy.append(a); cf.append(b)
    return np.concatenate(xy), np.concatenate(cf)


def targets(gt):
    """gt (B,2) pixel coords -> cell index, offsets in cell units relative to cell centre."""
    c = (gt + 0.5) / CELL
    gx = np.clip(np.floor(c[:, 0]).astype(int), 0, G - 1); gy = np.clip(np.floor(c[:, 1]).astype(int), 0, G - 1)
    off = np.stack([c[:, 0] - gx - 0.5, c[:, 1] - gy - 0.5], 1)
    return gy * G + gx, off.astype(np.float32)


def loss_fn(model, x, idx, off, w):
    out = model(x)
    B = out.shape[0]
    lg = out[..., 0].reshape(B, -1)
    ce = nn.losses.cross_entropy(lg, idx, reduction="none")
    po = out.reshape(B, -1, 3)[mx.arange(B), idx, 1:]
    l1 = mx.abs(po - off).sum(-1)
    return ((ce + l1) * w).sum() / mx.maximum(w.sum(), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=500)
    ap.add_argument("--val_scenes", type=int, default=40)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/ducknet")
    ap.add_argument("--log", default="results/tracker_train_log.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--aug", action="store_true", help="per-frame photometric jitter (follow-up variant)")
    args = ap.parse_args()
    t0 = time.time()
    X, GT, VIS = S.make_set("train", args.scenes)
    Xv, GTv, VISv = S.make_set("val", args.val_scenes)
    if args.aug:
        Xv = S.jitter_set(Xv, 4242)   # model selection on jittered val scenes
    X = X.reshape(-1, S.H, S.W, 3); GT = GT.reshape(-1, 2); VIS = VIS.reshape(-1)
    Xv = Xv.reshape(-1, S.H, S.W, 3); GTv = GTv.reshape(-1, 2); VISv = VISv.reshape(-1)
    print(f"data {X.shape} rendered in {time.time()-t0:.0f}s", flush=True)
    mx.random.seed(args.seed); rng = np.random.default_rng(args.seed)
    model = DuckNet(); mx.eval(model.parameters())
    sched = optim.cosine_decay(args.lr, args.steps, 1e-5)
    opt = optim.AdamW(learning_rate=sched, weight_decay=1e-4)
    lg = nn.value_and_grad(model, loss_fn)
    flog = open(args.log, "w"); best = 1e9; run = []
    for step in range(1, args.steps + 1):
        b = rng.integers(0, len(X), args.bs)
        x = X[b]; g = GT[b].copy(); w = VIS[b].astype(np.float32)
        flip = rng.random(args.bs) < 0.5
        x[flip] = x[flip, :, ::-1]; g[flip, 0] = S.W - 1 - g[flip, 0]
        if args.aug:
            x = np.stack([S.photometric_jitter(xi, rng) for xi in x])
        idx, off = targets(g)
        l, gr = lg(model, prep(x), mx.array(idx), mx.array(off), mx.array(w))
        opt.update(model, gr); mx.eval(model.parameters(), opt.state)
        run.append(l.item())
        if step % args.eval_every == 0 or step == args.steps:
            xy, _ = predict(model, Xv)
            err = np.linalg.norm(xy - GTv, axis=1)[VISv]
            rec = dict(step=step, wall=time.time() - t0, loss=float(np.mean(run[-args.eval_every:])),
                       val_mean_err=float(err.mean()), val_lost=float((err > 5).mean()),
                       val_clipped=float(np.minimum(err, 20).mean()))
            flog.write(json.dumps(rec) + "\n"); flog.flush(); print(rec, flush=True)
            if rec["val_clipped"] < best:
                best = rec["val_clipped"]
                model.save_weights(args.out + ".safetensors")
                json.dump(dict(step=step, cfg=vars(args), val=rec), open(args.out + ".json", "w"))
    print("best val clipped err", best, "wall", time.time() - t0)


if __name__ == "__main__":
    main()
