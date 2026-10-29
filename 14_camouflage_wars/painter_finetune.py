"""Follow-up round 1 retraining: fine-tune the seed-0 U-Net on a 50/50 mix of the original training
scenes and painter-picked creatures rendered from TRAINING families only (results/painter_train/,
produced by `painter.py --families train`). Held-out families are never touched. Selection: none
(last checkpoint); val IoU (seen families, original split) is logged before/after as a forgetting check.
Writes checkpoints/painter_round1/unet_ft.*"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import model as M
from train import augment, loss_fn, predict, iou

ap = argparse.ArgumentParser()
ap.add_argument("--painted", default="results/painter_train")
ap.add_argument("--ckpt", default="checkpoints/unet")
ap.add_argument("--out", default="checkpoints/painter_round1/unet_ft")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--key", default="painted_by_unet0", help="'random0' = control: same scenes, random skin draw")
a = ap.parse_args()
mx.random.seed(a.seed); rng = np.random.default_rng(a.seed)
d = np.load("data/train.npz"); Xtr, Ytr = d["X"], (d["Y"] > 0)
v = np.load("data/val.npz"); Xva, Yva = v["X"], v["Y"]
im = np.load(os.path.join(a.painted, "images.npz")); sc = np.load(os.path.join(a.painted, "scores.npz"))
Xp, Yp = im[a.key], sc["mask"]
net = M.load(a.ckpt); net.train()
val0 = iou(predict(net, Xva), Yva)
print(f"painted train scenes {len(Xp)}; val IoU before {val0:.4f}", flush=True)
sched = optim.join_schedules([optim.linear_schedule(1e-6, a.lr, 100), optim.cosine_decay(a.lr, max(1, a.steps - 100), 1e-5)], [100])
opt = optim.AdamW(learning_rate=sched, weight_decay=1e-4)
step_fn = nn.value_and_grad(net, loss_fn)
t0, log = time.time(), []
for step in range(1, a.steps + 1):
    h = a.bs // 2
    i1, i2 = rng.integers(0, len(Xtr), h), rng.integers(0, len(Xp), a.bs - h)
    xb = np.concatenate([Xtr[i1], Xp[i2]]).astype(np.float32) / 255
    yb = np.concatenate([Ytr[i1], Yp[i2]]).astype(np.float32)
    xb, yb = augment(xb, yb, rng)
    loss, g = step_fn(net, mx.array(np.ascontiguousarray(xb)), mx.array(np.ascontiguousarray(yb)))
    opt.update(net, g); mx.eval(net.parameters(), opt.state, loss)
    if step % 250 == 0 or step == a.steps:
        log.append(dict(step=step, loss=float(loss), minutes=(time.time() - t0) / 60))
        print(log[-1], flush=True)
net.eval()
val1 = iou(predict(net, Xva), Yva)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
M.save(net, a.out)
json.dump(dict(args=vars(a), val_iou_before=val0, val_iou_after=val1, log=log, minutes=(time.time() - t0) / 60),
          open(a.out + "_log.json", "w"), indent=1)
print(f"done; val IoU {val0:.4f} -> {val1:.4f}")
