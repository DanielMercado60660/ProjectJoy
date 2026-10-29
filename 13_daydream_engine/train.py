"""Train the Dynamics MLP with a multi-step (unrolled, open-loop) loss.

Each batch is a set of random windows of length K+1 from the training
trajectories. The model rolls out K steps from the first frame feeding its own
predictions back, and the loss is the mean squared error of every predicted
frame in normalised state units. K ramps from 1 to --unroll over the first
--ramp fraction of training (a curriculum so early training is stable).

Stops at --steps or --max-minutes, whichever comes first, and always writes a
checkpoint (weights + config json with step, seeds and losses).
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from model import Dynamics, set_norm, save, imagine

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="checkpoints/dynamics")
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--max-minutes", type=float, default=15.0)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--unroll", type=int, default=20)
ap.add_argument("--ramp", type=float, default=0.3)
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--log-every", type=int, default=200)
a = ap.parse_args()

mx.random.seed(a.seed)
rng = np.random.default_rng(a.seed)
tr = np.load(os.path.join(a.data, "train.npz"))["states"]
te = np.load(os.path.join(a.data, "test.npz"))["states"]
N, T, D = tr.shape
print(f"train {tr.shape}  test {te.shape}")

model = Dynamics(D, a.hidden, a.depth)
set_norm(model, tr)
mx.eval(model.parameters())
n_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
print(f"params {n_params}")

opt = optim.AdamW(learning_rate=a.lr, weight_decay=1e-5)


def progress(step, t0):
    """training progress in [0,1]: whichever of the step or wall-clock budget is further along."""
    return min(1.0, max(step / a.steps, (time.time() - t0) / 60 / a.max_minutes))
sd = model.s_sd


def loss_fn(model, win, k):
    s = win[:, 0]
    loss = 0.0
    for t in range(1, k + 1):
        s = model(s)
        loss = loss + mx.mean(((s - win[:, t]) / sd) ** 2)
    return loss / k


grad_fn = nn.value_and_grad(model, loss_fn)


def batch(k):
    i = rng.integers(0, N, a.batch)
    j = rng.integers(0, T - k, a.batch)
    idx = j[:, None] + np.arange(k + 1)[None]
    return mx.array(tr[i[:, None], idx])


# quick held-out monitor: 20-step open-loop position RMSE on 256 test scenes
mon_init, mon_true = te[:256, 0], te[:256, :21]
pos_mask = np.tile([1, 1, 0, 0], D // 4).astype(bool)


def monitor():
    pred = imagine(model, mon_init, 20)
    return float(np.sqrt(np.mean((pred[..., pos_mask] - mon_true[..., pos_mask]) ** 2)))


t0 = time.time()
log = []
step = 0
for step in range(1, a.steps + 1):
    p = progress(step, t0)
    k = max(1, min(a.unroll, int(round(a.unroll * p / a.ramp))))
    opt.learning_rate = a.lr * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * p)))
    win = batch(k)
    loss, grads = grad_fn(model, win, k)
    grads, gnorm = optim.clip_grad_norm(grads, 1.0)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state, loss)
    if step % a.log_every == 0 or step == 1:
        el = time.time() - t0
        rec = dict(step=step, k=k, lr=float(opt.learning_rate), loss=float(loss), gnorm=float(gnorm), test_rmse20=monitor(),
                   minutes=el / 60)
        log.append(rec)
        print(json.dumps(rec), flush=True)
    if p >= 1.0:
        print(f"time budget hit at step {step}")
        break

el = time.time() - t0
cfg = dict(state_dim=D, hidden=a.hidden, depth=a.depth, step=step, args=vars(a),
           params=n_params, train_minutes=el / 60, final_test_rmse20=monitor(), log=log,
           uptime=os.popen("uptime").read().strip())
save(model, a.out, cfg)
mx.save_safetensors(a.out + ".opt.safetensors", dict(tree_flatten(opt.state)))
print(f"saved {a.out} after {step} steps, {el/60:.1f} min, test_rmse20 {cfg['final_test_rmse20']:.4f}")
