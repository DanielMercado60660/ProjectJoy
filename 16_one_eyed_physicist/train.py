"""Train OneEyed on freshly rendered clips (infinite synthetic data, rendered on the GPU per step).

  ../.venv/bin/python train.py --minutes 17 --out checkpoints/main
  ../.venv/bin/python train.py --minutes 0.5 --out checkpoints/smoke     (smoke)
"""
import argparse, json, math, os, time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

import scene
from model import OneEyed, predict

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=17.0)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--width", type=int, default=32)
ap.add_argument("--feat", type=int, default=128)
ap.add_argument("--warmup_mse", type=int, default=400, help="steps of plain MSE before Gaussian NLL")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="checkpoints/main")
ap.add_argument("--val_n", type=int, default=512)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
mx.random.seed(args.seed)
rng = np.random.default_rng(args.seed + 12345)   # training seeds are disjoint from val (7) / test (1001, 2002)
Xv, pv = scene.make_set(7, args.val_n)

model = OneEyed(width=args.width, feat=args.feat)
mx.eval(model.parameters())
nparams = sum(v.size for _, v in tree_flatten(model.parameters()))
opt = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)


def loss_fn(model, x, e, use_nll):
    mu, lv = model(x)
    mse = ((mu - e) ** 2).mean()
    nll = (0.5 * (lv + (e - mu) ** 2 * mx.exp(-lv))).mean()
    return mx.where(use_nll, nll, mse), (mse, nll)


step_fn = nn.value_and_grad(model, loss_fn)


def train_step(x, e, use_nll):
    (loss, (mse, nll)), g = step_fn(model, x, e, use_nll)
    g, _ = optim.clip_grad_norm(g, 1.0)
    opt.update(model, g)
    return loss, mse, nll


def save(tag_dir, step, extra):
    model.save_weights(f"{tag_dir}/model.safetensors")
    cfg = dict(width=args.width, feat=args.feat, seed=args.seed, args=vars(args), **{**extra, "step": step})
    json.dump(cfg, open(f"{tag_dir}/config.json", "w"), indent=1)


log = open(f"{args.out}/train_log.jsonl", "w")
print(f"params {nparams}", flush=True)
t0 = time.time(); step = 0; budget = args.minutes * 60; best = 1e9; next_val = 0
while True:
    el = time.time() - t0
    if el > budget:
        break
    frac = el / budget
    opt.learning_rate = args.lr * (min(1.0, step / 200) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac))))
    p = scene.sample_params(rng, args.bs)
    x = scene.render(p, key_seed=int(rng.integers(1 << 30)))
    e = mx.array(p["e"].astype(np.float32))
    loss, mse, nll = train_step(x, e, mx.array(step >= args.warmup_mse))
    mx.eval(model.parameters(), opt.state, loss)
    step += 1
    if el >= next_val or step == 1:
        next_val = el + max(10.0, budget / 40)
        mu, sd = predict(model, Xv)
        err = mu - pv["e"]
        vmae = float(np.abs(err).mean()); vnll = float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + err ** 2 / (2 * sd ** 2)))
        cov90 = float(np.mean(np.abs(err) < 1.645 * sd))
        rec = dict(step=step, t=round(el, 1), lr=float(opt.learning_rate), loss=float(loss), mse=float(mse),
                   val_mae=vmae, val_nll=vnll, val_cov90=cov90)
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if step >= args.warmup_mse and vnll < best:
            best = vnll
            os.makedirs(f"{args.out}/best", exist_ok=True)
            save(f"{args.out}/best", step, dict(val_mae=vmae, val_nll=vnll))
mu, sd = predict(model, Xv)
err = mu - pv["e"]
final = dict(step=step, t=round(time.time() - t0, 1), val_mae=float(np.abs(err).mean()),
             val_nll=float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + err ** 2 / (2 * sd ** 2))),
             steps_per_s=step / (time.time() - t0), params=int(nparams))
print("final", json.dumps(final), flush=True); log.write(json.dumps(dict(final=final)) + "\n")
save(args.out, step, final)
