"""Train the learned terrain prior for the fog scenario: from the elevation map alone
(elevation + slope channels) predict P(water) and the expected land multiplier per cell.

  ../bin/gpu_run ../.venv/bin/python train_prior.py --steps 3000 [--resume]
Uses data/train.npz / data/val.npz (static channels only; maps 0..2999 / 500000+).
Checkpoint: checkpoints/prior.{safetensors,json} + prior_state.pkl (optimizer, step, RNG).
"""
import argparse, json, os, pickle, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from terra import learn

CK = "checkpoints"


def batch(st, idx, rng):
    x = st[idx][:, :2].astype(np.float32)          # elev, slope  (what a topo map shows)
    water = st[idx][:, 3].astype(np.float32)
    mland = st[idx][:, 2].astype(np.float32)        # (m-1)/2 on land, 0 on water
    if rng is not None:
        k = rng.integers(4)
        x, water, mland = np.rot90(x, k, (2, 3)), np.rot90(water, k, (1, 2)), np.rot90(mland, k, (1, 2))
        if rng.random() < 0.5:
            x, water, mland = x[..., ::-1], water[..., ::-1], mland[..., ::-1]
    return (mx.array(np.ascontiguousarray(x.transpose(0, 2, 3, 1))), mx.array(np.ascontiguousarray(water)),
            mx.array(np.ascontiguousarray(mland)))


def loss_fn(model, x, water, mland):
    y = model(x)
    bce = nn.losses.binary_cross_entropy(y[..., 0], water, with_logits=True, reduction="mean")
    land = 1.0 - water
    mse = (((mx.sigmoid(y[..., 1]) - mland) ** 2) * land).sum() / mx.maximum(land.sum(), 1.0)
    return bce + mse


def predict(model, elev, kind=None):
    """Returns (p_water, m_land) arrays for one map. Uses only elevation-derived channels."""
    st = learn.static_features(elev, kind if kind is not None else np.zeros(elev.shape, np.int8))
    x = mx.array(st[None, :2].transpose(0, 2, 3, 1))
    y = model(x)
    mx.eval(y)
    y = np.array(y[0], dtype=np.float64)
    return 1 / (1 + np.exp(-y[..., 0])), 1.0 + 2.0 / (1 + np.exp(-y[..., 1]))


def load_prior(tag="prior"):
    cfg = json.load(open(f"{CK}/{tag}.json"))
    m = learn.HNet(cin=2, base=cfg["base"], cout=2)
    m.load_weights(f"{CK}/{tag}.safetensors")
    m.eval()
    return m, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--data", default="data")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="prior")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = vars(a).copy()
    mx.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tr = np.load(f"{a.data}/train.npz")["static"]
    va = np.load(f"{a.data}/val.npz")["static"]
    model = learn.HNet(cin=2, base=a.base, cout=2)
    opt = optim.AdamW(learning_rate=optim.cosine_decay(a.lr, a.steps, end=a.lr * 0.02), weight_decay=1e-4)
    step = 0
    if a.resume and os.path.exists(f"{CK}/{a.tag}_state.pkl"):
        model.load_weights(f"{CK}/{a.tag}.safetensors")
        st = pickle.load(open(f"{CK}/{a.tag}_state.pkl", "rb"))
        opt.state = tree_unflatten([(k, mx.array(v)) for k, v in st["opt"].items()])
        step = st["step"]; rng.bit_generator.state = st["rng"]
        print(f"resumed at step {step}", flush=True)
    lg = nn.value_and_grad(model, loss_fn)
    t0, run = time.time(), None
    while step < a.steps:
        x, wt, ml = batch(tr, rng.integers(0, len(tr), a.bs), rng)
        loss, g = lg(model, x, wt, ml)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state, loss)
        step += 1
        run = float(loss) if run is None else 0.98 * run + 0.02 * float(loss)
        if step % 100 == 0:
            print(f"step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
        if step % 500 == 0 or step == a.steps:
            # validation: water accuracy / IoU and land-multiplier MAE
            inter = union = correct = tot = 0.0; mae = []
            for i in range(0, len(va), 16):
                x, wt, ml = batch(va, np.arange(i, min(len(va), i + 16)), None)
                y = model(x)
                pw = np.array(y[..., 0]) > 0
                wtn = np.array(wt) > 0.5
                inter += (pw & wtn).sum(); union += (pw | wtn).sum()
                correct += (pw == wtn).sum(); tot += wtn.size
                mae.append(np.abs(1 / (1 + np.exp(-np.array(y[..., 1]))) - np.array(ml))[~wtn].mean() * 2)
            cfg.update(step=step, val_water_iou=float(inter / union), val_water_acc=float(correct / tot),
                       val_mland_mae=float(np.mean(mae)))
            print(f"  val water IoU {inter / union:.3f} acc {correct / tot:.4f} land-mult MAE {np.mean(mae):.3f}", flush=True)
            os.makedirs(CK, exist_ok=True)
            model.save_weights(f"{CK}/{a.tag}.safetensors")
            pickle.dump({"opt": {k: np.array(v) for k, v in tree_flatten(opt.state)}, "step": step,
                         "rng": rng.bit_generator.state}, open(f"{CK}/{a.tag}_state.pkl", "wb"))
            json.dump(cfg, open(f"{CK}/{a.tag}.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
