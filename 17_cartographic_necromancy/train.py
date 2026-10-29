"""Train a small MLX U-Net to inpaint road pixels inside a rectangular hole.

Input (B,64,64,2) = [raster with hole zeroed, hole mask]; output logits (B,64,64,1).
Loss: BCE on hole pixels only (known pixels are copied from the input at eval time anyway).
Time-capped: --minutes. Checkpoints (weights + config json) every ~2 min and at the end.
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from maps import S, make_map, rasterize, TRAIN0

HOLE_TRAIN = (12, 24)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout, pytorch_compatible=True)
        self.n2 = nn.GroupNorm(8, cout, pytorch_compatible=True)

    def __call__(self, x):
        x = nn.silu(self.n1(self.c1(x)))
        return nn.silu(self.n2(self.c2(x)))


class UNet(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        c = [ch, ch * 2, ch * 4, ch * 4]
        self.d0 = Block(2, c[0]); self.d1 = Block(c[0], c[1]); self.d2 = Block(c[1], c[2]); self.mid = Block(c[2], c[3])
        self.u2 = Block(c[3] + c[2], c[2]); self.u1 = Block(c[2] + c[1], c[1]); self.u0 = Block(c[1] + c[0], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    @staticmethod
    def down(x):
        B, H, W, C = x.shape
        return x.reshape(B, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))

    @staticmethod
    def up(x):
        return mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)

    def __call__(self, x):
        h0 = self.d0(x)                 # 64
        h1 = self.d1(self.down(h0))     # 32
        h2 = self.d2(self.down(h1))     # 16
        m = self.mid(self.down(h2))     # 8 (receptive field covers a 24px hole)
        y = self.u2(mx.concatenate([self.up(m), h2], -1))
        y = self.u1(mx.concatenate([self.up(y), h1], -1))
        y = self.u0(mx.concatenate([self.up(y), h0], -1))
        return self.out(y)


def random_holes(rng, B, lo, hi, margin=3):
    m = np.zeros((B, S, S), np.float32)
    hs = rng.integers(lo, hi + 1, (B, 2))
    for b in range(B):
        h, w = hs[b]
        y = rng.integers(margin, S - margin - h + 1); x = rng.integers(margin, S - margin - w + 1)
        m[b, y:y + h, x:x + w] = 1
    return m


def make_input(ims, holes):
    x = np.stack([ims * (1 - holes), holes], -1).astype(np.float32)
    return mx.array(x)


def load_model(path):
    cfg = json.load(open(path + ".json"))
    model = UNet(cfg["ch"])
    model.load_weights(path + ".safetensors")
    model.eval()
    return model, cfg


def predict_prob(model, ims, holes, bs=256):
    out = []
    for i in range(0, len(ims), bs):
        lo = model(make_input(ims[i:i + bs].astype(np.float32), holes[i:i + bs].astype(np.float32)))
        out.append(np.array(mx.sigmoid(lo[..., 0])))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--max_steps", type=int, default=10**9)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="checkpoints/unet")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()
    ims = np.stack([rasterize(*make_map(TRAIN0 + s, "grid")) for s in range(a.n_train)]).astype(np.float32)
    print(f"generated {a.n_train} train maps in {time.time()-t0:.1f}s, road frac {ims.mean():.3f}", flush=True)
    mx.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = UNet(a.ch)
    est_steps = 7000
    sched = optim.cosine_decay(a.lr, est_steps, 1e-4)
    opt = optim.AdamW(learning_rate=lambda s: sched(mx.minimum(s, est_steps)), weight_decay=1e-4)
    cfg = vars(a) | dict(hole_train=HOLE_TRAIN, style="grid")

    def loss_fn(m, x, y, h):
        lo = m(x)[..., 0]
        l = nn.losses.binary_cross_entropy(lo, y, with_logits=True, reduction="none")
        return (l * h).sum() / h.sum()

    lg = nn.value_and_grad(model, loss_fn)

    def save(tag=""):
        model.save_weights(a.out + tag + ".safetensors")
        json.dump(cfg | dict(step=step, elapsed=time.time() - t0), open(a.out + tag + ".json", "w"))

    step, last_save, ema, log = 0, time.time(), None, []
    tstart = time.time()
    while time.time() - tstart < a.minutes * 60 and step < a.max_steps:
        idx = rng.integers(0, len(ims), a.bs)
        b = ims[idx]
        # augmentation: random flips / transpose
        if rng.random() < 0.5: b = b[:, ::-1]
        if rng.random() < 0.5: b = b[:, :, ::-1]
        if rng.random() < 0.5: b = b.transpose(0, 2, 1)
        b = np.ascontiguousarray(b)
        h = random_holes(rng, a.bs, *HOLE_TRAIN)
        x = make_input(b, h)
        loss, g = lg(model, x, mx.array(b), mx.array(h))
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state, loss)
        step += 1
        lv = loss.item(); ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        if step % 100 == 0 or step == 1:
            print(f"step {step} loss {lv:.4f} ema {ema:.4f} t {time.time()-tstart:.0f}s", flush=True)
            log.append((step, ema, time.time() - tstart))
        if time.time() - last_save > 120:
            save(); last_save = time.time()
    save()
    json.dump(log, open(a.out + "_trainlog.json", "w"))
    print(f"done {step} steps in {time.time()-tstart:.0f}s final ema {ema:.4f}", flush=True)


if __name__ == "__main__":
    main()
