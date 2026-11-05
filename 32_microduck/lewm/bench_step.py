"""Feasibility timing only (no data, no training): one LeWM-sized train step in MLX at Microduck sizes.

Encoder: ViT-tiny (12 layers, width 192, 3 heads, MLP 768) on one frame + [CLS] -> projector
(Linear 192->2048, BatchNorm, GELU, Linear 2048->192). Predictor: 6 layers, 16 heads x 64, MLP 2048,
width 192, AdaLN-zero action conditioning, causal over T frames. Loss: next-embedding MSE +
0.09 * SIGReg (1024 random directions, Epps-Pulley, 17 knots on [0, 3]). AdamW.
Usage: python lewm/bench_step.py [--img 96] [--patch 8] [--batch 128] [--T 4] [--steps 6] [--bf16]
"""
import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten


class Block(nn.Module):
    def __init__(self, d, heads, dh, mlp, adaln=False):
        super().__init__()
        self.h, self.dh = heads, dh
        self.n1, self.n2 = nn.LayerNorm(d, affine=not adaln), nn.LayerNorm(d, affine=not adaln)
        self.qkv, self.o = nn.Linear(d, 3 * heads * dh), nn.Linear(heads * dh, d)
        self.fc1, self.fc2 = nn.Linear(d, mlp), nn.Linear(mlp, d)
        self.ada = None
        if adaln:
            self.ada = nn.Linear(d, 6 * d)
            self.ada.weight = mx.zeros_like(self.ada.weight)
            self.ada.bias = mx.zeros_like(self.ada.bias)

    def __call__(self, x, cond=None, mask=None):
        B, L, _ = x.shape
        if self.ada is not None:
            s1, b1, g1, s2, b2, g2 = mx.split(self.ada(nn.silu(cond)), 6, axis=-1)
        else:
            s1 = b1 = s2 = b2 = 0.0
            g1 = g2 = 1.0
        h = self.n1(x) * (1 + s1) + b1
        q, k, v = mx.split(self.qkv(h).reshape(B, L, 3, self.h, self.dh).transpose(2, 0, 3, 1, 4), 3, axis=0)
        a = mx.fast.scaled_dot_product_attention(q[0], k[0], v[0], scale=self.dh ** -0.5, mask=mask)
        x = x + g1 * self.o(a.transpose(0, 2, 1, 3).reshape(B, L, -1))
        h = self.n2(x) * (1 + s2) + b2
        return x + g2 * self.fc2(nn.gelu(self.fc1(h)))


def mlp_proj(d, hid):
    return nn.Sequential(nn.Linear(d, hid), nn.BatchNorm(hid), nn.GELU(), nn.Linear(hid, d))


class LeWM(nn.Module):
    def __init__(self, img, patch, act_dim, T, d=192):
        super().__init__()
        n = (img // patch) ** 2
        self.patch = nn.Conv2d(3, d, patch, stride=patch)
        self.cls = mx.zeros((1, 1, d))
        self.pos = mx.random.normal((1, n + 1, d)) * 0.02
        self.enc = [Block(d, 3, 64, 4 * d) for _ in range(12)]
        self.enc_norm = nn.LayerNorm(d)
        self.proj = mlp_proj(d, 2048)
        self.act = nn.Sequential(nn.Linear(act_dim, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
        self.tpos = mx.random.normal((1, T, d)) * 0.02
        self.pred = [Block(d, 16, 64, 2048, adaln=True) for _ in range(6)]
        self.pred_norm = nn.LayerNorm(d)
        self.pred_proj = mlp_proj(d, 2048)

    def encode(self, x):                       # x: (N, H, W, 3)
        t = self.patch(x).reshape(x.shape[0], -1, self.cls.shape[-1])
        t = mx.concatenate([mx.broadcast_to(self.cls, (t.shape[0], 1, t.shape[-1])), t], axis=1) + self.pos
        for b in self.enc:
            t = b(t)
        return self.proj(self.enc_norm(t[:, 0]))

    def predict(self, z, a):                   # z: (B, T, d), a: (B, T, act_dim)
        B, T, d = z.shape
        c = self.act(a)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(z.dtype)
        h = z + self.tpos[:, :T]
        for b in self.pred:
            h = b(h, c, mask)
        return self.pred_proj(self.pred_norm(h).reshape(B * T, d)).reshape(B, T, d)


def sigreg(z, key, m=1024, knots=17):
    """Epps-Pulley test of N(0,1) on m random 1-D projections of z (N, d); averaged."""
    n, d = z.shape
    u = mx.random.normal((d, m), key=key)
    u = u / mx.linalg.norm(u, axis=0, keepdims=True)
    p = z @ u                                  # (N, m)
    t = mx.linspace(0, 3, knots)
    w = mx.exp(-0.5 * t ** 2)
    ang = p[:, :, None] * t                    # (N, m, K)
    re, im = mx.cos(ang).mean(0), mx.sin(ang).mean(0)
    err = (re - mx.exp(-0.5 * t ** 2)) ** 2 + im ** 2
    dt = t[1] - t[0]
    integ = ((err * w)[:, 1:] + (err * w)[:, :-1]).sum(-1) * dt / 2
    return (n * integ).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--act", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    mx.random.seed(0)
    model = LeWM(a.img, a.patch, a.act, a.T)
    if a.bf16:
        model.set_dtype(mx.bfloat16)
    nparam = sum(v.size for _, v in tree_flatten(model.parameters()))
    opt = optim.AdamW(learning_rate=5e-5, weight_decay=1e-3)
    dt = mx.bfloat16 if a.bf16 else mx.float32
    x = mx.random.uniform(shape=(a.batch, a.T, a.img, a.img, 3)).astype(dt)
    acts = mx.random.normal((a.batch, a.T, a.act)).astype(dt)

    def loss_fn(m, x, acts, key):
        z = m.encode(x.reshape(-1, a.img, a.img, 3)).reshape(a.batch, a.T, -1)
        zh = m.predict(z[:, :-1], acts[:, :-1])
        lp = ((zh - z[:, 1:]) ** 2).mean()
        ls = sigreg(z.reshape(-1, z.shape[-1]).astype(mx.float32), key)
        return lp + 0.09 * ls

    def step(x, acts, key):                    # eager (uncompiled) step: a conservative timing
        l, g = nn.value_and_grad(model, loss_fn)(model, x, acts, key)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        return l
    times = []
    for i in range(a.steps):
        t0 = time.perf_counter()
        l = step(x, acts, mx.random.key(i))
        mx.eval(l, model.parameters(), opt.state)
        times.append(time.perf_counter() - t0)
    steady = sorted(times[2:])[len(times[2:]) // 2] if len(times) > 2 else times[-1]
    imgs = a.batch * a.T
    print(f"params {nparam/1e6:.2f}M img {a.img} patch {a.patch} tokens {(a.img//a.patch)**2} batch {a.batch}x{a.T} "
          f"dtype {'bf16' if a.bf16 else 'fp32'} first {times[0]:.2f}s median_step {steady:.3f}s "
          f"-> {imgs/steady:.0f} frames/s, loss {float(l):.4f}, peak_mem {mx.get_peak_memory()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
