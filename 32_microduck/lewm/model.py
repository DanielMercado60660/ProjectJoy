"""R6 LeWM S1: LeWorldModel (arXiv 2603.19312) in MLX, sized for the Microduck.

Encoder: ViT-tiny (12 layers, width 192, 3 heads x 64, MLP 768) on 64x64 frames, patch 8 (64 tokens)
+ [CLS]; final LayerNorm; projector Linear 192->2048, BatchNorm, GELU, Linear 2048->192 (paper Sec 3.1:
the BN projector is what lets SIGReg act on the embedding).
Predictor: 6 layers, width 192, 16 heads x 64, MLP 2048, dropout 0.1, learned temporal positions,
causal over time, AdaLN-zero conditioning on an embedded action block; then pred_proj (same as projector).
Loss: next-embedding MSE + lam * SIGReg (Epps-Pulley on 1024 random unit directions, 17 knots on [0, 3],
window exp(-t^2/2), per time step), no EMA, no stop-gradient.
"""
import mlx.core as mx
import mlx.nn as nn

D = 192


class Block(nn.Module):
    def __init__(self, d, heads, dh, mlp, adaln=False, dropout=0.0):
        super().__init__()
        self.h, self.dh = heads, dh
        self.n1, self.n2 = nn.LayerNorm(d, affine=not adaln), nn.LayerNorm(d, affine=not adaln)
        self.qkv, self.o = nn.Linear(d, 3 * heads * dh), nn.Linear(heads * dh, d)
        self.fc1, self.fc2 = nn.Linear(d, mlp), nn.Linear(mlp, d)
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
        self.ada = None
        if adaln:                                   # AdaLN-zero: modulation starts at exactly zero
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
        a = self.o(a.transpose(0, 2, 1, 3).reshape(B, L, -1))
        if self.drop is not None:
            a = self.drop(a)
        x = x + g1 * a
        h = self.fc2(nn.gelu(self.fc1(self.n2(x) * (1 + s2) + b2)))
        if self.drop is not None:
            h = self.drop(h)
        return x + g2 * h


def projector(d=D, hid=2048):
    return nn.Sequential(nn.Linear(d, hid), nn.BatchNorm(hid), nn.GELU(), nn.Linear(hid, d))


class Encoder(nn.Module):
    def __init__(self, img=64, patch=8, d=D, depth=12):
        super().__init__()
        n = (img // patch) ** 2
        self.patch = nn.Conv2d(3, d, patch, stride=patch)
        self.cls = mx.zeros((1, 1, d))
        self.pos = mx.random.normal((1, n + 1, d)) * 0.02
        self.blocks = [Block(d, 3, 64, 4 * d) for _ in range(depth)]
        self.norm = nn.LayerNorm(d)
        self.proj = projector(d)

    def cls_features(self, x):                      # pre-projector [CLS] (reported-only probe row)
        t = self.patch(x)
        t = t.reshape(t.shape[0], -1, t.shape[-1])
        t = mx.concatenate([mx.broadcast_to(self.cls, (t.shape[0], 1, t.shape[-1])), t], axis=1) + self.pos
        for b in self.blocks:
            t = b(t)
        return self.norm(t[:, 0])

    def __call__(self, x):                          # x: (N, H, W, 3) float, normalised
        return self.proj(self.cls_features(x))


class Predictor(nn.Module):
    def __init__(self, act_dim, d=D, depth=6, max_T=16, dropout=0.1):
        super().__init__()
        self.act = nn.Sequential(nn.Linear(act_dim, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
        self.tpos = mx.random.normal((1, max_T, d)) * 0.02
        self.blocks = [Block(d, 16, 64, 2048, adaln=True, dropout=dropout) for _ in range(depth)]
        self.norm = nn.LayerNorm(d)
        self.proj = projector(d)

    def __call__(self, z, a):                       # z (B, T, d), a (B, T, act_dim) -> zhat for t+1
        B, T, d = z.shape
        c = self.act(a)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(z.dtype)
        h = z + self.tpos[:, :T]
        for b in self.blocks:
            h = b(h, c, mask)
        return self.proj(self.norm(h).reshape(B * T, d)).reshape(B, T, d)


class LeWM(nn.Module):
    def __init__(self, act_dim, img=64, patch=8):
        super().__init__()
        self.encoder = Encoder(img, patch)
        self.predictor = Predictor(act_dim)

    def encode_frames(self, frames_u8):
        """uint8 (..., H, W, 3) -> (..., D)."""
        sh = frames_u8.shape
        x = (frames_u8.reshape(-1, *sh[-3:]).astype(mx.float32) / 255.0 - 0.45) / 0.25
        return self.encoder(x).reshape(*sh[:-3], D)


def sigreg(z, key, m=1024, knots=17):
    """Epps-Pulley statistic of N(0,1) fit on m random 1-D projections of z (N, d), averaged."""
    n, d = z.shape
    u = mx.random.normal((d, m), key=key)
    u = u / mx.sqrt((u * u).sum(0, keepdims=True) + 1e-12)
    p = z @ u
    t = mx.linspace(0, 3, knots)
    phi0 = mx.exp(-0.5 * t ** 2)
    ang = p[:, :, None] * t
    re, im = mx.cos(ang).mean(0), mx.sin(ang).mean(0)
    err = ((re - phi0) ** 2 + im ** 2) * phi0      # weight w(t) = exp(-t^2/2)
    dt = 3.0 / (knots - 1)
    integ = (err[:, 1:] + err[:, :-1]).sum(-1) * dt / 2
    return (n * integ).mean()


def lewm_loss(model, frames, acts, key, lam=0.09, use_sigreg=True):
    """frames uint8 (B, T, H, W, 3); acts (B, T, A). Teacher forcing: predict z[1:] from z[:-1]."""
    z = model.encode_frames(frames)                  # (B, T, D)
    if not use_sigreg:                               # random-encoder control: frozen encoder
        z = mx.stop_gradient(z)
    zh = model.predictor(z[:, :-1], acts[:, :-1])
    lp = ((zh - z[:, 1:]) ** 2).mean()
    if not use_sigreg:
        return lp, (lp, mx.array(0.0))
    T = z.shape[1]
    keys = mx.random.split(key, T)
    ls = sum(sigreg(z[:, t], keys[t]) for t in range(T)) / T
    return lp + lam * ls, (lp, ls)
