"""All networks for the Monster Factory, in MLX (NHWC, images in [-1, 1]).

- AttrClassifier: evaluator CNN (6 attribute heads + 256-d feature vector)
- VAE: conv encoder/decoder, 128-d latent
- DCGAN: Generator / Discriminator
- UNet: small DDPM-style U-Net shared by DDPM and flow matching (optional attribute conditioning)
- count_flops(): analytic forward FLOP counter (conv / linear / attention matmuls)
"""
import math

import mlx.core as mx
import mlx.nn as nn

from data import N_CLASSES


def gn(c):
    return nn.GroupNorm(min(32, c // 4), c, pytorch_compatible=True)


# --------------------------------------------------------------------------- evaluator
class AttrClassifier(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        w = width
        chans = [(3, w), (w, w), (w, 2 * w), (2 * w, 2 * w), (2 * w, 4 * w), (4 * w, 4 * w)]
        self.convs = [nn.Conv2d(a, b, 3, padding=1) for a, b in chans]
        self.norms = [gn(b) for _, b in chans]
        self.head = nn.Linear(4 * w, sum(N_CLASSES))
        self.feat_dim = 4 * w

    def features(self, x):
        for i, (c, n) in enumerate(zip(self.convs, self.norms)):
            x = nn.silu(n(c(x)))
            if i % 2 == 1:
                x = nn.max_pool2d(x, 2) if hasattr(nn, "max_pool2d") else _maxpool2(x)
        return x.mean(axis=(1, 2))

    def __call__(self, x):
        f = self.features(x)
        return self.head(f), f


def _maxpool2(x):
    b, h, w, c = x.shape
    return x.reshape(b, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))


def split_logits(logits):
    out, i = [], 0
    for k in N_CLASSES:
        out.append(logits[:, i:i + k]); i += k
    return out


# --------------------------------------------------------------------------- VAE
class VAE(nn.Module):
    def __init__(self, ch=64, zdim=128, res=32):
        super().__init__()
        self.zdim, self.res = zdim, res
        self.n_down = int(math.log2(res // 4))
        enc, c = [], 3
        for i in range(self.n_down):
            co = ch * 2 ** i
            enc += [nn.Conv2d(c, co, 4, stride=2, padding=1), gn(co)]
            c = co
        self.enc = enc
        self.enc_out = nn.Linear(c * 16, 2 * zdim)
        self.top_c = c
        self.dec_in = nn.Linear(zdim, c * 16)
        dec = []
        for i in reversed(range(self.n_down)):
            co = ch * 2 ** (i - 1) if i > 0 else ch // 2
            dec += [nn.ConvTranspose2d(c, co, 4, stride=2, padding=1), gn(co)]
            c = co
        self.dec = dec
        self.dec_out = nn.Conv2d(c, 3, 3, padding=1)

    def encode(self, x):
        for i in range(0, len(self.enc), 2):
            x = nn.silu(self.enc[i + 1](self.enc[i](x)))
        h = self.enc_out(x.reshape(x.shape[0], -1))
        return h[:, :self.zdim], h[:, self.zdim:]

    def decode(self, z):
        x = self.dec_in(z).reshape(z.shape[0], 4, 4, self.top_c)
        for i in range(0, len(self.dec), 2):
            x = nn.silu(self.dec[i + 1](self.dec[i](x)))
        return mx.tanh(self.dec_out(x))


# --------------------------------------------------------------------------- DCGAN
class Generator(nn.Module):
    def __init__(self, zdim=128, ch=64, res=32):
        super().__init__()
        self.zdim = zdim
        n_up = int(math.log2(res // 4))
        c = ch * 2 ** (n_up - 1) * 2  # 256 at 32px
        self.top_c = c
        self.fc = nn.Linear(zdim, 4 * 4 * c)
        self.bn0 = nn.BatchNorm(c)
        ups, bns = [], []
        for i in range(n_up):
            co = c // 2
            ups.append(nn.ConvTranspose2d(c, co, 4, stride=2, padding=1)); bns.append(nn.BatchNorm(co))
            c = co
        self.ups, self.bns = ups, bns
        self.out = nn.Conv2d(c, 3, 3, padding=1)

    def __call__(self, z):
        x = nn.relu(self.bn0(self.fc(z).reshape(z.shape[0], 4, 4, self.top_c)))
        for u, b in zip(self.ups, self.bns):
            x = nn.relu(b(u(x)))
        return mx.tanh(self.out(x))


class Discriminator(nn.Module):
    def __init__(self, ch=64, res=32):
        super().__init__()
        n_down = int(math.log2(res // 4))
        convs, c = [], 3
        for i in range(n_down):
            co = ch * 2 ** i
            convs.append(nn.Conv2d(c, co, 4, stride=2, padding=1)); c = co
        self.convs = convs
        self.norms = [None] + [nn.LayerNorm(ch * 2 ** i) for i in range(1, n_down)]
        self.fc = nn.Linear(c * 16, 1)

    def __call__(self, x):
        for cv, nm in zip(self.convs, self.norms):
            x = cv(x)
            if nm is not None:
                x = nm(x)
            x = nn.leaky_relu(x, 0.2)
        return self.fc(x.reshape(x.shape[0], -1)).squeeze(-1)


# --------------------------------------------------------------------------- U-Net
def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t.astype(mx.float32)[:, None] * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout=0.0):
        super().__init__()
        self.n1, self.c1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(self, x, temb):
        h = self.c1(nn.silu(self.n1(x)))
        h = h + self.t(nn.silu(temb))[:, None, None, :]
        h = nn.silu(self.n2(h))
        if self.drop is not None:
            h = self.drop(h)
        h = self.c2(h)
        return h + (self.skip(x) if self.skip is not None else x)


class Attn(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.norm = gn(c)
        self.attn = nn.MultiHeadAttention(c, heads)

    def __call__(self, x):
        b, h, w, c = x.shape
        y = self.norm(x).reshape(b, h * w, c)
        return x + self.attn(y, y, y).reshape(b, h, w, c)


class UNet(nn.Module):
    """DDPM-style U-Net. t is a float in [0, 1000). cond: (B, 6) int attributes, -1 = dropped."""

    def __init__(self, ch=64, mults=(1, 2, 2), num_res=2, attn_res=(8,), res=32, dropout=0.0, cond=False):
        super().__init__()
        tdim = ch * 4
        self.ch, self.cond = ch, cond
        self.temb1, self.temb2 = nn.Linear(ch, tdim), nn.Linear(tdim, tdim)
        if cond:
            self.cond_emb = [nn.Embedding(k + 1, tdim) for k in N_CLASSES]  # index k = "null"
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        downs, chans, c, r = [], [ch], ch, res
        for i, m in enumerate(mults):
            for _ in range(num_res):
                blk = [ResBlock(c, ch * m, tdim, dropout)]
                c = ch * m
                if r in attn_res:
                    blk.append(Attn(c))
                downs.append(blk); chans.append(c)
            if i != len(mults) - 1:
                downs.append([nn.Conv2d(c, c, 3, stride=2, padding=1)]); chans.append(c); r //= 2
        self.downs = downs
        self.mid1, self.mid_attn, self.mid2 = ResBlock(c, c, tdim, dropout), Attn(c), ResBlock(c, c, tdim, dropout)
        ups = []
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(num_res + 1):
                blk = [ResBlock(c + chans.pop(), ch * m, tdim, dropout)]
                c = ch * m
                if r in attn_res:
                    blk.append(Attn(c))
                ups.append(blk)
            if i != 0:
                ups.append([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)]); r *= 2
        self.ups = ups
        self.norm_out, self.conv_out = gn(c), nn.Conv2d(c, 3, 3, padding=1)

    def __call__(self, x, t, cond=None):
        temb = self.temb2(nn.silu(self.temb1(timestep_embedding(t, self.ch))))
        if self.cond and cond is not None:
            for j, e in enumerate(self.cond_emb):
                idx = mx.where(cond[:, j] < 0, N_CLASSES[j], cond[:, j])
                temb = temb + e(idx)
        h = self.conv_in(x)
        hs = [h]
        for blk in self.downs:
            if isinstance(blk[0], ResBlock):
                h = blk[0](h, temb)
                for b in blk[1:]:
                    h = b(h)
            else:
                h = blk[0](h)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for blk in self.ups:
            if isinstance(blk[0], ResBlock):
                h = blk[0](mx.concatenate([h, hs.pop()], axis=-1), temb)
                for b in blk[1:]:
                    h = b(h)
            else:
                h = blk[1](blk[0](h))
        return self.conv_out(nn.silu(self.norm_out(h)))


# --------------------------------------------------------------------------- FLOP counting
def count_flops(fn, *args):
    """Run fn(*args) once with conv/linear/attention patched to count multiply-adds*2. Returns FLOPs."""
    total = [0]
    orig = {}

    def conv_call(self, x):
        y = orig["conv"](self, x)
        total[0] += 2 * y.size * self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
        return y

    def convt_call(self, x):
        y = orig["convt"](self, x)
        # each input element scatters into k*k*cout outputs
        total[0] += 2 * x.size * self.weight.shape[0] * self.weight.shape[1] * self.weight.shape[2]
        return y

    def lin_call(self, x):
        y = orig["lin"](self, x)
        total[0] += 2 * y.size * self.weight.shape[1]
        return y

    def mha_call(self, q, k, v, mask=None):
        y = orig["mha"](self, q, k, v, mask)
        b, l, c = q.shape
        total[0] += 2 * 2 * b * l * k.shape[1] * c  # QK^T and AV
        return y

    pairs = [("conv", nn.Conv2d, conv_call), ("convt", nn.ConvTranspose2d, convt_call),
             ("lin", nn.Linear, lin_call), ("mha", nn.MultiHeadAttention, mha_call)]
    for name, cls, fnew in pairs:
        orig[name] = cls.__call__
        cls.__call__ = fnew
    try:
        out = fn(*args)
        mx.eval(out)
    finally:
        for name, cls, _ in pairs:
            cls.__call__ = orig[name]
    return total[0]
