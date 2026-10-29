"""Monster Factory flow-matching U-Net + attribute classifier (copied from 29_image_gen/models.py and the
Euler sampler in train_gen.py), in three dialects: MLX (reference), NumPy (baseline) and torch (Core ML
conversion frontend only). The classifier is MLX only: it is the judge of the task check, not an athlete.

U-Net: 64*[1,2,2] channels, 2 res blocks per level, 4-head attention at 8x8, GroupNorm(min(32, c/4)),
6.47M params, 2.89 GFLOP per image per evaluation. Input x (B,32,32,3) NHWC in [-1,1]-ish, t (B,) float in
[0, 1000). Output v (B,32,32,3), the velocity. Flow sampler (Euler): x0 ~ N(0,1), for i < steps:
x += (1/steps) * v(x, 1000 * i/steps); then clip to [-1, 1].
"""
import json
import math
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten

CKPT = "../29_image_gen/checkpoints/flow"            # EMA weights behind every flow sample / demo
CLF = "../29_image_gen/checkpoints/classifier_32.safetensors"
N_CLASSES = [4, 6, 3, 2, 3, 3]                       # shape, color, eyes, horns, mouth, pattern
ATTR_NAMES = ["shape", "color", "eyes", "horns", "mouth", "pattern"]
RES = 32


def gn(c):
    return nn.GroupNorm(min(32, c // 4), c, pytorch_compatible=True)


def n_groups(c):
    return min(32, c // 4)


# ---------- MLX U-Net (same module layout as the original, so weights load by name) ----------
def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t.astype(mx.float32)[:, None] * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.c1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def __call__(self, x, temb):
        h = self.c1(nn.silu(self.n1(x)))
        h = h + self.t(nn.silu(temb))[:, None, None, :]
        h = self.c2(nn.silu(self.n2(h)))
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
    def __init__(self, ch=64, mults=(1, 2, 2), num_res=2, attn_res=(8,), res=32):
        super().__init__()
        tdim = ch * 4
        self.ch = ch
        self.temb1, self.temb2 = nn.Linear(ch, tdim), nn.Linear(tdim, tdim)
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        downs, chans, c, r = [], [ch], ch, res
        for i, m in enumerate(mults):
            for _ in range(num_res):
                blk = [ResBlock(c, ch * m, tdim)]
                c = ch * m
                if r in attn_res:
                    blk.append(Attn(c))
                downs.append(blk); chans.append(c)
            if i != len(mults) - 1:
                downs.append([nn.Conv2d(c, c, 3, stride=2, padding=1)]); chans.append(c); r //= 2
        self.downs = downs
        self.mid1, self.mid_attn, self.mid2 = ResBlock(c, c, tdim), Attn(c), ResBlock(c, c, tdim)
        ups = []
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(num_res + 1):
                blk = [ResBlock(c + chans.pop(), ch * m, tdim)]
                c = ch * m
                if r in attn_res:
                    blk.append(Attn(c))
                ups.append(blk)
            if i != 0:
                ups.append([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)]); r *= 2
        self.ups = ups
        self.norm_out, self.conv_out = gn(c), nn.Conv2d(c, 3, 3, padding=1)

    def __call__(self, x, t):
        # cast: the embedding is built in fp32; without it an fp16 net silently promotes back to fp32
        temb = self.temb2(nn.silu(self.temb1(timestep_embedding(t, self.ch).astype(x.dtype))))
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


def config(path=CKPT):
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


def load_mlx(path=CKPT, dtype=mx.float32):
    cfg = config(path)
    assert cfg["model"] == "flow" and not cfg["cond"]
    m = UNet(ch=cfg["ch"], mults=tuple(cfg["mults"]), res=cfg["res"])
    m.update(tree_unflatten(list(mx.load(os.path.join(path, "ema.safetensors")).items())))
    m.eval()
    if dtype != mx.float32:
        m.set_dtype(dtype)
    mx.eval(m.parameters())
    return m


def weights_numpy(path=CKPT):
    return {k: np.array(v, dtype=np.float32) for k, v in mx.load(os.path.join(path, "ema.safetensors")).items()}


def euler_times(steps):
    ts = np.linspace(0, 1, steps + 1)
    return [(float(ts[i]), float(ts[i + 1])) for i in range(steps)]


# ---------- MLX classifier (the judge) ----------
class AttrClassifier(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        w = width
        chans = [(3, w), (w, w), (w, 2 * w), (2 * w, 2 * w), (2 * w, 4 * w), (4 * w, 4 * w)]
        self.convs = [nn.Conv2d(a, b, 3, padding=1) for a, b in chans]
        self.norms = [gn(b) for _, b in chans]
        self.head = nn.Linear(4 * w, sum(N_CLASSES))

    def __call__(self, x):
        for i, (c, n) in enumerate(zip(self.convs, self.norms)):
            x = nn.silu(n(c(x)))
            if i % 2 == 1:
                b, h, w, ch = x.shape
                x = x.reshape(b, h // 2, 2, w // 2, 2, ch).max(axis=(2, 4))
        return self.head(x.mean(axis=(1, 2)))


def load_classifier(path=CLF):
    m = AttrClassifier()
    m.load_weights(path)
    m.eval()
    return m


def to_uint8(x):
    return ((np.clip(x, -1, 1) + 1) * 127.5).round().astype(np.uint8)


def classify(clf, x_uint8):
    """(N, 6) int attribute labels, on the MLX CPU stream (deterministic, full fp32)."""
    with mx.stream(mx.cpu):
        logits = clf(mx.array(x_uint8).astype(mx.float32) / 127.5 - 1)
        out, i = [], 0
        for k in N_CLASSES:
            out.append(mx.argmax(logits[:, i:i + k], axis=-1)); i += k
        return np.array(mx.stack(out, axis=-1))


# ---------- NumPy baseline ----------
def _conv(x, k, b, stride=1):
    """x (B,H,W,C), k (O,kh,kw,C) MLX layout, padding kh//2."""
    B, H, W, C = x.shape
    kh = k.shape[1]
    if kh == 1:
        return (x.reshape(-1, C) @ k[:, 0, 0, :].T + b).reshape(B, H, W, -1)
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    Ho, Wo = (H - 1) // stride + 1, (W - 1) // stride + 1
    cols = np.empty((B, Ho, Wo, 3, 3, C), np.float32)
    for dy in range(3):
        for dx in range(3):
            cols[:, :, :, dy, dx] = xp[:, dy:dy + stride * (Ho - 1) + 1:stride, dx:dx + stride * (Wo - 1) + 1:stride]
    return (cols.reshape(-1, 9 * C) @ k.reshape(k.shape[0], -1).T + b).reshape(B, Ho, Wo, -1)


def _gn(x, g, b, eps=1e-5):
    B, H, W, C = x.shape
    G = n_groups(C)
    xg = x.reshape(B, H * W, G, C // G)
    mu = xg.mean(axis=(1, 3), keepdims=True)
    var = xg.var(axis=(1, 3), keepdims=True)
    return ((xg - mu) / np.sqrt(var + eps)).reshape(B, H, W, C) * g + b


def _silu(x):
    return x / (1.0 + np.exp(-x))


class NumpyUNet:
    """Walks the structure of our MLX UNet; weights from the same EMA file."""

    def __init__(self, path=CKPT):
        self.w = weights_numpy(path)
        self.m = UNet(ch=config(path)["ch"], mults=tuple(config(path)["mults"]))  # structure only
        self.ch = self.m.ch

    def _lin(self, x, p):
        return x @ self.w[p + ".weight"].T + (self.w[p + ".bias"] if p + ".bias" in self.w else 0)

    def _res(self, x, temb, p, blk):
        w = self.w
        h = _conv(_silu(_gn(x, w[p + ".n1.weight"], w[p + ".n1.bias"])), w[p + ".c1.weight"], w[p + ".c1.bias"])
        h = h + self._lin(_silu(temb), p + ".t")[:, None, None, :]
        h = _conv(_silu(_gn(h, w[p + ".n2.weight"], w[p + ".n2.bias"])), w[p + ".c2.weight"], w[p + ".c2.bias"])
        sk = _conv(x, w[p + ".skip.weight"], w[p + ".skip.bias"]) if blk.skip is not None else x
        return h + sk

    def _attn(self, x, p, heads=4):
        w = self.w
        B, H, W, C = x.shape
        y = _gn(x, w[p + ".norm.weight"], w[p + ".norm.bias"]).reshape(B, H * W, C)
        q, k, v = (self._lin(y, f"{p}.attn.{n}_proj").reshape(B, H * W, heads, -1).transpose(0, 2, 1, 3)
                   for n in ("query", "key", "value"))
        s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(q.shape[-1])
        s = np.exp(s - s.max(-1, keepdims=True))
        o = (s / s.sum(-1, keepdims=True)) @ v
        o = self._lin(o.transpose(0, 2, 1, 3).reshape(B, H * W, C), p + ".attn.out_proj")
        return x + o.reshape(B, H, W, C)

    def __call__(self, x, t):
        w = self.w
        half = self.ch // 2
        freqs = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / half)
        a = t.astype(np.float32)[:, None] * freqs[None]
        temb = np.concatenate([np.cos(a), np.sin(a)], -1)
        temb = self._lin(_silu(self._lin(temb, "temb1")), "temb2")
        h = _conv(x, w["conv_in.weight"], w["conv_in.bias"])
        hs = [h]
        for i, blk in enumerate(self.m.downs):
            if isinstance(blk[0], ResBlock):
                h = self._res(h, temb, f"downs.{i}.0", blk[0])
                if len(blk) > 1:
                    h = self._attn(h, f"downs.{i}.1")
            else:
                h = _conv(h, w[f"downs.{i}.0.weight"], w[f"downs.{i}.0.bias"], stride=2)
            hs.append(h)
        h = self._res(h, temb, "mid1", self.m.mid1)
        h = self._attn(h, "mid_attn")
        h = self._res(h, temb, "mid2", self.m.mid2)
        for i, blk in enumerate(self.m.ups):
            if isinstance(blk[0], ResBlock):
                h = self._res(np.concatenate([h, hs.pop()], -1), temb, f"ups.{i}.0", blk[0])
                if len(blk) > 1:
                    h = self._attn(h, f"ups.{i}.1")
            else:
                h = np.repeat(np.repeat(h, 2, axis=1), 2, axis=2)
                h = _conv(h, w[f"ups.{i}.1.weight"], w[f"ups.{i}.1.bias"])
        h = _silu(_gn(h, w["norm_out.weight"], w["norm_out.bias"]))
        return _conv(h, w["conv_out.weight"], w["conv_out.bias"]).astype(np.float32)


# ---------- torch (Core ML frontend only) ----------
def torch_net(path=CKPT):
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F
    ref = UNet(ch=config(path)["ch"], mults=tuple(config(path)["mults"]))  # structure only
    w = weights_numpy(path)
    CH = ref.ch

    def tgn(c):
        return tnn.GroupNorm(n_groups(c), c)

    class TRes(tnn.Module):
        def __init__(s, blk):
            super().__init__()
            cin, cout = blk.c1.weight.shape[3], blk.c1.weight.shape[0]
            s.n1, s.c1 = tgn(cin), tnn.Conv2d(cin, cout, 3, padding=1)
            s.t = tnn.Linear(blk.t.weight.shape[1], cout)
            s.n2, s.c2 = tgn(cout), tnn.Conv2d(cout, cout, 3, padding=1)
            s.skip = tnn.Conv2d(cin, cout, 1) if blk.skip is not None else None

        def forward(s, x, temb):
            h = s.c1(F.silu(s.n1(x)))
            h = h + s.t(F.silu(temb))[:, :, None, None]
            h = s.c2(F.silu(s.n2(h)))
            return h + (s.skip(x) if s.skip is not None else x)

    class MHA(tnn.Module):
        def __init__(s, c):
            super().__init__()
            s.query_proj, s.key_proj = tnn.Linear(c, c, bias=False), tnn.Linear(c, c, bias=False)
            s.value_proj, s.out_proj = tnn.Linear(c, c, bias=False), tnn.Linear(c, c, bias=False)

    class TAttn(tnn.Module):
        def __init__(s, c, hw, heads=4):
            super().__init__()
            s.norm, s.attn = tgn(c), MHA(c)
            s.c, s.hw, s.heads = c, hw, heads

        def forward(s, x):  # x (B, C, H, W) with H = W = hw; reshape with -1 for the batch
            c, L, nh = s.c, s.hw * s.hw, s.heads
            y = s.norm(x).reshape(-1, c, L).transpose(1, 2)
            q = s.attn.query_proj(y).reshape(-1, L, nh, c // nh).transpose(1, 2)
            k = s.attn.key_proj(y).reshape(-1, L, nh, c // nh).transpose(1, 2)
            v = s.attn.value_proj(y).reshape(-1, L, nh, c // nh).transpose(1, 2)
            a = torch.softmax(q @ k.transpose(2, 3) * (1.0 / math.sqrt(c // nh)), dim=-1)
            o = s.attn.out_proj((a @ v).transpose(1, 2).reshape(-1, L, c))
            return x + o.transpose(1, 2).reshape(-1, c, s.hw, s.hw)

    class Down(tnn.Module):  # holds the stride-2 conv as index 0, like MLX's [Conv2d]
        pass

    def make_list(blocks, res_seq):
        out = tnn.ModuleList()
        for blk, r in zip(blocks, res_seq):
            if isinstance(blk[0], ResBlock):
                items = [TRes(blk[0])] + ([TAttn(blk[0].c1.weight.shape[0], r)] if len(blk) > 1 else [])
            elif len(blk) == 1:  # downsample conv
                c = blk[0].weight.shape[0]
                items = [tnn.Conv2d(c, c, 3, stride=2, padding=1)]
            else:  # upsample + conv
                c = blk[1].weight.shape[0]
                items = [tnn.Upsample(scale_factor=2, mode="nearest"), tnn.Conv2d(c, c, 3, padding=1)]
            out.append(tnn.ModuleList(items))
        return out

    def res_seq(blocks, r):
        """Spatial size at each block (only the attention blocks use it)."""
        seq = []
        for blk in blocks:
            if not isinstance(blk[0], ResBlock):
                r = r // 2 if len(blk) == 1 else r * 2
            seq.append(r)
        return seq, r

    dseq, r_mid = res_seq(ref.downs, RES)
    useq, _ = res_seq(ref.ups, r_mid)

    class Net(tnn.Module):
        def __init__(s):
            super().__init__()
            tdim = CH * 4
            s.temb1, s.temb2 = tnn.Linear(CH, tdim), tnn.Linear(tdim, tdim)
            s.conv_in = tnn.Conv2d(3, CH, 3, padding=1)
            s.downs = make_list(ref.downs, dseq)
            s.mid1, s.mid_attn, s.mid2 = TRes(ref.mid1), TAttn(ref.mid1.c1.weight.shape[0], r_mid), TRes(ref.mid2)
            s.ups = make_list(ref.ups, useq)
            c = ref.conv_out.weight.shape[3]
            s.norm_out, s.conv_out = tgn(c), tnn.Conv2d(c, 3, 3, padding=1)
            half = CH // 2
            s.register_buffer("freqs", torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half))

        def forward(s, x, t):  # x (B,32,32,3) NHWC like MLX, t (B,)
            a = t[:, None] * s.freqs[None]
            temb = s.temb2(F.silu(s.temb1(torch.cat([torch.cos(a), torch.sin(a)], dim=-1))))
            h = s.conv_in(x.permute(0, 3, 1, 2))
            hs = [h]
            for blk in s.downs:
                if isinstance(blk[0], TRes):
                    h = blk[0](h, temb)
                    for b in blk[1:]:
                        h = b(h)
                else:
                    h = blk[0](h)
                hs.append(h)
            h = s.mid2(s.mid_attn(s.mid1(h, temb)), temb)
            for blk in s.ups:
                if isinstance(blk[0], TRes):
                    h = blk[0](torch.cat([h, hs.pop()], dim=1), temb)
                    for b in blk[1:]:
                        h = b(h)
                else:
                    h = blk[1](blk[0](h))
            return s.conv_out(F.silu(s.norm_out(h))).permute(0, 2, 3, 1)

    net = Net().eval()
    sd = net.state_dict()
    for k in sd:
        if k == "freqs":
            continue
        a = w[k]
        sd[k] = torch.from_numpy((a.transpose(0, 3, 1, 2) if a.ndim == 4 else a).copy())
    net.load_state_dict(sd)
    return net
