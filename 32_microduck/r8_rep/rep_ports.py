"""R8 Gate C: MLX ports of the pretrained image encoders and of the official LeWorldModel (LeWM).

All three families take 224x224 RGB frames normalised with ImageNet statistics (DINOv2's preprocessor
config, V-JEPA 2's eval transforms and LeWM's `get_img_preprocessor` all use them). Inputs here are uint8
(N, 224, 224, 3); `normalize_u8` does the scaling.

* DINOv2 ViT-S/14 (`facebook/dinov2-small`, HF safetensors). Written from facebookresearch/dinov2
  (models/vision_transformer.py, layers/block.py; commit 7764ea0): pre-norm blocks with LayerScale, LN eps
  1e-6, exact GELU; the 37x37 position grid is resized to 16x16 by bicubic interpolation exactly as the
  official `interpolate_pos_encoding` does for the hub models (scale_factor (16 + 0.1) / 37, align_corners
  False, antialias False), done once in numpy at load time (`bicubic_resize`).
  Output: tokens (N, 257, 384) after the final norm; [CLS] = tokens[:, 0].
* V-JEPA 2.1 ViT-L/16, distilled from ViT-G (`dgrauet/vjepa-2.1-vitl-mlx`, the MLX conversion of
  `vjepa2_1_vitl_dist_vitG_384.pt`, key `ema_encoder`). Written from facebookresearch/vjepa2
  app/vjepa_2_1/models (vision_transformer.py, utils/modules.py; commit 204698b): the image path is
  `patch_embed_img` (Conv3d, tubelet 1) + `img_mod_embed`, no absolute positions; every block uses 3-axis
  RoPE on the first 60 of 64 head dims (20 each for time, height, width; interleaved pairs,
  omega_i = 10000^(-i/10)), with `interpolate_rope`: height/width positions scaled by (16-1)/(H_p-1), so a
  14x14 grid at 224 px spans the 16x16 pretraining range. Output = norms_block[-1](x) (the eval path),
  tokens (N, 196, 1024).
* LeWM (`quentinll/lewm-*`, `weights.pt` state_dicts). Written from lucas-maes/le-wm (module.py, jepa.py;
  commit 8edfeb3) and the HF ViT it instantiates through stable_pretraining.vit_hf ("tiny", patch 14,
  224 px; HF ViTConfig defaults: LN eps 1e-12, exact GELU, qkv bias, no pooler): embedding =
  projector([CLS] of the last hidden state); projector / pred_proj = Linear 192-2048, BatchNorm1d, GELU,
  Linear 2048-192; predictor = ARPredictor (learned 3-frame positions; 6 ConditionalBlocks: AdaLN-zero
  modulation from the action embedding, non-affine LN eps 1e-6 before the modulation, and inside each
  Attention / FeedForward the official extra affine LayerNorm (eps 1e-5); causal attention, 16 heads x 64;
  final LN) and the action Embedder (Conv1d k=1 to 10 dims, Linear 10-768, SiLU, Linear 768-192).
Torch is never used here; `rep_torchref.py` holds the torch reference modules for the parity tests only.
"""
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRE = ROOT / "data" / "pretrained"
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
LEWM_CKPTS = ("tworooms", "pusht", "cube", "reacher")


def normalize_u8(frames):
    """uint8 (..., H, W, 3) mx/np array -> float32 ImageNet-normalised mx array."""
    x = mx.array(frames).astype(mx.float32) / 255.0
    return (x - mx.array(IMNET_MEAN)) / mx.array(IMNET_STD)


# ----------------------------------------------------------------------------------------- shared ViT
class ViTAttention(nn.Module):
    """Fused-qkv multi-head self-attention (no RoPE)."""

    def __init__(self, dim, heads, bias=True):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).transpose(2, 0, 3, 1, 4)
        o = mx.fast.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=(C // self.heads) ** -0.5)
        return self.proj(o.transpose(0, 2, 1, 3).reshape(B, N, C))


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


def _conv_w(w):
    """torch Conv2d weight (O, I, kh, kw) -> MLX (O, kh, kw, I)."""
    return np.ascontiguousarray(np.asarray(w).transpose(0, 2, 3, 1))


# ----------------------------------------------------------------------------- bicubic (torch-exact)
def _cubic_coeffs(t, A=-0.75):
    def c1(x):
        return ((A + 2) * x - (A + 3)) * x * x + 1

    def c2(x):
        return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A
    return np.stack([c2(t + 1.0), c1(t), c1(1.0 - t), c2(2.0 - t)], -1)


def bicubic_resize(a, out_hw, scale_factor=None):
    """torch.nn.functional.interpolate(mode='bicubic', align_corners=False, antialias=False) on the first
    two axes of a (H, W, C) float array. With scale_factor (sy, sx) the source coordinates use 1/scale
    (torch's rule when a scale factor is passed), else in/out."""
    a = np.asarray(a, np.float32)
    for axis, n_out in enumerate(out_hw):
        n_in = a.shape[axis]
        scale = (1.0 / scale_factor[axis]) if scale_factor is not None else n_in / n_out
        real = np.float32(scale) * (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) - np.float32(0.5)
        i0 = np.floor(real).astype(np.int64)
        t = (real - i0).astype(np.float32)
        w = _cubic_coeffs(t).astype(np.float32)                   # (n_out, 4)
        idx = np.clip(i0[:, None] + np.arange(-1, 3)[None], 0, n_in - 1)
        g = np.take(a, idx.reshape(-1), axis=axis)
        shp = list(a.shape)
        shp[axis:axis + 1] = [n_out, 4]
        g = g.reshape(shp)
        wshape = [1] * g.ndim
        wshape[axis], wshape[axis + 1] = n_out, 4
        a = (g * w.reshape(wshape)).sum(axis + 1)
    return a


# --------------------------------------------------------------------------------------------- DINOv2
class DinoBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn = ViTAttention(dim, heads)
        self.mlp = MLP(dim, 4 * dim)
        self.ls1, self.ls2 = mx.ones((dim,)), mx.ones((dim,))

    def __call__(self, x):
        x = x + self.ls1 * self.attn(self.norm1(x))
        return x + self.ls2 * self.mlp(self.norm2(x))


class DINOv2(nn.Module):
    def __init__(self, img=224, patch=14, dim=384, depth=12, heads=6):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch, stride=patch)
        self.cls_token = mx.zeros((1, 1, dim))
        self.pos_embed = mx.zeros((1, 1 + (img // patch) ** 2, dim))
        self.blocks = [DinoBlock(dim, heads) for _ in range(depth)]
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def tokens(self, x):
        t = self.patch(x)
        t = t.reshape(t.shape[0], -1, t.shape[-1])
        t = mx.concatenate([mx.broadcast_to(self.cls_token, (t.shape[0], 1, t.shape[-1])), t], 1) + self.pos_embed
        for b in self.blocks:
            t = b(t)
        return self.norm(t)

    def __call__(self, frames_u8):
        return self.tokens(normalize_u8(frames_u8))


def dinov2_pos_224(pos, img=224, patch=14, offset=0.1):
    """Official DINOv2 interpolate_pos_encoding for a square img: (1, 1+M*M, d) -> (1, 1+g*g, d)."""
    pos = np.asarray(pos, np.float32)
    M = int(round(math.sqrt(pos.shape[1] - 1)))
    g = img // patch
    s = float(g + offset) / M
    grid = bicubic_resize(pos[0, 1:].reshape(M, M, -1), (g, g), scale_factor=(s, s))
    return np.concatenate([pos[:, :1], grid.reshape(1, g * g, -1)], 1)


def dinov2_weights_from_hf(w):
    """HF Dinov2Model state (name -> np) -> this module's names."""
    out = {"patch.weight": _conv_w(w["embeddings.patch_embeddings.projection.weight"]),
           "patch.bias": w["embeddings.patch_embeddings.projection.bias"],
           "cls_token": w["embeddings.cls_token"],
           "pos_embed": dinov2_pos_224(w["embeddings.position_embeddings"]),
           "norm.weight": w["layernorm.weight"], "norm.bias": w["layernorm.bias"]}
    i = 0
    while f"encoder.layer.{i}.norm1.weight" in w:
        p, q = f"encoder.layer.{i}.", f"blocks.{i}."
        out[q + "attn.qkv.weight"] = np.concatenate([w[p + f"attention.attention.{n}.weight"]
                                                     for n in ("query", "key", "value")], 0)
        out[q + "attn.qkv.bias"] = np.concatenate([w[p + f"attention.attention.{n}.bias"]
                                                   for n in ("query", "key", "value")], 0)
        out[q + "attn.proj.weight"] = w[p + "attention.output.dense.weight"]
        out[q + "attn.proj.bias"] = w[p + "attention.output.dense.bias"]
        for a, b in (("norm1", "norm1"), ("norm2", "norm2"), ("mlp.fc1", "mlp.fc1"), ("mlp.fc2", "mlp.fc2")):
            out[q + a + ".weight"], out[q + a + ".bias"] = w[p + b + ".weight"], w[p + b + ".bias"]
        out[q + "ls1"], out[q + "ls2"] = w[p + "layer_scale1.lambda1"], w[p + "layer_scale2.lambda1"]
        i += 1
    return out


def load_dinov2(path=PRE / "dinov2-small" / "model.safetensors"):
    w = {k: np.array(v) for k, v in mx.load(str(path)).items()}
    m = DINOv2()
    m.load_weights([(k, mx.array(v)) for k, v in dinov2_weights_from_hf(w).items()], strict=True)
    m.eval()
    return m


# ------------------------------------------------------------------------------------- V-JEPA 2.1 L
def vjepa_rope_tables(H, W, T=1, head_dim=64, pretrained_grid=16, interpolate=True):
    """cos/sin (T*H*W, 3*seg) for the d, h, w segments, float32 as the official code computes them."""
    seg = int(2 * ((head_dim // 3) // 2))
    ids = np.arange(T * H * W)
    tpf = H * W
    d = ids // tpf
    h = (ids - tpf * d) // W
    w_ = (ids - tpf * d) - W * h
    d, h, w_ = (x.astype(np.float32) for x in (d, h, w_))
    if interpolate:
        h = (h * np.float32(pretrained_grid - 1)) / np.float32(H - 1)
        w_ = (w_ * np.float32(pretrained_grid - 1)) / np.float32(W - 1)
    om = np.arange(seg // 2, dtype=np.float32) / np.float32(seg / 2.0)
    om = np.float32(1.0) / (np.float32(10000.0) ** om)
    cos, sin = [], []
    for p in (d, h, w_):
        f = (p[:, None] * om[None]).astype(np.float32)
        cos.append(np.repeat(np.cos(f), 2, -1))
        sin.append(np.repeat(np.sin(f), 2, -1))
    return np.concatenate(cos, -1).astype(np.float32), np.concatenate(sin, -1).astype(np.float32), seg


def rope_rotate(x, cos, sin):
    """Interleaved-pair rotation of the last axis (official rotate_queries_or_keys)."""
    x2 = x.reshape(*x.shape[:-1], -1, 2)
    y = mx.stack([-x2[..., 1], x2[..., 0]], axis=-1).reshape(x.shape)
    return x * cos + y * sin


class RopeAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x, cos, sin):
        B, N, C = x.shape
        hd = C // self.heads
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, hd).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        r = cos.shape[-1]
        q = mx.concatenate([rope_rotate(q[..., :r], cos, sin), q[..., r:]], -1)
        k = mx.concatenate([rope_rotate(k[..., :r], cos, sin), k[..., r:]], -1)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5)
        return self.proj(o.transpose(0, 2, 1, 3).reshape(B, N, C))


class VJBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn = RopeAttention(dim, heads)
        self.mlp = MLP(dim, 4 * dim)

    def __call__(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class VJEPA21(nn.Module):
    def __init__(self, img=224, patch=16, dim=1024, depth=24, heads=16):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch, stride=patch)   # = patch_embed_img (Conv3d, tubelet 1)
        self.img_mod_embed = mx.zeros((1, 1, dim))
        self.blocks = [VJBlock(dim, heads) for _ in range(depth)]
        self.norms_block = [nn.LayerNorm(dim, eps=1e-6) for _ in range(4)]
        g = img // patch
        cos, sin, _ = vjepa_rope_tables(g, g, 1, dim // heads)
        self._cos, self._sin = mx.array(cos), mx.array(sin)

    def tokens(self, x):
        t = self.patch(x)
        t = t.reshape(t.shape[0], -1, t.shape[-1]) + self.img_mod_embed
        for b in self.blocks:
            t = b(t, self._cos, self._sin)
        return self.norms_block[-1](t)

    def __call__(self, frames_u8):
        return self.tokens(normalize_u8(frames_u8))


def vjepa_weights_from_mlx_release(w):
    """dgrauet/vjepa-2.1-vitl-mlx encoder.safetensors -> this module's names (image path only)."""
    out = {"patch.weight": np.asarray(w["patch_embed_img.proj.weight"])[:, 0],   # (O, 1, kh, kw, I) -> (O, kh, kw, I)
           "patch.bias": w["patch_embed_img.proj.bias"], "img_mod_embed": w["img_mod_embed"]}
    for k, v in w.items():
        if k.startswith("blocks.") or k.startswith("norms_block."):
            out[k] = v
    return out


def load_vjepa21(path=PRE / "vjepa-2.1-vitl-mlx" / "encoder.safetensors", dtype=mx.float32):
    w = {k: np.array(v) for k, v in mx.load(str(path)).items()}
    m = VJEPA21()
    m.load_weights([(k, mx.array(v)) for k, v in vjepa_weights_from_mlx_release(w).items()], strict=True)
    if dtype != mx.float32:
        m.set_dtype(dtype)
    m.eval()
    return m


# ------------------------------------------------------------------------- random-init ViT control
class RandomViT(nn.Module):
    """Control arm: the LeWM encoder architecture (ViT-tiny/14 at 224, HF init: trunc-normal 0.02,
    zero biases, unit LayerNorms), never trained. Output tokens (N, 257, 192)."""

    def __init__(self, seed=0):
        super().__init__()
        self.enc = LeWMEncoder()
        init_hf_vit(self.enc, seed)

    def __call__(self, frames_u8):
        return self.enc.tokens(normalize_u8(frames_u8))


# ----------------------------------------------------------------------------------- official LeWM
class HFViTLayer(nn.Module):
    def __init__(self, dim=192, heads=3, eps=1e-12):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(dim, eps=eps)
        self.attn = ViTAttention(dim, heads)
        self.layernorm_after = nn.LayerNorm(dim, eps=eps)
        self.mlp = MLP(dim, 4 * dim)

    def __call__(self, x):
        x = x + self.attn(self.layernorm_before(x))
        return x + self.mlp(self.layernorm_after(x))


class LeWMEncoder(nn.Module):
    def __init__(self, img=224, patch=14, dim=192, depth=12, heads=3):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch, stride=patch)
        self.cls_token = mx.zeros((1, 1, dim))
        self.pos = mx.zeros((1, 1 + (img // patch) ** 2, dim))
        self.layers = [HFViTLayer(dim, heads) for _ in range(depth)]
        self.layernorm = nn.LayerNorm(dim, eps=1e-12)

    def tokens(self, x):
        t = self.patch(x)
        t = t.reshape(t.shape[0], -1, t.shape[-1])
        t = mx.concatenate([mx.broadcast_to(self.cls_token, (t.shape[0], 1, t.shape[-1])), t], 1) + self.pos
        for layer in self.layers:
            t = layer(t)
        return self.layernorm(t)

    def __call__(self, x):
        return self.tokens(x)[:, 0]


class ProjMLP(nn.Module):
    """module.MLP with BatchNorm1d: Linear, BN, GELU, Linear."""

    def __init__(self, d=192, hid=2048):
        super().__init__()
        self.l0, self.bn, self.l3 = nn.Linear(d, hid), nn.BatchNorm(hid), nn.Linear(hid, d)

    def __call__(self, x):
        return self.l3(nn.gelu(self.bn(self.l0(x))))


class Embedder(nn.Module):
    def __init__(self, input_dim, smoothed_dim=10, emb_dim=192, mlp_scale=4):
        super().__init__()
        self.patch = nn.Linear(input_dim, smoothed_dim)            # Conv1d kernel 1 over time
        self.e0, self.e2 = nn.Linear(smoothed_dim, mlp_scale * emb_dim), nn.Linear(mlp_scale * emb_dim, emb_dim)

    def __call__(self, a):
        return self.e2(nn.silu(self.e0(self.patch(a))))


def _dropout(x, p, training):
    if not training or p <= 0:
        return x
    keep = mx.random.bernoulli(1 - p, x.shape)
    return mx.where(keep, x / (1 - p), mx.zeros_like(x))


class PredAttention(nn.Module):
    def __init__(self, dim=192, heads=16, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads, self.dh, self.p = heads, dim_head, dropout
        self.norm = nn.LayerNorm(dim)                                # eps 1e-5, affine
        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, dim)

    def __call__(self, x):
        B, T, _ = x.shape
        x = self.norm(x)
        q, k, v = mx.split(self.to_qkv(x), 3, axis=-1)
        q, k, v = (t.reshape(B, T, self.heads, self.dh).transpose(0, 2, 1, 3) for t in (q, k, v))
        mask = mx.triu(mx.full((T, T), -mx.inf, dtype=x.dtype), k=1)
        if self.training and self.p > 0:                             # attention-weight dropout (SDPA dropout_p)
            a = mx.softmax(q @ k.transpose(0, 1, 3, 2) * self.dh ** -0.5 + mask, axis=-1)
            o = _dropout(a, self.p, True) @ v
        else:
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.dh ** -0.5, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return _dropout(self.to_out(o), self.p, self.training)


class PredFF(nn.Module):
    def __init__(self, dim=192, hidden=2048, dropout=0.0):
        super().__init__()
        self.p = dropout
        self.norm, self.l1, self.l4 = nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def __call__(self, x):
        h = _dropout(nn.gelu(self.l1(self.norm(x))), self.p, self.training)
        return _dropout(self.l4(h), self.p, self.training)


class CondBlock(nn.Module):
    def __init__(self, dim=192, heads=16, dim_head=64, mlp=2048, dropout=0.0):
        super().__init__()
        self.attn = PredAttention(dim, heads, dim_head, dropout)
        self.mlp = PredFF(dim, mlp, dropout)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.ada = nn.Linear(dim, 6 * dim)
        self.ada.weight = mx.zeros_like(self.ada.weight)
        self.ada.bias = mx.zeros_like(self.ada.bias)

    def __call__(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = mx.split(self.ada(nn.silu(c)), 6, axis=-1)
        x = x + g1 * self.attn(self.norm1(x) * (1 + sc1) + sh1)
        return x + g2 * self.mlp(self.norm2(x) * (1 + sc2) + sh2)


class ARPredictor(nn.Module):
    def __init__(self, num_frames=3, dim=192, depth=6, heads=16, dim_head=64, mlp=2048, dropout=0.1):
        super().__init__()
        self.pos = mx.zeros((1, num_frames, dim))
        self.layers = [CondBlock(dim, heads, dim_head, mlp, dropout) for _ in range(depth)]
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x, c):
        x = x + self.pos[:, :x.shape[1]]
        for b in self.layers:
            x = b(x, c)
        return self.norm(x)


class LeWMOfficial(nn.Module):
    def __init__(self, act_dim, dropout=0.1):
        super().__init__()
        self.encoder = LeWMEncoder()
        self.projector = ProjMLP()
        self.predictor = ARPredictor(dropout=dropout)
        self.action_encoder = Embedder(act_dim)
        self.pred_proj = ProjMLP()

    def cls(self, frames_u8):
        sh = frames_u8.shape
        x = normalize_u8(frames_u8.reshape(-1, *sh[-3:]))
        return self.encoder(x).reshape(*sh[:-3], -1)

    def encode(self, frames_u8):
        """uint8 (..., 224, 224, 3) -> embedding (..., 192) = projector([CLS])."""
        sh = frames_u8.shape
        c = self.cls(frames_u8).reshape(-1, 192)
        return self.projector(c).reshape(*sh[:-3], 192)

    def predict(self, emb, act):
        """emb (B, T, 192), act (B, T, act_dim) -> predicted next embeddings (B, T, 192)."""
        B, T, d = emb.shape
        h = self.predictor(emb, self.action_encoder(act))
        return self.pred_proj(h.reshape(B * T, d)).reshape(B, T, d)


def _np(t):
    return np.asarray(t, np.float32)


def lewm_weights_from_torch(sd, with_action_encoder=True):
    """Official LeWM state_dict (name -> np array) -> LeWMOfficial names."""
    o = {"encoder.patch.weight": _conv_w(sd["encoder.embeddings.patch_embeddings.projection.weight"]),
         "encoder.patch.bias": sd["encoder.embeddings.patch_embeddings.projection.bias"],
         "encoder.cls_token": sd["encoder.embeddings.cls_token"],
         "encoder.pos": sd["encoder.embeddings.position_embeddings"],
         "encoder.layernorm.weight": sd["encoder.layernorm.weight"],
         "encoder.layernorm.bias": sd["encoder.layernorm.bias"]}
    i = 0
    while f"encoder.encoder.layer.{i}.layernorm_before.weight" in sd:
        p, q = f"encoder.encoder.layer.{i}.", f"encoder.layers.{i}."
        for n in ("weight", "bias"):
            o[q + "attn.qkv." + n] = np.concatenate([sd[p + f"attention.attention.{x}.{n}"]
                                                    for x in ("query", "key", "value")], 0)
            o[q + "attn.proj." + n] = sd[p + "attention.output.dense." + n]
            o[q + "mlp.fc1." + n] = sd[p + "intermediate.dense." + n]
            o[q + "mlp.fc2." + n] = sd[p + "output.dense." + n]
            o[q + "layernorm_before." + n] = sd[p + "layernorm_before." + n]
            o[q + "layernorm_after." + n] = sd[p + "layernorm_after." + n]
        i += 1
    for src, dst in (("projector", "projector"), ("pred_proj", "pred_proj")):
        for n in ("weight", "bias"):
            o[f"{dst}.l0.{n}"] = sd[f"{src}.net.0.{n}"]
            o[f"{dst}.bn.{n}"] = sd[f"{src}.net.1.{n}"]
            o[f"{dst}.l3.{n}"] = sd[f"{src}.net.3.{n}"]
        o[f"{dst}.bn.running_mean"] = sd[f"{src}.net.1.running_mean"]
        o[f"{dst}.bn.running_var"] = sd[f"{src}.net.1.running_var"]
    o["predictor.pos"] = sd["predictor.pos_embedding"]
    o["predictor.norm.weight"] = sd["predictor.transformer.norm.weight"]
    o["predictor.norm.bias"] = sd["predictor.transformer.norm.bias"]
    i = 0
    while f"predictor.transformer.layers.{i}.attn.norm.weight" in sd:
        p, q = f"predictor.transformer.layers.{i}.", f"predictor.layers.{i}."
        o[q + "attn.to_qkv.weight"] = sd[p + "attn.to_qkv.weight"]
        for n in ("weight", "bias"):
            o[q + "attn.norm." + n] = sd[p + "attn.norm." + n]
            o[q + "attn.to_out." + n] = sd[p + "attn.to_out.0." + n]
            o[q + "mlp.norm." + n] = sd[p + "mlp.net.0." + n]
            o[q + "mlp.l1." + n] = sd[p + "mlp.net.1." + n]
            o[q + "mlp.l4." + n] = sd[p + "mlp.net.4." + n]
            o[q + "ada." + n] = sd[p + "adaLN_modulation.1." + n]
        i += 1
    if with_action_encoder:
        o["action_encoder.patch.weight"] = np.asarray(sd["action_encoder.patch_embed.weight"])[:, :, 0]
        o["action_encoder.patch.bias"] = sd["action_encoder.patch_embed.bias"]
        for n in ("weight", "bias"):
            o["action_encoder.e0." + n] = sd["action_encoder.embed.0." + n]
            o["action_encoder.e2." + n] = sd["action_encoder.embed.2." + n]
    return {k: _np(v) for k, v in o.items()}


def read_lewm_state(name):
    """The official weights.pt (a plain state_dict) -> numpy, read with torch.load(weights_only=True).
    Torch is used here ONLY to read the published file (owner-approved); nothing runs in torch."""
    import torch
    sd = torch.load(str(PRE / f"lewm-{name}" / "weights.pt"), map_location="cpu", weights_only=True)
    return {k: v.numpy() for k, v in sd.items() if v.dtype != torch.int64}


def lewm_state_cached(name):
    """numpy copy of the official state (data/r8_rep/lewm_np/<name>.npz, written once; the download directory
    stays as downloaded), so the training/eval path needs no torch at all."""
    p = ROOT / "data" / "r8_rep" / "lewm_np" / f"{name}.npz"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **read_lewm_state(name))
    z = np.load(p)
    return {k: z[k] for k in z.files}


def load_lewm_official(name, act_dim=None, with_action_encoder=True, dropout=0.1):
    sd = lewm_state_cached(name)
    a = sd["action_encoder.patch_embed.weight"].shape[1] if act_dim is None else act_dim
    m = LeWMOfficial(a, dropout=dropout)
    w = lewm_weights_from_torch(sd, with_action_encoder=with_action_encoder)
    m.load_weights([(k, mx.array(v)) for k, v in w.items()], strict=with_action_encoder)
    m.eval()
    return m


# ----------------------------------------------------------------------------------------- inits
def _trunc_normal(shape, std, key):
    return mx.clip(mx.random.normal(shape, key=key), -2.0, 2.0) * std


def init_hf_vit(enc, seed):
    """HF ViTPreTrainedModel._init_weights: Linear/Conv weights trunc-normal(0.02), biases 0, LN (1, 0),
    position embeddings and [CLS] trunc-normal(0.02)."""
    keys = iter(mx.random.split(mx.random.key(seed), 1000))
    enc.patch.weight = _trunc_normal(enc.patch.weight.shape, 0.02, next(keys))
    enc.patch.bias = mx.zeros_like(enc.patch.bias)
    enc.cls_token = _trunc_normal(enc.cls_token.shape, 0.02, next(keys))
    enc.pos = _trunc_normal(enc.pos.shape, 0.02, next(keys))
    for layer in enc.layers:
        for lin in (layer.attn.qkv, layer.attn.proj, layer.mlp.fc1, layer.mlp.fc2):
            lin.weight = _trunc_normal(lin.weight.shape, 0.02, next(keys))
            lin.bias = mx.zeros_like(lin.bias)
    return enc


def init_scratch_lewm(model, seed):
    """Scratch arm init, as the official code would build it untrained: HF ViT init for the encoder, torch
    defaults for Linear layers (MLX's nn.Linear default is the same U(+-1/sqrt(fan_in))), BN (1, 0, running
    0/1), AdaLN zero, predictor positions ~ N(0, 1) (torch.randn)."""
    mx.random.seed(seed)
    fresh = LeWMOfficial(model.action_encoder.patch.weight.shape[1])
    init_hf_vit(fresh.encoder, seed)
    fresh.predictor.pos = mx.random.normal(fresh.predictor.pos.shape, key=mx.random.key(seed + 17))
    from mlx.utils import tree_flatten
    model.load_weights(tree_flatten(fresh.parameters()), strict=True)
    return model


def param_count(m):
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(m.parameters()))


if __name__ == "__main__":
    for n in LEWM_CKPTS:
        m = load_lewm_official(n)
        print(n, param_count(m))
    print("dinov2", param_count(load_dinov2()))
    print("vjepa", param_count(load_vjepa21()))
    print(json.dumps({"ok": True}))
