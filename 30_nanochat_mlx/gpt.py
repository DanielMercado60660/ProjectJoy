"""
GPT model in MLX, ported from upstream nanochat/gpt.py.

Kept from upstream:
- rotary embeddings (base 100000, rotate-halves layout), no learned positional embeddings
- QK norm (parameter-free RMSNorm on q and k) with the 1.2 "sharper attention" scale
- untied token embedding / lm_head, norm right after the token embedding
- ReLU^2 MLP, no biases, parameter-free RMSNorm everywhere
- per-layer learnable residual scalars (resid_lambdas, x0_lambdas)
- logit softcap 15 (tanh), loss in fp32
- GQA support (n_kv_head), fp32 master weights with matmuls in the activation dtype
- upstream init (uniform for block matrices, zero output projections, std 0.001 lm_head)

Skipped (see README): value embeddings (ResFormer), smear gate, backout, sliding-window attention.
"""
import math
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 512

    def to_dict(self):
        return asdict(self)


def config_for_depth(depth, vocab_size=8192, seq_len=1024, aspect_ratio=64, head_dim=128):
    """upstream base_train: model_dim = depth * aspect_ratio, rounded up to head_dim multiple"""
    dim = ((depth * aspect_ratio + head_dim - 1) // head_dim) * head_dim
    nh = dim // head_dim
    return GPTConfig(sequence_len=seq_len, vocab_size=vocab_size, n_layer=depth, n_head=nh, n_kv_head=nh, n_embd=dim)


def norm(x):
    return mx.fast.rms_norm(x, None, 1e-6)


class Linear(nn.Module):
    """Bias-free linear; fp32 master weight, matmul in the input's dtype (upstream's trick)."""

    def __init__(self, din, dout):
        super().__init__()
        self.weight = mx.zeros((dout, din))

    def __call__(self, x):
        return x @ self.weight.astype(x.dtype).T


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head, self.n_kv_head = cfg.n_head, cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_q = Linear(cfg.n_embd, cfg.n_head * self.head_dim)
        self.c_k = Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim)
        self.c_v = Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim)
        self.c_proj = Linear(cfg.n_embd, cfg.n_embd)
        self.fused_qkv = False  # efficiency flag: one wide GEMM for q/k/v (same params, same math)

    def __call__(self, x, cache=None):
        B, T, C = x.shape
        if self.fused_qkv:
            nq, nkv = self.n_head * self.head_dim, self.n_kv_head * self.head_dim
            w = mx.concatenate([lin.weight.astype(x.dtype) for lin in (self.c_q, self.c_k, self.c_v)], axis=0)
            q, k, v = mx.split(x @ w.T, [nq, nq + nkv], axis=-1)
        else:
            q, k, v = self.c_q(x), self.c_k(x), self.c_v(x)
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache.offset
        # mx.fast.rope (traditional=False) rotates halves like upstream (by +theta instead of -theta;
        # only the relative rotation matters)
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=100000.0, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=100000.0, scale=1.0, offset=offset)
        q, k = norm(q) * 1.2, norm(k) * 1.2  # QK norm + sharper attention
        if cache is not None:
            k, v = cache.update(k, v)
        mask = "causal" if T > 1 else None
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(self.head_dim), mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = Linear(4 * cfg.n_embd, cfg.n_embd)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.square(nn.relu(x))
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

    def __call__(self, x, cache=None):
        x = x + self.attn(norm(x), cache)
        x = x + self.mlp(norm(x))
        return x


class LayerKV:
    """Per-layer KV cache: preallocated buffer, grown in chunks (like mlx-lm's KVCache)."""
    step = 256

    def __init__(self):
        self.k = self.v = None
        self.offset = 0

    def update(self, k, v):
        B, H, T, D = k.shape
        if self.k is None or self.offset + T > self.k.shape[2]:
            n = ((self.offset + T + self.step - 1) // self.step) * self.step
            nk = mx.zeros((B, H, n, D), dtype=k.dtype)
            nv = mx.zeros((B, H, n, D), dtype=v.dtype)
            if self.k is not None:
                nk[:, :, :self.offset] = self.k[:, :, :self.offset]
                nv[:, :, :self.offset] = self.v[:, :, :self.offset]
            self.k, self.v = nk, nv
        self.k[:, :, self.offset:self.offset + T] = k
        self.v[:, :, self.offset:self.offset + T] = v
        self.offset += T
        return self.k[:, :, :self.offset], self.v[:, :, :self.offset]


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.h = [Block(cfg) for _ in range(cfg.n_layer)]
        self.lm_head = Linear(cfg.n_embd, cfg.vocab_size)
        self.resid_lambdas = mx.ones((cfg.n_layer,))
        self.x0_lambdas = mx.zeros((cfg.n_layer,))
        self.compute_dtype = mx.bfloat16
        # efficiency flags (see efficiency/REPORT.md); defaults reproduce the pilot exactly
        # ce_mode: "kernel" = fused softcap+CE Metal kernel (fused_ce.py; same math, parity-checked,
        # +6% d6 tok/s; falls back to the reference on CPU), "ref" = the pilot's exact path,
        # "bf16cap" = capped logits in bf16 (CHANGES TRAINING numerics; slower than "kernel")
        self.ce_mode = "kernel"

    def set_fast_flags(self, fused_qkv=None, ce_mode=None):
        if fused_qkv is not None:
            for b in self.h:
                b.attn.fused_qkv = fused_qkv
        if ce_mode is not None:
            assert ce_mode in ("ref", "kernel", "bf16cap"), ce_mode
            self.ce_mode = ce_mode
        return self

    def init_weights(self, seed=0):
        """Same init as upstream GPT.init_weights (minus the skipped features)."""
        mx.random.seed(seed)
        cfg = self.config
        d = cfg.n_embd
        s = 3 ** 0.5 * d ** -0.5
        self.wte.weight = mx.random.normal((cfg.vocab_size, d)) * 0.8
        self.lm_head.weight = mx.random.normal((cfg.vocab_size, d)) * 0.001
        for b in self.h:
            for lin in (b.attn.c_q, b.attn.c_k, b.attn.c_v):
                lin.weight = mx.random.uniform(-s, s, lin.weight.shape)
            b.attn.c_proj.weight = mx.zeros(b.attn.c_proj.weight.shape)
            b.mlp.c_fc.weight = mx.random.uniform(-0.4 * s, 0.4 * s, b.mlp.c_fc.weight.shape)
            b.mlp.c_proj.weight = mx.zeros(b.mlp.c_proj.weight.shape)
        L = cfg.n_layer
        self.resid_lambdas = mx.array([1.15 - 0.10 * i / max(L - 1, 1) for i in range(L)])
        self.x0_lambdas = mx.array([0.20 - 0.15 * i / max(L - 1, 1) for i in range(L)])
        mx.eval(self.parameters())

    def __call__(self, idx, cache=None, raw_logits=False):
        """idx (B, T) int -> logits (B, T, V) float32 (softcapped); raw_logits: uncapped, compute dtype."""
        x = self.wte(idx).astype(self.compute_dtype)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.h):
            x = self.resid_lambdas[i].astype(x.dtype) * x + self.x0_lambdas[i].astype(x.dtype) * x0
            x = block(x, None if cache is None else cache[i])
        x = norm(x)
        if raw_logits:
            return self.lm_head(x)
        logits = self.lm_head(x).astype(mx.float32)
        softcap = 15.0
        return softcap * mx.tanh(logits / softcap)

    def make_cache(self):
        return [LayerKV() for _ in range(self.config.n_layer)]

    # ---- accounting (same conventions as upstream) ----
    def num_params(self):
        from mlx.utils import tree_flatten
        return sum(v.size for _, v in tree_flatten(self.parameters()))

    def num_matmul_params(self):
        from mlx.utils import tree_flatten
        return sum(v.size for k, v in tree_flatten(self.parameters()) if k.startswith("h.") or k.startswith("lm_head"))

    def num_scaling_params(self):
        """upstream: transformer matrices + lm_head"""
        return self.num_matmul_params()

    def estimate_flops(self):
        """Training FLOPs per token (fwd+bwd): 6 * matmul params + 12 * h * q * T attention."""
        c = self.config
        hd = c.n_embd // c.n_head
        return 6 * self.num_matmul_params() + c.n_layer * 12 * c.n_head * hd * c.sequence_len


def loss_fn(model, x, y, mask=None, reduction="mean"):
    """Cross entropy; mask (B,T) of 0/1 selects supervised positions (SFT). Returns mean loss."""
    mode = getattr(model, "ce_mode", "kernel")
    if mode == "kernel":  # fused softcap + CE (fp32 math inside one Metal kernel per direction)
        from fused_ce import capped_ce
        ce = capped_ce(model(x, raw_logits=True), y)
    elif mode == "bf16cap":  # CHANGES TRAINING: capped logits stored in bf16 before the fused CE
        raw = model(x, raw_logits=True)
        capped = 15.0 * mx.tanh(raw.astype(mx.float32) / 15.0)
        ce = mx.fast.cross_entropy(capped.astype(raw.dtype), y)
    else:
        logits = model(x)
        ce = nn.losses.cross_entropy(logits, y, reduction="none")
    if reduction == "none":
        return ce
    if mask is None:
        return ce.mean()
    mask = mask.astype(ce.dtype)
    return (ce * mask).sum() / mx.maximum(mask.sum(), 1.0)
