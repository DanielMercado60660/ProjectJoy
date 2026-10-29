"""Nanochat pilot GPT (copied from 30_nanochat_mlx/gpt.py), in three dialects:
MLX (reference), NumPy (baseline) and torch (Core ML conversion frontend only).

d6 pilot: 6 layers, 384 wide, 3 heads of 128, vocab 8192, 16.9M params. Rotary (base 1e5, rotate
halves), QK-norm x1.2 after rope, ReLU^2 MLP, parameter-free RMSNorm, per-layer resid/x0 lambdas,
untied lm_head, logit softcap 15. Linear layers are nn.Linear(bias=False) here (the original uses
its own bias-free Linear with the same weight layout) so that nn.quantize can find them.

Decode uses a pure function (token, past_k, past_v) -> logits, new_k, new_v with the cache passed
as arrays (concatenated each step) instead of the original's mutable preallocated LayerKV buffer:
same math, and it compiles.
"""
import json
import math
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

CKPT = "../30_nanochat_mlx/checkpoints/pilot/last"          # README: the 16.9M d6 pilot
VAL_TOKENS = "../30_nanochat_mlx/data/tokens/val.npy"       # held-out shard_06542, tokenized
TOKEN_BYTES = "../30_nanochat_mlx/checkpoints/tokenizer/token_bytes.json"
ROPE_BASE = 100000.0
SOFTCAP = 15.0


def load_config(path=CKPT):
    with open(os.path.join(path, "meta.json")) as f:
        return json.load(f)["config"]


def norm(x):
    return mx.fast.rms_norm(x, None, 1e-6)


# ---------- MLX ----------
class Attn(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh, self.hd = nh, d // nh
        self.c_q = nn.Linear(d, d, bias=False)
        self.c_k = nn.Linear(d, d, bias=False)
        self.c_v = nn.Linear(d, d, bias=False)
        self.c_proj = nn.Linear(d, d, bias=False)

    def __call__(self, x, past=None):
        B, T, C = x.shape
        q, k, v = (l(x).reshape(B, T, self.nh, self.hd).transpose(0, 2, 1, 3) for l in (self.c_q, self.c_k, self.c_v))
        off = 0 if past is None else past[0].shape[2]
        q = mx.fast.rope(q, self.hd, traditional=False, base=ROPE_BASE, scale=1.0, offset=off)
        k = mx.fast.rope(k, self.hd, traditional=False, base=ROPE_BASE, scale=1.0, offset=off)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        new_k, new_v = k, v
        if past is not None:
            k = mx.concatenate([past[0], k], axis=2)
            v = mx.concatenate([past[1], v], axis=2)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(self.hd),
                                                 mask="causal" if T > 1 else None)
        return self.c_proj(y.transpose(0, 2, 1, 3).reshape(B, T, C)), (new_k, new_v)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.c_fc = nn.Linear(d, 4 * d, bias=False)
        self.c_proj = nn.Linear(4 * d, d, bias=False)

    def __call__(self, x):
        return self.c_proj(mx.square(nn.relu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.attn = Attn(d, nh)
        self.mlp = MLP(d)

    def __call__(self, x, past=None):
        a, kv = self.attn(norm(x), past)
        x = x + a
        return x + self.mlp(norm(x)), kv


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, **_):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.h = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.resid_lambdas = mx.ones((n_layer,))
        self.x0_lambdas = mx.zeros((n_layer,))
        self.compute_dtype = mx.float32

    def forward(self, idx, past=None):
        """idx (B,T) int -> softcapped fp32 logits (B,T,V), list of per-layer (k, v) for these T tokens.
        past: list of per-layer (k, v) with shape (B, H, T_past, D), or None."""
        dt = self.compute_dtype
        x = norm(self.wte(idx).astype(dt))
        x0 = x
        kvs = []
        for i, b in enumerate(self.h):
            x = self.resid_lambdas[i].astype(dt) * x + self.x0_lambdas[i].astype(dt) * x0
            x, kv = b(x, None if past is None else past[i])
            kvs.append(kv)
        logits = self.lm_head(norm(x)).astype(mx.float32)
        return SOFTCAP * mx.tanh(logits / SOFTCAP), kvs

    def __call__(self, idx):
        return self.forward(idx)[0]


def load_mlx(dtype=mx.float32, path=CKPT):
    """Weights are fp32 masters. dtype is the compute (activation and weight) dtype."""
    m = GPT(**load_config(path))
    m.load_weights(os.path.join(path, "model.safetensors"))
    if dtype != mx.float32:
        m.set_dtype(dtype)
    m.compute_dtype = dtype
    m.eval()
    mx.eval(m.parameters())
    return m


def load_quantized(bits, group_size=64, path=CKPT):
    """nn.quantize every nn.Linear and the token embedding (MLX default predicate); activations fp32."""
    m = load_mlx(mx.float32, path)
    nn.quantize(m, group_size=group_size, bits=bits)
    assert isinstance(m.h[0].attn.c_q, nn.QuantizedLinear) and isinstance(m.lm_head, nn.QuantizedLinear)
    mx.eval(m.parameters())
    return m


# ---------- NumPy baseline ----------
def weights_numpy(path=CKPT):
    return {k: np.array(v, dtype=np.float32) for k, v in mx.load(os.path.join(path, "model.safetensors")).items()}


def _np_norm(x, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def rope_tables(T, hd, offset=0, base=ROPE_BASE):
    """cos, sin of shape (T, hd/2) for positions offset..offset+T-1 (same as mx.fast.rope, traditional=False)."""
    inv = base ** (-np.arange(0, hd // 2, dtype=np.float64) * 2.0 / hd)
    ang = np.arange(offset, offset + T, dtype=np.float64)[:, None] * inv[None]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def _np_rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class NumpyGPT:
    def __init__(self, path=CKPT):
        cfg = load_config(path)
        self.L, self.nh, self.d = cfg["n_layer"], cfg["n_head"], cfg["n_embd"]
        self.hd = self.d // self.nh
        w = weights_numpy(path)
        self.wte = w["wte.weight"]
        self.lm_head_t = np.ascontiguousarray(w["lm_head.weight"].T)
        self.rl, self.xl = w["resid_lambdas"], w["x0_lambdas"]
        self.layers = []
        for i in range(self.L):
            p = f"h.{i}."
            qkv = np.concatenate([w[p + f"attn.c_{n}.weight"] for n in "qkv"], 0)
            self.layers.append(dict(qkv=np.ascontiguousarray(qkv.T), proj=np.ascontiguousarray(w[p + "attn.c_proj.weight"].T),
                                    fc=np.ascontiguousarray(w[p + "mlp.c_fc.weight"].T),
                                    fc2=np.ascontiguousarray(w[p + "mlp.c_proj.weight"].T)))

    def forward(self, idx, past=None):
        """idx (1,T) -> logits (1,T,V) fp32, list of (k, v) (H, T, D) per layer. Batch 1 only."""
        idx = np.asarray(idx).reshape(-1)
        T = idx.shape[0]
        off = 0 if past is None else past[0][0].shape[1]
        cos, sin = rope_tables(T, self.hd, off)
        x = _np_norm(self.wte[idx])
        x0 = x
        kvs = []
        scale = 1.0 / math.sqrt(self.hd)
        for i, l in enumerate(self.layers):
            x = self.rl[i] * x + self.xl[i] * x0
            h = _np_norm(x) @ l["qkv"]                                          # (T, 3d)
            q, k, v = (h[:, j * self.d:(j + 1) * self.d].reshape(T, self.nh, self.hd).transpose(1, 0, 2) for j in range(3))
            q = _np_norm(_np_rope(q, cos, sin)) * 1.2
            k = _np_norm(_np_rope(k, cos, sin)) * 1.2
            kvs.append((k, v))
            if past is not None:
                k = np.concatenate([past[i][0], k], 1)
                v = np.concatenate([past[i][1], v], 1)
            s = (q @ k.transpose(0, 2, 1)) * scale                               # (H, T, Tk)
            if T > 1:
                Tk = k.shape[1]
                s = s + np.triu(np.full((T, Tk), -np.inf, np.float32), k=Tk - T + 1)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            y = (s @ v).transpose(1, 0, 2).reshape(T, self.d)
            x = x + y @ l["proj"]
            m = np.maximum(_np_norm(x) @ l["fc"], 0)
            x = x + (m * m) @ l["fc2"]
        logits = _np_norm(x) @ self.lm_head_t
        return (SOFTCAP * np.tanh(logits / SOFTCAP))[None], kvs


# ---------- torch (Core ML frontend only): static shapes ----------
def torch_prefill(T, path=CKPT):
    """idx (1,T) -> logits (1,T,V)."""
    return torch_gpt(T, path=path)


def torch_gpt(T, T_past=0, return_kv=False, path=CKPT):
    """T new tokens after T_past cached ones. Inputs idx (1,T) [, past_k, past_v (L,H,T_past,D)].
    Outputs logits (1,T,V) [, new_k, new_v (L,H,T,D) for these T tokens]."""
    import torch
    import torch.nn as tnn
    cfg = load_config(path)
    L, nh, d, V = cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["vocab_size"]
    hd = d // nh
    w = weights_numpy(path)
    cos, sin = rope_tables(T, hd, offset=T_past)

    def rms(x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)

    class Net(tnn.Module):
        def __init__(s):
            super().__init__()
            s.wte = tnn.Embedding(V, d)
            s.lin = tnn.ModuleDict()
            for i in range(L):
                for n in ("c_q", "c_k", "c_v", "c_proj"):
                    s.lin[f"{i}_attn_{n}"] = tnn.Linear(d, d, bias=False)
                s.lin[f"{i}_mlp_c_fc"] = tnn.Linear(d, 4 * d, bias=False)
                s.lin[f"{i}_mlp_c_proj"] = tnn.Linear(4 * d, d, bias=False)
            s.lm_head = tnn.Linear(d, V, bias=False)
            s.register_buffer("cos", torch.from_numpy(cos))
            s.register_buffer("sin", torch.from_numpy(sin))
            s.register_buffer("mask", torch.from_numpy(np.triu(np.full((T, T_past + T), -1e4, np.float32), k=T_past + 1)))
            s.rl = [float(a) for a in w["resid_lambdas"]]
            s.xl = [float(a) for a in w["x0_lambdas"]]

        def rope(s, x):
            h = hd // 2
            x1, x2 = x[..., :h], x[..., h:]
            return torch.cat([x1 * s.cos - x2 * s.sin, x1 * s.sin + x2 * s.cos], dim=-1)

        def forward(s, idx, past_k=None, past_v=None):
            x = rms(s.wte(idx.reshape(-1)))                                    # (T, d)
            x0 = x
            ks, vs = [], []
            for i in range(L):
                g = lambda n: s.lin[f"{i}_{n}"]
                x = s.rl[i] * x + s.xl[i] * x0
                h = rms(x)
                q, k, v = (g(f"attn_{n}")(h).reshape(T, nh, hd).permute(1, 0, 2) for n in ("c_q", "c_k", "c_v"))
                q = rms(s.rope(q)) * 1.2
                k = rms(s.rope(k)) * 1.2
                ks.append(k); vs.append(v)
                if past_k is not None:
                    k = torch.cat([past_k[i], k], dim=1)
                    v = torch.cat([past_v[i], v], dim=1)
                a = torch.softmax(q @ k.transpose(1, 2) * (1.0 / math.sqrt(hd)) + s.mask, dim=-1)
                x = x + g("attn_c_proj")((a @ v).permute(1, 0, 2).reshape(T, d))
                m = torch.relu(g("mlp_c_fc")(rms(x)))
                x = x + g("mlp_c_proj")(m * m)
            logits = s.lm_head(rms(x))
            logits = (SOFTCAP * torch.tanh(logits / SOFTCAP)).reshape(1, T, V)
            if return_kv:
                return logits, torch.stack(ks), torch.stack(vs)
            return logits

    net = Net().eval()
    with torch.no_grad():
        net.wte.weight.copy_(torch.from_numpy(w["wte.weight"]))
        net.lm_head.weight.copy_(torch.from_numpy(w["lm_head.weight"]))
        for i in range(L):
            for n in ("c_q", "c_k", "c_v", "c_proj"):
                net.lin[f"{i}_attn_{n}"].weight.copy_(torch.from_numpy(w[f"h.{i}.attn.{n}.weight"]))
            for n in ("c_fc", "c_proj"):
                net.lin[f"{i}_mlp_{n}"].weight.copy_(torch.from_numpy(w[f"h.{i}.mlp.{n}.weight"]))
    return net


# ---------- held-out data ----------
def val_tokens():
    return np.load(VAL_TOKENS, mmap_mode="r")


def token_bytes():
    with open(TOKEN_BYTES) as f:
        return np.array(json.load(f), dtype=np.int64)


def val_window(T, seed):
    """T+1 consecutive held-out tokens at a seed-determined offset (int32)."""
    a = val_tokens()
    s = int(np.random.default_rng(seed).integers(0, len(a) - T - 2))
    return np.asarray(a[s:s + T + 1], dtype=np.int32)
