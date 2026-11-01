"""Gemma 3 text decoder (270M config) in plain MLX, written from config.json.

Details that matter (all from the config / reference implementation):
- embeddings are multiplied by sqrt(hidden_size) (rounded through bf16, as upstream does)
- Gemma RMSNorm scales by (1 + weight); four norms per block (pre/post attention, pre/post MLP)
- per-head RMSNorm on queries and keys (QK-norm); attention scale = query_pre_attn_scalar ** -0.5
- grouped-query attention: 4 query heads share 1 key/value head, head_dim 256
- layer types: 5 sliding-window layers (window 512, RoPE base 1e4), then 1 global layer (RoPE base 1e6), x3
- MLP: down(gelu_tanh(gate(x)) * up(x)); LM head tied to the embedding

Checked against mlx-lm's gemma3_text in ../.venv-moshi by parity.py.
"""
import glob, json, os
import mlx.core as mx
import mlx.nn as nn

GEMMA_DIR = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-270m/snapshots/*"))


class Linear(nn.Linear):
    """Bias-free linear that keeps an fp32 master weight and multiplies in the activation dtype
    (bf16 compute when the activations are bf16; a no-op cast in fp32)."""
    def __call__(self, x):
        return x @ self.weight.astype(x.dtype).T


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = mx.zeros((d,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, (1.0 + self.weight).astype(x.dtype), self.eps)


def rope_at(x, pos, base):
    """RoPE at arbitrary per-token positions (non-traditional, rotate-half; equals mx.fast.rope at integer positions).
    x (B, H, L, D), pos (B, L) float. Used for T5Gemma-TTS-style progress positions."""
    D = x.shape[-1]
    half = D // 2
    inv = 1.0 / (base ** (mx.arange(0, half, dtype=mx.float32) * 2.0 / D))
    f = pos.astype(mx.float32)[..., None] * inv
    cos, sin = mx.cos(f)[:, None], mx.sin(f)[:, None]
    x1, x2 = x[..., :half].astype(mx.float32), x[..., half:].astype(mx.float32)
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, c, sliding):
        super().__init__()
        d, self.nh, self.nkv, self.hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.q_proj = Linear(d, self.nh * self.hd, bias=False)
        self.k_proj = Linear(d, self.nkv * self.hd, bias=False)
        self.v_proj = Linear(d, self.nkv * self.hd, bias=False)
        self.o_proj = Linear(self.nh * self.hd, d, bias=False)
        self.q_norm = RMSNorm(self.hd, c["rms_norm_eps"])
        self.k_norm = RMSNorm(self.hd, c["rms_norm_eps"])
        self.scale = c["query_pre_attn_scalar"] ** -0.5
        self.sliding = sliding
        self.window = c["sliding_window"]
        self.rope_base = c["rope_local_base_freq"] if sliding else c["rope_theta"]

    def __call__(self, x, mask, cache=None, offset=0, pos=None):
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.nh, self.hd)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.nkv, self.hd)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.nkv, self.hd).transpose(0, 2, 1, 3)
        if pos is None:
            q = mx.fast.rope(q, self.hd, traditional=False, base=self.rope_base, scale=1.0, offset=offset)
            k = mx.fast.rope(k, self.hd, traditional=False, base=self.rope_base, scale=1.0, offset=offset)
        else:  # explicit positions for these L tokens (keys are cached already rotated)
            q, k = rope_at(q, pos, self.rope_base), rope_at(k, pos, self.rope_base)
        if cache is not None:  # cache: dict, filled on prefill, extended on each decode step
            if "k" in cache:
                k = mx.concatenate([cache["k"], k], axis=2)
                v = mx.concatenate([cache["v"], v], axis=2)
            keep = self.window - 1 if self.sliding else k.shape[2]  # next query sees these + its own key
            cache["k"], cache["v"] = k[:, :, -keep:], v[:, :, -keep:]
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate_proj = Linear(d, h, bias=False)
        self.up_proj = Linear(d, h, bias=False)
        self.down_proj = Linear(h, d, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, c, sliding):
        super().__init__()
        eps = c["rms_norm_eps"]
        self.self_attn = Attention(c, sliding)
        self.mlp = MLP(c["hidden_size"], c["intermediate_size"])
        self.input_layernorm = RMSNorm(c["hidden_size"], eps)
        self.post_attention_layernorm = RMSNorm(c["hidden_size"], eps)
        self.pre_feedforward_layernorm = RMSNorm(c["hidden_size"], eps)
        self.post_feedforward_layernorm = RMSNorm(c["hidden_size"], eps)

    def __call__(self, x, mask, cache=None, offset=0, pos=None):
        h = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), mask, cache, offset, pos))
        return h + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(h)))


def make_masks(L, window, offset=0, kv_len=None, key_valid=None, prefix_len=None):
    """Additive masks (L, kv_len) for global (causal) and sliding (causal + window) layers.
    key_valid: optional (B, kv_len) bool for padded batches -> masks become (B, 1, L, kv_len).
    prefix_len: optional (B,) ints; positions < prefix_len attend to each other in both directions (prefix-LM,
    how Google made T5Gemma's encoder from Gemma). Everything after stays causal."""
    kv_len = kv_len or (offset + L)
    qpos = mx.arange(offset, offset + L)[:, None]
    kpos = mx.arange(kv_len)[None, :] + (offset + L - kv_len)
    causal = kpos <= qpos
    slide = causal & (kpos > qpos - window)
    if prefix_len is not None:
        pl = prefix_len[:, None, None]
        both = (qpos[None] < pl) & (kpos[None] < pl)           # (B, L, kv)
        causal = (causal[None] | both)[:, None]
        slide = (slide[None] | both)[:, None]
        if key_valid is not None:
            causal = causal & key_valid[:, None, None, :]
            slide = slide & key_valid[:, None, None, :]
    elif key_valid is not None:
        causal = causal[None, None] & key_valid[:, None, None, :]
        slide = slide[None, None] & key_valid[:, None, None, :]
    neg = mx.array(-1e9, mx.float32)
    return mx.where(causal, 0.0, neg), mx.where(slide, 0.0, neg)


class Gemma3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c["vocab_size"], c["hidden_size"])
        types = c.get("layer_types") or [
            "full_attention" if (i + 1) % c.get("_sliding_window_pattern", 6) == 0 else "sliding_attention"
            for i in range(c["num_hidden_layers"])]
        self.sliding = [t == "sliding_attention" for t in types]
        self.layers = [Block(c, s) for s in self.sliding]
        self.norm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.emb_scale = mx.array(c["hidden_size"] ** 0.5, mx.bfloat16).astype(mx.float32).item()
        self.compute_dtype = mx.float32  # set to mx.bfloat16 for bf16 activations with fp32 master weights

    def embed(self, tokens):
        return self.embed_tokens(tokens) * self.emb_scale

    def hidden(self, h, caches=None, offset=0, key_valid=None, pos=None, prefix_len=None):
        """h: (B, L, d) already-scaled input embeddings -> final-normed hidden states (B, L, d).
        Two modes: full forward / prefill (offset 0, caches empty or None), or one-token decode (L == 1)
        against filled caches. Sliding caches hold window-1 keys, so a decode step needs no mask."""
        h = h.astype(self.compute_dtype)
        L = h.shape[1]
        if offset == 0:
            gmask, smask = make_masks(L, self.c["sliding_window"], key_valid=key_valid, prefix_len=prefix_len)
            gmask, smask = gmask.astype(h.dtype), smask.astype(h.dtype)
        else:
            assert L == 1 and caches is not None and key_valid is None, "decode is one token at a time, unpadded"
            gmask = smask = None
        for i, layer in enumerate(self.layers):
            h = layer(h, smask if self.sliding[i] else gmask, None if caches is None else caches[i], offset, pos)
        return self.norm(h)

    def logits(self, tokens):
        return self.embed_tokens.as_linear(self.hidden(self.embed(tokens)))


def load_gemma(path=None, dtype=mx.bfloat16):
    path = path or GEMMA_DIR[0]
    c = json.load(open(os.path.join(path, "config.json")))
    m = Gemma3(c)
    w = {k.removeprefix("model."): v for k, v in mx.load(os.path.join(path, "model.safetensors")).items()}
    m.load_weights(list(w.items()), strict=True)
    m.set_dtype(dtype)
    mx.eval(m.parameters())
    return m, c


def load_tokenizer(path=None):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(path or GEMMA_DIR[0], "tokenizer.json"))
