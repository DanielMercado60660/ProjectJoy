"""Gemma Voice, flow-matching variant: Gemma 3 270M backbone + a small per-frame flow head on Mimi's continuous latent.

Idea from Kyutai's Pocket TTS / gradium phonon-31m (flow transformer predicts the Mimi latent one 80 ms frame at a
time) and VibeVoice (LLM backbone + diffusion head). No codebooks, so no 4-codebook quality cap: the target is the
512-dim latent in the quantizer's projected space ('c' in mimi_io.py), whose round trip scores Whisper CER 0.38 /
UTMOS 4.28 on val-100 (Mimi 4 codebooks: 0.70 / 3.11).

Sequence for one clip (positions are backbone time steps):
    [BOS, text chars ...] [AUDIO_START] [frame 0] [frame 1] ... [frame F-1]
- Frame inputs: in_proj(normalized latent) + speaker embedding. Training adds Gaussian noise to the fed-back latents
  (per-clip sigma ~ U(0, in_noise)) so the model tolerates its own imperfect outputs at inference.
- The hidden state at position tl+s (AUDIO_START for s=0) conditions the flow head that generates frame s, and a stop
  logit; position tl+F says "stop" (one positive per clip).
- Flow head: rectified flow (x_t = (1-t) noise + t x1, predict v = x1 - noise), an MLP with adaLN residual blocks
  conditioned on the hidden state and t. K noise draws per frame per step (the head is cheap next to the backbone).
- Latents are normalized per dimension (mean/std from the encoded sets), stored in each run's meta.
"""
import math, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from gemma import Gemma3, load_gemma
from tts import encode_text, BOS  # noqa: F401  (encode_text re-exported for train_flow / evaluate_flow)

D_LAT = 512


def timestep_embed(t, dim=256):
    half = dim // 2
    f = mx.exp(-math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half)
    a = (t[:, None] * 1000.0) * f[None]
    return mx.concatenate([mx.cos(a), mx.sin(a)], -1)


class AdaBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.norm = nn.LayerNorm(w, eps=1e-6, affine=False)
        self.mod = nn.Linear(w, 3 * w)
        self.mod.weight = mx.zeros_like(self.mod.weight); self.mod.bias = mx.zeros_like(self.mod.bias)
        self.fc1 = nn.Linear(w, 2 * w)
        self.fc2 = nn.Linear(2 * w, w)

    def __call__(self, x, c):
        sh, sc, g = mx.split(self.mod(nn.silu(c)), 3, axis=-1)
        return x + g * self.fc2(nn.silu(self.fc1(self.norm(x) * (1 + sc) + sh)))


class FlowHead(nn.Module):
    def __init__(self, d_cond, w=768, depth=6, D=D_LAT):
        super().__init__()
        self.inp = nn.Linear(D, w)
        self.cond = nn.Linear(d_cond, w)
        self.t1 = nn.Linear(256, w)
        self.t2 = nn.Linear(w, w)
        self.blocks = [AdaBlock(w) for _ in range(depth)]
        self.out_norm = nn.LayerNorm(w, eps=1e-6, affine=False)
        self.out_mod = nn.Linear(w, 2 * w)
        self.out_mod.weight = mx.zeros_like(self.out_mod.weight); self.out_mod.bias = mx.zeros_like(self.out_mod.bias)
        self.out = nn.Linear(w, D)
        self.out.weight = mx.zeros_like(self.out.weight); self.out.bias = mx.zeros_like(self.out.bias)

    def __call__(self, x, t, h):
        """x (N, D) noisy latent, t (N,) in [0, 1], h (N, d) backbone state -> velocity (N, D). fp32."""
        c = self.cond(h) + self.t2(nn.silu(self.t1(timestep_embed(t))))
        y = self.inp(x)
        for b in self.blocks:
            y = b(y, c)
        sh, sc = mx.split(self.out_mod(nn.silu(c)), 2, axis=-1)
        return self.out(self.out_norm(y) * (1 + sc) + sh)


class GemmaFlowTTS(nn.Module):
    def __init__(self, backbone: Gemma3, n_spk=1, head_w=768, head_depth=6):
        super().__init__()
        self.backbone = backbone
        d = backbone.c["hidden_size"]
        target = float(mx.std(backbone.embed_tokens.weight[:20000].astype(mx.float32)).item()) * backbone.emb_scale
        self.in_proj = nn.Linear(D_LAT, d)
        self.in_proj.weight = mx.random.normal((d, D_LAT)) * (target / math.sqrt(D_LAT))   # unit-std latent -> text scale
        self.in_proj.bias = mx.zeros((d,))
        self.audio_start = mx.random.normal((d,)) * target
        self.spk_emb = nn.Embedding(max(1, n_spk), d)
        self.spk_emb.weight = mx.zeros((max(1, n_spk), d))
        self.head = FlowHead(d, head_w, head_depth)
        self.stop = nn.Linear(d, 1)
        self.stop.weight = mx.zeros_like(self.stop.weight); self.stop.bias = mx.full((1,), -4.0)
        self.progress = 0.0     # T5Gemma-TTS-style options, as in tts.GemmaTTS
        self.prefix_lm = False

    def positions(self, tlen, S, T):
        """Progress positions (see tts.GemmaTTS.positions). S = audio-side positions per clip = F + 1."""
        P = self.progress
        j = mx.arange(T)[None, :].astype(mx.float32)
        tl = tlen[:, None].astype(mx.float32)
        Sf = S[:, None].astype(mx.float32)
        return mx.where(j < tl, j / mx.maximum(tl - 1, 1) * P, mx.minimum((j - tl) / mx.maximum(Sf - 1, 1), 1.0) * P)

    def inputs(self, text, text_len, lat_in, spk=None):
        """text (B, Lt), lat_in (B, La, D) normalized latents of frames 0..La-1 (fed at tl+1+s) -> (B, T, d)."""
        B, Lt = text.shape
        La = lat_in.shape[1]
        T = Lt + 1 + La
        te = self.backbone.embed(text)
        ae = self.in_proj(lat_in.astype(mx.float32))
        pos = mx.arange(T)[None, :]
        tl = text_len[:, None]
        bidx = mx.arange(B)[:, None]
        h_text = te[bidx, mx.broadcast_to(mx.clip(pos, 0, Lt - 1), (B, T))]
        h_audio = ae[bidx, mx.clip(pos - tl - 1, 0, La - 1)]
        is_text = (pos < tl)[..., None]
        is_start = (pos == tl)[..., None]
        h = mx.where(is_text, h_text.astype(mx.float32),
                     mx.where(is_start, self.audio_start.astype(mx.float32), h_audio))
        sv = self.spk_emb(spk if spk is not None else mx.zeros((B,), mx.int32)).astype(mx.float32)
        h = h + mx.where(is_text, 0.0, sv[:, None, :])
        return h.astype(self.backbone.compute_dtype)


def make_batch(items, store, mean, std, in_noise=0.0, rng=None):
    """items: (text_ids, offset, F, spk). store: (total_frames, 512) float16 memmap. Returns mx arrays:
    text (B,Lt), tlen (B,), lat_in (B,La,D) noisy inputs, lat_tgt (B,La,D) clean targets, flen (B,), spk (B,)."""
    B = len(items)
    Lt = int(math.ceil(max(len(it[0]) for it in items) / 8) * 8)
    La = int(math.ceil(max(it[2] for it in items) / 16) * 16)
    text = np.zeros((B, Lt), np.int32)
    tlen = np.zeros(B, np.int32)
    flen = np.zeros(B, np.int32)
    spk = np.zeros(B, np.int32)
    tgt = np.zeros((B, La, D_LAT), np.float32)
    for b, (t, off, F, sp) in enumerate(items):
        text[b, :len(t)] = t
        tlen[b], flen[b], spk[b] = len(t), F, sp
        tgt[b, :F] = (np.asarray(store[off:off + F], np.float32) - mean) / std
    lat_in = tgt
    if in_noise > 0:
        sig = rng.uniform(0, in_noise, (B, 1, 1)).astype(np.float32)
        lat_in = tgt + sig * rng.standard_normal(tgt.shape).astype(np.float32)
    return (mx.array(text), mx.array(tlen), mx.array(lat_in), mx.array(tgt), mx.array(flen), mx.array(spk))


def loss_fn(model, text, tlen, lat_in, lat_tgt, flen, spk, key, K=4, stop_w=1.0, pos_w=5.0):
    """Flow-matching MSE on every real frame (K noise draws each) + stop BCE. Returns (total, flow, stop)."""
    h = model.inputs(text, tlen, lat_in, spk)
    B, T, _ = h.shape
    La = lat_tgt.shape[1]
    pos = model.positions(tlen, flen + 1, T) if model.progress > 0 else None
    out = model.backbone.hidden(h, pos=pos, prefix_len=tlen if model.prefix_lm else None)
    bidx = mx.arange(B)[:, None]
    s = mx.arange(La + 1)[None, :]
    hs = out[bidx, mx.clip(tlen[:, None] + s, 0, T - 1)].astype(mx.float32)     # (B, La+1, d): predicts frame s / stop
    # stop: positions s <= F, positive at s == F
    st_logit = model.stop(hs)[..., 0]
    st_tgt = (s == flen[:, None]).astype(mx.float32)
    st_mask = (s <= flen[:, None]).astype(mx.float32)
    bce = nn.losses.binary_cross_entropy(st_logit, st_tgt, with_logits=True, reduction="none")
    bce = bce * (1 + (pos_w - 1) * st_tgt)
    stop = (bce * st_mask).sum() / st_mask.sum()
    # flow on frames s < F
    fm = (mx.arange(La)[None, :] < flen[:, None]).astype(mx.float32).reshape(-1)            # (B*La,)
    hc = mx.repeat(hs[:, :La].reshape(B * La, -1), K, axis=0)
    x1 = mx.repeat(lat_tgt.reshape(B * La, -1), K, axis=0)
    w = mx.repeat(fm, K, axis=0)
    k1, k2 = mx.random.split(key)
    x0 = mx.random.normal(x1.shape, key=k1)
    t = mx.sigmoid(mx.random.normal((x1.shape[0],), key=k2))                                 # logit-normal t
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    v = model.head(xt, t, hc)
    mse = ((v - (x1 - x0)) ** 2).mean(-1)
    flow = (mse * w).sum() / w.sum()
    return flow + stop_w * stop, flow, stop


def build_model(seed=0, n_spk=1, progress=0.0, prefix_lm=False, head_w=768, head_depth=6):
    mx.random.seed(seed)
    bb, _ = load_gemma(dtype=mx.float32)
    model = GemmaFlowTTS(bb, n_spk, head_w, head_depth)
    model.progress, model.prefix_lm = float(progress), bool(prefix_lm)
    model.set_dtype(mx.float32)
    model.backbone.embed_tokens.freeze()
    mx.eval(model.parameters())
    return model


# ----------------------------------------------------------------------------- generation
def sample_frame(model, h, key, temp=0.8, n_steps=16, h_unc=None, cfg=1.0):
    """h (1, d) fp32 -> one normalized latent (1, D): Euler integration of the flow from temp * N(0, I).
    With h_unc (state of the text-free stream) and cfg != 1: v = v_unc + cfg * (v_cond - v_unc)."""
    x = mx.random.normal((1, D_LAT), key=key) * temp
    dt = 1.0 / n_steps
    hh = mx.concatenate([h, h_unc], 0) if h_unc is not None else h
    for i in range(n_steps):
        if h_unc is None:
            v = model.head(x, mx.array([i * dt]), h)
        else:
            vv = model.head(mx.concatenate([x, x], 0), mx.array([i * dt, i * dt]), hh)
            v = vv[1:] + cfg * (vv[:1] - vv[1:])
        x = x + dt * v
    return x


def generate(model, text_ids, max_frames, temp=0.8, n_steps=16, seed=0, min_frames=1, spk=0, est_frames=None,
             stop_th=0.5, cfg=1.0):
    """One utterance, batch 1, KV cache. Returns (normalized latents (F, D) np.float32, info).
    cfg != 1 runs a second, text-free stream ([BOS] only; needs a model trained with --text_drop) for guidance."""
    t_start = time.perf_counter()
    key = mx.random.key(seed)
    bb = model.backbone
    caches = [dict() for _ in bb.layers]
    t = mx.array(text_ids)[None]
    sv = model.spk_emb(mx.array([spk]))[0].astype(mx.float32)
    h = mx.concatenate([bb.embed(t).astype(mx.float32), (model.audio_start.astype(mx.float32) + sv)[None, None]], axis=1)
    tl = len(text_ids)
    use_prog = model.progress > 0
    if use_prog:
        assert est_frames, "progress positions need an estimated frame count"
        P, S = model.progress, est_frames + 1
        ppos = mx.array([[i / max(tl - 1, 1) * P for i in range(tl)] + [0.0]])
        step_pos = lambda s: mx.array([[min(s / max(S - 1, 1), 1.0) * P]])
    out = bb.hidden(h.astype(bb.compute_dtype), caches, 0, pos=ppos if use_prog else None,
                    prefix_len=mx.array([tl]) if model.prefix_lm else None)[:, -1].astype(mx.float32)
    pos = tl + 1
    use_cfg = cfg != 1.0
    if use_cfg:   # text-free stream: [BOS] [AUDIO_START] frames..., positions as for a 1-token text
        caches_u = [dict() for _ in bb.layers]
        hu = mx.concatenate([bb.embed(mx.array([[BOS]])).astype(mx.float32),
                             (model.audio_start.astype(mx.float32) + sv)[None, None]], axis=1)
        out_u = bb.hidden(hu.astype(bb.compute_dtype), caches_u, 0, pos=mx.array([[0.0, 0.0]]) if use_prog else None,
                          prefix_len=mx.array([1]) if model.prefix_lm else None)[:, -1].astype(mx.float32)
        pos_u = 2
    frames, first_audio, hit_max = [], None, False
    for s in range(max_frames + 1):
        p_stop = float(mx.sigmoid(model.stop(out))[0, 0])
        if s >= min_frames and p_stop > stop_th:
            break
        if s == max_frames:
            hit_max = True
            break
        key, sub = mx.random.split(key)
        x = sample_frame(model, out, sub, temp, n_steps, out_u if use_cfg else None, cfg)
        mx.eval(x)
        frames.append(np.array(x[0]))
        if first_audio is None:
            first_audio = time.perf_counter() - t_start
        e = (model.in_proj(x) + sv)[None]                                             # (1, 1, d)
        out = bb.hidden(e.astype(bb.compute_dtype), caches, pos,
                        pos=step_pos(s + 1) if use_prog else None)[:, -1].astype(mx.float32)
        pos += 1
        if use_cfg:
            out_u = bb.hidden(e.astype(bb.compute_dtype), caches_u, pos_u,
                              pos=step_pos(s + 1) if use_prog else None)[:, -1].astype(mx.float32)
            pos_u += 1
    lat = np.stack(frames) if frames else np.zeros((0, D_LAT), np.float32)
    return lat, {"frames": len(frames), "hit_max": hit_max, "first_audio_s": first_audio}
