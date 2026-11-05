"""Follow-up acoustic model (MLX): 27's MAS-aligned character encoder + duration predictor, a FastPitch-style
pitch/energy adaptor, a speaker embedding, and a Matcha-TTS-style conditional flow-matching (OT-CFM) mel decoder.

  encoder: char embedding + conv prenet + FFT blocks -> h (B,N,d);  h_s = h + speaker embedding
  prior mean mu = Linear(h_s) in normalized-mel space; MAS on log N(x_t; mu_n, I) gives the hard monotonic alignment
      (exactly 27's alignment: model.align / mas.py), with 27's diagonal prior early in training
  duration / pitch / energy predictors: conv stacks on stop_gradient(h_s) (27's DurationPredictor class), MSE targets:
      log duration; per-character mean of speaker-normalized log-F0 over voiced frames (0 if none voiced);
      per-character mean of utterance-centred log frame energy (computed from the mel itself)
  h_v = h_s + conv(pitch) + conv(energy)  (teacher-forced with targets in training, predictions at inference)
  decoder: vector field v(x_t, t | mu_y, h_v_y, speaker) where *_y are gathered to frames by the alignment. A transformer
      over frames (pre-LN attention + conv FFN, kernel 3) with adaLN-zero modulation from (time, speaker) per block.
  OT-CFM loss (Lipman et al. 2023; Matcha-TTS): x_t = (1 - (1 - s) t) x0 + t x1, target u = x1 - (1 - s) x0, s = 1e-4.
  Sampling: Euler ODE from x0 ~ N(0, temp^2 I), n_steps evaluations.
"""
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from model import FFTBlock, DurationPredictor, masks, pos_enc, N_MELS

def cap_memory(limit_gb=16, cache_gb=2):
    """Every MLX process in the follow-up calls this first: uncapped MLX buffer caches ran this laptop out of memory."""
    mx.set_memory_limit(limit_gb << 30)
    mx.set_cache_limit(cache_gb << 30)


SIG_MIN = 1e-4
E_SCALE = 2.0  # utterance-centred log energy is divided by this (keeps the target near unit scale)


def t_embed(t, dim=256):
    half = dim // 2
    f = mx.exp(-math.log(10000.0) * mx.arange(half) / half)
    a = (t[:, None] * 1000.0) * f[None]
    return mx.concatenate([mx.sin(a), mx.cos(a)], -1)


class AdaBlock(nn.Module):
    """Pre-LN transformer block (attention + conv FFN) with adaLN-zero shift/scale/gate from a conditioning vector."""

    def __init__(self, d, heads, ff, k=3, drop=0.0):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d, affine=False)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d, affine=False)
        self.c1 = nn.Conv1d(d, ff, k, padding=k // 2)
        self.c2 = nn.Conv1d(ff, d, 1)
        self.ada = nn.Linear(d, 6 * d)
        self.ada.weight = mx.zeros_like(self.ada.weight)
        self.ada.bias = mx.zeros_like(self.ada.bias)
        self.drop = nn.Dropout(drop)

    def __call__(self, x, c, mask, amask):
        B, L, d = x.shape
        s1, b1, g1, s2, b2, g2 = mx.split(self.ada(nn.silu(c))[:, None, :], 6, axis=-1)
        h = self.ln1(x) * (1 + s1) + b1
        q, k, v = mx.split(self.qkv(h), 3, axis=-1)
        sh = lambda t: t.reshape(B, L, self.heads, d // self.heads).transpose(0, 2, 1, 3)
        a = mx.fast.scaled_dot_product_attention(sh(q), sh(k), sh(v), scale=1.0 / math.sqrt(d // self.heads), mask=amask)
        x = x + g1 * self.drop(self.o(a.transpose(0, 2, 1, 3).reshape(B, L, d)))
        h = (self.ln2(x) * (1 + s2) + b2) * mask
        h = self.c2(self.drop(nn.gelu(self.c1(h))))
        return (x + g2 * self.drop(h)) * mask


class FlowDecoder(nn.Module):
    def __init__(self, d_cond, d=384, layers=6, heads=4, ff=1536, k=3, drop=0.0):
        super().__init__()
        self.d = d
        self.inp = nn.Linear(2 * N_MELS + d_cond, d)
        self.t1 = nn.Linear(256, d)
        self.t2 = nn.Linear(d, d)
        self.spk = nn.Linear(d_cond, d)
        self.blocks = [AdaBlock(d, heads, ff, k, drop) for _ in range(layers)]
        self.ln = nn.LayerNorm(d, affine=False)
        self.ada = nn.Linear(d, 2 * d)
        self.ada.weight = mx.zeros_like(self.ada.weight)
        self.ada.bias = mx.zeros_like(self.ada.bias)
        self.out = nn.Linear(d, N_MELS)

    def __call__(self, xt, t, mu_y, h_y, spk, fm, am):
        c = self.t2(nn.silu(self.t1(t_embed(t)))) + self.spk(spk)
        x = self.inp(mx.concatenate([xt, mu_y, h_y], -1))
        x = (x + pos_enc(x.shape[1], self.d)[None]) * fm
        for b in self.blocks:
            x = b(x, c, fm, am)
        s, sh = mx.split(self.ada(nn.silu(c))[:, None, :], 2, axis=-1)
        return self.out(self.ln(x) * (1 + s) + sh) * fm


class FMTTS(nn.Module):
    def __init__(self, n_sym, n_spk=1, d=384, enc_layers=6, heads=4, ff=1024, enc_k=5, dec_d=384, dec_layers=6,
                 dec_heads=4, dec_ff=1536, dec_k=3, drop=0.1, dec_drop=0.0, variance=True):
        super().__init__()
        self.cfg = dict(n_sym=n_sym, n_spk=n_spk, d=d, enc_layers=enc_layers, heads=heads, ff=ff, enc_k=enc_k, dec_d=dec_d,
                        dec_layers=dec_layers, dec_heads=dec_heads, dec_ff=dec_ff, dec_k=dec_k, drop=drop,
                        dec_drop=dec_drop, variance=variance)
        self.d = d
        self.variance = variance
        self.emb = nn.Embedding(n_sym, d)
        self.pre = nn.Conv1d(d, d, 5, padding=2)
        self.enc = [FFTBlock(d, heads, ff, enc_k, drop) for _ in range(enc_layers)]
        self.enc_ln = nn.LayerNorm(d)
        self.spk_emb = nn.Embedding(n_spk, d)
        self.spk_emb.weight = self.spk_emb.weight * 0.1
        self.mu = nn.Linear(d, N_MELS)
        self.dur = DurationPredictor(d, drop=drop)
        if variance:
            self.pitch = DurationPredictor(d, drop=drop)
            self.energy = DurationPredictor(d, drop=drop)
            self.p_emb = nn.Conv1d(1, d, 3, padding=1)
            self.e_emb = nn.Conv1d(1, d, 3, padding=1)
        self.dec = FlowDecoder(d, dec_d, dec_layers, dec_heads, dec_ff, dec_k, dec_drop)

    def encode(self, tok, tlen, spk):
        B, N = tok.shape
        m, am = masks(tlen, N)
        x = self.emb(tok) * math.sqrt(self.d)
        x = (x + nn.relu(self.pre(x * m))) * m
        x = x + pos_enc(N, self.d)[None]
        for b in self.enc:
            x = b(x, m, am)
        s = self.spk_emb(spk)  # (B, d)
        h = (self.enc_ln(x) + s[:, None, :]) * m
        return h, self.mu(h) * m, m, s

    def log_lik(self, mu, x):
        xx = mx.sum(x * x, -1)[:, None, :]
        mm = mx.sum(mu * mu, -1)[:, :, None]
        return -0.5 * (xx + mm - 2.0 * (mu @ x.transpose(0, 2, 1)))

    def add_variance(self, h, m, p, e):
        """p, e (B,N) per-character targets or predictions -> h + embeddings."""
        if not self.variance:
            return h
        return (h + self.p_emb(p[..., None]) + self.e_emb(e[..., None])) * m


def char_means(idx, fm2, vals, wts, N):
    """Per-character weighted mean of frame values. idx (B,T) frame->char, fm2 (B,T) frame mask, vals/wts (B,T)."""
    A = (idx[..., None] == mx.arange(N)[None, None, :]).astype(mx.float32) * fm2[..., None]  # (B,T,N)
    num = mx.sum(A * (vals * wts)[..., None], 1)
    den = mx.sum(A * wts[..., None], 1)
    return num / mx.maximum(den, 1.0)


def frame_energy(x, mean, std):
    """normalized mel (B,T,80) -> log mean linear-mel energy per frame (B,T)."""
    raw = x * std + mean
    return mx.log(mx.mean(mx.exp(raw), -1) + 1e-5)


def loss_fn(model, tok, tlen, spk, x, flen, idx, dur, lf, voiced, mel_mean, mel_std, x0, t):
    """x normalized mel (B,T,80); idx (B,T) MAS alignment; dur (B,N); lf (B,T) speaker-normalized log-F0; voiced (B,T)
    0/1; x0 (B,T,80) noise; t (B,) flow times."""
    B, T, _ = x.shape
    N = tok.shape[1]
    h, mu, tm, s = model.encode(tok, tlen, spk)
    fm2 = (mx.arange(T)[None, :] < flen[:, None]).astype(mx.float32)
    fm = fm2[..., None]
    nf = mx.sum(fm)
    gi = idx[..., None].astype(mx.int32)
    mu_y = mx.take_along_axis(mu, gi, axis=1)
    l_prior = 0.5 * mx.sum(((mu_y - x) ** 2) * fm) / (nf * N_MELS)
    hs = mx.stop_gradient(h)
    tmask = tm[..., 0]
    l_dur = mx.sum(((model.dur(hs, tm) - mx.log(mx.maximum(dur, 1.0))) ** 2) * tmask) / mx.sum(tmask)
    l_p = l_e = mx.array(0.0)
    if model.variance:
        p_t = char_means(idx, fm2, lf, voiced * fm2, N)
        e_f = frame_energy(x, mel_mean, mel_std)
        e_f = (e_f - mx.sum(e_f * fm2, 1, keepdims=True) / flen[:, None]) / E_SCALE
        e_t = char_means(idx, fm2, e_f, fm2, N)
        p_t, e_t = mx.stop_gradient(p_t), mx.stop_gradient(e_t)
        l_p = mx.sum(((model.pitch(hs, tm) - p_t) ** 2) * tmask) / mx.sum(tmask)
        l_e = mx.sum(((model.energy(hs, tm) - e_t) ** 2) * tmask) / mx.sum(tmask)
        h = model.add_variance(h, tm, p_t, e_t)
    h_y = mx.take_along_axis(h, gi, axis=1) * fm
    tt = t[:, None, None]
    xt = (1 - (1 - SIG_MIN) * tt) * x0 + tt * x
    u = x - (1 - SIG_MIN) * x0
    _, am = masks(flen, T)
    v = model.dec(xt * fm, t, mu_y, h_y, s, fm, am)
    l_cfm = mx.sum(((v - u) ** 2) * fm) / (nf * N_MELS)
    total = l_cfm + l_prior + l_dur + l_p + l_e
    return total, (l_cfm, l_prior, l_dur, l_p, l_e)


def align(model, tok, tlen, spk, x, flen, prior_w=0.0):
    """MAS (numba, CPU) on the model's speaker-conditioned prior; same procedure as 27's model.align."""
    from mas import maximum_path_batch
    _, mu, _, _ = model.encode(tok, tlen, spk)
    ll = model.log_lik(mu, x)
    mx.eval(ll)
    ll = np.array(ll, dtype=np.float32)
    tl, fl = np.array(tlen), np.array(flen)
    if prior_w > 0:
        B, N, T = ll.shape
        n = np.arange(N)[None, :, None] / np.maximum(tl[:, None, None] - 1, 1)
        tt = np.arange(T)[None, None, :] / np.maximum(fl[:, None, None] - 1, 1)
        ll = ll - prior_w * (n - tt) ** 2 / (2 * 0.05 ** 2)
    idx = maximum_path_batch(np.ascontiguousarray(ll), tl.astype(np.int64), fl.astype(np.int64))
    B, N = tok.shape
    dur = np.zeros((B, N), np.float32)
    for b in range(B):
        dur[b, : tl[b]] = np.bincount(idx[b, : fl[b]], minlength=tl[b])[: tl[b]]
    return idx, dur


def sample_ode(model, mu_y, h_y, s, n_steps, temp, key, solver="euler"):
    T = mu_y.shape[1]
    fm = mx.ones((1, T, 1))
    _, am = masks(mx.array([T]), T)
    x = mx.random.normal(mu_y.shape, key=key) * temp
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = mx.array([i * dt])
        v = model.dec(x, t, mu_y, h_y, s, fm, am)
        if solver == "midpoint":
            xm = x + 0.5 * dt * v
            v = model.dec(xm, t + 0.5 * dt, mu_y, h_y, s, fm, am)
        x = x + dt * v
    return x


def infer(model, tok, spk=0, pace=1.0, rounding="round", n_steps=10, temp=0.667, seed=0, solver="euler",
          pitch_shift=0.0, energy_shift=0.0, durations=None):
    """tok (1,N) int -> normalized mel (T,80) numpy, durations (N,)."""
    tlen = mx.array([tok.shape[1]])
    sp = mx.array([spk])
    h, mu, tm, s = model.encode(tok, tlen, sp)
    if durations is None:
        d = np.exp(np.array(model.dur(h, tm)[0])) * pace
        d = np.maximum(1, np.ceil(d) if rounding == "ceil" else np.round(d)).astype(np.int64)
    else:
        d = np.asarray(durations, np.int64)
    if model.variance:
        p = model.pitch(h, tm) + pitch_shift
        e = model.energy(h, tm) + energy_shift
        h = model.add_variance(h, tm, p, e)
    idx = mx.array(np.repeat(np.arange(len(d)), d)[None], dtype=mx.int32)[..., None]
    mu_y = mx.take_along_axis(mu, idx, axis=1)
    h_y = mx.take_along_axis(h, idx, axis=1)
    x = sample_ode(model, mu_y, h_y, s, n_steps, temp, mx.random.key(seed), solver)
    mx.eval(x)
    return np.array(x[0]), d


def load_fm(path):
    import json
    cfg = json.load(open(path.replace(".safetensors", ".json")))
    m = FMTTS(**cfg["model"])
    m.load_weights(path)
    m.eval()
    mx.eval(m.parameters())
    return m, cfg
