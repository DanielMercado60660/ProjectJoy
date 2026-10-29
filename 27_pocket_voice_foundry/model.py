"""Acoustic model (MLX): characters -> durations -> 80-band log-mel (28's mel definition).

FastSpeech 2-style non-autoregressive model, with the alignment learned Grad-TTS / Glow-TTS style:
  encoder (char embedding + FFT blocks) -> hidden h (N, d) and prior mean mu = Linear(h) in normalized-mel space
  MAS on log N(x_t; mu_n, I) gives a hard monotonic alignment -> per-character durations (training targets)
  length regulator (gather h by the alignment) + frame positional encoding -> decoder FFT blocks -> mel
  duration predictor (convs on stop_gradient(h)) -> log duration
Losses: L1(mel) + prior MSE(mu vs aligned mel) + MSE(log duration). No pitch/energy predictors.
"""
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn

N_MELS = 80


def sinusoid(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d // 2)[None, :]
    ang = pos / np.power(10000.0, 2 * i / d)
    pe = np.concatenate([np.sin(ang), np.cos(ang)], -1).astype(np.float32)
    return mx.array(pe)


_PE = {}


def pos_enc(T, d):
    key = (d,)
    if key not in _PE or _PE[key].shape[0] < T:
        _PE[key] = sinusoid(max(T, 2048), d)
    return _PE[key][:T]


class FFTBlock(nn.Module):
    """Pre-LN transformer block with a conv feed-forward (FastSpeech)."""

    def __init__(self, d, heads=2, ff=1024, k=9, drop=0.1):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.c1 = nn.Conv1d(d, ff, k, padding=k // 2)
        self.c2 = nn.Conv1d(ff, d, 1)
        self.drop = nn.Dropout(drop)

    def __call__(self, x, mask, amask):  # x (B,L,d), mask (B,L,1) float, amask (B,1,1,L) additive
        B, L, d = x.shape
        h = self.ln1(x)
        q, k, v = mx.split(self.qkv(h), 3, axis=-1)
        sh = lambda t: t.reshape(B, L, self.heads, d // self.heads).transpose(0, 2, 1, 3)
        a = mx.fast.scaled_dot_product_attention(sh(q), sh(k), sh(v), scale=1.0 / math.sqrt(d // self.heads), mask=amask)
        a = a.transpose(0, 2, 1, 3).reshape(B, L, d)
        x = x + self.drop(self.o(a))
        h = self.ln2(x) * mask
        h = self.c2(self.drop(nn.relu(self.c1(h))))
        return (x + self.drop(h)) * mask


class DurationPredictor(nn.Module):
    def __init__(self, d, hid=256, k=3, drop=0.1):
        super().__init__()
        self.c1 = nn.Conv1d(d, hid, k, padding=k // 2)
        self.ln1 = nn.LayerNorm(hid)
        self.c2 = nn.Conv1d(hid, hid, k, padding=k // 2)
        self.ln2 = nn.LayerNorm(hid)
        self.out = nn.Linear(hid, 1)
        self.drop = nn.Dropout(drop)

    def __call__(self, h, mask):
        x = self.drop(self.ln1(nn.relu(self.c1(h * mask))))
        x = self.drop(self.ln2(nn.relu(self.c2(x * mask))))
        return self.out(x * mask)[..., 0]  # (B, N) log duration


def masks(lens, L):
    m = (mx.arange(L)[None, :] < lens[:, None]).astype(mx.float32)  # (B, L)
    amask = mx.where(m[:, None, None, :] > 0, 0.0, -1e9).astype(mx.float32)
    return m[..., None], amask


class Acoustic(nn.Module):
    def __init__(self, n_sym, d=256, enc_layers=4, dec_layers=4, heads=2, ff=1024, drop=0.1):
        super().__init__()
        self.cfg = dict(n_sym=n_sym, d=d, enc_layers=enc_layers, dec_layers=dec_layers, heads=heads, ff=ff, drop=drop)
        self.d = d
        self.emb = nn.Embedding(n_sym, d)
        self.pre = nn.Conv1d(d, d, 5, padding=2)
        self.enc = [FFTBlock(d, heads, ff, 9, drop) for _ in range(enc_layers)]
        self.enc_ln = nn.LayerNorm(d)
        self.mu = nn.Linear(d, N_MELS)
        self.dur = DurationPredictor(d, drop=drop)
        self.dec = [FFTBlock(d, heads, ff, 9, drop) for _ in range(dec_layers)]
        self.dec_ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, N_MELS)

    def encode(self, tok, tlen):
        B, N = tok.shape
        m, am = masks(tlen, N)
        x = self.emb(tok) * math.sqrt(self.d)
        x = (x + nn.relu(self.pre(x * m))) * m
        x = x + pos_enc(N, self.d)[None]
        for b in self.enc:
            x = b(x, m, am)
        h = self.enc_ln(x) * m
        return h, self.mu(h) * m, m

    def decode(self, h, idx, flen):
        """h (B,N,d), idx (B,T) token index per frame, flen (B,) -> mel (B,T,80) normalized."""
        B, T = idx.shape
        m, am = masks(flen, T)
        x = mx.take_along_axis(h, idx[..., None].astype(mx.int32), axis=1)
        x = (x + pos_enc(T, self.d)[None]) * m
        for b in self.dec:
            x = b(x, m, am)
        return self.out(self.dec_ln(x)) * m

    def log_lik(self, mu, x):
        """mu (B,N,80), x (B,T,80) -> (B,N,T) of -0.5||x_t - mu_n||^2."""
        xx = mx.sum(x * x, -1)[:, None, :]
        mm = mx.sum(mu * mu, -1)[:, :, None]
        return -0.5 * (xx + mm - 2.0 * (mu @ x.transpose(0, 2, 1)))


def loss_fn(model, tok, tlen, x, flen, idx, dur):
    """x normalized mel (B,T,80); idx (B,T) alignment from MAS; dur (B,N) frame counts (>=1 on valid)."""
    h, mu, tm = model.encode(tok, tlen)
    fm = (mx.arange(x.shape[1])[None, :] < flen[:, None]).astype(mx.float32)[..., None]
    nf = mx.sum(fm)
    mel = model.decode(h, idx, flen)
    l_mel = mx.sum(mx.abs(mel - x) * fm) / (nf * N_MELS)
    mu_al = mx.take_along_axis(mu, idx[..., None].astype(mx.int32), axis=1)
    l_prior = 0.5 * mx.sum(((mu_al - x) ** 2) * fm) / (nf * N_MELS)
    logd = model.dur(mx.stop_gradient(h), tm)
    tmask = tm[..., 0]
    l_dur = mx.sum(((logd - mx.log(mx.maximum(dur, 1.0))) ** 2) * tmask) / mx.sum(tmask)
    return l_mel + l_prior + l_dur, (l_mel, l_prior, l_dur)


def align(model, tok, tlen, x, flen, prior_w=0.0):
    """Run MAS (numba, CPU) on the model's current prior. Returns idx (B,T) int32, dur (B,N) float32 numpy."""
    from mas import maximum_path_batch
    _, mu, _ = model.encode(tok, tlen)
    ll = model.log_lik(mu, x)
    mx.eval(ll)
    ll = np.array(ll, dtype=np.float32)
    tl, fl = np.array(tlen), np.array(flen)
    if prior_w > 0:  # diagonal prior for early training: penalize distance from the straight-line alignment
        B, N, T = ll.shape
        n = np.arange(N)[None, :, None] / np.maximum(tl[:, None, None] - 1, 1)
        t = np.arange(T)[None, None, :] / np.maximum(fl[:, None, None] - 1, 1)
        ll = ll - prior_w * (n - t) ** 2 / (2 * 0.05 ** 2)
    idx = maximum_path_batch(np.ascontiguousarray(ll), tl.astype(np.int64), fl.astype(np.int64))
    B, N = tok.shape
    dur = np.zeros((B, N), np.float32)
    for b in range(B):
        dur[b, : tl[b]] = np.bincount(idx[b, : fl[b]], minlength=tl[b])[: tl[b]]
    return idx, dur


def infer(model, tok, pace=1.0, rounding="round"):
    """tok (1,N) -> normalized mel (T,80) numpy, durations (N,)."""
    tlen = mx.array([tok.shape[1]])
    h, _, tm = model.encode(tok, tlen)
    logd = model.dur(h, tm)[0]
    d = np.exp(np.array(logd)) * pace
    d = np.maximum(1, np.ceil(d) if rounding == "ceil" else np.round(d)).astype(np.int64)
    idx = np.repeat(np.arange(len(d)), d)[None]
    mel = model.decode(h, mx.array(idx, dtype=mx.int32), mx.array([idx.shape[1]]))
    mx.eval(mel)
    return np.array(mel[0]), d


def load_acoustic(path):
    import json
    cfg = json.load(open(path.replace(".safetensors", ".json")))
    m = Acoustic(**cfg["model"])
    m.load_weights(path)
    m.eval()
    mx.eval(m.parameters())
    return m, cfg
