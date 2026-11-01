# PROVENANCE: copied from ../27_pocket_voice_foundry/model.py on 2026-10-30 (lines 1-62 and masks(), only what ../27_pocket_voice_foundry/ctc.py needs; not modified).
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



def masks(lens, L):
    m = (mx.arange(L)[None, :] < lens[:, None]).astype(mx.float32)  # (B, L)
    amask = mx.where(m[:, None, None, :] > 0, 0.0, -1e9).astype(mx.float32)
    return m[..., None], amask

