# PROVENANCE: copied verbatim from ../28_vocoder_thunderdome/griffin_lim.py on 2026-10-24 (not modified below this line).
"""Classical baseline: log-mel -> linear magnitude (non-negative least squares through the mel
filterbank) -> fast Griffin-Lim (Perraudin et al. 2013, momentum 0.99). Pure numpy, CPU."""
import numpy as np
from mel import MEL_FB, stft, istft, HOP


def mel_to_linear(logmel, n_iter=60):
    """(T, 80) log-mel -> (T, 513) magnitude. pinv init, then multiplicative NNLS updates."""
    M = np.exp(logmel.astype(np.float64))  # (T, 80)
    fb = MEL_FB.astype(np.float64)  # (80, 513)
    S = np.maximum(M @ np.linalg.pinv(fb).T, 1e-8)
    num = M @ fb  # (T, 513)
    G = fb.T @ fb
    for _ in range(n_iter):
        S *= num / np.maximum(S @ G, 1e-12)
    return S.astype(np.float32)


def griffin_lim(logmel, n_iter=64, momentum=0.99, seed=0, length=None):
    mag = mel_to_linear(logmel)
    T = len(mag)
    length = length or (T - 1) * HOP
    rng = np.random.default_rng(seed)
    angles = np.exp(2j * np.pi * rng.random(mag.shape))
    prev = np.zeros_like(mag, dtype=np.complex128)
    x = istft(mag * angles, length)
    for _ in range(n_iter):
        reb = stft(x)[:T]
        if reb.shape[0] < T:
            reb = np.pad(reb, ((0, T - reb.shape[0]), (0, 0)))
        angles = reb - (momentum / (1 + momentum)) * prev
        prev = reb
        angles = angles / np.maximum(np.abs(angles), 1e-16)
        x = istft(mag * angles, length)
    return x
