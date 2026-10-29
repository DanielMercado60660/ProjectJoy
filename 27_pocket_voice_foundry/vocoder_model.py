# PROVENANCE: copied from ../28_vocoder_thunderdome/model.py on 2026-10-24 (Vocoder + load_vocoder; unmodified).
"""Small Vocos-style neural vocoder in MLX: log-mel frames -> ConvNeXt backbone (frame rate) ->
per-frame complex STFT (log-magnitude + phase) -> inverse STFT (n_fft 1024, hop 256) -> waveform.

Working at frame rate (62.5 frames/s) instead of sample rate is what makes it trainable in ~15 minutes.
Plus the differentiable MLX STFT used by the multi-resolution STFT + mel L1 training loss.
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mel import N_FFT, HOP, N_MELS, MEL_FB, LOG_FLOOR

N_BINS = N_FFT // 2 + 1


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mult=3, k=7):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * mult)
        self.pw2 = nn.Linear(dim * mult, dim)
        self.gamma = mx.full((dim,), 1e-2)

    def __call__(self, x):  # (B, T, C)
        return x + self.gamma * self.pw2(nn.gelu(self.pw1(self.norm(self.dw(x)))))


def _ola_envelope(n_frames):
    w2 = np.hanning(N_FFT + 1)[:-1] ** 2
    n = N_FFT + HOP * (n_frames - 1)
    env = np.zeros(n)
    for i in range(n_frames):
        env[i * HOP:i * HOP + N_FFT] += w2
    return env


class Vocoder(nn.Module):
    def __init__(self, dim=256, n_blocks=8):
        super().__init__()
        self.cfg = {"dim": dim, "n_blocks": n_blocks}
        self.inp = nn.Conv1d(N_MELS, dim, 7, padding=3)
        self.norm_in = nn.LayerNorm(dim)
        self.blocks = [ConvNeXtBlock(dim) for _ in range(n_blocks)]
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2 * N_BINS)
        self._win = mx.array(np.hanning(N_FFT + 1)[:-1].astype(np.float32))

    def spec(self, mel):  # mel (B, T, 80) -> real, imag (B, T, 513)
        h = self.norm_in(self.inp(mel))
        for b in self.blocks:
            h = b(h)
        o = self.head(self.norm_out(h))
        logmag, ph = o[..., :N_BINS], o[..., N_BINS:]
        mag = mx.minimum(mx.exp(logmag), 1e2)
        return mag * mx.cos(ph), mag * mx.sin(ph)

    def __call__(self, mel):
        """mel (B, T, 80) -> waveform (B, (T-1)*HOP). Same centering as mel.stft()."""
        re, im = self.spec(mel)
        B, T, _ = re.shape
        frames = mx.fft.irfft(re + 1j * im, n=N_FFT, axis=-1) * self._win  # (B, T, 1024)
        k = N_FFT // HOP  # 4 chunks per frame
        ch = frames.reshape(B, T, k, HOP)
        out = mx.zeros((B, T + k - 1, HOP))
        for j in range(k):
            out = out + mx.pad(ch[:, :, j, :], [(0, 0), (j, k - 1 - j), (0, 0)])
        out = out.reshape(B, -1)  # length N_FFT + HOP*(T-1)
        env = mx.array(np.maximum(_ola_envelope(T), 1e-8).astype(np.float32))
        out = out / env
        return out[:, N_FFT // 2: N_FFT // 2 + (T - 1) * HOP]


# ---------------- differentiable STFT + losses ----------------
_IDX_CACHE = {}


def mx_stft_mag(x, n_fft, hop):
    """x (B, L) -> |STFT| (B, frames, n_fft//2+1); zero-padded centering, Hann window."""
    B, L = x.shape
    xp = mx.pad(x, [(0, 0), (n_fft // 2, n_fft // 2)])
    n_frames = 1 + (L + n_fft - n_fft) // hop
    key = (n_fft, hop, n_frames)
    if key not in _IDX_CACHE:
        idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
        _IDX_CACHE[key] = (mx.array(idx.astype(np.int32)), mx.array(np.hanning(n_fft + 1)[:-1].astype(np.float32)))
    idx, win = _IDX_CACHE[key]
    fr = xp[:, idx] * win
    S = mx.fft.rfft(fr, axis=-1)
    return mx.sqrt(mx.real(S) ** 2 + mx.imag(S) ** 2 + 1e-9)


RESOLUTIONS = [(512, 128), (1024, 256), (2048, 512)]
_FB = mx.array(MEL_FB.T)  # (513, 80)


def vocoder_loss(y_hat, y):
    sc, lm = 0.0, 0.0
    for n_fft, hop in RESOLUTIONS:
        a, b = mx_stft_mag(y_hat, n_fft, hop), mx_stft_mag(y, n_fft, hop)
        sc = sc + mx.sqrt(mx.sum((b - a) ** 2)) / mx.sqrt(mx.sum(b ** 2) + 1e-9)
        lm = lm + mx.mean(mx.abs(mx.log(a + 1e-5) - mx.log(b + 1e-5)))
    sc, lm = sc / len(RESOLUTIONS), lm / len(RESOLUTIONS)
    A, Bm = mx_stft_mag(y_hat, N_FFT, HOP), mx_stft_mag(y, N_FFT, HOP)
    mel_l1 = mx.mean(mx.abs(mx.log(mx.maximum(A @ _FB, LOG_FLOOR)) - mx.log(mx.maximum(Bm @ _FB, LOG_FLOOR))))
    total = sc + lm + 2.0 * mel_l1
    return total, (sc, lm, mel_l1)


def load_vocoder(path):
    import json
    cfg = json.load(open(path.replace(".safetensors", ".json")))
    m = Vocoder(**cfg["model"])
    m.load_weights(path)
    mx.eval(m.parameters())
    return m, cfg
