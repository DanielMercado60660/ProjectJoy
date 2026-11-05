"""Round 6 vocoder: standalone loader + CLI. Copy this one file plus a checkpoint (checkpoints/r6_*.safetensors + .json)
into another project; it needs only numpy, mlx (and soundfile/scipy for the CLI). No imports from the rest of 28.

The fixed mel it inverts (identical to 28's mel.py, checked bit-exactly by test_r6.py):
16 kHz, n_fft 1024, Hann 1024, hop 256, centred (reflect-padded) frames, 80 Slaney mel bands 0-8 kHz (Slaney norm),
feature = ln(max(mel_filterbank @ |STFT|, 1e-5)), shape (frames, 80), frames = n_samples // 256 + 1.

Generator: Vocos-style. Conv(80->dim, k7) + LayerNorm -> n_blocks ConvNeXt blocks at frame rate (62.5 frames/s) ->
LayerNorm -> Linear to log-magnitude + phase for 513 bins -> inverse STFT (n_fft 1024, hop 256) -> waveform of
(frames - 1) * 256 samples, sample t*256 centred on frame t (same centring as the analysis STFT).

Python use:
    from r6_vocoder import load, vocode, log_mel
    G = load("checkpoints/r6_gan.safetensors")
    wav16k = vocode(G, logmel)                      # logmel: numpy (frames, 80) float32
CLI (analysis-synthesis of a wav, or a saved mel):
    ../.venv/bin/python r6_vocoder.py --ckpt checkpoints/r6_gan.safetensors --wav in.wav --out out.wav
    ../.venv/bin/python r6_vocoder.py --ckpt checkpoints/r6_gan.safetensors --mel mel.npy --out out.wav
"""
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn

SR, N_FFT, HOP, N_MELS, FMAX, LOG_FLOOR = 16000, 1024, 256, 80, 8000.0, 1e-5
N_BINS = N_FFT // 2 + 1


# ---------------- the fixed mel (verbatim maths of 28's mel.py) ----------------
def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64); f_sp = 200.0 / 3; min_log_mel = 1000.0 / f_sp; logstep = np.log(6.4) / 27.0
    return np.where(f >= 1000.0, min_log_mel + np.log(np.maximum(f, 1e-9) / 1000.0) / logstep, f / f_sp)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64); f_sp = 200.0 / 3; min_log_mel = 1000.0 / f_sp; logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, 1000.0 * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filterbank():
    fft_freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    mel_pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(FMAX), N_MELS + 2))
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[:, None] - fft_freqs[None, :]
    w = np.maximum(0, np.minimum(-ramps[:-2] / fdiff[:-1, None], ramps[2:] / fdiff[1:, None]))
    enorm = 2.0 / (mel_pts[2:N_MELS + 2] - mel_pts[:N_MELS])
    return (w * enorm[:, None]).astype(np.float32)


MEL_FB = mel_filterbank()


def log_mel(x):
    """16 kHz float waveform -> (len(x)//256 + 1, 80) log-mel (the fixed definition)."""
    win = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    xp = np.pad(np.asarray(x, np.float32), (N_FFT // 2, N_FFT // 2), mode="reflect")
    n = 1 + (len(xp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    mag = np.abs(np.fft.rfft(xp[idx] * win[None, :], axis=-1))
    return np.log(np.maximum(mag @ MEL_FB.T, LOG_FLOOR)).astype(np.float32)


# ---------------- generator ----------------
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mult=3, k=7):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * mult)
        self.pw2 = nn.Linear(dim * mult, dim)
        self.gamma = mx.full((dim,), 1e-2)

    def __call__(self, x):
        return x + self.gamma * self.pw2(nn.gelu(self.pw1(self.norm(self.dw(x)))))


class Vocoder(nn.Module):
    """Same parameter layout as 28's model.Vocoder, so round-1/3 checkpoints load into it."""

    def __init__(self, dim=384, n_blocks=8, mult=3):
        super().__init__()
        self.cfg = {"dim": dim, "n_blocks": n_blocks, "mult": mult}
        self.inp = nn.Conv1d(N_MELS, dim, 7, padding=3)
        self.norm_in = nn.LayerNorm(dim)
        self.blocks = [ConvNeXtBlock(dim, mult) for _ in range(n_blocks)]
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2 * N_BINS)
        self._win = mx.array(np.hanning(N_FFT + 1)[:-1].astype(np.float32))
        self._env = {}

    def head_out(self, mel):
        """mel (B, T, 80) -> (log-magnitude, phase), each (B, T, 513), on the analysis STFT's frame grid."""
        h = self.norm_in(self.inp(mel))
        for b in self.blocks:
            h = b(h)
        o = self.head(self.norm_out(h))
        return o[..., :N_BINS], o[..., N_BINS:]

    def spec(self, mel):
        logmag, ph = self.head_out(mel)
        mag = mx.minimum(mx.exp(logmag), 1e2)
        return mag * mx.cos(ph), mag * mx.sin(ph)

    def _envelope(self, T):
        if T not in self._env:
            w2 = np.hanning(N_FFT + 1)[:-1] ** 2
            env = np.zeros(N_FFT + HOP * (T - 1))
            for i in range(T):
                env[i * HOP:i * HOP + N_FFT] += w2
            self._env[T] = mx.array(np.maximum(env, 1e-8).astype(np.float32))
        return self._env[T]

    def __call__(self, mel):
        """mel (B, T, 80) -> waveform (B, (T-1)*256)."""
        return self.istft(*self.spec(mel))

    def istft(self, re, im):
        B, T, _ = re.shape
        frames = mx.fft.irfft(re + 1j * im, n=N_FFT, axis=-1) * self._win
        k = N_FFT // HOP
        ch = frames.reshape(B, T, k, HOP)
        out = mx.zeros((B, T + k - 1, HOP))
        for j in range(k):
            out = out + mx.pad(ch[:, :, j, :], [(0, 0), (j, k - 1 - j), (0, 0)])
        out = out.reshape(B, -1) / self._envelope(T)
        return out[:, N_FFT // 2: N_FFT // 2 + (T - 1) * HOP]


def load(path):
    """Load a generator from <path>.safetensors (+ sibling .json holding {"model": {dim, n_blocks[, mult]}})."""
    cfg = json.load(open(path.replace(".safetensors", ".json")))
    G = Vocoder(**cfg["model"])
    G.load_weights(path)
    mx.eval(G.parameters())
    G.meta = cfg
    return G


def vocode(G, logmel):
    """numpy (frames, 80) log-mel -> numpy float32 waveform at 16 kHz, (frames-1)*256 samples."""
    y = G(mx.array(np.asarray(logmel, np.float32)[None]))
    mx.eval(y)
    return np.array(y[0])


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav", help="input wav (any rate; resampled to 16 kHz) -> its mel -> vocoder")
    ap.add_argument("--mel", help="input .npy log-mel (frames, 80) on the fixed mel")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import soundfile as sf
    if a.wav:
        from math import gcd
        from scipy.signal import resample_poly
        x, sr = sf.read(a.wav, dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        if sr != SR:
            g = gcd(SR, sr); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        m = log_mel(x[: len(x) // HOP * HOP])
    else:
        m = np.load(a.mel)
    y = vocode(load(a.ckpt), m)
    sf.write(a.out, np.clip(y, -1, 1), SR)
    print(f"{a.out}: {len(y) / SR:.2f} s from {len(m)} mel frames")


if __name__ == "__main__":
    main()
