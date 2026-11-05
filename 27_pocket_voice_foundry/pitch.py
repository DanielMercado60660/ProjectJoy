"""Frame-level F0 tracker in numpy (no pitch tracker is installed), aligned to mel.py's frames.

Normalized autocorrelation per frame (Boersma 1993 style): Hann-windowed 1024-sample frame, centered and reflect-padded
exactly like mel.stft, so frame t here is mel frame t. The frame's autocorrelation is divided by the window's own
autocorrelation to undo the taper. Lag search covers 60-500 Hz with parabolic peak interpolation. A frame is voiced when
the peak is >= VOICING and the frame is no more than 45 dB below the clip's loudest frame. Octave guard against
sub-octave picks: take the *first* lag scoring >= 0.9 of the best peak (then its local max). Finally a 5-frame median filter on voiced log-F0.

f0(x) -> (log_f0 (T,) float32 with 0 where unvoiced, voiced (T,) bool)
"""
import numpy as np
from scipy.ndimage import median_filter
from mel import SR, N_FFT, HOP

FMIN, FMAX = 60.0, 500.0
VOICING = 0.45
_W = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
_NF = 2 * N_FFT
_RW = np.fft.irfft(np.abs(np.fft.rfft(_W, _NF)) ** 2, _NF)[:N_FFT]
_RW = (_RW / _RW[0]).astype(np.float64)
LAG_LO, LAG_HI = int(SR / FMAX), int(np.ceil(SR / FMIN))


def f0(x):
    x = np.pad(x.astype(np.float32), (N_FFT // 2, N_FFT // 2), mode="reflect")
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(1, keepdims=True)
    fr = fr * _W[None]
    r = np.fft.irfft(np.abs(np.fft.rfft(fr, _NF, axis=1)) ** 2, _NF, axis=1)[:, :LAG_HI + 2]
    e = r[:, 0]
    rn = r / np.maximum(e[:, None], 1e-12) / _RW[None, :LAG_HI + 2]
    seg = rn[:, LAG_LO:LAG_HI + 1]
    best = seg.max(1)
    # first lag within 0.9 of the best peak, then the local max just after it (avoids sub-octave picks)
    first = np.argmax(seg >= 0.9 * best[:, None], 1)
    win = np.clip(first[:, None] + np.arange(12)[None], 0, seg.shape[1] - 1)
    k = win[np.arange(n), np.argmax(seg[np.arange(n)[:, None], win], 1)]
    lag = k + LAG_LO
    peak = seg[np.arange(n), k]
    # parabolic interpolation around the chosen lag
    l = np.clip(lag, 1, LAG_HI)
    a, b, c = rn[np.arange(n), l - 1], rn[np.arange(n), l], rn[np.arange(n), l + 1]
    den = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / den, 0.0)
    off = np.clip(off, -0.5, 0.5)
    hz = SR / (l + off)
    db = 10 * np.log10(np.maximum(e, 1e-12))
    voiced = (peak >= VOICING) & (db > db.max() - 45) & (hz >= FMIN) & (hz <= FMAX)
    lf = np.where(voiced, np.log(hz), 0.0)
    if voiced.sum() > 5:
        sm = median_filter(np.where(voiced, lf, np.nan_to_num(np.median(lf[voiced]))), size=5, mode="nearest")
        lf = np.where(voiced, sm, 0.0)
    return lf.astype(np.float32), voiced


if __name__ == "__main__":
    t = np.arange(SR * 2) / SR
    for hz in (80, 150, 220, 400):
        y = sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 6)) * 0.3 + 0.01 * np.random.default_rng(0).standard_normal(len(t))
        lf, v = f0(y)
        print(hz, "Hz -> median", round(float(np.exp(np.median(lf[v]))), 1), "voiced frac", round(float(v.mean()), 2))
