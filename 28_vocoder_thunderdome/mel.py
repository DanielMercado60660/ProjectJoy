"""The one fixed acoustic representation every vocoder in this project reconstructs from.

sr 16000 Hz, n_fft 1024, win 1024 (Hann), hop 256, 80 mel bands (Slaney scale, Slaney norm),
fmin 0, fmax 8000, feature = natural log(max(mel_power_magnitude, 1e-5)) on the *magnitude* STFT.
Waveforms are LJSpeech 22.05 kHz resampled with scipy resample_poly(320, 441).
"""
import numpy as np
from scipy.signal import resample_poly

SR = 16000
N_FFT = 1024
WIN = 1024
HOP = 256
N_MELS = 80
FMIN = 0.0
FMAX = 8000.0
LOG_FLOOR = 1e-5


def hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-9) / min_log_hz) / logstep, mels)


def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """(n_mels, n_fft//2+1) Slaney-normalised triangular filters (same as librosa defaults)."""
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_pts[2:n_mels + 2] - mel_pts[:n_mels])
    return (w * enorm[:, None]).astype(np.float32)


MEL_FB = mel_filterbank()
WINDOW = np.hanning(WIN + 1)[:-1].astype(np.float32)  # periodic Hann


def stft(x, n_fft=N_FFT, hop=HOP, win=None):
    """Centered (reflect-padded) STFT -> complex (frames, n_fft//2+1)."""
    if win is None:
        win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    x = np.pad(x.astype(np.float32), (n_fft // 2, n_fft // 2), mode="reflect")
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(x[idx] * win[None, :], axis=-1)


def istft(S, length, n_fft=N_FFT, hop=HOP, win=None):
    """Weighted overlap-add inverse of stft() (frames, bins) -> waveform of `length`. Needs hop | n_fft."""
    if win is None:
        win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    T = len(S)
    k = n_fft // hop
    frames = (np.fft.irfft(S, n=n_fft, axis=-1) * win[None, :]).reshape(T, k, hop)
    w2 = (win ** 2).reshape(k, hop)
    out = np.zeros((T + k - 1, hop))
    wsum = np.zeros((T + k - 1, hop))
    for j in range(k):
        out[j:j + T] += frames[:, j]
        wsum[j:j + T] += w2[j]
    out = (out / np.maximum(wsum, 1e-8)).reshape(-1)
    out = out[n_fft // 2:n_fft // 2 + length]
    if len(out) < length:
        out = np.pad(out, (0, length - len(out)))
    return out.astype(np.float32)


def log_mel(x):
    """waveform (16 kHz float) -> (frames, 80) log-mel. frames = len(x)//HOP + 1."""
    mag = np.abs(stft(x))
    return np.log(np.maximum(mag @ MEL_FB.T, LOG_FLOOR)).astype(np.float32)


def load_ljspeech_16k(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    assert sr == 22050, sr
    return resample_poly(x, 320, 441).astype(np.float32)
