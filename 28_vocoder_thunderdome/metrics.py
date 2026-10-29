"""Objective quality proxies, all computed at 16 kHz on held-out clips against the ground truth wav.

- mrstft: multi-resolution STFT distance = mean over (512/128, 1024/256, 2048/512) of
  spectral convergence + mean |log mag diff|  (lower is better). Note: this is also the neural
  vocoder's training loss, so it favours the neural model by construction.
- mel_l1: mean |log-mel(output) - log-mel(gt)| with the project's one mel definition.
- pitch: YIN f0 (60-400 Hz, 10 ms hop): RMSE in cents on frames voiced in both, gross pitch error
  (>20% off) %, voicing decision error %, and periodicity RMSE (1 - min CMND, on GT-voiced frames).
- transients: high-band (>2 kHz) envelope around GT onsets: pre-onset energy excess (dB, energy in
  the 30..5 ms before each onset relative to GT; positive = smeared/pre-echo) and 10-90% rise time (ms).
"""
import numpy as np
from scipy.signal import butter, sosfilt
from mel import stft, log_mel, SR

RES = [(512, 128), (1024, 256), (2048, 512)]


def mrstft(y, x):
    sc, lm = [], []
    for n, h in RES:
        w = np.hanning(n + 1)[:-1].astype(np.float32)
        a, b = np.abs(stft(y, n, h, w)), np.abs(stft(x, n, h, w))
        sc.append(np.linalg.norm(b - a) / (np.linalg.norm(b) + 1e-9))
        lm.append(np.mean(np.abs(np.log(a + 1e-5) - np.log(b + 1e-5))))
    return float(np.mean(sc) + np.mean(lm))


def mel_l1(y, x):
    return float(np.mean(np.abs(log_mel(y) - log_mel(x))))


def yin(x, fmin=60, fmax=400, hop=160, win=1024, thresh=0.15):
    tau_min, tau_max = SR // fmax, SR // fmin
    xp = np.pad(x, (0, win))
    n = 1 + (len(x) - 1) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    fr = xp[idx].astype(np.float64)
    W = win // 2
    # difference function via autocorrelation (FFT)
    F = np.fft.rfft(fr, 2 * win)
    acf = np.fft.irfft(F * np.conj(F))[:, :W + 1]
    cs = np.cumsum(fr ** 2, axis=1)
    e0 = cs[:, W - 1:W]  # energy of first W samples
    # energy of window shifted by tau: sum x[t+tau]^2, t<W
    etau = np.stack([cs[:, tau + W - 1] - (cs[:, tau - 1] if tau > 0 else 0) for tau in range(W + 1)], 1)
    d = e0 + etau - 2 * acf
    d[:, 0] = 0
    cm = np.cumsum(d[:, 1:], axis=1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * np.arange(1, W + 1)[None, :] / np.maximum(cm, 1e-12)
    seg = cmnd[:, tau_min:tau_max + 1]
    f0 = np.zeros(n); per = np.clip(1 - seg.min(1), 0, 1)
    rms = np.sqrt(np.mean(fr ** 2, 1))
    for i in range(n):
        below = np.where(seg[i] < thresh)[0]
        if len(below) == 0 or rms[i] < 0.01 * rms.max():
            continue
        j = below[0]
        while j + 1 < len(seg[i]) and seg[i, j + 1] < seg[i, j]:
            j += 1
        t = j + tau_min
        if 1 <= j < len(seg[i]) - 1:  # parabolic interpolation
            a, b, c = seg[i, j - 1], seg[i, j], seg[i, j + 1]
            den = a - 2 * b + c
            t = t + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0)
        f0[i] = SR / t
    return f0, per


def pitch_metrics(y, x):
    fy, py = yin(y); fx, px = yin(x)
    n = min(len(fy), len(fx)); fy, fx, py, px = fy[:n], fx[:n], py[:n], px[:n]
    both = (fy > 0) & (fx > 0)
    cents = 1200 * np.log2(fy[both] / fx[both]) if both.any() else np.array([0.0])
    gpe = np.mean(np.abs(fy[both] / fx[both] - 1) > 0.2) * 100 if both.any() else 0.0
    vde = np.mean((fy > 0) != (fx > 0)) * 100
    gv = fx > 0
    per_rmse = float(np.sqrt(np.mean((py[gv] - px[gv]) ** 2))) if gv.any() else 0.0
    fine = np.abs(cents) < 1200 * np.log2(1.2)
    return {"f0_rmse_cents": float(np.sqrt(np.mean(cents[fine] ** 2))) if fine.any() else 0.0,
            "gpe_pct": float(gpe), "vde_pct": float(vde), "periodicity_rmse": per_rmse}


_SOS = butter(4, 2000, "highpass", fs=SR, output="sos")


def hf_env_db(x, hop=16, win=32):
    h = sosfilt(_SOS, x.astype(np.float64))
    n = (len(h) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    e = np.mean(h[idx] ** 2, 1)
    return 10 * np.log10(e + 1e-10)  # 1 ms resolution


def gt_onsets(env, rise_db=15, min_gap=50):
    ons, last = [], -10 ** 9
    for t in range(30, len(env) - 30):
        if env[t] - env[t - 10:t].min() > rise_db and env[t] >= env[t - 3:t + 4].max() - 1e-9 \
                and env[t] > env.max() - 45 and t - last > min_gap:
            ons.append(t); last = t
    return ons


def rise_time(env, t):
    lo_i = t - 15 + int(np.argmin(env[t - 15:t + 1]))
    pk_i = lo_i + int(np.argmax(env[lo_i:t + 15]))
    lo, pk = env[lo_i], env[pk_i]
    seg = env[lo_i:pk_i + 1]
    a = np.argmax(seg >= lo + 0.1 * (pk - lo)); b = np.argmax(seg >= lo + 0.9 * (pk - lo))
    return float(b - a)


def transient_metrics(y, x):
    ex, ey = hf_env_db(x), hf_env_db(y)
    n = min(len(ex), len(ey)); ex, ey = ex[:n], ey[:n]
    ons = [t for t in gt_onsets(ex) if 30 <= t < n - 30]
    if not ons:
        return {"n_onsets": 0, "pre_onset_excess_db": float("nan"), "post_onset_excess_db": float("nan"),
                "hf_level_db": float("nan"), "rise_ms": float("nan"), "rise_ms_gt": float("nan")}
    pre = [10 * np.log10(np.mean(10 ** (ey[t - 30:t - 5] / 10)) / np.mean(10 ** (ex[t - 30:t - 5] / 10))) for t in ons]
    post = [10 * np.log10(np.mean(10 ** (ey[t + 5:t + 30] / 10)) / np.mean(10 ** (ex[t + 5:t + 30] / 10))) for t in ons]
    hf = 10 * np.log10(np.mean(10 ** (ey / 10)) / np.mean(10 ** (ex / 10)))
    return {"n_onsets": len(ons), "pre_onset_excess_db": float(np.mean(pre)),
            "post_onset_excess_db": float(np.mean(post)), "hf_level_db": float(hf),
            "rise_ms": float(np.mean([rise_time(ey, t) for t in ons])),
            "rise_ms_gt": float(np.mean([rise_time(ex, t) for t in ons]))}
