"""Tiny numpy drum synth: turns a (hits, vel, off) grid into a 44.1 kHz WAV.

Kept fully separate from the model: it only ever sees the symbolic grid.
The same recipes are mirrored in JavaScript inside media/demo.html.
"""
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

SR = 44100


def _env(n, decay_s):
    t = np.arange(n) / SR
    return np.exp(-t / decay_s)


def _noise(n, rng):
    return rng.uniform(-1, 1, n)


def _hp(x, hz):
    b, a = butter(2, hz / (SR / 2), "high")
    return lfilter(b, a, x)


def _bp(x, lo, hi):
    b, a = butter(2, [lo / (SR / 2), hi / (SR / 2)], "band")
    return lfilter(b, a, x)


def _sweep(n, f0, f1, k):
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t * k)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def make_kit(seed=0):
    """Return a list of 9 one-shot samples in instrument order (see data.INSTRUMENTS)."""
    rng = np.random.default_rng(seed)
    kit = []
    n = int(0.45 * SR)
    kick = _sweep(n, 160, 45, 30) * _env(n, 0.18)
    kick[:200] += _noise(200, rng) * 0.3 * _env(200, 0.002)
    kit.append(kick)
    n = int(0.25 * SR)
    snare = 0.6 * _hp(_noise(n, rng), 1500) * _env(n, 0.07) + 0.5 * _sweep(n, 260, 180, 40) * _env(n, 0.05)
    kit.append(snare)
    n = int(0.3 * SR)
    burst = np.zeros(n)
    for d in (0.0, 0.010, 0.021):
        i = int(d * SR)
        burst[i:] += _env(n - i, 0.006 if d < 0.02 else 0.09)
    kit.append(_bp(_noise(n, rng), 800, 2500) * burst * 2.0)
    n = int(0.08 * SR)
    kit.append(_hp(_noise(n, rng), 7000) * _env(n, 0.018))
    n = int(0.5 * SR)
    kit.append(_hp(_noise(n, rng), 6000) * _env(n, 0.16))
    n = int(0.5 * SR)
    kit.append(_sweep(n, 160, 95, 12) * _env(n, 0.22))
    kit.append(_sweep(n, 240, 160, 12) * _env(n, 0.18))
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    metal = sum(np.sign(np.sin(2 * np.pi * f * t)) for f in (523, 811, 1077, 1463, 1879)) / 5
    kit.append(_hp(metal, 3000) * _env(n, 0.45) * 0.35 + _hp(_noise(n, rng), 8000) * _env(n, 0.3) * 0.1)
    n = int(2.0 * SR)
    kit.append(_hp(_noise(n, rng), 4000) * _env(n, 0.7) * 0.6)
    return [k.astype(np.float32) for k in kit]


GAIN = np.array([1.0, 0.8, 0.7, 0.35, 0.3, 0.7, 0.7, 0.35, 0.35])
_KIT = None


def _kit(kit):
    global _KIT
    if kit is not None:
        return kit
    if _KIT is None:
        _KIT = make_kit()
    return _KIT


def render_sequence(grids, bpm=110, loops_each=1, tail_s=1.0, kit=None):
    """Render several (hits, vel, off) grids back to back into one mono float32 waveform.

    bpm may be a single number or one per grid (for tempo morphs).
    """
    kit = _kit(kit)
    bpms = np.broadcast_to(np.asarray(bpm, float), (len(grids),))
    steps = [60.0 / b / 4 for b in bpms]
    total_s = sum(h.shape[0] * loops_each * st for (h, _, _), st in zip(grids, steps))
    n = int((total_s + tail_s) * SR)
    out = np.zeros(n, np.float32)
    t0 = 0.0  # seconds
    for (h, v, o), step in zip(grids, steps):
        for _ in range(loops_each):
            for s, i in zip(*np.nonzero(h > 0.5)):
                a = max(0, int((t0 + (s + float(o[s, i])) * step) * SR))
                smp = kit[i] * (GAIN[i] * float(v[s, i]))
                b = min(n, a + len(smp))
                out[a:b] += smp[: b - a]
            t0 += h.shape[0] * step
    return np.tanh(out * 0.9).astype(np.float32)


def render(hits, vel, off, bpm=110, loops=2, tail_s=1.0, kit=None):
    """Render one [32, 9] grid, looped `loops` times."""
    return render_sequence([(hits, vel, off)], bpm, loops, tail_s, kit)


def write(path, wave):
    sf.write(path, wave, SR, subtype="PCM_16")
