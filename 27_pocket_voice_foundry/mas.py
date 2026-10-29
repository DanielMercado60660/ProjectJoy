"""Monotonic alignment search (Glow-TTS, Kim et al. 2020) and DTW, in numba.

maximum_path: value (N tokens, T frames) log-likelihoods -> per-frame token index (T,), the monotonic,
non-skipping path (every token gets >= 1 frame, starts at token 0, ends at token N-1) with max total value.
"""
import numpy as np
import numba
from numba import njit, prange


@njit(cache=True)
def _mp_one(value, n, t, out):
    Q = np.full((n, t), -1e30, dtype=np.float32)
    Q[0, 0] = value[0, 0]
    for y in range(1, t):
        lo = max(0, n + y - t)
        hi = min(n - 1, y)
        for x in range(lo, hi + 1):
            stay = Q[x, y - 1]
            move = Q[x - 1, y - 1] if x > 0 else -1e30
            Q[x, y] = (stay if stay > move else move) + value[x, y]
    x = n - 1
    for y in range(t - 1, -1, -1):
        out[y] = x
        if x > 0 and y > 0 and (x == y or Q[x - 1, y - 1] > Q[x, y - 1]):
            x -= 1


@njit(parallel=True, cache=True)
def maximum_path_batch(values, ns, ts):
    """values (B, Nmax, Tmax) float32; returns idx (B, Tmax) int32 (padding frames -> 0)."""
    B = values.shape[0]
    out = np.zeros((B, values.shape[2]), dtype=np.int32)
    for b in prange(B):
        _mp_one(values[b], ns[b], ts[b], out[b])
    return out


def maximum_path(value):
    v = np.ascontiguousarray(value[None].astype(np.float32))
    return maximum_path_batch(v, np.array([value.shape[0]]), np.array([value.shape[1]]))[0]


def durations_from_idx(idx, n):
    return np.bincount(idx, minlength=n)[:n]


@njit(cache=True)
def dtw_cost(a, b):
    """Mean per-step L1-per-dim cost along the DTW path between (Ta, D) and (Tb, D); returns (cost, path_len)."""
    Ta, Tb, D = a.shape[0], b.shape[0], a.shape[1]
    C = np.empty((Ta, Tb), dtype=np.float64)
    for i in range(Ta):
        for j in range(Tb):
            s = 0.0
            for k in range(D):
                s += abs(a[i, k] - b[j, k])
            C[i, j] = s / D
    A = np.full((Ta + 1, Tb + 1), 1e30)
    L = np.zeros((Ta + 1, Tb + 1))
    A[0, 0] = 0.0
    for i in range(1, Ta + 1):
        for j in range(1, Tb + 1):
            best, bl = A[i - 1, j - 1], L[i - 1, j - 1]
            if A[i - 1, j] < best:
                best, bl = A[i - 1, j], L[i - 1, j]
            if A[i, j - 1] < best:
                best, bl = A[i, j - 1], L[i, j - 1]
            A[i, j] = best + C[i - 1, j - 1]
            L[i, j] = bl + 1
    return A[Ta, Tb] / L[Ta, Tb], L[Ta, Tb]


@njit(cache=True)
def dtw_cost_euclid(a, b):
    """Same DTW but Euclidean frame distance (for mel-cepstral distortion)."""
    Ta, Tb, D = a.shape[0], b.shape[0], a.shape[1]
    A = np.full((Ta + 1, Tb + 1), 1e30)
    L = np.zeros((Ta + 1, Tb + 1))
    A[0, 0] = 0.0
    for i in range(1, Ta + 1):
        for j in range(1, Tb + 1):
            s = 0.0
            for k in range(D):
                d = a[i - 1, k] - b[j - 1, k]
                s += d * d
            c = np.sqrt(s)
            best, bl = A[i - 1, j - 1], L[i - 1, j - 1]
            if A[i - 1, j] < best:
                best, bl = A[i - 1, j], L[i - 1, j]
            if A[i, j - 1] < best:
                best, bl = A[i, j - 1], L[i, j - 1]
            A[i, j] = best + c
            L[i, j] = bl + 1
    return A[Ta, Tb] / L[Ta, Tb]
