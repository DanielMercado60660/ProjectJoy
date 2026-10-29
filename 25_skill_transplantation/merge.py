"""Training-free ways to combine two specialists that branched from one shared start theta0.

All functions take/return flat dicts of numpy arrays (common.PARAM_KEYS).
  interpolate(A, B, alpha)          (1-alpha) A + alpha B; alpha=0.5 is naive weight averaging
  task_arithmetic(t0, [A, B], lam)  t0 + lam * sum_i (theta_i - t0)
  ties(t0, [A, B], lam, k)          TIES-merging: trim each task vector to its top-k fraction by
                                    magnitude, elect a sign per entry, average the agreeing entries
  dare(t0, [A, B], lam, p, seed)    DARE: drop a random fraction p of each task vector, rescale by
                                    1/(1-p), then task arithmetic
"""
import numpy as np

from common import ACTOR_KEYS, LOGSTD_MAX, LOGSTD_MIN, PARAM_KEYS


def _fix(p):
    p["log_std"] = np.clip(p["log_std"], LOGSTD_MIN, LOGSTD_MAX).astype(np.float32)
    return {k: v.astype(np.float32) for k, v in p.items()}


def interpolate(A, B, alpha):
    return _fix({k: (1 - alpha) * A[k] + alpha * B[k] for k in PARAM_KEYS})


def task_vectors(t0, thetas):
    return [{k: th[k] - t0[k] for k in PARAM_KEYS} for th in thetas]


def task_arithmetic(t0, thetas, lam):
    tv = task_vectors(t0, thetas)
    return _fix({k: t0[k] + lam * sum(t[k] for t in tv) for k in PARAM_KEYS})


def _flat(d, keys=PARAM_KEYS):
    return np.concatenate([d[k].ravel() for k in keys])


def _unflat(v, like):
    out, i = {}, 0
    for k in PARAM_KEYS:
        n = like[k].size
        out[k] = v[i:i + n].reshape(like[k].shape)
        i += n
    return out


def ties(t0, thetas, lam, k=0.2):
    tv = [_flat(t) for t in task_vectors(t0, thetas)]
    trimmed = []
    for t in tv:
        thr = np.quantile(np.abs(t), 1 - k)
        trimmed.append(np.where(np.abs(t) >= thr, t, 0.0))
    T = np.stack(trimmed)
    sign = np.sign(T.sum(0))
    agree = (np.sign(T) == sign[None]) & (T != 0)
    cnt = np.maximum(agree.sum(0), 1)
    merged = (T * agree).sum(0) / cnt
    m = _unflat(merged, t0)
    return _fix({kk: t0[kk] + lam * m[kk] for kk in PARAM_KEYS})


def dare(t0, thetas, lam, p=0.9, seed=0):
    rng = np.random.default_rng(seed)
    tv = task_vectors(t0, thetas)
    out = {}
    for kk in PARAM_KEYS:
        s = 0.0
        for t in tv:
            mask = rng.random(t[kk].shape) >= p
            s = s + t[kk] * mask / (1 - p)
        out[kk] = t0[kk] + lam * s
    return _fix(out)


def delta_stats(t0, A, B):
    """Size and overlap of the two task vectors, actor parameters only (the part that acts)."""
    ta, tb = [_flat(t, ACTOR_KEYS) for t in task_vectors(t0, [A, B])]
    w0 = _flat(t0, ACTOR_KEYS)
    cos = float(ta @ tb / (np.linalg.norm(ta) * np.linalg.norm(tb) + 1e-12))
    sign_conflict = float(np.mean(np.sign(ta) != np.sign(tb)))
    return dict(norm_tau_fwd=float(np.linalg.norm(ta)), norm_tau_bwd=float(np.linalg.norm(tb)),
                norm_theta0=float(np.linalg.norm(w0)), cos_tau=cos, sign_conflict_frac=sign_conflict,
                dist_specialists=float(np.linalg.norm(ta - tb)))
