"""Classical baseline: threshold/centroid tracking -> known-g parabola fits before and after the
first bounce -> e = |v_out| / |v_in| at ground contact.  Uncertainty from a residual bootstrap.

Uses the same known geometry the network implicitly learns (scale, ground row, ball radius, g,
fps).  It is NOT told the exposure time or noise level (the network isn't either).
"""
import numpy as np
from scipy.ndimage import uniform_filter

import scene

PRIOR_MU = 0.5 * sum(scene.E_RANGE)
PRIOR_SD = (scene.E_RANGE[1] - scene.E_RANGE[0]) / np.sqrt(12)
MAX_EXPO = scene.EXPO_RANGE[1] / scene.FPS     # conservative exclusion window around the impact


def track(frames_u8):
    """(T,H,W) uint8 -> (T,2) centre estimates in metres (x, y), NaN where lost."""
    f = frames_u8.astype(np.float32) / 255.0
    g = int(scene.GROUND_ROW)
    out = np.full((len(f), 2), np.nan)
    for i, im in enumerate(f):
        sm = uniform_filter(im[:g], size=3)
        bg = np.median(sm)
        thr = bg + 0.5 * (sm.max() - bg)
        m = sm > thr
        if m.sum() == 0:
            continue
        # keep the largest blob's neighbourhood: window around the max pixel
        r0, c0 = np.unravel_index(np.argmax(sm), sm.shape)
        win = np.zeros_like(m); win[max(0, r0 - 6):r0 + 7, max(0, c0 - 6):c0 + 7] = True
        w = (im[:g] - bg).clip(0) * (m & win)
        if w.sum() <= 0:
            continue
        rr, cc = np.mgrid[0:g, 0:im.shape[1]] + 0.5
        cy = (w * rr).sum() / w.sum(); cx = (w * cc).sum() / w.sum()
        out[i] = (cx / scene.SCALE, (scene.GROUND_ROW - cy) / scene.SCALE)
    return out


def _fit(t, y):
    """Known-g parabola y = a + b t - g/2 t^2  -> (a, b, residuals)."""
    A = np.stack([np.ones_like(t), t], 1)
    z = y + 0.5 * scene.G * t ** 2
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    return coef, z - A @ coef


def _contact(coef, which):
    """Times where the fitted parabola meets y = R; 'late' root for pre, 'early' root for post."""
    a, b = coef
    disc = b ** 2 + 2 * scene.G * (a - scene.R)
    if disc < 0:
        return np.nan                         # parabola never reaches the ground: use apex
    s = np.sqrt(disc)
    r1, r2 = (b - s) / scene.G, (b + s) / scene.G
    return r2 if which == "late" else r1


def _e_from(cpre, cpost):
    t_in = _contact(cpre, "late"); t_out = _contact(cpost, "early")
    if not (np.isfinite(t_in) and np.isfinite(t_out)):
        return np.nan, np.nan
    v_in = cpre[1] - scene.G * t_in; v_out = cpost[1] - scene.G * t_out
    return -v_out / v_in, t_in


def estimate(frames_u8, n_boot=64, rng=None):
    """-> dict(e, sd, ok, track, segments). Falls back to the prior when the fit is impossible."""
    rng = rng or np.random.default_rng(0)
    xy = track(frames_u8)
    y = xy[:, 1]; t = np.arange(len(y)) / scene.FPS
    good = np.isfinite(y)
    fail = dict(e=PRIOR_MU, sd=PRIOR_SD, ok=False, track=xy, pre=None, post=None, t_c=np.nan)
    # first bounce: first local minimum of the height track that is near the ground
    k = None
    for i in range(1, len(y) - 1):
        if good[i - 1] and good[i] and good[i + 1] and y[i] <= y[i - 1] and y[i] < y[i + 1] \
                and y[i] < scene.R + 0.15:
            k = i; break
    if k is None:
        return fail
    # second bounce (end of the post segment)
    k2 = len(y)
    for i in range(k + 2, len(y) - 1):
        if good[i - 1] and good[i] and good[i + 1] and y[i] <= y[i - 1] and y[i] < y[i + 1] \
                and y[i] < scene.R + 0.15:
            k2 = i; break
    pre = np.arange(0, k); post = np.arange(k + 1, k2)
    for _ in range(2):                         # refit after excluding frames whose exposure covers the impact
        pre = pre[good[pre]]; post = post[good[post]]
        if len(pre) < 3 or len(post) < 2:
            return fail
        cpre, _ = _fit(t[pre], y[pre]); cpost, _ = _fit(t[post], y[post])
        e, tc = _e_from(cpre, cpost)
        if not np.isfinite(e):
            return fail
        pre = np.arange(0, k + 1); pre = pre[t[pre] + MAX_EXPO < tc]
        post = np.arange(k, k2); post = post[t[post] > tc]
    pre = pre[good[pre]]; post = post[good[post]]
    if len(pre) < 3 or len(post) < 2:
        return fail
    cpre, rpre = _fit(t[pre], y[pre]); cpost, rpost = _fit(t[post], y[post])
    e, tc = _e_from(cpre, cpost)
    if not np.isfinite(e):
        return fail
    # residual bootstrap (pooled residual scale; post segment can be as short as 2 points)
    res = np.concatenate([rpre, rpost])
    dof = max(1, len(res) - 4)
    s = np.sqrt((res ** 2).sum() / dof)
    s = max(s, 0.25 / scene.SCALE)             # floor: quarter-pixel localisation
    Apre = cpre[0] + cpre[1] * t[pre] - 0.5 * scene.G * t[pre] ** 2
    Apost = cpost[0] + cpost[1] * t[post] - 0.5 * scene.G * t[post] ** 2
    eb = []
    for _ in range(n_boot):
        c1, _ = _fit(t[pre], Apre + s * rng.standard_normal(len(pre)))
        c2, _ = _fit(t[post], Apost + s * rng.standard_normal(len(post)))
        eb.append(_e_from(c1, c2)[0])
    eb = np.array(eb); eb = eb[np.isfinite(eb)]
    sd = float(np.std(eb)) if len(eb) > 8 else PRIOR_SD
    return dict(e=float(e), sd=max(sd, 1e-3), ok=True, track=xy, pre=(pre, cpre), post=(post, cpost), t_c=tc)


def estimate_many(X):
    rng = np.random.default_rng(0)
    rs = [estimate(x, rng=rng) for x in X]
    return np.array([r["e"] for r in rs]), np.array([r["sd"] for r in rs]), np.array([r["ok"] for r in rs])


if __name__ == "__main__":
    X, p = scene.make_set(99, 300)
    e, sd, ok = estimate_many(X)
    err = e - p["e"]
    print("ok frac", ok.mean(), "MAE", np.abs(err).mean(), "MAE(ok)", np.abs(err[ok]).mean(),
          "median", np.median(np.abs(err)), "cov90", np.mean(np.abs(err) < 1.645 * sd))
    for lo, hi in [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95)]:
        m = (p["e"] >= lo) & (p["e"] < hi)
        print(lo, hi, "MAE", np.abs(err[m]).mean(), "fail", 1 - ok[m].mean())
