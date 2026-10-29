"""Mixture densities, calibration and causal one-look policies; NumPy/SciPy only."""
import numpy as np
from scipy.special import logsumexp, ndtr, ndtri
import metrics as old_metrics
from r5_data import LOOK_FIRST, LOOK_LAST


def validate(dist):
    w, m, c = dist
    assert m.shape == w.shape + (2,) and c.shape == w.shape + (2, 2)
    assert np.isfinite(w).all() and np.isfinite(m).all() and np.isfinite(c).all()
    assert (w >= 0).all() and np.allclose(w.sum(-1), 1, atol=1e-5)
    assert (np.linalg.eigvalsh(c) > 0).all()


def moments(dist):
    w, m, c = dist
    mean = (w[..., None] * m).sum(-2)
    delta = m - mean[..., None, :]
    cov = (w[..., None, None] * (c + delta[..., :, None] * delta[..., None, :])).sum(-3)
    return mean, cov


def component_logpdf(dist, target):
    _, m, c = dist
    d = target[..., None, :] - m
    a, b, e = c[..., 0, 0], c[..., 0, 1], c[..., 1, 1]
    det = np.maximum(a * e - b * b, 1e-20)
    q = (e * d[..., 0] ** 2 - 2*b*d[..., 0]*d[..., 1] + a*d[..., 1] ** 2) / det
    return -np.log(2*np.pi) - .5*np.log(det) - .5*q


def nll(dist, target):
    return -logsumexp(np.log(np.maximum(dist[0], 1e-300)) + component_logpdf(dist, target), axis=-1)


def calibration_d2(dist, target):
    """Rosenblatt x-then-y transform: calibrated mixtures give chi-square(2).

    Unlike a moment ellipse, this is valid for separated modes. Coverage at 1/4 has
    nominal mass .393/.865. It is an equivalent calibration diagnostic, not a HPD area.
    """
    w, m, c = dist
    dx = target[..., None, 0] - m[..., 0]
    vx = c[..., 0, 0]
    ux = (w * ndtr(dx / np.sqrt(vx))).sum(-1)
    log_wx = np.log(np.maximum(w, 1e-300)) - .5*(np.log(2*np.pi*vx) + dx*dx/vx)
    wx = np.exp(log_wx - logsumexp(log_wx, axis=-1, keepdims=True))
    cy = m[..., 1] + c[..., 1, 0] / vx * dx
    vy = np.maximum(c[..., 1, 1] - c[..., 1, 0]**2 / vx, 1e-12)
    uy = (wx * ndtr((target[..., None, 1] - cy) / np.sqrt(vy))).sum(-1)
    return ndtri(np.clip(ux, 1e-12, 1-1e-12))**2 + ndtri(np.clip(uy, 1e-12, 1-1e-12))**2


def wall_scores(dist, present, occ):
    """Spread weighted by current posterior wall mass. No target or future data."""
    w, m, c = dist
    _, cov = moments(dist)
    spread = np.trace(cov, axis1=-2, axis2=-1)
    scores = []
    for a, b in occ:
        if a < 0:
            scores.append(np.full(spread.shape[:-1], -np.inf))
            continue
        sx = np.sqrt(c[..., 0, 0])
        mass = (w * (ndtr((b-m[..., 0])/sx) - ndtr((a-m[..., 0])/sx))).sum(-1)
        scores.append((spread * mass * present).sum(-1))
    return np.stack(scores, -1)


def uncertainty_look(dist, present, occ, threshold):
    # Decide the view at t from the posterior through t-1. Force a look at the
    # deadline: identical cost/coverage to random, with no hindsight or abstention.
    scores = wall_scores(dist, present, occ)
    last = min(LOOK_LAST, len(scores)-2)
    for t in range(LOOK_FIRST, last+1):
        wall = int(np.argmax(scores[t-1]))
        if scores[t-1, wall] >= threshold or t == last:
            return t, wall
    raise ValueError('sequence too short for peek window')


def summarize(mask, err, loss, d2):
    if not mask.any():
        return {'n': 0, 'error_px': None, 'nll': None, 'coverage1': None, 'coverage2': None}
    return dict(n=int(mask.sum()), error_px=float(err[mask].mean()),
                median_error_px=float(np.median(err[mask])), nll=float(loss[mask].mean()),
                coverage1=float((d2[mask] <= 1).mean()), coverage2=float((d2[mask] <= 4).mean()))


def score_sequence(dist, gt, vis, present):
    validate(dist)
    mean, cov = moments(dist)
    err = np.linalg.norm(mean - gt, axis=-1)
    loss = nll(dist, gt)
    d2 = calibration_d2(dist, gt)
    mask = np.broadcast_to(present, vis.shape)
    re = np.zeros_like(mask)
    re[1:] = (vis[:-1] == 0) & (vis[1:] > 0) & present
    out = {name: summarize(sel, err, loss, d2) for name, sel in
           [('all', mask), ('visible', mask & (vis >= .999)),
            ('hidden', mask & (vis == 0)), ('reappear', re)]}
    out['identity'] = old_metrics.id_switches(mean[None], gt[None], present[None])
    out['initial_min_xgap_px'] = float(np.diff(np.sort(gt[0, present, 0])).min())
    out['spread_px2'] = float(np.trace(cov, axis1=-2, axis2=-1)[mask].mean())
    return out


def peek_value(before, after, gt, present, look):
    """Matched natural-view counterfactual at t, then suffix t+1:T, same sequence."""
    t, wall = look
    bm, bc = moments(before)
    am, ac = moments(after)
    suffix = np.broadcast_to(present, gt[t+1:].shape[:-1])
    eb = np.linalg.norm(bm[t+1:] - gt[t+1:], axis=-1)
    ea = np.linalg.norm(am[t+1:] - gt[t+1:], axis=-1)
    lb, la = nll(before, gt)[t+1:], nll(after, gt)[t+1:]
    b_id = old_metrics.id_switches(bm[None, t+1:], gt[None, t+1:], present[None])
    a_id = old_metrics.id_switches(am[None, t+1:], gt[None, t+1:], present[None])
    bspread = float(np.trace(bc[t], axis1=-2, axis2=-1)[present].mean())
    aspread = float(np.trace(ac[t], axis1=-2, axis2=-1)[present].mean())
    return dict(time=t, wall=wall, cost_views=1, request_coverage=1.,
                spread_before_px2=bspread, spread_after_px2=aspread,
                spread_reduction_px2=bspread-aspread,
                later_error_gain_px=float((eb-ea)[suffix].mean()),
                later_nll_gain=float((lb-la)[suffix].mean()),
                later_confusion_gain=b_id['confused_frames']-a_id['confused_frames'],
                final_swap_gain=b_id['swapped_at_end']-a_id['swapped_at_end'])


def aggregate(rows):
    out = {}
    for group in ('all', 'visible', 'hidden', 'reappear'):
        total = sum(r[group]['n'] for r in rows)
        out[group] = {'n': total}
        for key in ('error_px', 'nll', 'coverage1', 'coverage2'):
            out[group][key] = (sum(r[group][key]*r[group]['n'] for r in rows if r[group]['n'])/total
                               if total else None)
    out['identity'] = {k: float(np.mean([r['identity'][k] for r in rows])) for k in rows[0]['identity']}
    if 'diagnostics' in rows[0]:
        out['diagnostics'] = {k: float(np.mean([r['diagnostics'][k] for r in rows]))
                              for k in rows[0]['diagnostics']}
    out['visible_by_initial_xgap'] = {}
    for label, lo, hi in [('<1', 0, 1), ('1-3', 1, 3), ('3-8', 3, 8), ('>=8', 8, np.inf)]:
        group = [r for r in rows if lo <= r['initial_min_xgap_px'] < hi]
        count = sum(r['visible']['n'] for r in group)
        out['visible_by_initial_xgap'][label] = dict(sequences=len(group), n=count,
            error_px=(sum(r['visible']['n']*r['visible']['error_px'] for r in group)/count if count else None))
    if 'peek' in rows[0]:
        out['peek'] = {k: float(np.mean([r['peek'][k] for r in rows])) for k in rows[0]['peek']
                       if k not in ('time', 'wall')}
    return out


def paired_ci(values, seed=501, draws=1000):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return None
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(0, len(x), (draws, len(x)))].mean(-1)
    return dict(mean=float(x.mean()), low=float(np.quantile(samples, .025)),
                high=float(np.quantile(samples, .975)), n=len(x))
