"""Shared metrics for any tracker that outputs per-frame, per-slot Gaussians (pixels)."""
import itertools

import numpy as np

import sim

BUCKETS = [(1, 2), (3, 4), (5, 7), (8, 11), (12, 15), (16, 40)]
CHI1, CHI2 = 1.0, 4.0           # 1-sigma / 2-sigma ellipse in 2D: P = 0.393 / 0.865
EXP1, EXP2 = 1 - np.exp(-0.5), 1 - np.exp(-2.0)


def per_frame(mean, cov, gt):
    d = gt - mean
    cov = cov + np.eye(2) * 1e-6
    inv = np.linalg.inv(cov)
    d2 = np.einsum("...i,...ij,...j->...", d, inv, d)
    nll = np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(cov)) + 0.5 * d2
    err = np.sqrt((d ** 2).sum(-1))
    return err, d2, nll


def summarize(sel, err, d2, nll):
    if sel.sum() == 0:
        return dict(n=0)
    return dict(n=int(sel.sum()), err=float(err[sel].mean()), err_med=float(np.median(err[sel])),
                nll=float(np.mean(np.clip(nll[sel], -50, 1e4))), nll_med=float(np.median(nll[sel])),
                cov1=float((d2[sel] <= CHI1).mean()), cov2=float((d2[sel] <= CHI2).mean()))


def id_switches(mean, gt, present, margin=2.0):
    """A frame is identity-confused if some permutation of the predicted slots matches the true
    balls better than the identity by more than `margin` px (summed). Switch events count
    transitions into a confused state.  Returns (frac_confused_frames, switches_per_seq,
    frac_seqs_with_switch, frac_seqs_swapped_at_end)."""
    N, Tn, K = gt.shape[:3]
    conf = np.zeros((N, Tn), bool)
    for n in range(N):
        ks = np.where(present[n])[0]
        perms = [p for p in itertools.permutations(range(len(ks))) if p != tuple(range(len(ks)))]
        m, g = mean[n][:, ks], gt[n][:, ks]
        base = np.sqrt(((m - g) ** 2).sum(-1)).sum(-1)
        best = np.full(Tn, np.inf)
        for p in perms:
            best = np.minimum(best, np.sqrt(((m[:, list(p)] - g) ** 2).sum(-1)).sum(-1))
        conf[n] = best < base - margin
    sw = (conf[:, 1:] & ~conf[:, :-1]).sum(1) + conf[:, 0]
    return dict(confused_frames=float(conf.mean()), switches_per_seq=float(sw.mean()),
                seqs_with_switch=float((sw > 0).mean()), swapped_at_end=float(conf[:, -1].mean()))


def evaluate(mean, cov, d, idx):
    """mean (N,T,K,2), cov (N,T,K,2,2) for sequences idx of split d."""
    gt = d["pos"][idx].astype(np.float64)
    vis = d["vis"][idx]
    present = d["present"][idx]
    pres = np.broadcast_to(present[:, None, :], vis.shape)
    err, d2, nll = per_frame(mean, cov, gt)
    out = {}
    out["all"] = summarize(pres, err, d2, nll)
    out["visible"] = summarize(pres & (vis >= 0.999), err, d2, nll)
    out["partial"] = summarize(pres & (vis > 0) & (vis < 0.999), err, d2, nll)
    out["hidden"] = summarize(pres & (vis == 0), err, d2, nll)
    # occlusion runs
    rows = []
    for n in range(len(idx)):
        for (k, s, L, end) in sim.hidden_runs((vis[n] == 0) & present[n][None]):
            if end >= vis.shape[1] or s == 0:
                continue
            rows.append((n, k, s, L, end))
    by = {}
    for lo, hi in BUCKETS:
        rs = [r for r in rows if lo <= r[3] <= hi]
        if not rs:
            continue
        hid_sel = np.zeros_like(pres)
        last_sel = np.zeros_like(pres)
        re_sel = np.zeros_like(pres)
        for (n, k, s, L, end) in rs:
            hid_sel[n, s:end, k] = True
            last_sel[n, end - 1, k] = True
            re_sel[n, end, k] = True
        by[f"{lo}-{hi}"] = dict(runs=len(rs), hidden=summarize(hid_sel, err, d2, nll),
                                last_hidden=summarize(last_sel, err, d2, nll),
                                reappear=summarize(re_sel, err, d2, nll))
    out["by_occlusion_len"] = by
    allr = np.zeros_like(pres)
    lastr = np.zeros_like(pres)
    for (n, k, s, L, end) in rows:
        allr[n, end, k] = True
        lastr[n, end - 1, k] = True
    out["reappear"] = summarize(allr, err, d2, nll)
    out["last_hidden"] = summarize(lastr, err, d2, nll)
    out["n_runs"] = len(rows)
    if "turned" in d and d["turned"][idx].any():
        # hard variant: split occlusions by whether the ball secretly changed velocity behind the wall
        turned = d["turned"][idx]
        for tag, want in [("turned", True), ("straight", False)]:
            hs, ls, rs_ = (np.zeros_like(pres) for _ in range(3))
            nr = 0
            for (n, k, s, L, end) in rows:
                if bool(turned[n, s - 1:end, k].any()) != want:
                    continue
                nr += 1
                hs[n, s:end, k] = True
                ls[n, end - 1, k] = True
                rs_[n, end, k] = True
            out[f"runs_{tag}"] = dict(runs=nr, hidden=summarize(hs, err, d2, nll),
                                      last_hidden=summarize(ls, err, d2, nll), reappear=summarize(rs_, err, d2, nll))
    out["identity"] = id_switches(mean, gt, present)
    return out
