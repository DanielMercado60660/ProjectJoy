"""Pure-numpy metrics over drum grids. hits arrays are [B, 32, 9] of 0/1."""
import numpy as np

KICK, SNARE, CLAP = 0, 1, 2
BEAT = np.arange(32) % 4 == 0  # quarter-note positions


def flat(h):
    return h.reshape(len(h), -1).astype(np.float32)


def hamming_matrix(a, b):
    """Pairwise Hamming distances between two sets of hit grids."""
    A, Bm = flat(a), flat(b)
    return A.sum(1)[:, None] + Bm.sum(1)[None, :] - 2 * A @ Bm.T


def nn_hamming(gen, train, batch=512):
    """Distance from each generated grid to its nearest training grid."""
    out = []
    for i in range(0, len(gen), batch):
        out.append(hamming_matrix(gen[i:i + batch], train).min(1))
    return np.concatenate(out)


def novelty(gen, train):
    d = nn_hamming(gen, train)
    return {"nn_hamming_mean": float(d.mean()), "nn_hamming_median": float(np.median(d)),
            "exact_copy_pct": float(100 * (d == 0).mean()), "near_copy_le2_pct": float(100 * (d <= 2).mean())}


def diversity(gen, style):
    """Mean pairwise Hamming between generated grids of the same style, and % unique grids."""
    vals = []
    for s in np.unique(style):
        g = gen[style == s]
        if len(g) < 2:
            continue
        D = hamming_matrix(g, g)
        vals.append(D[np.triu_indices(len(g), 1)].mean())
    uniq = len(np.unique(flat(gen), axis=0)) / len(gen)
    return {"pairwise_hamming_within_style": float(np.mean(vals)), "unique_pct": float(100 * uniq)}


def pulse(h):
    """Timing-consistency stats for a set of grids (all from one style)."""
    h = h.astype(np.float32)
    dens = h.sum(2)  # hits per step [B, 32]
    on, offb = dens[:, BEAT].mean(), dens[:, ~BEAT].mean()
    kick = h[:, :, KICK]
    return {
        "beat_offbeat_density_ratio": float(on / max(offb, 1e-6)),
        "kick_on_beat_frac": float(kick[:, BEAT].sum() / max(kick.sum(), 1)),
        "kick_on_downbeat_pct": float(100 * kick[:, 0].mean()),
        "bar_repeat_hamming": float(np.abs(h[:, :16] - h[:, 16:]).sum((1, 2)).mean()),
        "hits_mean": float(h.sum((1, 2)).mean()),
        "empty_or_sparse_pct": float(100 * (h.sum((1, 2)) < 4).mean()),
    }


def position_profile(h):
    """Average onsets per 16th position (folded over the two bars) per instrument: [16, 9]."""
    h = h.astype(np.float32)
    return 0.5 * (h[:, :16] + h[:, 16:]).mean(0)


def profile_l1(gen, real):
    """Mean |difference| of per-cell hit frequency (32x9) between two sets of one style."""
    return float(np.abs(gen.astype(np.float32).mean(0) - real.astype(np.float32).mean(0)).mean())


def swing(h, off):
    """Mean microtiming offset on odd 16ths minus even 16ths (in 16th-note units)."""
    odd = np.arange(32) % 2 == 1
    m = h > 0
    o_odd = off[:, odd][m[:, odd]]
    o_even = off[:, ~odd][m[:, ~odd]]
    if len(o_odd) == 0 or len(o_even) == 0:
        return float("nan")
    return float(o_odd.mean() - o_even.mean())
