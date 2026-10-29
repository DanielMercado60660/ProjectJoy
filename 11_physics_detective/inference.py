"""Inference for the one-unknown world: exact grid posterior, least-squares baseline,
and expected-information-gain (EIG) experiment selection."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import norm

from physics import (DESIGNS, MU_HI, MU_LO, NT, SIGMA, VIEW, loglik_frames, observe,
                     world1_positions)

GRID = np.linspace(MU_LO, MU_HI, 1500)


class GridPosterior:
    """Exact Bayesian posterior over mu_k on a fine grid (uniform prior on [MU_LO, MU_HI])."""

    def __init__(self, grid=GRID):
        self.grid = grid
        self.logp = np.zeros_like(grid)

    def update(self, F, T, y, cens):
        self.logp = self.logp + loglik_frames(y, cens, world1_positions(F, T, self.grid))
        self.logp -= self.logp.max()

    @property
    def w(self):
        w = np.exp(self.logp)
        return w / w.sum()

    def mean(self):
        return float((self.w * self.grid).sum())

    def sd(self):
        w = self.w
        m = (w * self.grid).sum()
        return float(np.sqrt((w * (self.grid - m) ** 2).sum()))

    def quantile(self, q):
        c = np.cumsum(self.w)
        c = c / c[-1]
        return np.interp(q, c, self.grid)

    def interval(self, level):
        a = (1 - level) / 2
        return float(self.quantile(a)), float(self.quantile(1 - a))

    def particles(self, K=64):
        """K equal-weight points at posterior quantiles: a compact stand-in for EIG."""
        return self.quantile((np.arange(K) + 0.5) / K)

    def predict(self, F, T):
        """Posterior-predictive mean trajectory for designs (F, T) arrays of shape (n,)."""
        X = world1_positions(np.asarray(F)[:, None], np.asarray(T)[:, None], self.grid[None, :])  # (n,G,NT)
        return np.einsum("g,ngt->nt", self.w, X)


def eig(particles, designs=DESIGNS, M=2, rng=None, sim=world1_positions):
    """Expected information gain about the hidden parameter for each candidate design.

    Nested Monte Carlo with the posterior particles used both as the "true world" samples and
    as the mixture for the marginal likelihood:
        EIG(d) ~= mean_{i,m} [ log p(y_im | th_i, d) - log mean_j p(y_im | th_j, d) ]
    with y_im simulated from th_i. Common random numbers across designs keep the ranking stable.
    `particles` is a 1-D array (or a tuple of arrays for multi-parameter sims, passed to sim).
    """
    rng = rng or np.random.default_rng(0)
    if not isinstance(particles, tuple):
        particles = (particles,)
    K = len(particles[0])
    eps = rng.standard_normal((K, M, NT))
    out = np.empty(len(designs))
    for s in range(0, len(designs), 25):
        d = designs[s:s + 25]
        X = sim(d[:, 0, None], d[:, 1, None], *[p[None, :] for p in particles])     # (D,K,NT)
        Y = X[:, :, None, :] + SIGMA * eps[None]                                     # (D,K,M,NT)
        C = Y > VIEW
        L = loglik_frames(Y[:, :, :, None, :], C[:, :, :, None, :], X[:, None, None, :, :])  # (D,K,M,K)
        own = np.einsum("dimi->dim", L)                                             # L[d,i,m,i]
        out[s:s + 25] = (own - logsumexp(L, axis=3) + np.log(K)).mean(axis=(1, 2))
    return out


# ---------------------------------------------------------------- conventional baseline
def lsq_fit(data, grid=np.linspace(MU_LO, MU_HI, 300)):
    """Conventional least-squares fitter: minimise squared error on the frames the camera saw
    (out-of-view frames are dropped, as a practitioner typically would), then a Laplace /
    Gauss-Newton standard error from the residual variance and a numerical Jacobian.
    data: list of (F, T, y, cens). Returns (mu_hat, se)."""
    if not data:
        return 0.5 * (MU_LO + MU_HI), np.inf

    def resid(mu):
        r = [np.where(c, 0.0, np.nan_to_num(y) - world1_positions(F, T, mu)) for F, T, y, c in data]
        return np.concatenate(r)

    sse = np.array([np.sum(resid(m) ** 2) for m in grid])
    i = int(np.argmin(sse))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda m: np.sum(resid(m) ** 2), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-6})
    mu = float(res.x)
    n = sum(int((~c).sum()) for _, _, _, c in data)
    s2 = res.fun / max(n - 1, 1)
    h = 1e-4
    J = (resid(min(mu + h, MU_HI)) - resid(max(mu - h, MU_LO))) / (min(mu + h, MU_HI) - max(mu - h, MU_LO))
    jtj = float(J @ J)
    se = np.sqrt(s2 / jtj) if jtj > 1e-12 and s2 > 0 else np.inf
    return mu, se


def lsq_interval(mu, se, level):
    z = norm.ppf(0.5 + level / 2)
    if not np.isfinite(se):
        return MU_LO, MU_HI
    return mu - z * se, mu + z * se


# ---------------------------------------------------------------- design strategies
FIXED_DESIGN = (8.0, 0.3)   # "textbook protocol": a medium push, repeated


def choose_design(strategy, post, rng, K=64, M=2):
    if strategy == "active":
        scores = eig(post.particles(K), DESIGNS, M=M, rng=rng)
        return DESIGNS[int(np.argmax(scores))], scores
    if strategy == "random":
        return DESIGNS[rng.integers(len(DESIGNS))], None
    if strategy == "fixed":
        return np.array(FIXED_DESIGN), None
    raise ValueError(strategy)


def run_experiment(mu_true, F, T, rng):
    x = world1_positions(F, T, mu_true)
    y, c = observe(x, rng)
    return y, c
