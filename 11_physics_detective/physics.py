"""The hidden world: a block on a table, pushed by a robot finger, watched by a noisy camera.

Everything here is analytic and vectorised with numpy broadcasting, so the same function
simulates one experiment or a (designs x hypotheses x frames) tensor of them.

Physics (Coulomb friction, rigid block, flat table):
  * A push applies a known commanded force F (N) for T seconds, then lets go.
  * The actuator delivers k*F, where k is the pusher gain (k = 1 unless the pusher is
    "uncalibrated", see the non-identifiable world).
  * If k*F <= mu_s*m*g the block never breaks free and stays at x = 0.
  * Otherwise it accelerates at a1 = k*F/m - mu_k*g for T seconds, then decelerates at
    mu_k*g until it stops.
Camera:
  * Frames at t = 0.1, 0.2, ..., 2.0 s (20 frames).
  * Each frame reports y = x + N(0, SIGMA^2).
  * The camera only sees x in [.., VIEW] (VIEW = 1.0 m). If the noisy reading falls past the
    edge of view, the frame is reported as "out of view" (censored), not as a number.
    This is what makes experiment choice matter: push too softly and nothing moves, push too
    hard and the block leaves the picture before it shows you how it stops.
"""
import numpy as np
from scipy.special import log_ndtr

G = 9.81
SIGMA = 0.08          # camera noise, metres (a cheap webcam tracker)
VIEW = 1.0            # camera field of view edge, metres
TIMES = np.round(np.arange(1, 21) * 0.1, 10)   # 20 frames, 0.1 .. 2.0 s
NT = len(TIMES)
MU_RATIO = 1.2        # first world: mu_s = 1.2 * mu_k (known); only mu_k is hidden
MU_LO, MU_HI = 0.05, 0.8   # prior range of mu_k (uniform)

# The experiment menu: commanded force F (N) and push duration T (s).
F_GRID = np.linspace(1.0, 15.0, 15)
T_GRID = np.round(np.linspace(0.1, 1.0, 10), 3)
DESIGNS = np.array([(f, t) for f in F_GRID for t in T_GRID])   # (150, 2)


def positions(F, T, mu_k, mu_s, m=1.0, k=1.0, times=TIMES):
    """Noise-free block position at each frame. All args broadcast; frames on the last axis."""
    F, T, mu_k, mu_s, m, k = [np.asarray(a, dtype=np.float64)[..., None] for a in (F, T, mu_k, mu_s, m, k)]
    t = np.asarray(times, dtype=np.float64)
    moves = k * F > mu_s * m * G
    a1 = np.where(moves, k * F / m - mu_k * G, 0.0)
    a2 = mu_k * G
    v1 = a1 * T
    x1 = 0.5 * a1 * T ** 2
    t_stop = v1 / a2
    tau = np.clip(t - T, 0.0, t_stop)
    x_push = 0.5 * a1 * np.minimum(t, T) ** 2
    x_after = x1 + v1 * tau - 0.5 * a2 * tau ** 2
    return np.where(t <= T, x_push, x_after)


def observe(x, rng):
    """Turn true positions into camera readings. Returns (y, censored) with y=nan when censored."""
    y = x + SIGMA * rng.standard_normal(np.shape(x))
    cens = y > VIEW
    return np.where(cens, np.nan, y), cens


def loglik_frames(y, cens, x):
    """Log-likelihood of camera readings (y, cens) given predicted positions x, summed over frames.
    y, cens broadcast against x; frame axis last."""
    ll_obs = -0.5 * ((np.nan_to_num(y, nan=0.0) - x) / SIGMA) ** 2 - np.log(SIGMA * np.sqrt(2 * np.pi))
    ll_cen = log_ndtr((x - VIEW) / SIGMA)
    return np.where(cens, ll_cen, ll_obs).sum(-1)


def world1_positions(F, T, mu_k):
    """First world: one hidden number mu_k, with mu_s = MU_RATIO * mu_k, m = k = 1."""
    return positions(F, T, mu_k, MU_RATIO * np.asarray(mu_k))


def rest_distance(F, T, mu_k, mu_s, m=1.0, k=1.0):
    """Where the block finally stops (no camera limits)."""
    F, T, mu_k, mu_s, m, k = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in (F, T, mu_k, mu_s, m, k)])
    moves = k * F > mu_s * m * G
    a1 = np.where(moves, k * F / m - mu_k * G, 0.0)
    v1 = a1 * T
    return 0.5 * a1 * T ** 2 + v1 ** 2 / (2 * mu_k * G)
