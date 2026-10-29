"""Structural model variant ("struct") for Paper Plane Darwin (round-3 follow-up).

The original model (`sim.py`) makes span free: a 50 cm wing weighs the same per area as a 10 cm one
and never breaks, so every evolved design sat at the 50 cm bound.  This variant keeps `sim.py`
exactly as it was (old results stay reproducible) and adds a **wing root bending-strength limit**:

  * The wing is the same flat 3 mm foam sheet the original mass model already implies
    (sim.AREAL = 0.12 kg/m^2 = 3 mm x 40 kg/m^3, Depron/XPS-like).  No spar.
  * Root bending capacity of a flat sheet of chord c, thickness t:  M_cap = SIGMA * c * t^2 / 6.
  * Root bending moment in flight from the actual wing lift L_w (|L|, elliptic spanwise loading:
    half-lift L/2 acts at 4(b/2)/(3 pi) from the root), with inertial relief from the wing's own
    mass (only the non-wing mass hangs off the root):  M = |L_w| * b / (3 pi) * (1 - m_wing / m).
  * If M > M_cap at any step the wing folds: the flight ends there (status -2) and the range is the
    downrange x at the break.  (Conservative: the wreck gets no further glide credit.)

Physics consequence: allowed lift  L_max = 3 pi SIGMA c t^2 / (6 b (1 - m_w/m)), i.e. the safe
load factor falls as span/chord and total weight rise.  Long, thin, heavily ballasted wings break on
fast throws and gusts; short wide wings don't.  Span is no longer free.

Constants (stated before any struct search was run, not fitted):  T_SHEET = 3 mm.  Allowable stress
SIGMA = 0.25 MPa = 0.5 MPa (order of extruded-polystyrene foam-sheet flexural strength) / safety
factor 2.  This is the HEADLINE level.  0.5 MPa (no safety factor) and 0.125 MPa (weak/damp sheet)
are run as sensitivity levels.  All three are assumptions, not measurements.  A probe before the run
showed 0.5 MPa never binds on the six training launches (old CMA best peaks at 0.59 of capacity).
Note that pure mass penalties would barely matter here: from a fixed height, glide range is set by
L/D, not weight, so a strength *limit* is the structural cost that actually bites.
"""
import math

import numba as nb
import numpy as np

import sim

T_SHEET = 0.003       # wing sheet thickness, m (consistent with sim.AREAL at 40 kg/m^3)
SIGMA = 0.25e6        # headline allowable bending stress (0.5 MPa strength / safety factor 2), Pa
N_STATS = sim.N_STATS + 1   # + max root-moment / capacity ratio over the flight
STAT_NAMES = sim.STAT_NAMES + ["max_load_ratio"]


@nb.njit(cache=True)
def capacity(P, sigma):
    """(M_cap [N m], moment arm factor k so that M = k * |L_w|)."""
    chord, span, m = P[6], P[17], P[0]
    m_w = sim.AREAL * P[4]
    m_cap = sigma * chord * T_SHEET * T_SHEET / 6.0
    k = span / (3.0 * math.pi) * (1.0 - m_w / m)
    return m_cap, k


def describe(genes, sigma=SIGMA):
    d = sim.describe(genes)
    P = sim.decode(np.asarray(genes, dtype=np.float64))
    m_cap, k = capacity(P, sigma)
    d.update(root_capacity_Nm=float(m_cap), safe_load_factor=float(m_cap / k / (P[0] * sim.G)))
    return d


@nb.njit(cache=True, inline="always")
def _wing_lift(st, P, U, gust, clam, cd0m):
    wx, wz = sim._wind(st[0], st[1], U, gust)
    fx, fz, tau, cl, V = sim._surface(st[2], st[3], st[4], st[5], P[2], wx, wz, P[12], P[4], P[6],
                                      clam * P[10], P[8], cd0m, 0.175, 0.0)
    return 0.5 * sim.RHO * V * V * P[4] * abs(cl)


@nb.njit(cache=True)
def simulate(P, cond, gust, dt, tmax, method, traj, rec_every, sigma):
    """Same as sim.simulate plus the wing-strength check after every step (see module doc)."""
    h0, v0, g0, a0, U, clam, cd0m, q0 = (cond[0], cond[1], cond[2], cond[3], cond[4],
                                         cond[5], cond[6], cond[7])
    m_cap, kmom = capacity(P, sigma)
    st = np.empty(6)
    st[0] = 0.0
    st[1] = h0
    st[2] = v0 * math.cos(g0)
    st[3] = v0 * math.sin(g0)
    st[4] = g0 + a0
    st[5] = q0
    E0 = sim.energy(st, P)
    Eprev = E0
    max_dE = 0.0
    n_gain = 0
    zmax = h0
    t = 0.0
    nsteps = int(tmax / dt + 0.5)
    status = 0.0
    q2sum = 0.0
    q2n = 0
    vsum = 0.0
    vmin = 1e9
    xr = 0.0
    nrec = 0
    tol = 1e-9 * E0
    max_load = kmom * _wing_lift(st, P, U, gust, clam, cd0m) / m_cap
    i = 0
    for i in range(nsteps):
        if traj.shape[0] > 0 and i % rec_every == 0 and nrec < traj.shape[0]:
            traj[nrec, 0] = t
            traj[nrec, 1] = st[0]
            traj[nrec, 2] = st[1]
            traj[nrec, 3] = st[4]
            traj[nrec, 4] = math.sqrt(st[2] ** 2 + st[3] ** 2)
            traj[nrec, 5] = Eprev
            traj[nrec, 6] = st[5]
            traj[nrec, 7] = sim._wrap(st[4] + P[12] - math.atan2(st[3], st[2]))
            nrec += 1
        new = sim.step(st, dt, P, U, gust, clam, cd0m, method)
        t += dt
        ok = True
        for j in range(6):
            if not math.isfinite(new[j]):
                ok = False
        V = math.sqrt(new[2] ** 2 + new[3] ** 2) if ok else 1e9
        if (not ok) or V > 200.0 or abs(new[5]) > 1e4:
            status = -1.0
            xr = st[0]
            break
        E = sim.energy(new, P)
        dE = E - Eprev
        if dE > tol:
            n_gain += 1
        if dE / E0 > max_dE:
            max_dE = dE / E0
        Eprev = E
        if new[1] > zmax:
            zmax = new[1]
        if t > 2.0:
            q2sum += new[5] ** 2
            q2n += 1
        vsum += V
        if V < vmin:
            vmin = V
        if new[1] <= 0.0:
            f = st[1] / (st[1] - new[1])
            xr = st[0] + f * (new[0] - st[0])
            t = t - dt + f * dt
            status = 1.0
            st = new
            break
        load = kmom * _wing_lift(new, P, U, gust, clam, cd0m) / m_cap
        if load > max_load:
            max_load = load
        if load > 1.0:            # wing folds: flight over at this point
            status = -2.0
            st = new
            xr = st[0]
            break
        st = new
        xr = st[0]
    if traj.shape[0] > 0 and nrec < traj.shape[0]:
        traj[nrec, 0] = t
        traj[nrec, 1] = xr
        traj[nrec, 2] = max(st[1], 0.0)
        traj[nrec, 3] = st[4]
        traj[nrec, 4] = math.sqrt(st[2] ** 2 + st[3] ** 2)
        traj[nrec, 5] = Eprev
        traj[nrec, 6] = st[5]
        traj[nrec, 7] = 0.0
        nrec += 1
    out = np.empty(N_STATS)
    out[0] = xr
    out[1] = t
    out[2] = status
    out[3] = max_dE
    out[4] = (Eprev - E0) / E0
    out[5] = zmax - h0
    out[6] = math.sqrt(q2sum / q2n) if q2n > 0 else 0.0
    out[7] = vsum / max(i + 1, 1)
    out[8] = vmin
    out[9] = n_gain
    out[10] = max_load
    return out


def evaluate_batch(genes, conds, gusts, dt, tmax, method, sigma=SIGMA):
    """Drop-in for sim.evaluate_batch under the struct model; stats (N, K, N_STATS)."""
    return _evaluate_batch(np.ascontiguousarray(genes, dtype=np.float64), conds, gusts, dt, tmax,
                           method, float(sigma))


@nb.njit(cache=True, parallel=True)
def _evaluate_batch(genes, conds, gusts, dt, tmax, method, sigma):
    N = genes.shape[0]
    K = conds.shape[0]
    out = np.empty((N, K, N_STATS))
    empty = np.empty((0, 8))
    for idx in nb.prange(N * K):
        n = idx // K
        k = idx % K
        P = sim.decode(genes[n])
        out[n, k, :] = simulate(P, conds[k], gusts[k], dt, tmax, method, empty, 1, sigma)
    return out


def fly(genes, cond, gust, dt=0.005, tmax=60.0, method=0, rec_every=4, sigma=SIGMA):
    P = sim.decode(np.asarray(genes, dtype=np.float64))
    n = int(tmax / dt / rec_every) + 3
    traj = np.zeros((n, 8))
    stats = simulate(P, np.asarray(cond, dtype=np.float64), np.asarray(gust, dtype=np.float64),
                     dt, tmax, method, traj, rec_every, float(sigma))
    last = int(np.argmax(traj[1:, 0] == 0.0)) + 1 if np.any(traj[1:, 0] == 0.0) else n
    return stats, traj[:last]
