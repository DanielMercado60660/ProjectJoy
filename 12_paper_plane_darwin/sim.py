"""Paper Plane Darwin: a deliberately simple 2D longitudinal glider simulator (numba).

Rigid body in the x-z plane (x forward/downrange, z up) with three degrees of freedom:
position (x, z), pitch angle theta, and their rates.  Two lifting surfaces (wing, tail) each
get lift perpendicular to, and drag opposite to, the air velocity *at that surface* (so pitch
rate gives the tail its damping), plus a little fuselage drag at the centre of gravity.

Why this is energy-safe in still air: every aerodynamic force is either perpendicular to the
local air velocity (lift, zero work) or opposite to it (drag, negative work), so the exact
continuous dynamics can never gain mechanical energy.  Any energy gain we see is integration
error, which is exactly what the sanity checks look for.  There is deliberately no airfoil
Cm0 couple, because a pure couple can do positive work on a rotating body.

Design genome (8 genes in [0, 1]) -> physical glider via `decode`.
"""
import math

import numba as nb
import numpy as np

RHO = 1.225          # air density, kg/m^3
G = 9.81             # gravity, m/s^2
NU = 1.5e-5          # kinematic viscosity of air, m^2/s
AREAL = 0.12         # wing/tail sheet areal density, kg/m^2 (thin balsa / foam card)
FUSE_LIN = 0.015     # fuselage stick linear density, kg/m
TAIL_AR = 4.0        # tailplane aspect ratio (fixed)
OSWALD = 0.8

GENE_NAMES = ["span_m", "chord_m", "tail_area_ratio", "tail_arm_m",
              "ballast_kg", "ballast_pos_m", "wing_incidence_deg", "tail_incidence_deg"]
LO = np.array([0.10, 0.03, 0.05, 0.05, 0.000, 0.00, -4.0, -8.0])
HI = np.array([0.50, 0.15, 0.60, 0.35, 0.020, 0.20, 8.0, 4.0])
N_GENES = 8
N_PHYS = 18
N_COND = 8
N_GUST = 4
N_STATS = 10
STAT_NAMES = ["range_m", "flight_time_s", "status", "max_step_dE_rel", "net_dE_rel",
              "peak_climb_m", "pitch_rate_rms", "mean_airspeed", "min_airspeed",
              "n_energy_gain_steps"]


def genes_to_phys_units(g):
    """Gene vector in [0,1] -> human units (m, kg, deg)."""
    g = np.clip(np.asarray(g, dtype=np.float64), 0.0, 1.0)
    return LO + (HI - LO) * g


def phys_units_to_genes(u):
    return (np.asarray(u, dtype=np.float64) - LO) / (HI - LO)


@nb.njit(cache=True)
def decode(g):
    """Genes -> physical parameter array P (mass, inertia, surface positions, ...)."""
    gg = np.minimum(np.maximum(g, 0.0), 1.0)
    span = 0.10 + 0.40 * gg[0]
    chord = 0.03 + 0.12 * gg[1]
    str_ = 0.05 + 0.55 * gg[2]
    lt = 0.05 + 0.30 * gg[3]
    mb = 0.020 * gg[4]
    xb = 0.20 * gg[5]
    iw = (-4.0 + 12.0 * gg[6]) * math.pi / 180.0
    it = (-8.0 + 12.0 * gg[7]) * math.pi / 180.0

    sw = span * chord
    st = str_ * sw
    ct = math.sqrt(st / TAIL_AR)
    arw = span / chord
    # body x positions relative to the wing aerodynamic centre (quarter chord), forward positive
    x_wing_mass = -0.25 * chord          # mid chord
    x_tail_ac = -lt
    x_tail_mass = -lt - 0.25 * ct
    x_nose = xb + 0.02
    x_tailend = -lt - 0.75 * ct
    lf = x_nose - x_tailend
    x_fuse_mass = 0.5 * (x_nose + x_tailend)

    mw = AREAL * sw
    mt = AREAL * st
    mf = FUSE_LIN * lf
    m = mw + mt + mf + mb
    xcg = (mw * x_wing_mass + mt * x_tail_mass + mf * x_fuse_mass + mb * xb) / m
    inertia = (mw * ((x_wing_mass - xcg) ** 2 + chord * chord / 12.0)
               + mt * ((x_tail_mass - xcg) ** 2 + ct * ct / 12.0)
               + mf * ((x_fuse_mass - xcg) ** 2 + lf * lf / 12.0)
               + mb * (xb - xcg) ** 2 + 1e-7)
    aw = 0.9 * 2.0 * math.pi * arw / (arw + 2.0)
    at = 0.9 * 2.0 * math.pi * TAIL_AR / (TAIL_AR + 2.0)
    deda = 2.0 * aw / (math.pi * arw)
    vh = lt * st / (chord * sw)
    x_np = -chord * vh * (at / aw) * (1.0 - deda)
    sm = (xcg - x_np) / chord                     # static margin, fraction of chord (+ = stable)
    cda_f = 2e-5 + 0.004 * 0.01 * lf             # fuselage drag area, m^2

    P = np.empty(N_PHYS)
    P[0] = m
    P[1] = inertia
    P[2] = 0.0 - xcg          # wing AC relative to CG
    P[3] = x_tail_ac - xcg    # tail AC relative to CG
    P[4] = sw
    P[5] = st
    P[6] = chord
    P[7] = ct
    P[8] = arw
    P[9] = TAIL_AR
    P[10] = aw
    P[11] = at
    P[12] = iw
    P[13] = it
    P[14] = cda_f
    P[15] = sm
    P[16] = xcg
    P[17] = span
    return P


@nb.njit(cache=True, inline="always")
def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@nb.njit(cache=True, inline="always")
def _coeffs(alpha, a, ar, cd0, alpha_stall):
    """Lift/drag coefficients with a smooth blend into flat-plate post-stall behaviour.
    CD >= cd0 > 0 always, so the surface can only dissipate energy."""
    s = 1.0 / (1.0 + math.exp(-(abs(alpha) - alpha_stall) / 0.03))
    cl_att = a * alpha
    cl_ps = 1.1 * math.sin(2.0 * alpha)
    cl = (1.0 - s) * cl_att + s * cl_ps
    cd_att = cd0 + cl_att * cl_att / (math.pi * OSWALD * ar)
    sa = math.sin(alpha)
    cd_ps = cd0 + 1.8 * sa * sa
    cd = (1.0 - s) * cd_att + s * cd_ps
    return cl, cd


@nb.njit(cache=True, inline="always")
def _wind(x, z, U, gust):
    wx = U
    wz = 0.0
    for k in range(gust.shape[0]):
        kk = gust[k, 0]
        wx += gust[k, 1] * math.sin(kk * x + gust[k, 2])
        wz += gust[k, 3] * math.sin(kk * x + gust[k, 4])
    return wx, wz


@nb.njit(cache=True, inline="always")
def _surface(vx, vz, th, q, xr, wx, wz, inc, S, chord, a, ar, cd0m, alpha_stall, eps):
    """Force (Fx, Fz) and pitching torque about CG from one lifting surface at body x = xr."""
    c = math.cos(th)
    s = math.sin(th)
    rx = xr * c
    rz = xr * s
    ux = vx - q * rz - wx       # surface velocity relative to the air
    uz = vz + q * rx - wz
    V2 = ux * ux + uz * uz + 1e-12
    V = math.sqrt(V2)
    gam = math.atan2(uz, ux)
    alpha = _wrap(th + inc - gam - eps)
    re = max(V * chord / NU, 1000.0)
    cd0 = cd0m * (2.0 * 1.328 / math.sqrt(re) + 0.008)
    cl, cd = _coeffs(alpha, a, ar, cd0, alpha_stall)
    qS = 0.5 * RHO * V2 * S
    ex = ux / V
    ez = uz / V
    fx = qS * (-cl * ez - cd * ex)
    fz = qS * (cl * ex - cd * ez)
    tau = rx * fz - rz * fx
    return fx, fz, tau, cl, V


@nb.njit(cache=True)
def deriv(st, P, U, gust, clam, cd0m):
    x, z, vx, vz, th, q = st[0], st[1], st[2], st[3], st[4], st[5]
    wx, wz = _wind(x, z, U, gust)
    fxw, fzw, tw, clw, Vw = _surface(vx, vz, th, q, P[2], wx, wz, P[12], P[4], P[6],
                                      clam * P[10], P[8], cd0m, 0.175, 0.0)
    eps = 2.0 * clw / (math.pi * P[8])
    eps = min(max(eps, -0.3), 0.3)
    fxt, fzt, tt, clt, Vt = _surface(vx, vz, th, q, P[3], wx, wz, P[13], P[5], P[7],
                                      clam * P[11], P[9], cd0m, 0.21, eps)
    ux = vx - wx
    uz = vz - wz
    Vf = math.sqrt(ux * ux + uz * uz)
    df = 0.5 * RHO * P[14] * Vf * cd0m
    fx = fxw + fxt - df * ux
    fz = fzw + fzt - df * uz
    m = P[0]
    out = np.empty(6)
    out[0] = vx
    out[1] = vz
    out[2] = fx / m
    out[3] = fz / m - G
    out[4] = q
    out[5] = (tw + tt) / P[1]
    return out


@nb.njit(cache=True)
def energy(st, P):
    return 0.5 * P[0] * (st[2] ** 2 + st[3] ** 2) + 0.5 * P[1] * st[5] ** 2 + P[0] * G * st[1]


@nb.njit(cache=True)
def step(st, dt, P, U, gust, clam, cd0m, method):
    if method == 0:   # classical RK4
        k1 = deriv(st, P, U, gust, clam, cd0m)
        k2 = deriv(st + 0.5 * dt * k1, P, U, gust, clam, cd0m)
        k3 = deriv(st + 0.5 * dt * k2, P, U, gust, clam, cd0m)
        k4 = deriv(st + dt * k3, P, U, gust, clam, cd0m)
        return st + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:             # explicit (forward) Euler: the "cheap and dangerous" integrator
        return st + dt * deriv(st, P, U, gust, clam, cd0m)


@nb.njit(cache=True)
def simulate(P, cond, gust, dt, tmax, method, traj, rec_every):
    """Fly one glider from a launch condition until it touches the ground or tmax.

    cond = [h0, v0, gamma0, alpha0, U_wind, cl_alpha_mult, cd0_mult, q0]
    Returns stats (see STAT_NAMES).  If traj has rows, records [t,x,z,theta,V,E,q,alpha_w]."""
    h0, v0, g0, a0, U, clam, cd0m, q0 = (cond[0], cond[1], cond[2], cond[3], cond[4],
                                         cond[5], cond[6], cond[7])
    st = np.empty(6)
    st[0] = 0.0
    st[1] = h0
    st[2] = v0 * math.cos(g0)
    st[3] = v0 * math.sin(g0)
    st[4] = g0 + a0
    st[5] = q0
    E0 = energy(st, P)
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
    for i in range(nsteps):
        if traj.shape[0] > 0 and i % rec_every == 0 and nrec < traj.shape[0]:
            traj[nrec, 0] = t
            traj[nrec, 1] = st[0]
            traj[nrec, 2] = st[1]
            traj[nrec, 3] = st[4]
            traj[nrec, 4] = math.sqrt(st[2] ** 2 + st[3] ** 2)
            traj[nrec, 5] = Eprev
            traj[nrec, 6] = st[5]
            traj[nrec, 7] = _wrap(st[4] + P[12] - math.atan2(st[3], st[2]))
            nrec += 1
        new = step(st, dt, P, U, gust, clam, cd0m, method)
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
        E = energy(new, P)
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
            # linear interpolation of the touchdown point
            f = st[1] / (st[1] - new[1])
            xr = st[0] + f * (new[0] - st[0])
            t = t - dt + f * dt
            status = 1.0
            st = new
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
    return out


@nb.njit(cache=True, parallel=True)
def evaluate_batch(genes, conds, gusts, dt, tmax, method):
    """genes (N, 8), conds (K, 8), gusts (K, N_GUST, 5) -> stats (N, K, N_STATS)."""
    N = genes.shape[0]
    K = conds.shape[0]
    out = np.empty((N, K, N_STATS))
    empty = np.empty((0, 8))
    for idx in nb.prange(N * K):
        n = idx // K
        k = idx % K
        P = decode(genes[n])
        out[n, k, :] = simulate(P, conds[k], gusts[k], dt, tmax, method, empty, 1)
    return out


def fly(genes, cond, gust, dt=0.005, tmax=60.0, method=0, rec_every=4):
    """Single flight with trajectory recording (for plots and checks)."""
    P = decode(np.asarray(genes, dtype=np.float64))
    n = int(tmax / dt / rec_every) + 3
    traj = np.zeros((n, 8))
    stats = simulate(P, np.asarray(cond, dtype=np.float64), np.asarray(gust, dtype=np.float64),
                     dt, tmax, method, traj, rec_every)
    last = int(np.argmax(traj[1:, 0] == 0.0)) + 1 if np.any(traj[1:, 0] == 0.0) else n
    return stats, traj[:last]


# ---------------------------------------------------------------- launch conditions / gusts

def make_gust(rng, amp):
    """Frozen gust field: 4 spatial sine modes, wavelengths 3-40 m, horizontal + vertical."""
    g = np.zeros((N_GUST, 5))
    if amp <= 0:
        return g
    lam = rng.uniform(3.0, 40.0, N_GUST)
    g[:, 0] = 2 * np.pi / lam
    g[:, 1] = amp * rng.normal(0, 1, N_GUST) / np.sqrt(N_GUST)
    g[:, 2] = rng.uniform(0, 2 * np.pi, N_GUST)
    g[:, 3] = 0.7 * amp * rng.normal(0, 1, N_GUST) / np.sqrt(N_GUST)
    g[:, 4] = rng.uniform(0, 2 * np.pi, N_GUST)
    return g


NOMINAL = np.array([10.0, 6.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])   # 10 m balcony, 6 m/s level throw


def training_conditions(seed=0):
    """6 fixed training conditions: nominal still air + 5 mild perturbations (seeded gusts)."""
    rng = np.random.default_rng(seed)
    conds = [NOMINAL.copy()]
    gusts = [np.zeros((N_GUST, 5))]
    specs = [(10.0, 4.5, 0.0, 0.0, 0.0, 0.0), (10.0, 8.0, -0.1, 0.0, 0.0, 0.0),
             (10.0, 6.0, 0.15, 0.05, -1.0, 0.5), (10.0, 6.0, -0.1, -0.05, 0.5, 0.5),
             (10.0, 6.0, 0.0, 0.0, 0.0, 0.8)]
    for h, v, gam, a0, U, amp in specs:
        conds.append(np.array([h, v, gam, a0, U, 1.0, 1.0, 0.0]))
        gusts.append(make_gust(rng, amp))
    return np.array(conds), np.array(gusts)


def heldout_conditions(n=64, seed=12345):
    """Held-out launches: wider speeds/angles/heights, head/tail wind, stronger gusts, and
    +-15% lift slope / +-30% parasite drag model uncertainty.  Never seen during search."""
    rng = np.random.default_rng(seed)
    conds, gusts = [], []
    for _ in range(n):
        h = rng.uniform(6.0, 14.0)
        v = rng.uniform(3.5, 9.0)
        gam = rng.uniform(-0.25, 0.25)
        a0 = rng.uniform(-0.1, 0.1)
        U = rng.uniform(-1.5, 1.0)
        clam = rng.uniform(0.85, 1.15)
        cd0m = rng.uniform(0.7, 1.3)
        q0 = rng.uniform(-1.0, 1.0)
        conds.append(np.array([h, v, gam, a0, U, clam, cd0m, q0]))
        gusts.append(make_gust(rng, rng.uniform(0.3, 1.5)))
    return np.array(conds), np.array(gusts)


def still_air(n=1):
    return np.repeat(NOMINAL[None], n, 0), np.zeros((n, N_GUST, 5))


# ---------------------------------------------------------------- stability analysis

def glide_eigen(genes, dt=0.002):
    """Linearised longitudinal stability around the steady glide in still air.

    Finds the equilibrium by flying from 2 km altitude for 120 s, then takes a finite-difference
    Jacobian of (vx, vz, theta, q).  Returns (eigs, eq_state, converged)."""
    P = decode(np.asarray(genes, dtype=np.float64))
    gust = np.zeros((N_GUST, 5))
    st = np.array([0.0, 2000.0, 6.0, 0.0, 0.0, 0.0])
    for _ in range(int(120.0 / dt)):
        st = step(st, dt, P, 0.0, gust, 1.0, 1.0, 0)
        if not np.all(np.isfinite(st)) or np.hypot(st[2], st[3]) > 200:
            return None, st, False
    d = deriv(st, P, 0.0, gust, 1.0, 1.0)
    resid = np.abs(d[2:]).max()
    idx = [2, 3, 4, 5]
    J = np.zeros((4, 4))
    for j, sj in enumerate(idx):
        h = 1e-6 * max(1.0, abs(st[sj]))
        sp = st.copy(); sp[sj] += h
        sm = st.copy(); sm[sj] -= h
        J[:, j] = (deriv(sp, P, 0.0, gust, 1.0, 1.0)[idx] - deriv(sm, P, 0.0, gust, 1.0, 1.0)[idx]) / (2 * h)
    eig = np.linalg.eigvals(J)
    return eig, st, bool(resid < 1e-3)


def describe(genes):
    u = genes_to_phys_units(genes)
    P = decode(np.asarray(genes, dtype=np.float64))
    d = {k: float(v) for k, v in zip(GENE_NAMES, u)}
    d.update(mass_g=float(P[0] * 1000), inertia_kgm2=float(P[1]), aspect_ratio=float(P[8]),
             static_margin=float(P[15]), wing_loading_Nm2=float(P[0] * G / P[4]))
    return d
