"""Round-5 NumPy physics, experiment bookkeeping and metrics. Never imports MLX."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import sim

SEEDS = dict(train=51001, val=51002, tune=51003, test=51901, actions=51902)
SCALE = np.tile([0.06, 0.06, 1.0, 1.0], 3).astype(np.float32)
TRAIN_DEFAULTS = dict(features="raw", depth=2, collision_weight=1.0, contact_fraction=.5,
                      decay_steps=0, min_lr_fraction=.1)
TURN2_SEEDS = dict(scene=62001, actions=62002, target=62101, forecast=62201, tune=53003)


class BudgetExpired(RuntimeError):
    pass


class Budget:
    """Cooperative wall guard, including setup; callers use small bounded work units."""
    def __init__(self, minutes):
        if not 0 < minutes <= 45:
            raise ValueError("--max-minutes must be in (0, 45]")
        self.start = time.monotonic()
        self.seconds = 60 * minutes

    def check(self):
        if time.monotonic() - self.start >= self.seconds:
            raise BudgetExpired("wall-clock budget reached")

    @property
    def elapsed(self):
        return time.monotonic() - self.start


def write_json(path, value):
    """Commit a complete JSON file atomically; never emit NaN/Infinity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def walls_step(state, xp=np):
    """Exact walls-only frame, NumPy/MLX array API; arbitrary leading dimensions."""
    s = state.reshape(*state.shape[:-1], 3, 4)
    p, v = s[..., :2], s[..., 2:]
    h = sim.FRAME_DT / sim.SUBSTEPS
    gravity = xp.array([0.0, sim.G * h], dtype=state.dtype)
    for _ in range(sim.SUBSTEPS):
        v = v - gravity
        p = p + v * h
        lo, hi = p < sim.R, p > 1 - sim.R
        hit = (lo & (v < 0)) | (hi & (v > 0))
        v = xp.where(hit, -sim.E_WALL * v, v)
        p = xp.where(lo, 2 * sim.R - p, p)
        p = xp.where(hi, 2 * (1 - sim.R) - p, p)
    return xp.concatenate([p, v], axis=-1).reshape(state.shape)


def pair_features(state, base, xp=np):
    """Directed edges: start/end relative state + pair centre (12 inputs).

    The symmetric local gate is a conservative one-frame travel bound, not a
    ground-truth collision flag. It can admit near misses and miss chain contacts.
    """
    s = state.reshape(*state.shape[:-1], 3, 4)
    b = base.reshape(*base.shape[:-1], 3, 4)
    si, sj = s[..., :, None, :], s[..., None, :, :]
    bi, bj = b[..., :, None, :], b[..., None, :, :]
    norm = xp.array([0.12, 0.12, 3.0, 3.0], dtype=state.dtype)
    centre = xp.array([0.5, 0.5, 0.0, 0.0], dtype=state.dtype)
    feat = xp.concatenate([(sj - si) / norm, (bj - bi) / norm,
                           (0.5 * (si + sj) - centre) / norm], axis=-1)
    # The gate is boolean, so it carries no gradient. Stop it explicitly: in MLX the
    # vjp of sqrt at the zero self-distance is 0 * inf = NaN (round-5 runner fix).
    sg = getattr(xp, "stop_gradient", lambda a: a)
    g = sg(s)
    distance = xp.sqrt(xp.sum((g[..., None, :, :2] - g[..., :, None, :2]) ** 2, axis=-1))
    speed = xp.sqrt(xp.sum(g[..., 2:] ** 2, axis=-1))
    bound = 2 * sim.R + sim.FRAME_DT * (speed[..., :, None] + speed[..., None, :]) + 0.01
    gate = (distance <= bound) & (1 - xp.eye(3, dtype=state.dtype) > 0)
    return feat, gate


def antisymmetric_correction(edge, gate, xp=np):
    """Conserve summed residual state; identical network for every directed pair."""
    swap = xp.swapaxes(edge, -3, -2)
    correction = 0.5 * (edge - swap) * gate[..., None]
    return xp.sum(correction, axis=-2)


def geometry_pair_features(state, base, xp=np):
    """24 features: original 12 plus safe contact geometry; still no impulse law.

    Every differentiated norm has an epsilon *inside* sqrt. This includes self
    edges and zero relative speed; boolean travel gates remain stop-gradient.
    """
    features, gate = pair_features(state, base, xp)
    s = state.reshape(*state.shape[:-1], 3, 4)
    b = base.reshape(*base.shape[:-1], 3, 4)
    dp = s[..., None, :, :2] - s[..., :, None, :2]
    dv = s[..., None, :, 2:] - s[..., :, None, 2:]
    db = b[..., None, :, :2] - b[..., :, None, :2]
    d = xp.sqrt(xp.sum(dp ** 2, axis=-1, keepdims=True) + 1e-8)
    end_d = xp.sqrt(xp.sum(db ** 2, axis=-1, keepdims=True) + 1e-8)
    normal = dp / d
    approach = xp.sum(dv * normal, axis=-1, keepdims=True)
    tangent = dv - approach * normal
    t = xp.clip(-xp.sum(dp * dv, axis=-1, keepdims=True) /
                (xp.sum(dv ** 2, axis=-1, keepdims=True) + 1e-8), 0, sim.FRAME_DT)
    closest = xp.sqrt(xp.sum((dp + t * dv) ** 2, axis=-1, keepdims=True) + 1e-8)
    extra = xp.concatenate([d / (2 * sim.R), end_d / (2 * sim.R), normal, approach / 3,
                            tangent / 3, t / sim.FRAME_DT, closest / (2 * sim.R),
                            xp.maximum(2 * sim.R - d, 0) / (2 * sim.R), db / end_d], axis=-1)
    return xp.concatenate([features, extra], axis=-1), gate


def learning_rate_at(step, initial, decay_steps=0, min_fraction=.1):
    """Step-indexed schedule, so kill/resume has no wall-time-dependent drift."""
    if decay_steps == 0:
        return float(initial)
    progress = min(max(step / decay_steps, 0), 1)
    return float(initial * (min_fraction + (1 - min_fraction) * .5 * (1 + np.cos(np.pi * progress))))


def require_fresh_turn2(seed, action_seed, legacy_data=""):
    used = {0, 1, 2, 51901, 51902, 52001, 52101, 53003, 53004, 53005} | set(SEEDS.values())
    if legacy_data or any(s in used for s in (seed, seed + 100, seed + 200, action_seed)):
        raise ValueError("turn-2 headline requires new scene/target/forecast/action seeds and no legacy data")
    if len({seed, seed + 100, seed + 200, action_seed}) != 4:
        raise ValueError("turn-2 streams must have distinct seeds")


def physics_step(state, restitution=1.0, separation=1.0, balls=True):
    """True integrator family, with the two collision parameters supplied by a fit.

    Known radius, gravity and wall law match the hybrid's supplied information.
    Sequential contact order intentionally matches sim.py, including triple contacts.
    """
    s = np.asarray(state, dtype=np.float64).reshape(-1, 3, 4)
    p, v = s[..., :2].copy(), s[..., 2:].copy()
    h = sim.FRAME_DT / sim.SUBSTEPS
    for _ in range(sim.SUBSTEPS):
        v[..., 1] -= sim.G * h
        p += v * h
        lo, hi = p < sim.R, p > 1 - sim.R
        v = np.where((lo & (v < 0)) | (hi & (v > 0)), -sim.E_WALL * v, v)
        p = np.where(lo, 2 * sim.R - p, p)
        p = np.where(hi, 2 * (1 - sim.R) - p, p)
        if not balls:
            continue
        for i, j in ((0, 1), (0, 2), (1, 2)):
            dp = p[:, j] - p[:, i]
            dist = np.linalg.norm(dp, axis=-1) + 1e-12
            normal = dp / dist[:, None]
            approach = np.sum((v[:, j] - v[:, i]) * normal, axis=-1)
            hit = (dist < 2 * sim.R) & (approach < 0)
            impulse = (-(1 + restitution) * approach * hit / 2)[:, None] * normal
            overlap = (separation * (2 * sim.R - dist) * hit / 2)[:, None] * normal
            v[:, i] -= impulse
            v[:, j] += impulse
            p[:, i] -= overlap
            p[:, j] += overlap
    return np.concatenate([p, v], axis=-1).reshape(state.shape)


def physics_rollout(init, horizon, parameters=None, *, walls=False, final=False, budget=None):
    s = np.asarray(init, dtype=np.float64)
    out = [s]
    parameters = parameters or dict(restitution=1.0, separation=1.0)
    for _ in range(horizon):
        if budget:
            budget.check()
        s = physics_step(s, **parameters, balls=not walls)
        if not final:
            out.append(s)
    return s if final else np.stack(out, axis=1)


def parameter_candidates(seed=51004, n=80):
    """Broad coarse grid, then random exploration; includes the exact-law hypothesis."""
    grid = [(e, s) for e in (0.5, 0.75, 1.0, 1.25)
            for s in (0.5, 0.75, 1.0, 1.25, 1.5)]
    # Put the full coarse grid first; n must be >=20 in a substantive run.
    rng = np.random.default_rng(seed)
    pairs = grid + list(zip(rng.uniform(0.5, 1.25, max(0, n - len(grid))),
                            rng.uniform(0.5, 1.5, max(0, n - len(grid)))))
    return [dict(restitution=float(e), separation=float(s)) for e, s in pairs[:n]]


def sample_windows(states, flags, count, horizon, rng, scene_pool=None, return_flags=False):
    """Half uniform windows, half ball-contact windows; same sampler for every fit.

    Bootstrap scene identities (with replacement) precede window sampling. The
    loss is deliberately collision-balanced, not an estimate of natural MSE.
    """
    pool = np.arange(len(states)) if scene_pool is None else scene_pool
    room = states.shape[1] - horizon
    if room < 1 or len(pool) < 1:
        raise ValueError("empty data or window exceeds data horizon")
    i = rng.choice(pool, count)
    t = rng.integers(0, room, count)
    # A collision anywhere in the window qualifies. Restrict to bootstrap scenes.
    contact = np.zeros((len(pool), room), bool)
    for k in range(horizon):
        contact |= flags[pool, k:k + room, 1]
    pi, pt = np.nonzero(contact)
    if len(pi):
        idx = rng.integers(0, len(pi), count // 2)
        i[:count // 2], t[:count // 2] = pool[pi[idx]], pt[idx]
    windows = states[i[:, None], t[:, None] + np.arange(horizon + 1)]
    if return_flags:
        events = flags[i[:, None], t[:, None] + np.arange(horizon), 1]
        return windows, events
    return windows


def stats(values):
    v = np.asarray(values, float)
    if not len(v):
        return dict(n=0, mean=None, median=None, nonfinite=0)
    finite = np.isfinite(v)
    # A divergent rollout must not disappear from the mean.
    return dict(n=int(v.size), mean=float(v.mean()) if finite.all() else None,
                median=float(np.median(v)) if np.isfinite(np.median(v)) else None,
                nonfinite=int((~finite).sum()))


def pos_error(prediction, truth):
    assert prediction.shape == truth.shape and prediction.shape[-1] == 12
    d = (prediction - truth).reshape(*truth.shape[:-1], 3, 4)[..., :2]
    return np.linalg.norm(d, axis=-1).mean(axis=-1)


def forecast_metrics(prediction, truth, flags):
    error = pos_error(prediction, truth)
    horizons = sorted(set([1, 2, 5, 10, 20, 40, 50, truth.shape[1] - 1]))
    result = {}
    for h in horizons:
        if h >= truth.shape[1]:
            continue
        any_hit = flags[:, :h].any(axis=(1, 2))
        ball_hit = flags[:, :h, 1].any(axis=1)
        result[str(h)] = dict(all=stats(error[:, h]),
                             collision=stats(error[any_hit, h]),
                             no_collision=stats(error[~any_hit, h]),
                             ball_collision=stats(error[ball_hit, h]),
                             no_ball_collision=stats(error[~ball_hit, h]))
    return result


def average_ranks(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    start = np.r_[0, np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1]
    end = np.r_[start[1:], len(x)]
    ranks = np.empty(len(x), float)
    for lo, hi in zip(start, end):
        ranks[order[lo:hi]] = 0.5 * (lo + hi - 1)
    return ranks


def uncertainty_metrics(disagreement, error):
    u, e = np.asarray(disagreement), np.asarray(error)
    valid = np.isfinite(u) & np.isfinite(e)
    u, e = u[valid], e[valid]
    corr = None
    if len(u) > 1 and np.ptp(u) > 0 and np.ptp(e) > 0:
        corr = float(np.corrcoef(average_ranks(u), average_ranks(e))[0, 1])
    bad = e >= sim.R
    auc = None
    if bad.any() and (~bad).any():
        ranks = average_ranks(u) + 1
        n1, n0 = int(bad.sum()), int((~bad).sum())
        auc = float((ranks[bad].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    # Use value quantiles, not arbitrary splitting of tied zero uncertainties.
    edges = np.unique(np.quantile(u, np.linspace(0, 1, 6))) if len(u) else []
    bins = []
    for k in range(max(0, len(edges) - 1)):
        mask = (u >= edges[k]) & ((u <= edges[k + 1]) if k == len(edges) - 2 else (u < edges[k + 1]))
        bins.append(dict(disagreement=stats(u[mask]), error=stats(e[mask])))
    return dict(n=int(valid.sum()), invalid=int((~valid).sum()), spearman=corr,
                auc_error_ge_radius=auc, quantile_bins=bins)


def planning_metrics(rows):
    dream = np.array([np.inf if r["dream"] is None else r["dream"] for r in rows])
    real = np.array([r["real"] for r in rows])
    ds, rs = dream < sim.R, real < sim.R
    fooled = ds & ~rs
    return dict(n=len(rows), coverage=1.0, dream_success=float(ds.mean()),
                real_success=float(rs.mean()), fooled_rate=float(fooled.mean()),
                fooled_given_dream_success=float(fooled.sum() / ds.sum()) if ds.any() else None,
                severe_fooled_rate=float((ds & (real >= 2 * sim.R)).mean()),
                dream_distance=stats(dream), real_distance=stats(real), gap=stats(real - dream),
                invalid_candidates=sum(r.get("invalid_candidates", 0) for r in rows),
                all_invalid_tasks=sum(r.get("all_invalid", False) for r in rows))


def paired_test(a, b, seed=51005):
    """Exact McNemar; paired bootstrap CI on success-rate difference. Task is unit."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    assert a.shape == b.shape and len(a)
    win, lose = int((a & ~b).sum()), int((b & ~a).sum())
    n = win + lose
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(win, lose) + 1)) / 2 ** n) if n else 1.0
    delta = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(axis=1)
    return dict(a_only=win, b_only=lose, p_exact=p, difference=float(delta.mean()),
                difference_ci95=np.quantile(boot, [.025, .975]).tolist())


def holm_adjust(tests):
    order = sorted(tests, key=lambda k: tests[k]["p_exact"])
    last = 0.0
    for i, key in enumerate(order):
        last = max(last, min(1.0, (len(order) - i) * tests[key]["p_exact"]))
        tests[key]["p_holm"] = last
    return tests


def candidates(task_id, count, seed=SEEDS["actions"]):
    """Nested random shots; increasing count cannot change the prefix."""
    return np.random.default_rng(np.random.SeedSequence([seed, task_id])).uniform(-1.5, 1.5, (count, 2))


def launch_states(scene, velocities):
    x = np.broadcast_to(scene, (len(velocities), 12)).copy()
    x[:, 2:4] = velocities
    return x


def cem_proposal(velocities, scores, count, rng):
    elite = velocities[np.argsort(scores, kind="stable")[:max(2, len(scores) // 8)]]
    return np.clip(rng.normal(elite.mean(0), np.maximum(elite.std(0), 0.08), (count, 2)), -1.5, 1.5)
