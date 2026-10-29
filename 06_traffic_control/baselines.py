"""Cheap signal controllers. Each is a function env -> desired phase (B, 4); 0 = NS green, 1 = EW green.
All of them decide every K steps (10 s), the same rate as the learned controller."""
import numpy as np
from sim import N_INT, K, LANE_CAP, DOWN, INTERNAL_SRC, N_LANES


def fixed_time(plan):
    """plan: list of 4 (ew_decisions, ns_decisions) green lengths, one per intersection, in units of
    10-s decisions. Cycle = ew + ns decisions (the 2-s yellow comes out of the next green)."""
    ew = np.array([p[0] for p in plan]); ns = np.array([p[1] for p in plan])
    def ctrl(env):
        d = env.t // K
        pos = d % (ew + ns)
        return np.broadcast_to(np.where(pos < ew, 1, 0), (env.B, N_INT))
    return ctrl


def fixed_time_batch(ew, ns):
    """Vectorised fixed-time: ew, ns arrays of shape (B, 4) -> one plan per batch entry (for tuning)."""
    def ctrl(env):
        d = env.t // K
        return np.where(d % (ew + ns) < ew, 1, 0)
    return ctrl


def _phase_scores(env, use_downstream):
    q = env.ready.astype(float)
    if use_downstream:
        dq = np.zeros_like(q)
        dq[:, INTERNAL_SRC] = env.cnt[:, DOWN[INTERNAL_SRC]]
        q = q - dq
    q = q.reshape(env.B, N_INT, 4)
    return q[..., 0] + q[..., 1], q[..., 2] + q[..., 3]


def longest_queue_first(env):
    ns, ew = _phase_scores(env, False)
    return np.where(ew > ns, 1, np.where(ns > ew, 0, env.phase))


def max_pressure(env):
    """Varaiya-style max-pressure: serve the phase with the largest (upstream queue - downstream load)."""
    ns, ew = _phase_scores(env, True)
    return np.where(ew > ns, 1, np.where(ns > ew, 0, env.phase))


def max_pressure_guard(max_wait_s=90.0):
    """Max-pressure, but any approach whose front vehicle has waited > max_wait_s gets served now."""
    def ctrl(env):
        p = max_pressure(env)
        hw = env.head_wait().reshape(env.B, N_INT, 4)
        ns_old = np.maximum(hw[..., 0], hw[..., 1]); ew_old = np.maximum(hw[..., 2], hw[..., 3])
        p = np.where((ns_old > max_wait_s) & (ns_old >= ew_old), 0, p)
        p = np.where((ew_old > max_wait_s) & (ew_old > ns_old), 1, p)
        return p
    return ctrl
