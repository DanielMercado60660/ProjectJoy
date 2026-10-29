"""Cue-initialized Kalman and joint bootstrap particle filter. No simulator labels.

The PF models the *joint* state so collisions retain slot identity, and sums all
injective detection associations (including missed detections and clutter).
The reported density is a weighted particle KDE with val-tuned bandwidth.
"""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp, ndtr, ndtri
import sim
from kalman import KF, FULL_AREA, detect
from r5_data import CUE_SIGMA, view_occ


def physical_step(pos, vel, was_hidden, occ, rng, q=0., turn_probability=.3):
    """Vectorized simulator-family transition; same move/bounce/collide/turn order.

    pos, vel: (particles, live_slots, 2). Opacity is physical, never changed by a peek.
    """
    pos = pos + vel
    vel = vel.copy()
    for axis in range(2):
        lo, hi = pos[..., axis] < sim.R, pos[..., axis] > sim.W-sim.R
        pos[..., axis] = np.where(lo, 2*sim.R-pos[..., axis], pos[..., axis])
        pos[..., axis] = np.where(hi, 2*(sim.W-sim.R)-pos[..., axis], pos[..., axis])
        vel[..., axis] *= np.where(lo | hi, -1, 1)
    for i in range(pos.shape[1]):
        for j in range(i+1, pos.shape[1]):
            delta = pos[:, j]-pos[:, i]
            distance = np.linalg.norm(delta, axis=-1)
            normal = delta / np.maximum(distance[:, None], 1e-6)
            relative = ((vel[:, i]-vel[:, j])*normal).sum(-1)
            relative = np.where((distance > 1e-6) & (distance < 2*sim.R) & (relative > 0), relative, 0)
            vel[:, i] -= relative[:, None]*normal
            vel[:, j] += relative[:, None]*normal
    # Same sample-grid definition of fully hidden as sim.occ_visible_frac, without
    # constructing the full 25-point chord array for every particle.
    hidden = np.zeros(pos.shape[:2], bool)
    for a, b in occ:
        if a >= 0:
            hidden |= (pos[..., 0] - .98*sim.R >= a) & (pos[..., 0] + .98*sim.R < b)
    turn = hidden & ~was_hidden & (rng.random(hidden.shape) < turn_probability)
    new_vel = rng.uniform(.6, 2.2, vel.shape) * rng.choice([-1, 1], vel.shape)
    vel = np.where(turn[..., None], new_vel, vel)
    if q:
        vel += rng.normal(0, q, vel.shape)
    return pos, vel, hidden


def detection_probability(pos, occ, miss):
    frac = np.ones(pos.shape[:-1])
    for a, b in occ:
        if a >= 0:
            hidden_extent = np.maximum(0, np.minimum(pos[..., 0]+sim.R, b)-np.maximum(pos[..., 0]-sim.R, a))
            frac -= hidden_extent/(2*sim.R)
    return np.clip((1-miss)*np.clip(frac, 0, 1), .002, .998)


def association_loglik(pos, detections, occ, r, miss):
    """Exact sum over partial one-to-one assignments for <=3 objects.

    Visible blobs can merge on contact: missed-object likelihood explicitly permits
    fewer detections than objects. Uniform clutter permits extra detector components.
    """
    count, slots = pos.shape[:2]
    pd = detection_probability(pos, occ, miss)
    absent = np.log1p(-pd)
    base = absent.sum(-1)
    if not detections:
        return base
    # There can be tiny disconnected raster fragments; keep 3 largest components.
    detections = sorted(detections, key=lambda d: d[1], reverse=True)[:3]
    z = np.array([d[0] for d in detections])
    variance = r*r * FULL_AREA / np.maximum([d[1] for d in detections], 3)
    delta = pos[:, :, None, :] - z[None, None]
    lp = (-np.log(2*np.pi*variance)[None, None, :]
          - (delta*delta).sum(-1)/(2*variance[None, None, :])
          + np.log(pd)[..., None] - absent[..., None])
    hypotheses = []
    for assignment in itertools.product(range(-1, slots), repeat=len(z)):
        used = [a for a in assignment if a >= 0]
        if len(set(used)) != len(used):
            continue
        ll = base.copy()
        for j, slot in enumerate(assignment):
            ll += np.log(.02/(64*64)) if slot < 0 else lp[:, slot, j]
        hypotheses.append(ll)
    return logsumexp(np.stack(hypotheses), axis=0)


def systematic_resample(weights, rng):
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1
    return np.searchsorted(cumulative, (rng.random()+np.arange(len(weights)))/len(weights))


def initial_velocity_proposal(pos, detections, rng, r):
    """Observation-guided *importance* proposal for the first transition only.

    The simulator's initial component velocity is uniform on [-2.2,-.6] U [.6,2.2].
    Draw from .2*prior + .8*truncated-N(detection - position, measurement sigma),
    and return exact log(prior/proposal). This uses frame 1 only when processing
    frame 1, never future data. The prior component supports bounce/merge cases.
    """
    velocity = rng.uniform(.6, 2.2, pos.shape)*rng.choice([-1, 1], pos.shape)
    correction = np.zeros(pos.shape[0])
    if not detections:
        return velocity, correction
    z = np.array([d[0] for d in detections])
    slots, dets = linear_sum_assignment(np.linalg.norm(pos.mean(0)[:, None]-z[None], axis=-1))
    for slot, det in zip(slots, dets):
        sigma = r*np.sqrt(FULL_AREA/max(detections[det][1], 3))
        center = z[det]-pos[:, slot]
        # Each component is sampled independently; mixture proposal has full
        # support on the original bounded velocity prior even at box bounces.
        for axis in range(2):
            mu = center[:, axis]
            lower = ndtr((np.array([-2.2, .6])[None]-mu[:, None])/sigma)
            upper = ndtr((np.array([-.6, 2.2])[None]-mu[:, None])/sigma)
            masses = np.maximum(upper-lower, 0)
            total = masses.sum(-1)
            usable = total > 1e-10
            interval = (rng.random(len(pos)) >= masses[:, 0]/np.maximum(total, 1e-300)).astype(int)
            ix = np.arange(len(pos))
            quantile = lower[ix, interval]+rng.random(len(pos))*masses[ix, interval]
            guided = mu+sigma*ndtri(np.clip(quantile, 1e-15, 1-1e-15))
            use = usable & (rng.random(len(pos)) < .8)
            velocity[:, slot, axis] = np.where(use, guided, velocity[:, slot, axis])
            v = velocity[:, slot, axis]
            pdf = np.exp(-.5*((v-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi)*np.maximum(total, 1e-300))
            proposal = np.where(usable, .2/3.2+.8*pdf, 1/3.2)
            correction += -np.log(3.2)-np.log(proposal)
    return velocity, correction


def particle_track(frames, cue, present, occ, look=None, seed=0, particles=256,
                   q=.15, r=1., miss=.15, bandwidth=1.5):
    ids = np.flatnonzero(present)
    rng = np.random.default_rng(seed)
    pos = cue[ids][None] + rng.normal(0, CUE_SIGMA, (particles, len(ids), 2))
    vel = rng.uniform(.6, 2.2, pos.shape) * rng.choice([-1, 1], pos.shape)
    hidden = np.zeros(pos.shape[:-1], bool)
    weights = np.ones(particles)/particles
    all_w, all_m, all_c, ess_values = [], [], [], []
    for t, frame in enumerate(frames):
        # Per-time streams provide a reproducible common-random-number comparison.
        rng = np.random.default_rng(seed + 1009*(t+1))
        dets = detect(frame, sim.SAME, .35, True)
        correction = np.zeros(particles)
        if t == 1:
            vel, correction = initial_velocity_proposal(pos, dets, rng, r)
        if t:
            pos, vel, hidden = physical_step(pos, vel, hidden, occ, rng, q)
        observed_occ = view_occ(occ, look[1]) if look is not None and t == look[0] else occ
        ll = np.log(np.maximum(weights, 1e-300)) + correction + association_loglik(pos, dets, observed_occ, r, miss)
        weights = np.exp(ll-logsumexp(ll))
        ess = 1/(weights*weights).sum()
        ess_values.append(ess/particles)
        means = np.full((sim.K, particles, 2), 32.)
        means[ids] = pos.transpose(1, 0, 2)
        all_m.append(means)
        all_w.append(np.broadcast_to(weights, (sim.K, particles)).copy())
        all_c.append(np.broadcast_to(np.eye(2)*bandwidth**2, (sim.K, particles, 2, 2)).copy())
        if ess < .5*particles:
            ix = systematic_resample(weights, rng)
            pos, vel, hidden = pos[ix].copy(), vel[ix].copy(), hidden[ix].copy()
            weights = np.ones(particles)/particles
    return (np.stack(all_w), np.stack(all_m), np.stack(all_c)), {
        'mean_ess_fraction': float(np.mean(ess_values)), 'min_ess_fraction': float(np.min(ess_values))}


def kalman_track(frames, cue, present, occ, look=None, q=.5, r=7.):
    ids = np.flatnonzero(present)
    filters = {k: KF(q, r, 2.) for k in ids}
    for k, f in filters.items():
        f.init(cue[k], FULL_AREA)
        f.P[:2, :2] = np.eye(2)*CUE_SIGMA**2
    means, covs = [], []
    for t, frame in enumerate(frames):
        if t:
            for f in filters.values():
                f.predict(True)
        dets = detect(frame, sim.SAME, .35, True)
        if dets:
            cost = np.array([[filters[k].maha(*d) for d in dets] for k in ids])
            rows, cols = linear_sum_assignment(cost)
            for a, b in zip(rows, cols):
                if len(dets) >= len(ids) or cost[a, b] < 13.8:
                    filters[ids[a]].update(*dets[b])
        m = np.full((sim.K, 1, 2), 32.)
        c = np.tile(np.eye(2)*1024, (sim.K, 1, 1, 1))
        for k, f in filters.items():
            m[k, 0], c[k, 0] = f.x[:2], f.P[:2, :2] + np.eye(2)*1e-6
        means.append(m)
        covs.append(c)
    return (np.ones((len(frames), sim.K, 1)), np.stack(means), np.stack(covs)), {}
