"""Invisible spatial patches / progress-triggered mid-lap changes, strict walls.

The only new sensor is realized yaw rate over the PREVIOUS step. No appearance
channel exists. The independently seeded paint attribute is render-only.
"""
import numpy as np
from racing.sim import CAR_R, DT, G, VecRace
from racing.track import SPLITS, make_track, make_tracks, pack
from r5_common import GROUPS, SCENARIO_SEEDS

OBS = 14
STATE_FIELDS = ('x', 'y', 'yaw', 'v', 'steer_prev', 'pedal_prev', 'idx', 's_frac',
                'prog', 'lat', 'done', 'crashed', 'track_id', 'grip', 't', 'wall_hits')
SURFACE_FIELDS = ('mode', 'high', 'low', 'start', 'width', 'up', 'paint', 'omega', 'last_grip')


# Fixed before examining any R5 driving outcomes (turn 1b). Retry only exhausted
# rejection sampling; retain geometry and all original validity checks. The high
# namespace and 256-wide blocks keep each requested seed's retries separate.
TRACK_RETRY_BASE = 2**32
TRACK_RETRY_LIMIT = 32


def make_r5_track(seed, geometry):
    requested = SPLITS[geometry] + seed
    for retry in range(TRACK_RETRY_LIMIT + 1):
        effective = requested if retry == 0 else TRACK_RETRY_BASE + 256 * requested + retry
        local_seed = effective - SPLITS[geometry]
        try:
            track = make_track(local_seed, geometry)
        except RuntimeError as exc:
            # Do not mask unrelated generator/programming errors.
            if str(exc) != f'could not generate a valid track for seed {local_seed}':
                raise
            if retry == TRACK_RETRY_LIMIT:
                raise RuntimeError(f'R5 {geometry} requested seed {requested} exhausted '
                                   f'{TRACK_RETRY_LIMIT} deterministic replacements') from exc
        else:
            record = dict(geometry=geometry, requested_seed=requested,
                          effective_seed=track.seed, retry=retry)
            if retry:
                print(f'R5 track replacement: {record}', flush=True)
            return track, record


def make_bank(split, n):
    """n layouts/geometry; legacy uses the unchanged original 100/50 seeds."""
    start = {'train': 30000, 'val': 40000, 'test': 70000, 'legacy': 0}[split]
    regular = 'heldout' if split in ('test', 'legacy') else 'train'
    regular_n = 2 * n if split == 'legacy' else n
    tracks, manifest = [], []
    for geometry, count in ((regular, regular_n), ('hard', n)):
        if split == 'legacy':
            # No fallback whatsoever for the original exhibit's tracks.
            group = make_tracks(geometry, count, start)
            tracks.extend(group)
            manifest.extend(dict(geometry=geometry, requested_seed=t.seed,
                                 effective_seed=t.seed, retry=0) for t in group)
        else:
            for seed in range(start, start + count):
                track, record = make_r5_track(seed, geometry)
                tracks.append(track)
                manifest.append(record)
    bank = pack(tracks)
    bank.r5_regular_count, bank.r5_hard_count = regular_n, n
    bank.r5_track_manifest = manifest
    return bank


def suite(bank, split, repeats=1):
    """Balanced six groups; every validation call uses ALL groups and tracks."""
    normal_n = getattr(bank, 'r5_regular_count', len(bank.tracks) // 2)
    hard_n = len(bank.tracks) - normal_n
    ids, modes, labels = [], [], []
    for g in GROUPS:
        hard = g.startswith('hard')
        n = hard_n if hard else normal_n
        for _ in range(repeats):
            ids.extend(np.arange(n) + (normal_n if hard else 0))
            modes.extend([{'patch': 0, 'switch': 1, 'static': 2}[g.split('_')[1]]] * n)
            labels.extend([g] * n)
    return np.array(ids), np.array(modes), np.array(labels), SCENARIO_SEEDS[split]


class SurfaceRace:
    def __init__(self, bank, n):
        self.base = VecRace(bank, n)
        self.E = n
        for k in SURFACE_FIELDS:
            setattr(self, k, np.zeros(n, dtype=int if k == 'mode' else float))

    def reset(self, rng, ids=None, mask=None, modes=None):
        mask = np.ones(self.E, bool) if mask is None else mask
        ids = rng.integers(len(self.base.bank.tracks), size=self.E) if ids is None else np.asarray(ids)
        # All random draws have fixed length, independent of paint and controller.
        self.mode[mask] = (rng.choice(3, self.E, p=[.4, .4, .2]) if modes is None else modes)[mask]
        self.high[mask] = rng.uniform(.9, 1.1, self.E)[mask]
        self.low[mask] = rng.uniform(.4, .65, self.E)[mask]
        self.start[mask] = rng.uniform(.22, .38, self.E)[mask]
        self.width[mask] = rng.uniform(.24, .32, self.E)[mask]
        self.up[mask] = rng.integers(0, 2, self.E)[mask]
        # Paint uses a separate stream; it cannot affect physics or observation.
        self.paint[mask] = np.mod(ids[mask] * .61803398875 + .314159, 1)
        self.omega[mask] = 0
        self.base.reset(ids, mask=mask)
        self.base.grip[mask] = self.grip_now()[mask]
        self.last_grip[mask] = self.base.grip[mask]
        return self.observe()

    def grip_now(self):
        b = self.base
        # Patches are fixed in road coordinates, including when a car reverses.
        position = (b.idx + b.s_frac) / b.bank.n_pts[b.track_id]
        wet_patch = (position >= self.start) & (position < self.start + self.width)
        switched = b.prog / b.bank.length[b.track_id] >= self.start
        wet_switch = np.where(self.up > .5, ~switched, switched)
        wet = np.where(self.mode == 0, wet_patch, np.where(self.mode == 1, wet_switch, False))
        return np.where(wet, self.low, self.high)

    def observe(self):
        return np.column_stack([self.base.observe(), self.omega]).astype(np.float32)

    def step(self, action):
        b = self.base
        before = b.yaw.copy()
        b.grip[:] = self.grip_now()
        self.last_grip[:] = b.grip
        obs, dp, done = b.step(action)
        self.omega[:] = (b.yaw - before) / DT  # radians/s, measured after action
        return np.column_stack([obs, self.omega]).astype(np.float32), dp, done

    def state(self):
        return {**{'base/' + k: getattr(self.base, k).copy() for k in STATE_FIELDS},
                **{'surface/' + k: getattr(self, k).copy() for k in SURFACE_FIELDS}}

    def restore(self, state):
        for k in STATE_FIELDS:
            getattr(self.base, k)[:] = state['base/' + k]
        for k in SURFACE_FIELDS:
            getattr(self, k)[:] = state['surface/' + k]


def fixed_surfaces(env, ids, modes, scenario_seed):
    """Evaluation/CEM scenarios keyed by track+mode, independent of batch size.

    This lets the six-scene demo replay the EXACT cases from the headline test.
    PPO reset still draws online episodes from its checkpointed RNG stream.
    """
    cache = {}
    for e, (tid, mode) in enumerate(zip(ids, modes)):
        key = (env.base.bank.tracks[int(tid)].seed, int(mode))
        if key not in cache:
            rng = np.random.default_rng(np.random.SeedSequence([scenario_seed, *key]))
            cache[key] = (rng.uniform(.9, 1.1), rng.uniform(.4, .65),
                          rng.uniform(.22, .38), rng.uniform(.24, .32), rng.integers(0, 2))
        for name, value in zip(('high', 'low', 'start', 'width', 'up'), cache[key]):
            getattr(env, name)[e] = value
    env.base.grip[:] = env.grip_now()
    env.last_grip[:] = env.base.grip


def finish_flags(progress, length, crashed):
    return (progress >= length) & ~crashed


def normal_envelope(env):
    """Scoring only, never a policy input: central half of road and useful safe speed.

    Normal = |lateral|/room <= .5 and 0.6 <= v/v_ref <= 1.3,
    v_ref = clip(sqrt(mu*g / max(|local curvature|,.01)), 6, 30).
    This is a safety-envelope latency, not a claim of latent system identification.
    """
    b = env.base
    k = np.array([abs(b.bank.tracks[t].curvature[i]) for t, i in zip(b.track_id, b.idx)])
    vref = np.clip(np.sqrt(np.maximum(env.last_grip * G / np.maximum(k, .01), 1e-8)), 6, 30)
    return ((np.abs(b.lat) / (b.bank.half_width[b.track_id] - CAR_R) <= .5)
            & (b.v >= .6 * vref) & (b.v <= 1.3 * vref) & ~b.crashed)
