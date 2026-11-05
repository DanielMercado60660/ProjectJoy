"""Round 6 shared pieces: R6Arena (env.Arena + mirrored observations + ball-delay history + per-env opponent
kinds), the 27-d observation, the MLX actor-critic, a numpy forward for frozen archive snapshots, and the
opponent dispatcher used by training and eval.

Mirroring: robot 1 sees the world rotated by 180 degrees (x -> -x, y -> -y, heading + pi), so in its own
frame it also attacks +x. Rotation keeps handedness, so (left, right) wheel commands transfer unchanged and
a frozen copy of the learner can drive slot 1 (self-play). Slot 1 still moves at the env's V_OPP (0.7x).
"""
import json, os, pickle
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from env import Arena, W, H, R_ROBOT, R_BALL, T_MAX
import scripted
import r6_opponents as ro

OBS6 = 27
HIST = 8
NOMINAL_FRAC = 0.35
# opponent kinds (slot 1). Scripted legacy members first, then randomized families, then self-play.
KINDS = ["goalie", "keeper", "chaser", "goalie_r", "keeper_r", "chaser_r", "marker", "wall", "self_arch", "self_cur",
         "def_arch", "def_cur"]   # def_*: the learned defender (r6_train --learn-def 1): archived / current
K = {k: i for i, k in enumerate(KINDS)}
STRIKER_KINDS = {"chaser", "chaser_r"}   # start like the chaser (anywhere in the right half)


class R6Arena(Arena):
    def __init__(self, n, seed=0, opponent="goalie", mix=None):
        self.mix = mix                                  # None: fixed opponent (eval); else probs over KINDS
        self.kind = np.zeros(n, dtype=np.int64)
        self.arch_id = np.zeros(n, dtype=np.int64)
        self.n_arch = 0
        self.darch_id = np.zeros(n, dtype=np.int64)
        self.n_darch = 0
        self.P = None
        self.hist_b = np.zeros((HIST, n, 2))
        self.hist_v = np.zeros((HIST, n, 2))
        self.head = 0
        self.prev_act = np.zeros((n, 2, 2))
        self.zone_recover = np.zeros(n, dtype=bool)
        super().__init__(n, seed=seed, opponent=opponent)

    def reset(self, mask):
        k = int(mask.sum())
        if k == 0:
            return
        super().reset(mask)
        r = self.rng
        if self.P is None:
            self.P = ro.sample_params(np.random.default_rng(0), self.n)
        if self.mix is not None:
            p = np.asarray(self.mix, float)
            if self.n_arch == 0:                         # no snapshots yet: self_arch -> self_cur
                p = p.copy(); p[K["self_cur"]] += p[K["self_arch"]]; p[K["self_arch"]] = 0
            if self.n_darch == 0:
                p = p.copy(); p[K["def_cur"]] += p[K["def_arch"]]; p[K["def_arch"]] = 0
            self.kind[mask] = r.choice(len(KINDS), k, p=p / p.sum())
            self.arch_id[mask] = r.integers(0, max(self.n_arch, 1), k)
            self.darch_id[mask] = r.integers(0, max(self.n_darch, 1), k)
            newp = ro.sample_params(r, k)
            # pilot-1 lesson (dev, vs the training goalie): the nominal opponent (full speed, no delay, no noise)
            # sat at the corner of the randomization box, so it was almost never trained on. 35% of episodes are
            # now nominal; the rest use speed U[0.75, 1] with delay and noise as before.
            nom = r.random(k) < NOMINAL_FRAC
            newp["speed"] = np.where(nom, 1.0, 0.75 + (newp["speed"] - 0.7) / 0.3 * 0.25)
            newp["delay"] = np.where(nom, 0, newp["delay"])
            newp["noise"] = np.where(nom, 0.0, newp["noise"])
            for key in self.P:
                self.P[key][mask] = newp[key]
            st = mask & np.isin(self.kind, [K[s] for s in STRIKER_KINDS])
            m = int(st.sum())
            if m:   # striker start, as Arena does for the chaser
                p1 = np.stack([r.uniform(0.4, 0.85, m), r.uniform(-0.4, 0.4, m)], 1)
                ball = self.ball[st]
                d = p1 - ball
                dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
                too = dist[:, 0] < R_ROBOT + R_BALL + 0.03
                p1[too] = ball[too] + d[too] / dist[too] * (R_ROBOT + R_BALL + 0.03)
                self.pos[st, 1] = p1
        self.hist_b[:, mask] = self.ball[mask]
        self.hist_v[:, mask] = self.bvel[mask]
        self.prev_act[mask] = 0
        self.zone_recover[mask] = False

    def step(self, a0, a1):
        out = super().step(a0, a1)
        self.prev_act[:, 0] = np.clip(a0, -1, 1)
        self.prev_act[:, 1] = np.clip(a1, -1, 1)
        self.head = (self.head + 1) % HIST
        self.hist_b[self.head] = self.ball
        self.hist_v[self.head] = self.bvel
        return out

    def delayed_view(self, idx, delay):
        """View of envs idx where the ball is seen `delay` steps late (per env)."""
        h = (self.head - delay) % HIST
        return SimpleNamespace(n=len(idx), pos=self.pos[idx], th=self.th[idx], ball=self.hist_b[h, idx],
                               bvel=self.hist_v[h, idx], last_touch=self.last_touch[idx])

    def obs_r6(self, robot=0):
        """27-d observation from `robot`'s point of view (robot 1: world rotated by 180 degrees)."""
        o = [robot, 1 - robot]
        sg = 1.0 if robot == 0 else -1.0
        p = sg * self.pos[:, o]
        th = self.th[:, o] + (0.0 if robot == 0 else np.pi)
        b, bv = sg * self.ball, sg * self.bvel
        ov = sg * self.vel[:, 1 - robot]
        c0, s0 = np.cos(th[:, 0]), np.sin(th[:, 0])

        def ego(v):
            return np.stack([c0 * v[:, 0] + s0 * v[:, 1], -s0 * v[:, 0] + c0 * v[:, 1]], 1)

        goal, own = np.array([W, 0.0]), np.array([-W, 0.0])
        x = np.concatenate([
            p[:, 0], np.stack([c0, s0], 1), b, bv, ego(b - p[:, 0]), ego(bv),
            p[:, 1], np.stack([np.cos(th[:, 1]), np.sin(th[:, 1])], 1), ego(p[:, 1] - p[:, 0]),
            ego(goal - p[:, 0]), ego(own - p[:, 0]), (self.t / T_MAX)[:, None],
            ego(ov), self.prev_act[:, robot]], 1)
        return x.astype(np.float32)


# ----------------------------------------------------------------------------- networks
class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden):
        super().__init__()
        self.l1 = nn.Linear(d_in, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, d_out)

    def __call__(self, x):
        return self.l3(mx.tanh(self.l2(mx.tanh(self.l1(x)))))


class AC(nn.Module):
    """Tanh-squashed Gaussian mean (round-4 fix), state-independent log std, separate value net."""
    def __init__(self, hidden=128):
        super().__init__()
        self.pi = MLP(OBS6, 2, hidden)
        self.v = MLP(OBS6, 1, hidden)
        self.log_std = mx.zeros((2,)) - 0.5

    def __call__(self, x):
        return mx.tanh(self.pi(x)), self.v(x)[:, 0]


def logprob(mean, log_std, a):
    return mx.sum(-0.5 * ((a - mean) / mx.exp(log_std)) ** 2 - log_std - 0.5 * np.log(2 * np.pi), axis=-1)


class RunningNorm:
    def __init__(self, d):
        self.mean, self.var, self.count = np.zeros(d), np.ones(d), 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)


def snapshot(model, norm):
    """Frozen numpy copy of the policy (for the self-play archive and for eval)."""
    f = {k: np.array(v) for k, v in tree_flatten(model.pi.parameters())}
    return dict(w=f, mean=norm.mean.astype(np.float32).copy(), var=norm.var.astype(np.float32).copy())


def np_policy(snap, x_raw):
    x = np.clip((x_raw - snap["mean"]) / np.sqrt(snap["var"] + 1e-8), -10, 10)
    w = snap["w"]
    h = np.tanh(x @ w["l1.weight"].T + w["l1.bias"])
    h = np.tanh(h @ w["l2.weight"].T + w["l2.bias"])
    return np.tanh(h @ w["l3.weight"].T + w["l3.bias"])


def load_snap(path, key="snap"):
    """Policy snapshot from an r6 checkpoint (.pkl, full state; key 'snap' = attacker, 'dsnap' = learned
    defender) or from an archive snapshot file."""
    d = pickle.load(open(path, "rb"))
    return d[key] if key in d and d[key] is not None else d["snap"]


def r6_policy(snap, robot=0):
    """Deterministic (mean-action) policy function for eval: arena -> (n,2)."""
    return lambda a: np_policy(snap, a.obs_r6(robot))


# ----------------------------------------------------------------------------- opponents for slot 1
def pool_actions(env, archive, cur_fn, rng, darchive=()):
    """Slot-1 actions for a training R6Arena with mixed kinds. archive: list of learner snapshots;
    cur_fn(obs_raw) -> actions of the current learner (used for self_cur); darchive: learned-defender snapshots.
    def_cur envs are left at 0 here: the trainer fills them with the current defender's sampled actions."""
    n = env.n
    act = np.zeros((n, 2))
    P = env.P
    for name in ("goalie", "keeper", "chaser"):
        idx = np.flatnonzero(env.kind == K[name])
        if len(idx):
            v = env.delayed_view(idx, P["delay"][idx])
            act[idx] = scripted.OPPONENTS[name](v)
    for name, fn in ro.POOL_FAMILIES.items():
        idx = np.flatnonzero(env.kind == K[name])
        if len(idx):
            v = env.delayed_view(idx, P["delay"][idx])
            sub = {k: P[k][idx] for k in P}
            act[idx] = fn(v, ro.family_params(name, sub))
    sp = np.isin(env.kind, [K["self_arch"], K["self_cur"], K["def_arch"], K["def_cur"]])
    if sp.any():
        o1 = env.obs_r6(1)
        idx = np.flatnonzero(env.kind == K["self_cur"])
        if len(idx):
            act[idx] = cur_fn(o1[idx])
        for j in np.unique(env.arch_id[env.kind == K["self_arch"]]):
            idx = np.flatnonzero((env.kind == K["self_arch"]) & (env.arch_id == j))
            act[idx] = np_policy(archive[j], o1[idx])
        for j in np.unique(env.darch_id[env.kind == K["def_arch"]]):
            idx = np.flatnonzero((env.kind == K["def_arch"]) & (env.darch_id == j))
            act[idx] = np_policy(darchive[j], o1[idx])
    scr = ~sp
    act[scr] *= P["speed"][scr, None]
    act[scr] += rng.normal(0, 1, (int(scr.sum()), 2)) * P["noise"][scr, None]
    return np.clip(act, -1, 1)


def save_pickle(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def tree_np(t):
    """Recursively convert an MLX tree (dicts/lists of mx.array) to numpy."""
    if isinstance(t, dict):
        return {k: tree_np(v) for k, v in t.items()}
    if isinstance(t, (list, tuple)):
        return [tree_np(v) for v in t]
    if isinstance(t, mx.array):
        return np.array(t)
    return t


def tree_mx(t):
    if isinstance(t, dict):
        return {k: tree_mx(v) for k, v in t.items()}
    if isinstance(t, (list, tuple)):
        return [tree_mx(v) for v in t]
    if isinstance(t, np.ndarray):
        return mx.array(t)
    return t
