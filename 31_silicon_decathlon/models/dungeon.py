"""Dungeon navigator (05_dungeon/dungeon/policy.py RecurrentPolicy): MLP encoder -> GRU(128) cell ->
policy / value heads, one recurrent step. Checkpoint: solver_cur3 (PPO-GRU after 3 curriculum rounds,
the first/default policy of 05's demo.html).

  obs (B, 55) float32 (5x5 wall window, 5x5 goal window, prev-action one-hot, bumped flag), h (B, 128)
  -> logits (B, 4) over N/E/S/W, value (B,), h_next (B, 128)

GRU (05's own cell, PyTorch gate convention): r = sig(xr + hr), z = sig(xz + hz),
n = tanh(xn + r * hn), h' = (1 - z) * n + z * h, with x2h / h2h Linear layers (biases on both).
Dialects: MLX (reference), NumPy (baseline), torch (Core ML frontend only; plain matmuls).
Also vendored: 05's vectorised grid-maze env (dungeon/env.py: make_obs, VecMaze without novelty /
auto-reset) to run fixed held-out levels read from ../05_dungeon/data/test_*_500.npz.
"""
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

CKPT = "../05_dungeon/checkpoints/solver_cur3/weights.safetensors"
DATA = "../05_dungeon/data"
FAMILIES = ("dfs", "braid", "rooms", "caves", "spiral")   # caves, spiral never trained on
SIZE, VIEW = 13, 5
R = VIEW // 2
OBS_DIM = 2 * VIEW * VIEW + 4 + 1
HID = 128
MAX_STEPS = 200
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int64)  # N E S W


# ---------- vendored env (dungeon/env.py) ----------
def make_obs(walls_pad, pos, goal, prev_a, bumped):
    n = pos.shape[0]
    ar = np.arange(VIEW)
    rr = pos[:, 0, None, None] + ar[None, :, None]
    cc = pos[:, 1, None, None] + ar[None, None, :]
    win = walls_pad[np.arange(n)[:, None, None], rr, cc].astype(np.float32)
    gwin = np.zeros((n, VIEW, VIEW), np.float32)
    rel = goal - pos + R
    vis = (rel >= 0).all(1) & (rel < VIEW).all(1)
    gwin[np.nonzero(vis)[0], rel[vis, 0], rel[vis, 1]] = 1.0
    pa = np.zeros((n, 4), np.float32)
    has = prev_a >= 0
    pa[np.nonzero(has)[0], prev_a[has]] = 1.0
    return np.concatenate([win.reshape(n, -1), gwin.reshape(n, -1), pa, bumped[:, None].astype(np.float32)], 1)


class VecMaze:
    """n parallel fixed levels, one episode each (no auto-reset, no novelty bonus)."""

    def __init__(self, levels, max_steps=MAX_STEPS):
        n = len(levels["walls"])
        self.n, self.T = n, max_steps
        self.walls = levels["walls"].astype(np.uint8).copy()
        self.walls_pad = np.ones((n, SIZE + 2 * R, SIZE + 2 * R), np.uint8)
        self.walls_pad[:, R:R + SIZE, R:R + SIZE] = self.walls
        self.pos = levels["start"].astype(np.int64).copy()
        self.goal = levels["goal"].astype(np.int64).copy()
        self.t = np.zeros(n, np.int64)
        self.prev_a = -np.ones(n, np.int64)
        self.bumped = np.zeros(n, bool)
        self.done = np.zeros(n, bool)

    def obs(self):
        return make_obs(self.walls_pad, self.pos, self.goal, self.prev_a, self.bumped)

    def step(self, a):
        a = np.asarray(a, np.int64)
        live = ~self.done
        new = self.pos + DIRS[a]
        blocked = self.walls[np.arange(self.n), new[:, 0], new[:, 1]] == 1
        move = live & ~blocked
        self.pos[move] = new[move]
        self.bumped = live & blocked
        self.prev_a = np.where(live, a, self.prev_a)
        self.t += live
        success = live & (self.pos == self.goal).all(1)
        timeout = live & ~success & (self.t >= self.T)
        self.done |= success | timeout
        return self.obs(), success


def load_levels(per_family, start=0):
    """Interleaved fixed held-out levels: level i is family i % 5, index start + i // 5."""
    sets = [np.load(os.path.join(DATA, f"test_{f}_500.npz")) for f in FAMILIES]
    out = {k: [] for k in ("walls", "start", "goal")}
    for j in range(per_family):
        for s in sets:
            for k in out:
                out[k].append(s[k][start + j])
    return {k: np.stack(v) for k, v in out.items()}


# ---------- MLX ----------
class GRUCell(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.x2h = nn.Linear(d_in, 3 * d_h)
        self.h2h = nn.Linear(d_h, 3 * d_h)

    def __call__(self, x, h):
        xr, xz, xn = mx.split(self.x2h(x), 3, axis=-1)
        hr, hz, hn = mx.split(self.h2h(h), 3, axis=-1)
        r = mx.sigmoid(xr + hr)
        z = mx.sigmoid(xz + hz)
        n = mx.tanh(xn + r * hn)
        return (1 - z) * n + z * h


class RecurrentPolicy(nn.Module):
    def __init__(self, hidden=HID):
        super().__init__()
        self.enc1 = nn.Linear(OBS_DIM, hidden)
        self.enc2 = nn.Linear(hidden, hidden)
        self.gru = GRUCell(hidden, hidden)
        self.pi1 = nn.Linear(hidden, hidden)
        self.pi2 = nn.Linear(hidden, 4)
        self.v1 = nn.Linear(hidden, hidden)
        self.v2 = nn.Linear(hidden, 1)

    def __call__(self, obs, h):
        x = nn.relu(self.enc2(nn.relu(self.enc1(obs))))
        h = self.gru(x, h)
        return self.pi2(nn.tanh(self.pi1(h))), self.v2(nn.tanh(self.v1(h)))[..., 0], h


def load_mlx(path=CKPT):
    m = RecurrentPolicy()
    m.load_weights(path)
    m.eval()
    return m


def weights_numpy(path=CKPT):
    return {k: np.array(v, np.float32) for k, v in mx.load(path).items()}


# ---------- NumPy ----------
class NumpyPolicy:
    def __init__(self, path=CKPT):
        w = weights_numpy(path)
        self.L = {k[:-7]: (w[k].T.copy(), w[k[:-7] + ".bias"]) for k in w if k.endswith(".weight")}

    def _lin(self, name, x):
        W, b = self.L[name]
        return x @ W + b

    def __call__(self, obs, h):
        relu = lambda a: np.maximum(a, 0)
        sig = lambda a: 1.0 / (1.0 + np.exp(-a))
        x = relu(self._lin("enc2", relu(self._lin("enc1", obs))))
        gx, gh = self._lin("gru.x2h", x), self._lin("gru.h2h", h)
        r = sig(gx[:, :HID] + gh[:, :HID])
        z = sig(gx[:, HID:2 * HID] + gh[:, HID:2 * HID])
        n = np.tanh(gx[:, 2 * HID:] + r * gh[:, 2 * HID:])
        h2 = (1 - z) * n + z * h
        logits = self._lin("pi2", np.tanh(self._lin("pi1", h2)))
        value = self._lin("v2", np.tanh(self._lin("v1", h2)))[:, 0]
        return logits, value, h2


# ---------- torch (Core ML frontend only) ----------
def torch_net(path=CKPT):
    import torch
    import torch.nn as tnn
    w = weights_numpy(path)

    class Net(tnn.Module):
        def __init__(s):
            super().__init__()
            s.enc1, s.enc2 = tnn.Linear(OBS_DIM, HID), tnn.Linear(HID, HID)
            s.x2h, s.h2h = tnn.Linear(HID, 3 * HID), tnn.Linear(HID, 3 * HID)
            s.pi1, s.pi2 = tnn.Linear(HID, HID), tnn.Linear(HID, 4)
            s.v1, s.v2 = tnn.Linear(HID, HID), tnn.Linear(HID, 1)

        def forward(s, obs, h):
            x = torch.relu(s.enc2(torch.relu(s.enc1(obs))))
            gx, gh = s.x2h(x), s.h2h(h)
            r = torch.sigmoid(gx[:, :HID] + gh[:, :HID])
            z = torch.sigmoid(gx[:, HID:2 * HID] + gh[:, HID:2 * HID])
            n = torch.tanh(gx[:, 2 * HID:] + r * gh[:, 2 * HID:])
            h2 = (1 - z) * n + z * h
            return s.pi2(torch.tanh(s.pi1(h2))), s.v2(torch.tanh(s.v1(h2))).reshape(-1), h2

    net = Net().eval()
    sd = {k.replace("gru.", ""): torch.from_numpy(v.copy()) for k, v in w.items()}
    net.load_state_dict(sd)
    return net
