"""R8 Gate C: 224 px window sampler for LeWM training / gates (episodes/r8_rep).

Episodes: index rows without error. Sets: "train" = split train, configs 0000-0269 (all recorded plans);
"heldout" = split train, configs 0270-0299; "dev" = the 80 p2_dev rooms (split != train).
Actions: cmd_exec columns [vx, wz, neck_p, head_p, head_y, head_r] (vy is always 0, done ends the
episode), z-scored with the training set's per-column mean / std (the official LeWM recipe normalises
each action column), then concatenated over the frameskip: block i of a window = the commands executed
between frame i and frame i+1 (= R6's windows).
Frames are memory-mapped from each episode's frames224.npy; nothing is decoded or cached in RAM.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EP = ROOT / "episodes" / "r8_rep"
ACT_COLS = [0, 2, 3, 4, 5, 6]
HOLDOUT_CFG = 270


def read_index():
    """Index rows; a line still being appended by a running recorder (not yet complete JSON) is skipped."""
    rows = []
    for line in open(EP / "index.jsonl"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def episode_list(which, task="set_lamp"):
    """which: train | heldout | dev; task: set_lamp (the gated set; the protocol trains and gates on set_lamp
    only) or go_and_return (held-out rows reported separately) or None (any)."""
    out = []
    for r in read_index():
        if "error" in r or (task is not None and r["task"] != task):
            continue
        sp = r.get("split", "train")
        if which == "dev":
            ok = sp != "train"
        elif which == "train":
            ok = sp == "train" and r["cfg"] < HOLDOUT_CFG
        elif which == "heldout":
            ok = sp == "train" and r["cfg"] >= HOLDOUT_CFG
        else:
            raise ValueError(which)
        if ok:
            out.append({"name": r["name"], "path": r["path"], "ticks": r["ticks"], "cfg": r["cfg"], "task": r["task"],
                        "k": r["k"], "noisy": r.get("noisy", False)})
    return sorted(out, key=lambda r: r["name"])


def act_norm(eps):
    A = np.concatenate([np.nan_to_num(np.load(ROOT / e["path"] / "meta.npz")["cmd_exec"][:, ACT_COLS]) for e in eps])
    # sd floored at 0.01 so a column the bot never moves (exactly 0 in training) cannot explode elsewhere
    return A.mean(0).astype(np.float32), np.maximum(A.std(0), 0.01).astype(np.float32)


class Episodes:
    def __init__(self, eps, norm):
        self.eps = eps
        self.mu, self.sd = np.asarray(norm[0], np.float32), np.asarray(norm[1], np.float32)
        self._fr = [None] * len(eps)
        self.act = []
        for e in eps:
            a = np.nan_to_num(np.load(ROOT / e["path"] / "meta.npz")["cmd_exec"][:, ACT_COLS])
            self.act.append(((a - self.mu) / self.sd).astype(np.float32))

    def frames(self, i):
        if self._fr[i] is None:
            self._fr[i] = np.load(ROOT / self.eps[i]["path"] / "frames224.npy", mmap_mode="r")
        return self._fr[i]


class Windows:
    """Windows of T frames at frameskip fs (ticks s, s+fs, ...); action block i = act[s+i*fs : s+(i+1)*fs]."""

    def __init__(self, E, T=4, fs=5):
        self.E, self.T, self.fs = E, T, fs
        n = np.array([max(0, len(a) - (T - 1) * fs) for a in E.act], np.int64)
        self.ep_ids = np.nonzero(n > 0)[0]
        self.counts = n[self.ep_ids]
        self.cum = np.cumsum(self.counts)

    def __len__(self):
        return int(self.cum[-1]) if len(self.cum) else 0

    def locate(self, flat):
        flat = np.asarray(flat)
        k = np.searchsorted(self.cum, flat, side="right")
        off = flat - np.concatenate([[0], self.cum])[k]
        return self.ep_ids[k], off

    def get(self, flat, frames=True):
        e_ids, off = self.locate(flat)
        B, T, fs = len(e_ids), self.T, self.fs
        F = np.empty((B, T, 224, 224, 3), np.uint8) if frames else None
        A = np.zeros((B, T, fs * 6), np.float32)
        for b, (e, s) in enumerate(zip(e_ids, off)):
            s = int(s)
            if frames:
                F[b] = self.E.frames(e)[s + fs * np.arange(T)]
            a = self.E.act[e]
            for i in range(T):
                blk = a[s + i * fs: s + (i + 1) * fs]
                A[b, i, :len(blk) * 6] = blk.reshape(-1)       # the last block may run off the end (never an input)
        return F, A, e_ids


def frame_index(E):
    """(episode, tick) for every frame of every episode."""
    return [(i, t) for i, a in enumerate(E.act) for t in range(len(a))]
