"""R6 LeWM S1: frame cache and window sampler.

Cache parts live in episodes/cache/lewm_<part>/ (gitignored with episodes/):
  frames.u8   (N, 64, 64, 3) uint8 memmap, box-downsampled from the stored 96 px frames (as P2 does)
  meta.npz    act (N, 6) float32 = cmd_exec [vx, wz, neck_p, head_p, head_y, head_r] scaled to ~[-1, 1];
              yaw (N,) float32 truth yaw (EVALUATION-ONLY probe label, never a training signal);
              ep_start, ep_len, ep_cfg (config number), ep_src (source id), ep_split (0 train, 1 dev)
  episodes.jsonl  one row per episode (path, config, task, source)

    ../bin/gpu_run ../.venv/bin/python lewm/data.py build base     # index + dagger 1-3 + recovery
    ../bin/gpu_run ../.venv/bin/python lewm/data.py build noisy    # episodes/noisy_index.jsonl
    ../bin/gpu_run ../.venv/bin/python lewm/data.py build dev      # episodes/p2_dev (80 dev demos)
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
CACHE = HERE / "episodes" / "cache"
IMG = 64
ACT_SCALE = np.array([0.4, 0.6, *([0.704, 0.704, 0.896, 0.1984])], np.float32)  # vx, wz, head bounds
ACT_COLS = [0, 2, 3, 4, 5, 6]                       # cmd_exec columns: vx, wz, 4 head deltas
HOLDOUT_CFG = 270                                   # train configs >= 270 never train LeWM
SOURCES = {"base": ["index", "dagger_index", "dagger2_index", "dagger3_index", "recovery_index"],
           "noisy": ["noisy_index"]}


def episode_rows(part):
    if part == "dev":
        out = []
        for p in sorted((HERE / "episodes" / "p2_dev").rglob("*.npz")):
            cid = p.name.split("_")[1]
            out.append({"path": str(p.relative_to(HERE)), "config": cid, "task": "set_lamp", "src": "p2_dev"})
        return out
    out, seen = [], set()
    for name in SOURCES[part]:
        for line in open(HERE / "episodes" / f"{name}.jsonl"):
            r = json.loads(line)
            if "error" in r or r["path"] in seen:
                continue
            assert r.get("split", "train") == "train" and r["config"].startswith("train-"), r["config"]
            seen.add(r["path"])
            out.append({"path": r["path"], "config": r["config"], "task": r["task"], "src": name})
    return out


def load_episode(path):
    z = np.load(HERE / path)
    fr = z["frames"]
    small = np.stack([np.asarray(Image.fromarray(f).resize((IMG, IMG), Image.Resampling.BOX)) for f in fr])
    act = np.nan_to_num(z["cmd_exec"][:, ACT_COLS]) / ACT_SCALE
    return small, act.astype(np.float32), z["truth"][:, 2].astype(np.float32)


def build(part, procs=4):
    rows = episode_rows(part)
    d = CACHE / f"lewm_{part}"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "frames.u8.tmp"
    f = open(tmp, "wb")
    act, yaw, starts, lens = [], [], [], []
    off = 0
    with Pool(procs) as pool:
        for i, (fr, a, y) in enumerate(pool.imap(load_episode, [r["path"] for r in rows], chunksize=2)):
            f.write(np.ascontiguousarray(fr).tobytes())
            act.append(a)
            yaw.append(y)
            starts.append(off)
            lens.append(len(fr))
            off += len(fr)
            if i % 200 == 0:
                print(f"{part}: {i}/{len(rows)} episodes, {off} frames", flush=True)
    f.close()
    os.replace(tmp, d / "frames.u8")
    cfg = [int(r["config"].rsplit("-", 1)[1]) for r in rows]
    src_names = sorted({r["src"] for r in rows})
    np.savez(d / "meta.npz", act=np.concatenate(act), yaw=np.concatenate(yaw), ep_start=np.array(starts, np.int64),
             ep_len=np.array(lens, np.int64), ep_cfg=np.array(cfg, np.int32),
             ep_src=np.array([src_names.index(r["src"]) for r in rows], np.int16),
             ep_split=np.full(len(rows), 1 if part == "dev" else 0, np.int8), src_names=np.array(src_names))
    with open(d / "episodes.jsonl", "w") as g:
        for r, s, n in zip(rows, starts, lens):
            g.write(json.dumps({**r, "start": s, "len": n}) + "\n")
    print(json.dumps({"part": part, "episodes": len(rows), "frames": off,
                      "gb": round(off * IMG * IMG * 3 / 1e9, 2)}), flush=True)


class Part:
    """One cache part, memory-mapped."""

    def __init__(self, part):
        d = CACHE / f"lewm_{part}"
        m = np.load(d / "meta.npz")
        self.act, self.yaw = m["act"], m["yaw"]
        self.ep_start, self.ep_len, self.ep_cfg = m["ep_start"], m["ep_len"], m["ep_cfg"]
        self.ep_src, self.src_names = m["ep_src"], list(m["src_names"])
        self.frames = np.memmap(d / "frames.u8", np.uint8, "r", shape=(int(self.ep_len.sum()), IMG, IMG, 3))
        self.name = part


class Windows:
    """Windows of T frames at frameskip fs: frames s, s+fs, ..., s+(T-1)fs; action block i =
    act[s+i*fs : s+(i+1)*fs] flattened (the commands executed between frame i and frame i+1)."""

    def __init__(self, parts, T=4, fs=5, select=None, seed=0):
        self.parts, self.T, self.fs = parts, T, fs
        self.rng = np.random.default_rng(seed)
        idx = []
        for pi, p in enumerate(parts):
            for e in range(len(p.ep_len)):
                if select is not None and not select(p, e):
                    continue
                n = int(p.ep_len[e]) - (T - 1) * fs
                if n > 0:
                    idx.append((pi, int(p.ep_start[e]), n, e))
        self.eps = idx
        self.counts = np.array([n for _, _, n, _ in idx], np.int64)
        self.cum = np.cumsum(self.counts)

    def __len__(self):
        return int(self.cum[-1]) if len(self.cum) else 0

    def locate(self, flat):
        k = np.searchsorted(self.cum, flat, side="right")
        off = flat - np.concatenate([[0], self.cum])[k]
        return k, off

    def get(self, flat):
        """flat window ids -> (frames (B,T,64,64,3) uint8, actions (B,T,fs*6), ep index list)."""
        k, off = self.locate(np.asarray(flat))
        F = np.empty((len(k), self.T, IMG, IMG, 3), np.uint8)
        A = np.empty((len(k), self.T, self.fs * 6), np.float32)
        for b, (kk, o) in enumerate(zip(k, off)):
            pi, st, _, _ = self.eps[kk]
            p = self.parts[pi]
            s = st + int(o)
            ticks = s + self.fs * np.arange(self.T)
            F[b] = p.frames[ticks]
            for i in range(self.T):
                blk = p.act[s + i * self.fs: s + (i + 1) * self.fs]
                if len(blk) < self.fs:   # last block may run off the episode; it is never used as input
                    blk = np.concatenate([blk, np.zeros((self.fs - len(blk), 6), np.float32)])
                A[b, i] = blk.reshape(-1)
        return F, A, k

    def sample(self, B, rng=None):
        rng = self.rng if rng is None else rng
        return self.get(rng.integers(0, len(self), B))


def train_select(p, e):
    return p.name != "dev" and int(p.ep_cfg[e]) < HOLDOUT_CFG


def heldout_select(p, e):
    return p.name != "dev" and int(p.ep_cfg[e]) >= HOLDOUT_CFG


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
