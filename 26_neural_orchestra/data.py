"""Procedural drum-pattern dataset for Neural Orchestra.

Every pattern is a 2-bar, 16th-note grid (32 steps) over 9 instruments, stored as three arrays:
  hits  [32, 9]  0/1 onsets
  vel   [32, 9]  velocity in (0, 1], 0 where no hit
  off   [32, 9]  microtiming offset in units of one 16th note, in [-0.5, 0.5], 0 where no hit

Styles are hand-written "drummer rules": a pool of kick/snare/hat options per style, plus
variation rules (ghost notes, hat subdivision, open-hat accents, ride swaps, crashes, bar-2
mutations, fills, swing and humanised timing/velocity). Patterns are de-duplicated on the hit
grid and then split 80/10/10 into train/val/test with a fixed seed, so no exact hit grid
appears in two splits.

Run:  ../.venv/bin/python data.py            -> data/drums.npz
"""
import argparse
import os

import numpy as np

INSTRUMENTS = ["kick", "snare", "clap", "closed_hat", "open_hat", "low_tom", "high_tom", "ride", "crash"]
SHORT = ["BD", "SD", "CP", "CH", "OH", "LT", "HT", "RD", "CR"]
KICK, SNARE, CLAP, CH, OH, LT, HT, RIDE, CRASH = range(9)
STYLES = ["rock", "funk", "house", "breakbeat", "hiphop", "bossa", "halftime", "reggaeton"]
TEMPO = {"rock": 110, "funk": 100, "house": 124, "breakbeat": 136, "hiphop": 90,
         "bossa": 128, "halftime": 140, "reggaeton": 96}
STEPS, N_INST = 32, 9
# base velocity per instrument (accent rules move it around)
BASE_VEL = np.array([0.95, 0.9, 0.85, 0.6, 0.7, 0.8, 0.8, 0.6, 0.85])


class Bar:
    """Mutable 16-step bar: dict-free numpy grid plus velocity scale per hit."""

    def __init__(self):
        self.h = np.zeros((16, N_INST), np.uint8)
        self.v = np.zeros((16, N_INST), np.float32)

    def put(self, inst, steps, vel=None):
        for s in steps:
            self.h[s % 16, inst] = 1
            self.v[s % 16, inst] = BASE_VEL[inst] if vel is None else vel


def _choice(rng, options, p=None):
    return options[rng.choice(len(options), p=p)]


def _hats(rng, b, mode, inst=CH, accent=True):
    if mode == "8ths":
        steps = range(0, 16, 2)
    elif mode == "16ths":
        steps = range(16)
    elif mode == "quarters":
        steps = range(0, 16, 4)
    elif mode == "offbeat":
        steps = range(2, 16, 4)
    else:
        return
    for s in steps:
        v = BASE_VEL[inst]
        if accent:
            v *= 1.0 if s % 4 == 0 else (0.85 if s % 2 == 0 else 0.6)
        b.put(inst, [s], v)


def _ghosts(rng, b, density, inst=SNARE, avoid=()):
    for s in range(16):
        if s in avoid or b.h[s, inst]:
            continue
        if rng.random() < density:
            b.put(inst, [s], rng.uniform(0.2, 0.4))


def _open_hat(rng, b, candidates, p):
    for s in candidates:
        if rng.random() < p:
            b.h[s, CH] = 0
            b.v[s, CH] = 0
            b.put(OH, [s])


def style_bar(style, rng):
    """One bar of the style's groove, plus per-pattern feel parameters."""
    b = Bar()
    feel = {"swing": 0.0, "human": rng.uniform(0.01, 0.05), "lag": {}, "fill_p": 0.3, "crash_p": 0.3}
    if style == "rock":
        b.put(KICK, _choice(rng, [[0, 8], [0, 8, 10], [0, 7, 8], [0, 2, 8, 10], [0, 8, 11], [0, 3, 8, 10], [0, 6, 8]]))
        b.put(SNARE, [4, 12])
        if rng.random() < 0.25:
            _hats(rng, b, _choice(rng, ["8ths", "quarters"]), inst=RIDE)
        else:
            _hats(rng, b, _choice(rng, ["8ths", "16ths", "quarters"], p=[0.65, 0.25, 0.1]))
        _ghosts(rng, b, rng.uniform(0, 0.08), avoid=(4, 12))
        _open_hat(rng, b, [14], 0.25)
        feel.update(fill_p=0.5, crash_p=0.5)
    elif style == "funk":
        b.put(KICK, _choice(rng, [[0, 3, 10], [0, 6, 10], [0, 3, 7, 10], [0, 2, 7, 10, 13], [0, 6, 9, 14], [0, 10, 11], [0, 3, 6, 10, 11]]))
        b.put(SNARE, [4, 12])
        _hats(rng, b, _choice(rng, ["16ths", "8ths"], p=[0.7, 0.3]))
        _ghosts(rng, b, rng.uniform(0.15, 0.4), avoid=(4, 12))
        _open_hat(rng, b, [6, 10, 14, 2], rng.uniform(0.1, 0.4))
        feel.update(swing=rng.uniform(0.0, 0.15), fill_p=0.3, crash_p=0.2)
    elif style == "house":
        b.put(KICK, [0, 4, 8, 12])
        if rng.random() < 0.15:
            b.put(KICK, [14], 0.6)
        b.put(CLAP if rng.random() < 0.8 else SNARE, [4, 12])
        if rng.random() < 0.6:
            _hats(rng, b, _choice(rng, ["16ths", "8ths"]), accent=True)
        if rng.random() < 0.85:
            for s in (2, 6, 10, 14):
                b.h[s, CH] = 0
                b.v[s, CH] = 0
            b.put(OH, [2, 6, 10, 14])
        if rng.random() < 0.25:
            _hats(rng, b, _choice(rng, ["offbeat", "8ths"]), inst=RIDE)
        if rng.random() < 0.3:
            b.put(CLAP, [rng.integers(13, 16)], 0.5)
        _ghosts(rng, b, rng.uniform(0, 0.08), inst=CLAP, avoid=(4, 12))
        if rng.random() < 0.3:  # percussive tom hits
            b.put(_choice(rng, [LT, HT]), rng.choice(16, rng.integers(1, 3), replace=False), 0.6)
        feel.update(human=rng.uniform(0.0, 0.015), swing=_choice(rng, [0.0, 0.0, 0.1]), fill_p=0.15, crash_p=0.25)
    elif style == "breakbeat":
        b.put(KICK, _choice(rng, [[0, 2, 10], [0, 2, 10, 11], [0, 10], [0, 2, 7, 10], [0, 6, 10]]))
        b.put(SNARE, _choice(rng, [[4, 12], [4, 7, 9, 12, 15], [4, 9, 12], [4, 12, 14], [4, 7, 12]]))
        if rng.random() < 0.3:
            _hats(rng, b, "8ths", inst=RIDE)
        else:
            _hats(rng, b, _choice(rng, ["8ths", "16ths"], p=[0.7, 0.3]))
        _ghosts(rng, b, rng.uniform(0.05, 0.2), avoid=(4, 12))
        _open_hat(rng, b, [10, 14], 0.2)
        feel.update(swing=rng.uniform(0.0, 0.12), fill_p=0.5, crash_p=0.35)
    elif style == "hiphop":
        b.put(KICK, _choice(rng, [[0, 3, 10], [0, 7, 10], [0, 10, 11], [0, 2, 7, 10], [0, 3, 6, 10, 14], [0, 9, 10]]))
        b.put(SNARE, [4, 12])
        if rng.random() < 0.4:
            b.put(CLAP, [4, 12], 0.7)
        _hats(rng, b, _choice(rng, ["8ths", "16ths", "quarters"], p=[0.6, 0.3, 0.1]))
        _ghosts(rng, b, rng.uniform(0, 0.08), avoid=(4, 12))
        _open_hat(rng, b, [6, 14], 0.1)
        feel.update(swing=rng.uniform(0.15, 0.35), lag={SNARE: rng.uniform(0.03, 0.12), CLAP: 0.06},
                    fill_p=0.1, crash_p=0.1)
    elif style == "bossa":
        kick = [0, 3, 4, 7, 8, 11, 12, 15]
        if rng.random() < 0.4:
            kick = [s for s in kick if s % 4 == 0 or rng.random() < 0.7]
        b.put(KICK, kick, 0.75)
        # cross-stick clave on the snare channel (velocity kept soft)
        b.put(SNARE, _choice(rng, [[0, 3, 6, 10, 13], [0, 3, 6, 10, 12], [2, 6, 10, 13], [0, 3, 6, 9, 12]]), 0.55)
        _hats(rng, b, _choice(rng, ["8ths", "16ths"]), inst=RIDE if rng.random() < 0.5 else CH)
        b.put(CH, [4, 12], 0.4)  # hi-hat foot
        if rng.random() < 0.3:
            b.put(LT, [rng.integers(0, 16)], 0.6)
        feel.update(fill_p=0.15, crash_p=0.1)
    elif style == "halftime":
        b.put(KICK, _choice(rng, [[0, 10], [0, 3, 10], [0, 7, 10, 11], [0, 14], [0, 6, 11], [0, 2, 11]]))
        b.put(SNARE, [8])
        if rng.random() < 0.3:
            b.put(CLAP, [8], 0.8)
        _hats(rng, b, _choice(rng, ["8ths", "16ths", "quarters"], p=[0.45, 0.4, 0.15]))
        _ghosts(rng, b, rng.uniform(0, 0.1), avoid=(8,))
        _open_hat(rng, b, [6, 14], 0.2)
        feel.update(swing=rng.uniform(0, 0.08), fill_p=0.3, crash_p=0.3)
    elif style == "reggaeton":
        b.put(KICK, [0, 4, 8, 12])
        b.put(SNARE, _choice(rng, [[3, 6, 11, 14], [3, 6, 10, 14], [3, 7, 11, 14], [3, 6, 11]]))
        if rng.random() < 0.4:
            b.put(CLAP, [s for s in range(16) if b.h[s, SNARE]], 0.7)
        if rng.random() < 0.7:
            _hats(rng, b, _choice(rng, ["8ths", "16ths", "offbeat"]))
        _ghosts(rng, b, rng.uniform(0, 0.08), avoid=(3, 6, 11, 14))
        _open_hat(rng, b, [2, 10, 14], 0.15)
        if rng.random() < 0.3:
            b.put(_choice(rng, [LT, HT]), rng.choice(16, rng.integers(1, 3), replace=False), 0.6)
        feel.update(swing=rng.uniform(0, 0.08), fill_p=0.2, crash_p=0.2)
    else:
        raise ValueError(style)
    return b, feel


def _fill(rng, h, v):
    """Overwrite the end of bar 2 with a snare/tom fill."""
    length = _choice(rng, [4, 8], p=[0.6, 0.4])
    start = STEPS - length
    kind = _choice(rng, ["snare", "toms", "mixed"])
    h[start:, [SNARE, CLAP, CH, OH, RIDE, LT, HT]] = 0
    v[start:, [SNARE, CLAP, CH, OH, RIDE, LT, HT]] = 0
    density = _choice(rng, [1, 2])  # every 16th or every 8th
    for i, s in enumerate(range(start, STEPS, density)):
        frac = i / max(1, (length // density) - 1)
        if kind == "snare":
            inst = SNARE
        elif kind == "toms":
            inst = HT if frac < 0.5 else LT
        else:
            inst = _choice(rng, [SNARE, HT, LT])
        h[s, inst] = 1
        v[s, inst] = 0.55 + 0.4 * frac  # crescendo


def make_pattern(style, rng):
    b, feel = style_bar(style, rng)
    h = np.concatenate([b.h, b.h]).astype(np.uint8)
    v = np.concatenate([b.v, b.v]).astype(np.float32)
    # bar-2 mutations: nudge one kick, re-roll some ghost notes
    if rng.random() < 0.35:
        ks = np.flatnonzero(h[16:, KICK]) + 16
        ks = ks[ks % 4 != 0]
        if len(ks):
            s = rng.choice(ks)
            h[s, KICK] = 0
            v[s, KICK] = 0
            t = s + _choice(rng, [-1, 1])
            if 16 < t < 32:
                h[t, KICK] = 1
                v[t, KICK] = BASE_VEL[KICK]
    if rng.random() < 0.25:
        s = 16 + rng.integers(0, 16)
        if not h[s, KICK]:
            h[s, KICK] = 1
            v[s, KICK] = 0.7
    if rng.random() < feel["fill_p"]:
        _fill(rng, h, v)
    if rng.random() < feel["crash_p"]:
        h[0, CRASH] = 1
        v[0, CRASH] = BASE_VEL[CRASH]
        h[0, CH] = 0
        v[0, CH] = 0
    # humanise: velocity jitter, swing on odd 16ths, per-instrument lag, timing jitter
    v = np.where(h > 0, np.clip(v + rng.normal(0, 0.05, v.shape), 0.1, 1.0), 0).astype(np.float32)
    off = np.zeros((STEPS, N_INST), np.float32)
    odd = (np.arange(STEPS) % 2 == 1)[:, None]
    off += odd * feel["swing"]
    for inst, lag in feel["lag"].items():
        off[:, inst] += lag
    off += rng.normal(0, feel["human"], off.shape)
    off = np.where(h > 0, np.clip(off, -0.5, 0.5), 0).astype(np.float32)
    return h, v, off


def generate(n_per_style=1100, seed=0, max_tries=20):
    """Draw patterns until each style has n_per_style distinct hit grids (or tries run out)."""
    rng = np.random.default_rng(seed)
    H, V, O, S = [], [], [], []
    for si, style in enumerate(STYLES):
        seen = set()
        for _ in range(n_per_style * max_tries):
            h, v, o = make_pattern(style, rng)
            key = h.tobytes()
            if key in seen:
                continue
            seen.add(key)
            H.append(h); V.append(v); O.append(o); S.append(si)
            if len(seen) == n_per_style:
                break
    H, V, O, S = np.stack(H), np.stack(V), np.stack(O), np.array(S, np.int32)
    # de-duplicate on the hit grid (keep first occurrence, any style)
    _, keep = np.unique(H.reshape(len(H), -1), axis=0, return_index=True)
    keep = np.sort(keep)
    H, V, O, S = H[keep], V[keep], O[keep], S[keep]
    perm = np.random.default_rng(seed + 1).permutation(len(H))
    n = len(H)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    split = np.empty(n, np.int8)  # 0 train, 1 val, 2 test
    split[perm[:n_tr]] = 0
    split[perm[n_tr:n_tr + n_va]] = 1
    split[perm[n_tr + n_va:]] = 2
    return dict(hits=H, vel=V, off=O, style=S, split=split)


def load(path="data/drums.npz"):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def subset(d, which):
    idx = {"train": 0, "val": 1, "test": 2}[which]
    m = d["split"] == idx
    return {k: d[k][m] for k in ("hits", "vel", "off", "style")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_style", type=int, default=1100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/drums.npz")
    a = ap.parse_args()
    d = generate(a.n_per_style, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, **d)
    n = len(d["hits"])
    print(f"{n} distinct hit grids (target {a.n_per_style} per style)")
    for si, s in enumerate(STYLES):
        m = d["style"] == si
        print(f"  {s:10s} {m.sum():5d}  train/val/test = "
              + "/".join(str(((d['split'] == k) & m).sum()) for k in range(3))
              + f"  mean hits {d['hits'][m].sum((1, 2)).mean():.1f}")


if __name__ == "__main__":
    main()
