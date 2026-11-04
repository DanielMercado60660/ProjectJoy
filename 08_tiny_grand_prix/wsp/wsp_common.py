"""Shared helpers for the Round-6 weight-space-programming (WSP) experiment.

Weights format used everywhere in wsp/: an .npz with W1 (13,32), b1 (32,), W2 (32,32), b2 (32,),
W3 (32,2), b3 (2,), float32. Forward pass: h1 = tanh(obs @ W1 + b1); h2 = tanh(h1 @ W2 + b2);
action = tanh(h2 @ W3 + b3). Identical to racing/policy.py's flat-vector format (same order).
"""
import os
import sys

import numpy as np

WSP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WSP)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

NAMES = ["W1", "b1", "W2", "b2", "W3", "b3"]
SHAPES = [(13, 32), (32,), (32, 32), (32,), (32, 2), (2,)]
INPUTS = ["ray_m90", "ray_m60", "ray_m35", "ray_m18", "ray_m06", "ray_p06", "ray_p18", "ray_p35",
          "ray_p60", "ray_p90", "speed", "prev_steer", "prev_pedal"]


def theta_to_dict(theta):
    out, o = {}, 0
    for n, s in zip(NAMES, SHAPES):
        k = int(np.prod(s))
        out[n] = np.asarray(theta[o:o + k], np.float32).reshape(s)
        o += k
    assert o == len(theta)
    return out


def dict_to_theta(d):
    return np.concatenate([np.asarray(d[n], np.float32).reshape(-1) for n in NAMES]).astype(np.float32)


def load_weights(path):
    """Accepts a wsp npz (W1..b3) or an original checkpoint (theta + meta)."""
    d = np.load(path, allow_pickle=False)
    if "theta" in d.files:
        return theta_to_dict(d["theta"])
    for n, s in zip(NAMES, SHAPES):
        if n not in d.files:
            raise ValueError(f"{path}: missing array {n}")
        if tuple(d[n].shape) != s:
            raise ValueError(f"{path}: {n} has shape {d[n].shape}, expected {s}")
        if not np.all(np.isfinite(d[n])):
            raise ValueError(f"{path}: {n} has non-finite values")
    return {n: d[n].astype(np.float32) for n in NAMES}


def save_weights(path, d):
    np.savez(path, **{n: np.asarray(d[n], np.float32) for n in NAMES})


class WeightsController:
    """Controller with the same interface as racing.policy.MLPController; optional unit ablation:
    ablate = {"h1": {unit: value}, "h2": {unit: value}} replaces that unit's tanh output."""

    def __init__(self, d, ablate=None, name="wsp"):
        self.d = {k: np.asarray(v, np.float32) for k, v in d.items()}
        self.ablate = ablate or {}
        self.name = name
        self.record = None  # set to [] to record hidden activations

    def __call__(self, obs, env=None):
        d = self.d
        h1 = np.tanh(obs.astype(np.float32) @ d["W1"] + d["b1"])
        for u, val in self.ablate.get("h1", {}).items():
            h1[:, u] = val
        h2 = np.tanh(h1 @ d["W2"] + d["b2"])
        for u, val in self.ablate.get("h2", {}).items():
            h2[:, u] = val
        if self.record is not None:
            self.record.append((h1.copy(), h2.copy()))
        return np.tanh(h2 @ d["W3"] + d["b3"])


def dump_text(d, title="policy"):
    """Human/LLM-readable dump with labeled rows and columns (units 0-based)."""
    L = [f"# {title}: 13 -> 32 -> 32 -> 2 tanh MLP, float32, units are 0-based",
         "# h1 = tanh(obs @ W1 + b1); h2 = tanh(h1 @ W2 + b2); action = tanh(h2 @ W3 + b3)",
         "# action[0] = steer, action[1] = pedal", ""]
    f = lambda x: f"{x:+.4f}"
    L.append("## W1 (13 inputs x 32 h1 units): row = input, column j = h1[j]")
    L.append("input        " + " ".join(f"h1[{j}]".rjust(7) for j in range(32)))
    for i, nm in enumerate(INPUTS):
        L.append(nm.ljust(12) + " " + " ".join(f(x).rjust(7) for x in d["W1"][i]))
    L.append("")
    L.append("## b1 (32): " + " ".join(f"h1[{j}]={f(x)}" for j, x in enumerate(d["b1"])))
    L.append("")
    L.append("## W2 (32 h1 units x 32 h2 units): row i = h1[i], column j = h2[j]")
    L.append("from\\to   " + " ".join(f"h2[{j}]".rjust(7) for j in range(32)))
    for i in range(32):
        L.append(f"h1[{i}]".ljust(9) + " " + " ".join(f(x).rjust(7) for x in d["W2"][i]))
    L.append("")
    L.append("## b2 (32): " + " ".join(f"h2[{j}]={f(x)}" for j, x in enumerate(d["b2"])))
    L.append("")
    L.append("## W3 (32 h2 units x 2 outputs)")
    L.append("from      steer    pedal")
    for i in range(32):
        L.append(f"h2[{i}]".ljust(9) + " " + f(d["W3"][i, 0]).rjust(8) + " " + f(d["W3"][i, 1]).rjust(8))
    L.append("")
    L.append(f"## b3 (2): steer={f(d['b3'][0])} pedal={f(d['b3'][1])}")
    return "\n".join(L) + "\n"
