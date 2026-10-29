"""Atomic single-generation checkpoints and wall-clock guards; NumPy only."""
import json
import hashlib
import os
from pathlib import Path
import time
import numpy as np


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
    os.replace(temporary, path)


def save_checkpoint(path, payload):
    """No pickle. One NPZ contains typed trees, weights, optimizer and RNGs.

    Only latest working state and validation-selected state are retained; a
    failed write cannot destroy the previous generation. Never writes siblings.
    """
    arrays = {}

    def encode(x):
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            key = f"a{len(arrays)}"
            arrays[key] = np.asarray(x)
            if arrays[key].dtype.hasobject:
                raise TypeError("object arrays are not checkpointable")
            return {"array": key}
        if isinstance(x, dict):
            return {"dict": [[k, encode(v)] for k, v in x.items()]}
        if isinstance(x, (list, tuple)):
            return {"tuple" if isinstance(x, tuple) else "list": [encode(v) for v in x]}
        if isinstance(x, np.generic):
            return x.item()
        return x

    template = encode(payload)
    arrays["template"] = np.asarray(json.dumps(template, allow_nan=False))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as f:
        np.savez(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as z:
        def decode(x):
            if not isinstance(x, dict):
                return x
            if "array" in x:
                return z[x["array"]].copy()
            if "dict" in x:
                return {k: decode(v) for k, v in x["dict"]}
            if "tuple" in x:
                return tuple(decode(v) for v in x["tuple"])
            return [decode(v) for v in x["list"]]
        return decode(json.loads(str(z["template"])))


class Deadline:
    def __init__(self, max_minutes):
        if max_minutes <= 0:
            raise ValueError("--max-minutes must be positive")
        self.start = time.monotonic()
        self.end = self.start + max_minutes*60

    @property
    def elapsed(self):
        return time.monotonic()-self.start

    @property
    def remaining(self):
        return self.end-time.monotonic()

    def check(self, reserve=0):
        if self.remaining <= reserve:
            raise TimeoutError("wall-clock guard")
