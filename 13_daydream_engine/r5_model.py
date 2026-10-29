"""MLX-only neural paths. Imported only by Claude's training/evaluation commands."""
from pathlib import Path
import uuid

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from r5_core import (walls_step, pair_features, geometry_pair_features, antisymmetric_correction,
                     write_json, read_json)


class HybridEnsemble(nn.Module):
    """Independent batched MLPs; shared pair law within each member.

    Input (members, batch, 12). The walls integrator is exact. A bounded,
    antisymmetric edge correction is added once per recorded frame. This is
    permutation equivariant, but does not encode the true ball-contact law.
    """
    def __init__(self, members=5, hidden=32, seed=51010, features="raw", depth=2):
        super().__init__()
        self.members = members
        self.hidden = hidden
        self.seed = seed
        if features not in ("raw", "geometry") or depth < 1:
            raise ValueError("unknown features or invalid depth")
        self.features, self.depth = features, depth
        self.input_dim = 12 if features == "raw" else 24
        dims = [self.input_dim] + [hidden] * depth + [4]
        weights = [[] for _ in range(len(dims) - 1)]
        for e in range(members):
            mx.random.seed(seed + e)
            for k, (nin, nout) in enumerate(zip(dims[:-1], dims[1:])):
                weights[k].append(mx.random.normal((nin, nout)) * (nin ** -0.5) * (0.05 if k == len(dims) - 2 else 1.0))
        self.weights = [mx.stack(w) for w in weights]
        self.biases = [mx.zeros((members, 1, d)) for d in dims[1:]]

    def __call__(self, state):
        assert state.ndim == 3 and state.shape[0] == self.members and state.shape[-1] == 12
        base = walls_step(state, mx)
        features, gate = (pair_features if self.features == "raw" else geometry_pair_features)(state, base, mx)
        h = features.reshape(self.members, -1, self.input_dim)
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = mx.matmul(h, w) + b
            h = nn.silu(h) if k < len(self.weights) - 1 else mx.tanh(h)
        # Head-on launches can exchange >3 velocity units; do not cap the learned
        # collision impulse below the simulator's normal operating range.
        scale = mx.array([0.08, 0.08, 4.0, 4.0], dtype=state.dtype)
        edge = h.reshape(self.members, state.shape[1], 3, 3, 4) * scale
        correction = antisymmetric_correction(edge, gate, mx).reshape(state.shape)
        return base + correction


def save_checkpoint(directory, model, optimizer, metadata):
    """Immutable generation + atomic pointer: an interrupted save cannot mix states."""
    directory = Path(directory)
    generation = f"step_{metadata['step']:08d}_{uuid.uuid4().hex[:8]}"
    dest = directory / generation
    dest.mkdir(parents=True, exist_ok=False)
    mx.eval(model.parameters(), optimizer.state)
    model.save_weights(str(dest / "model.safetensors"))
    mx.save_safetensors(str(dest / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    # All training randomness is NumPy; MLX randomness is used only at init.
    write_json(dest / "state.json", metadata)
    write_json(directory / "latest.json", {"generation": generation})
    print(f"CHECKPOINT step={metadata['step']} path={dest}", flush=True)
    return dest


def checkpoint_path(path):
    path = Path(path)
    return path / read_json(path / "latest.json")["generation"] if (path / "latest.json").exists() else path


def load_checkpoint(path, optimizer=None):
    source = checkpoint_path(path)
    meta = read_json(source / "state.json")
    c = meta["config"]
    model = HybridEnsemble(c["members"], c["hidden"], c["seed"], c.get("features", "raw"), c.get("depth", 2))
    model.load_weights(str(source / "model.safetensors"))
    if optimizer is not None:
        optimizer.state = tree_unflatten(list(mx.load(str(source / "optimizer.safetensors")).items()))
    mx.eval(model.parameters())
    return model, meta


class Predictor:
    def __init__(self, model, chunk=512):
        self.model, self.chunk = model, chunk
        # Captured parameters are explicitly registered; no frozen-weight compile.
        self.step = mx.compile(lambda s: self.model(s), inputs=[self.model.state])

    def rollout(self, init, horizon, final=False, budget=None):
        outputs = []
        for start in range(0, len(init), self.chunk):
            s = mx.array(np.broadcast_to(init[start:start + self.chunk],
                                        (self.model.members, min(self.chunk, len(init) - start), 12)).copy(),
                         dtype=mx.float32)
            history = [np.array(s)] if not final else None
            for _ in range(horizon):
                if budget:
                    budget.check()
                s = self.step(s)
                mx.eval(s)
                if not final:
                    history.append(np.array(s))
            outputs.append(np.array(s) if final else np.stack(history, axis=2))
        return np.concatenate(outputs, axis=1)
