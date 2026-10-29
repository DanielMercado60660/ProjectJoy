"""Small mistake head and complete two-slot resumable state. Claude runs this module."""
import json
import shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from r5_core import learner_features

CONFIGS = [dict(width=w, lr=lr, wd=wd, seed=s)
           for w, lr, wd in ((64, .001, .001), (128, .0003, .001), (128, .001, .01))
           for s in (51, 73)]


class Head(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.a = nn.Linear(117, width)
        self.b = nn.Linear(width, width)
        self.c = nn.Linear(width, 1)

    def __call__(self, x):
        assert x.ndim == 2 and x.shape[1] == 117
        return self.c(nn.relu(self.b(nn.relu(self.a(x))))).squeeze(-1)


def loss(model, x, y):
    z = model(x)
    # Stable BCE; no sqrt/norm at zero and no class weighting (calibrated target).
    return mx.mean(mx.logaddexp(mx.zeros_like(z), z) - y*z)


class Trainer:
    def __init__(self, directory, config, mean, std):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config, self.mean, self.std = config, mean, std
        self.models, self.opts, self.grad = [], [], []
        self.steps = [0] * len(CONFIGS)
        self.rng = np.random.default_rng(50241089)
        self.best = float("inf")
        for c in CONFIGS:
            mx.random.seed(c["seed"])
            model = Head(c["width"])
            mx.eval(model.parameters())
            opt = optim.AdamW(learning_rate=c["lr"], weight_decay=c["wd"])
            opt.init(model.trainable_parameters())
            self.models.append(model); self.opts.append(opt)
            self.grad.append(nn.value_and_grad(model, loss))
        self.slot = 0
        active = self.directory / "active.json"
        if active.exists():
            self.slot = json.loads(active.read_text())["slot"]
            path = self.directory / ("slot" + str(self.slot))
            meta = json.loads((path / "meta.json").read_text())
            assert meta["config"] == config, "Resume config/data mismatch"
            assert meta["candidate_configs"] == CONFIGS, "Resume candidate config mismatch"
            self.steps, self.best = meta["steps"], meta["best"]
            if self.best is None:
                self.best = float("inf")
            self.rng.bit_generator.state = meta["numpy_rng"]
            for i, (model, opt) in enumerate(zip(self.models, self.opts)):
                model.load_weights(str(path / f"model{i}.safetensors"))
                opt.state = tree_unflatten(list(mx.load(str(path / f"opt{i}.safetensors")).items()))
            saved = mx.load(str(path / "rng.safetensors"))
            for k in range(len(mx.random.state)):
                mx.random.state[k][:] = saved[str(k)]  # MLX 0.32: never assign state[k]
            mx.eval(mx.random.state)
            # Restore the selection paired with the last complete snapshot if a
            # SIGKILL interrupted exporting a newer selection.
            selected = self.directory.parent / "selected"
            for name in ("head.safetensors", "head.json", "normalizer.npz"):
                source = path / ("selected_"+name)
                if source.exists():
                    selected.mkdir(exist_ok=True)
                    shutil.copy2(source, selected/name)

    def train(self, i, x, y, steps, batch, deadline):
        import time
        latest = None
        for _ in range(steps):
            if time.monotonic() >= deadline:
                break
            idx = self.rng.integers(len(y), size=batch)
            val, grads = self.grad[i](self.models[i], mx.array(x[idx]), mx.array(y[idx]))
            self.opts[i].update(self.models[i], grads)
            mx.eval(self.models[i].parameters(), self.opts[i].state, val)
            latest = float(val.item())
            if not np.isfinite(latest):
                raise FloatingPointError(f"Nonfinite head loss at {i}:{self.steps[i]}")
            self.steps[i] += 1
        return latest

    def predict(self, i, x):
        chunks = []
        for start in range(0, len(x), 1024):
            p = mx.sigmoid(self.models[i](mx.array(x[start:start+1024])))
            mx.eval(p)
            chunks.append(np.array(p))
        return np.concatenate(chunks)

    def save(self):
        self.slot = 1 - self.slot
        path = self.directory / ("slot" + str(self.slot))
        path.mkdir(exist_ok=True)
        for i, (model, opt) in enumerate(zip(self.models, self.opts)):
            model.save_weights(str(path / f"model{i}.safetensors"))
            mx.save_safetensors(str(path / f"opt{i}.safetensors"), dict(tree_flatten(opt.state)))
        mx.save_safetensors(str(path / "rng.safetensors"),
                            {str(i): v for i, v in enumerate(mx.random.state)})
        meta = dict(config=self.config, candidate_configs=CONFIGS, steps=self.steps,
                    best=self.best if np.isfinite(self.best) else None, numpy_rng=self.rng.bit_generator.state)
        (path / "meta.json").write_text(json.dumps(meta))
        selected = self.directory.parent / "selected"
        for name in ("head.safetensors", "head.json", "normalizer.npz"):
            if (selected/name).exists():
                shutil.copy2(selected/name, path/("selected_"+name))
        tmp = self.directory / "active.tmp"
        tmp.write_text(json.dumps(dict(slot=self.slot)))
        tmp.replace(self.directory / "active.json")

    def select(self, i, score, directory):
        if score >= self.best:
            return False
        self.best = score
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.models[i].save_weights(str(path / "head_next.safetensors"))
        (path / "head_next.safetensors").replace(path / "head.safetensors")
        np.savez(path / "normalizer.npz", mean=self.mean, std=self.std)
        (path / "head.json").write_text(json.dumps(dict(model=CONFIGS[i], candidate=i,
                     step=self.steps[i], validation_score=score, config=self.config), indent=2))
        return True


def predictor(directory):
    path = Path(directory)
    meta = json.loads((path / "head.json").read_text())
    model = Head(meta["model"]["width"])
    model.load_weights(str(path / "head.safetensors")); model.eval()
    norm = np.load(path / "normalizer.npz")
    mean, std = norm["mean"], norm["std"]

    def predict(x):
        out = mx.sigmoid(model(mx.array((learner_features(x)-mean)/std)))
        mx.eval(out)
        p = np.asarray(out)
        assert p.shape == (len(x),) and np.isfinite(p).all()
        return p
    return predict
