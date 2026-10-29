"""Bounded, crash-safe generations. Only selected/ needs archiving/committing.

Write the inactive resume slot completely, then atomically change current.json.
A kill while saving cannot mix weights and optimizer from different generations.
"""
import json
import shutil
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from r5_data import json_write


def current(out):
    out = Path(out)
    return out / json.loads((out / "current.json").read_text())["slot"]


def save(out, model, opt, meta, rng, final=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    old = current(out).name if (out / "current.json").exists() else "resume_b"
    slot = "resume_b" if old == "resume_a" else "resume_a"
    dest = out / slot
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir()
    mx.eval(model.parameters(), opt.state, mx.random.state)
    model.save_weights(str(dest / "model.safetensors"))
    mx.save_safetensors(str(dest / "optim.safetensors"), dict(tree_flatten(opt.state)))
    mx.save_safetensors(str(dest / "rng.safetensors"),
                        {str(k): v for k, v in enumerate(mx.random.state)})
    json_write(dest / "meta.json", {**meta, "numpy_rng": rng.bit_generator.state})
    json_write(out / "current.json", dict(slot=slot, step=meta["step"]))
    if final:
        staging = out / "selected.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(dest, staging)
        selected = out / "selected"
        # current.json still points to a complete resume generation throughout this export.
        if selected.exists():
            shutil.rmtree(selected)
        staging.rename(selected)


def restore(path, model, opt, rng):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    model.load_weights(str(path / "model.safetensors"))
    opt.init(model.trainable_parameters())
    opt.state = tree_unflatten(list(mx.load(str(path / "optim.safetensors")).items()))
    state = mx.load(str(path / "rng.safetensors"))
    # mlx 0.32: _RandomState has no item assignment; update each key array in place
    # (round-5 runner fix, verified to reproduce the next draw exactly).
    for k in range(len(state)):
        mx.random.state[k][:] = state[str(k)]
    rng.bit_generator.state = meta["numpy_rng"]
    mx.eval(model.parameters(), opt.state, mx.random.state)
    return meta
