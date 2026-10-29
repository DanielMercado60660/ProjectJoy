"""Checkpoint save/load: model weights + optimizer state (safetensors) + JSON meta."""
import json
import os

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from gpt import GPT, GPTConfig


def save(path, model, opt=None, meta=None):
    os.makedirs(path, exist_ok=True)
    tmp = path + ".tmp_meta.json"
    model.save_weights(os.path.join(path, "model.safetensors"))
    if opt is not None:
        flat = dict(tree_flatten(opt.state))
        mx.save_safetensors(os.path.join(path, "optim.safetensors"), flat)
    meta = dict(meta or {})
    meta["config"] = model.config.to_dict()
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, os.path.join(path, "meta.json"))  # meta written last = checkpoint complete


def load_meta(path):
    with open(os.path.join(path, "meta.json")) as f:
        return json.load(f)


def load_model(path, dtype=mx.bfloat16):
    meta = load_meta(path)
    model = GPT(GPTConfig(**meta["config"]))
    model.load_weights(os.path.join(path, "model.safetensors"))
    model.compute_dtype = dtype
    mx.eval(model.parameters())
    return model, meta


def load_optimizer(path, opt, model):
    """opt must be built for `model`; we init it so the state tree exists, then overwrite."""
    opt.init(model.trainable_parameters())
    want = dict(tree_flatten(opt.state))
    got = mx.load(os.path.join(path, "optim.safetensors"))
    missing = set(want) - set(got)
    assert not missing, f"optimizer state keys missing from checkpoint: {sorted(missing)[:5]}"
    opt.state = tree_unflatten(list(got.items()))
