"""Small residual policy/value CNN in MLX, plus save/load helpers.

Input: (B, 6, 7, 2) NHWC, plane 0 = stones of the player to move, plane 1 = opponent.
Output: policy logits (B, 7), value (B,) in [-1, 1] for the player to move.
"""
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from c4 import W, H


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm(ch)

    def __call__(self, x):
        y = nn.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return nn.relu(x + y)


class PVNet(nn.Module):
    def __init__(self, channels=64, blocks=5, value_hidden=64):
        super().__init__()
        self.config = dict(channels=channels, blocks=blocks, value_hidden=value_hidden)
        self.stem = nn.Conv2d(2, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm(channels)
        self.blocks = [ResBlock(channels) for _ in range(blocks)]
        self.p_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.p_bn = nn.BatchNorm(2)
        self.p_fc = nn.Linear(2 * H * W, W)
        self.v_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm(1)
        self.v_fc1 = nn.Linear(H * W, value_hidden)
        self.v_fc2 = nn.Linear(value_hidden, 1)

    def __call__(self, x):
        x = nn.relu(self.stem_bn(self.stem(x)))
        for b in self.blocks:
            x = b(x)
        p = nn.relu(self.p_bn(self.p_conv(x)))
        p = self.p_fc(p.reshape(p.shape[0], -1))
        v = nn.relu(self.v_bn(self.v_conv(x)))
        v = nn.relu(self.v_fc1(v.reshape(v.shape[0], -1)))
        v = mx.tanh(self.v_fc2(v)).squeeze(-1)
        return p, v


def n_params(model):
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def make_evaluator(model):
    """Returns evaluate(feats np (b,H,W,2)) -> (logits np, values np), eval mode.
    Batches are padded to a power of two so the compiled graph is reused."""
    model.eval()

    @mx.compile
    def fwd(x):
        return model(x)

    def evaluate(feats):
        b = feats.shape[0]
        pb = 1 << max(3, (b - 1).bit_length())
        if pb != b:
            pad = np.zeros((pb, H, W, 2), np.float32)
            pad[:b] = feats
            feats = pad
        p, v = fwd(mx.array(feats))
        mx.eval(p, v)
        return np.array(p)[:b], np.array(v)[:b]
    return evaluate


def save_model(model, path_prefix, extra=None):
    """Writes <prefix>.safetensors (weights incl. BN stats) and <prefix>.json (config)."""
    model.save_weights(path_prefix + ".safetensors")
    meta = dict(config=model.config)
    if extra:
        meta.update(extra)
    with open(path_prefix + ".json", "w") as f:
        json.dump(meta, f, indent=1)


def load_model(path_prefix):
    with open(path_prefix + ".json") as f:
        meta = json.load(f)
    model = PVNet(**meta["config"])
    model.load_weights(path_prefix + ".safetensors")
    model.eval()
    return model, meta
