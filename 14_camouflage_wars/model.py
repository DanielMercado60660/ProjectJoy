"""Tiny U-Net segmenter in MLX (NHWC)."""
import json
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout, pytorch_compatible=True)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout, pytorch_compatible=True)

    def __call__(self, x):
        x = nn.gelu(self.n1(self.c1(x)))
        return nn.gelu(self.n2(self.c2(x)))


def pool(x):
    B, H, W, C = x.shape
    return x.reshape(B, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))


def up(x):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, 2, W, 2, C))
    return x.reshape(B, 2 * H, 2 * W, C)


class UNet(nn.Module):
    def __init__(self, width=24, depth=4):
        super().__init__()
        chs = [width * min(2 ** i, 8) for i in range(depth + 1)]  # e.g. 24,48,96,192,192
        self.config = dict(width=width, depth=depth)
        self.enc = [Block(3 if i == 0 else chs[i - 1], chs[i]) for i in range(depth + 1)]
        self.dec = [Block(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(depth))]
        self.head = nn.Conv2d(chs[0], 1, 1)

    def __call__(self, x):
        skips = []
        for i, b in enumerate(self.enc):
            x = b(x if i == 0 else pool(x))
            skips.append(x)
        x = skips.pop()
        for b in self.dec:
            x = b(mx.concatenate([up(x), skips.pop()], axis=-1))
        return self.head(x)[..., 0]  # logits (B,H,W)


def n_params(model):
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def save(model, path):
    model.save_weights(path + ".safetensors")
    with open(path + ".json", "w") as f:
        json.dump(model.config, f)


def load(path):
    with open(path + ".json") as f:
        cfg = json.load(f)
    m = UNet(**cfg)
    m.load_weights(path + ".safetensors")
    m.eval()
    return m
