"""Neural Cellular Automaton core (MLX), after Mordvintsev et al., "Growing Neural Cellular Automata" (Distill 2020).

State: (B, H, W, 16) NHWC. Channels 0-2 = RGB (premultiplied by alpha), 3 = alpha, 4-15 = hidden.
Perception: identity + Sobel-x + Sobel-y per channel -> 48 features (fixed, not learned).
Update: per-cell MLP 48 -> 128 -> 16 (1x1 convs), last layer zero-initialised, stochastic fire mask,
alive masking via 3x3 max-pooled alpha > 0.1 before and after the update.
"""
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn

CHANNELS = 16
HIDDEN = 128
FIRE_RATE = 0.5
ALIVE_THRESH = 0.1
TARGET_SIZE = 40
TARGET_PAD = 16
EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
EMOJIS = {"lizard": "\U0001F98E", "butterfly": "\U0001F98B", "octopus": "\U0001F419", "mushroom": "\U0001F344"}


# ----------------------------------------------------------------------------- target
def load_target(name="lizard", size=TARGET_SIZE, pad=TARGET_PAD):
    """Render an Apple Color Emoji to a premultiplied RGBA float array of shape (size+2pad, size+2pad, 4).
    Falls back to a procedural pixel-art creature if the emoji font is unavailable."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(EMOJI_FONT, 160)  # Apple Color Emoji only has bitmap strikes (160 works)
        im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((0, 0), EMOJIS[name], font=font, embedded_color=True)
        im = im.crop(im.getbbox())
        im.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        a = np.asarray(canvas).astype(np.float32) / 255.0
    except Exception:  # pragma: no cover - fallback path
        a = procedural_creature(size)
    a[..., :3] *= a[..., 3:4]  # premultiply
    a = np.pad(a, ((pad, pad), (pad, pad), (0, 0)))
    return a.astype(np.float32)


def procedural_creature(size=TARGET_SIZE):
    """Simple symmetric pixel-art blob creature (fallback target, unpremultiplied RGBA)."""
    rng = np.random.default_rng(0)
    a = np.zeros((size, size, 4), np.float32)
    yy, xx = np.mgrid[:size, :size]
    c = size / 2 - 0.5
    body = ((xx - c) / (size * 0.35)) ** 2 + ((yy - c) / (size * 0.4)) ** 2 < 1
    a[body] = [0.3, 0.7, 0.3, 1.0]
    for ex in (c - 5, c + 5):
        eye = (xx - ex) ** 2 + (yy - c + 4) ** 2 < 6
        a[eye] = [1, 1, 1, 1]
    a[..., :3] += rng.normal(0, 0.03, a[..., :3].shape) * a[..., 3:4]
    return np.clip(a, 0, 1)


def make_seed(h, w, n=1):
    x = np.zeros((n, h, w, CHANNELS), np.float32)
    x[:, h // 2, w // 2, 3:] = 1.0
    return x


def to_rgb(x):
    """State (..., >=4) -> RGB composited on white, clipped to [0,1]."""
    x = np.asarray(x)
    rgb, a = x[..., :3], np.clip(x[..., 3:4], 0, 1)
    return np.clip(1.0 - a + rgb, 0, 1)


# ----------------------------------------------------------------------------- model
def perception_kernel():
    """Fixed conv weight of shape (48, 3, 3, 16) [MLX: out, kh, kw, in].
    Output channel layout: [identity(0..15), sobel_x(0..15), sobel_y(0..15)]."""
    ident = np.zeros((3, 3), np.float32)
    ident[1, 1] = 1
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8.0
    sy = sx.T.copy()
    w = np.zeros((3 * CHANNELS, 3, 3, CHANNELS), np.float32)
    for k, f in enumerate([ident, sx, sy]):
        for c in range(CHANNELS):
            w[k * CHANNELS + c, :, :, c] = f
    return mx.array(w)


def max_pool3(a):
    """3x3 max pool, stride 1, same size (pad with -inf). a: (B,H,W)."""
    p = mx.pad(a, [(0, 0), (1, 1), (1, 1)], constant_values=-1e9)
    H, W = a.shape[1], a.shape[2]
    out = p[:, 0:H, 0:W]
    for dy in range(3):
        for dx in range(3):
            if dy == 0 and dx == 0:
                continue
            out = mx.maximum(out, p[:, dy:dy + H, dx:dx + W])
    return out


def alive_mask(x):
    return max_pool3(x[..., 3]) > ALIVE_THRESH  # (B,H,W) bool


class NCA(nn.Module):
    def __init__(self, channels=CHANNELS, hidden=HIDDEN, fire_rate=FIRE_RATE):
        super().__init__()
        self.channels, self.hidden, self.fire_rate = channels, hidden, fire_rate
        self.fc1 = nn.Linear(3 * channels, hidden)
        self.fc2 = nn.Linear(hidden, channels, bias=False)
        self.fc2.weight = mx.zeros_like(self.fc2.weight)  # "do nothing" initial behaviour
        self._kernel = perception_kernel()  # leading underscore: not a trainable parameter

    def perceive(self, x):
        return mx.conv2d(x, self._kernel, padding=1)

    def __call__(self, x, fire_rate=None, fire_mask=None):
        fire_rate = self.fire_rate if fire_rate is None else fire_rate
        pre = alive_mask(x)
        y = self.perceive(x)
        dx = self.fc2(nn.relu(self.fc1(y)))
        if fire_mask is None:
            fire_mask = mx.random.uniform(shape=x.shape[:3]) <= fire_rate
        x = x + dx * fire_mask[..., None]
        post = alive_mask(x)
        life = mx.logical_and(pre, post)
        return x * life[..., None]

    def rollout(self, x, n, **kw):
        for _ in range(n):
            x = self(x, **kw)
        return x


# ----------------------------------------------------------------------------- checkpoints / export
def save_weights(model, path):
    model.save_weights(path)


def load_model(path):
    m = NCA()
    m.load_weights(path)
    mx.eval(m.parameters())
    return m


def export_json(model, target, path=None):
    """Weights + a deterministic self-check vector for the JS port."""
    w1 = np.array(model.fc1.weight)  # (hidden, 48)
    b1 = np.array(model.fc1.bias)
    w2 = np.array(model.fc2.weight)  # (16, hidden)
    rng = np.random.default_rng(0)
    h, w = target.shape[:2]
    x = np.zeros((1, h, w, CHANNELS), np.float32)
    cy, cx = h // 2, w // 2
    x[0, cy - 4:cy + 5, cx - 4:cx + 5] = rng.uniform(0, 1, (9, 9, CHANNELS)).astype(np.float32)
    # reference on the CPU stream: MLX GPU float32 matmuls on this machine differ by ~1e-3 relative
    with mx.stream(mx.cpu):
        y = np.array(model(mx.array(x), fire_mask=mx.ones((1, h, w), dtype=mx.bool_)))
    d = {
        "channels": CHANNELS, "hidden": int(w1.shape[0]), "fire_rate": model.fire_rate,
        "alive_thresh": ALIVE_THRESH, "height": h, "width": w,
        "w1": np.round(w1, 6).flatten().tolist(), "b1": np.round(b1, 6).tolist(),
        "w2": np.round(w2, 6).flatten().tolist(),
        "target": np.round(target, 3).flatten().tolist(),
        "check": {"cy": cy, "cx": cx, "in": np.round(x[0, cy - 4:cy + 5, cx - 4:cx + 5], 6).flatten().tolist(),
                  "out": np.round(y[0, cy - 6:cy + 7, cx - 6:cx + 7], 5).flatten().tolist()},
    }
    if path:
        with open(path, "w") as f:
            json.dump(d, f)
    return d
