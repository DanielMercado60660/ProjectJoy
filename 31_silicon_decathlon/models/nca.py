"""Self-Healing Pixels NCA (copied from 10_self_healing_pixels/nca.py + damage.half), in three dialects:
MLX (reference), NumPy (baseline) and torch (Core ML conversion frontend only).

State (B, 72, 72, 16) NHWC: channels 0-2 premultiplied RGB, 3 alpha, 4-15 hidden.
One update step: perceive (identity + Sobel x/y per channel -> 48), per-cell MLP 48 -> 128 -> 16,
stochastic fire mask, alive masking (3x3 max-pooled alpha > 0.1 before and after).
The fire mask is an INPUT here ((B, 72, 72) float 0/1, fixed seed), so every athlete computes the same thing.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np

CKPT = "../10_self_healing_pixels/checkpoints/lizard_regen/model.safetensors"  # README: demo pair is lizard_regen/persist
CHANNELS, HIDDEN, FIRE_RATE, ALIVE_THRESH = 16, 128, 0.5, 0.1
TARGET_SIZE, TARGET_PAD = 40, 16
SIZE = TARGET_SIZE + 2 * TARGET_PAD  # 72: the demo grid
EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"


# ---------- target (copied from nca.load_target, lizard only) ----------
def load_target(size=TARGET_SIZE, pad=TARGET_PAD):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(EMOJI_FONT, 160)
    im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((0, 0), "\U0001F98E", font=font, embedded_color=True)
    im = im.crop(im.getbbox())
    im.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    a = np.asarray(canvas).astype(np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return np.pad(a, ((pad, pad), (pad, pad), (0, 0))).astype(np.float32)


def make_seed(h, w, n=1):
    x = np.zeros((n, h, w, CHANNELS), np.float32)
    x[:, h // 2, w // 2, 3:] = 1.0
    return x


def half_damage(h, w, side):
    """damage.half with a fixed side (0 left, 1 right, 2 top, 3 bottom)."""
    m = np.zeros((h, w), bool)
    if side == 0:
        m[:, : w // 2] = True
    elif side == 1:
        m[:, w // 2:] = True
    elif side == 2:
        m[: h // 2] = True
    else:
        m[h // 2:] = True
    return m


def fire_masks(n_steps, batch, seed, h=SIZE, w=SIZE):
    rng = np.random.default_rng(seed)
    return (rng.random((n_steps, batch, h, w)) <= FIRE_RATE).astype(np.float32)


# ---------- MLX (same module layout as the original, so weights load by name) ----------
def perception_kernel_np():
    ident = np.zeros((3, 3), np.float32)
    ident[1, 1] = 1
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8.0
    sy = sx.T.copy()
    w = np.zeros((3 * CHANNELS, 3, 3, CHANNELS), np.float32)  # (out, kh, kw, in)
    for k, f in enumerate([ident, sx, sy]):
        for c in range(CHANNELS):
            w[k * CHANNELS + c, :, :, c] = f
    return w


def _max_pool3(a):
    p = mx.pad(a, [(0, 0), (1, 1), (1, 1)], constant_values=-1e9)
    H, W = a.shape[1], a.shape[2]
    out = p[:, 0:H, 0:W]
    for dy in range(3):
        for dx in range(3):
            if dy or dx:
                out = mx.maximum(out, p[:, dy:dy + H, dx:dx + W])
    return out


def _alive(x):
    return _max_pool3(x[..., 3]) > ALIVE_THRESH


class NCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * CHANNELS, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, CHANNELS, bias=False)
        self._kernel = mx.array(perception_kernel_np())

    def __call__(self, x, fire_mask):
        """x (B,H,W,16), fire_mask (B,H,W) 0/1 of x's dtype."""
        pre = _alive(x)
        y = mx.conv2d(x, self._kernel.astype(x.dtype), padding=1)
        dx = self.fc2(nn.relu(self.fc1(y)))
        x = x + dx * fire_mask[..., None]
        life = mx.logical_and(pre, _alive(x))
        return x * life[..., None]


def load_mlx(path=CKPT, dtype=mx.float32):
    m = NCA()
    m.load_weights(path)
    if dtype != mx.float32:
        m.set_dtype(dtype)
    mx.eval(m.parameters())
    return m


def weights_numpy(path=CKPT):
    return {k: np.array(v, dtype=np.float32) for k, v in mx.load(path).items()}


# ---------- NumPy baseline ----------
class NumpyNCA:
    def __init__(self, path=CKPT):
        w = weights_numpy(path)
        self.w1t, self.b1 = np.ascontiguousarray(w["fc1.weight"].T), w["fc1.bias"]
        self.w2t = np.ascontiguousarray(w["fc2.weight"].T)

    @staticmethod
    def _alive(a):  # a (B,H,W) alpha
        p = np.pad(a, ((0, 0), (1, 1), (1, 1)), constant_values=-1e9)
        H, W = a.shape[1:]
        m = p[:, 0:H, 0:W].copy()
        for dy in range(3):
            for dx in range(3):
                if dy or dx:
                    np.maximum(m, p[:, dy:dy + H, dx:dx + W], out=m)
        return m > ALIVE_THRESH

    def __call__(self, x, fire_mask):
        B, H, W, C = x.shape
        pre = self._alive(x[..., 3])
        p = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        s = lambda dy, dx: p[:, dy:dy + H, dx:dx + W]
        # cross-correlation with the Sobel kernels of nca.perception_kernel (sx[dy,dx], sy = sx.T), /8
        gx = ((s(0, 2) - s(0, 0)) + 2 * (s(1, 2) - s(1, 0)) + (s(2, 2) - s(2, 0))) / 8.0
        gy = ((s(2, 0) - s(0, 0)) + 2 * (s(2, 1) - s(0, 1)) + (s(2, 2) - s(0, 2))) / 8.0
        y = np.concatenate([x, gx, gy], axis=-1).reshape(-1, 3 * C)
        h = np.maximum(y @ self.w1t + self.b1, 0)
        dx = (h @ self.w2t).reshape(B, H, W, C)
        x = x + dx * fire_mask[..., None]
        life = pre & self._alive(x[..., 3])
        return (x * life[..., None]).astype(np.float32)


# ---------- torch (Core ML frontend only) ----------
def torch_net(path=CKPT):
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F
    w = weights_numpy(path)
    k = torch.from_numpy(perception_kernel_np().transpose(0, 3, 1, 2).copy())  # (48,16,3,3) OIHW

    class Net(tnn.Module):
        def __init__(s):
            super().__init__()
            s.register_buffer("k", k)
            s.fc1 = tnn.Conv2d(3 * CHANNELS, HIDDEN, 1)
            s.fc2 = tnn.Conv2d(HIDDEN, CHANNELS, 1, bias=False)
            s.fc1.weight.data = torch.from_numpy(w["fc1.weight"][:, :, None, None].copy())
            s.fc1.bias.data = torch.from_numpy(w["fc1.bias"].copy())
            s.fc2.weight.data = torch.from_numpy(w["fc2.weight"][:, :, None, None].copy())

        def alive(s, x):
            return (F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > ALIVE_THRESH).float()

        def forward(s, x, fire_mask):  # NHWC in/out like MLX; fire_mask (B,H,W)
            x = x.permute(0, 3, 1, 2)
            pre = s.alive(x)
            y = F.conv2d(x, s.k, padding=1)
            dx = s.fc2(torch.relu(s.fc1(y)))
            x = x + dx * fire_mask.unsqueeze(1)
            x = x * (pre * s.alive(x))
            return x.permute(0, 2, 3, 1)

    return Net().eval()
