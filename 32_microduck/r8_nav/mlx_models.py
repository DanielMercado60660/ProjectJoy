"""MLX ports of the published ViNT and NoMaD navigation models (visualnav-transformer, MIT; weights from
data/pretrained/*/ *_weights.npz written by r8_nav/weights.py). Pure MLX + numpy at runtime; no torch.

Shared piece: EfficientNet-b0 as efficientnet_pytorch builds it with EfficientNet.from_name(..., in_channels=3|6):
TF-"SAME" convolutions with STATIC padding computed for a 224 x 224 input whatever the real input size is (read back
from the pickled ViNT object: stem (0,1,0,1), stride-2 k3 (0,1,0,1), stride-2 k5 (1,2,1,2), stride-1 symmetric),
swish everywhere, squeeze-excite, identity skip on the repeated blocks. ViNT keeps BatchNorm (eps 1e-3, running
stats); NoMaD swaps every BatchNorm for GroupNorm(C // 16 groups, eps 1e-5).

ViNT (context 5, 85x64 input, 512-d tokens): obs encoder per frame (6 frames, oldest first), an obs+goal encoder on
the 6-channel [current frame, goal] stack, sinusoidal positions, 4 pre-norm transformer layers (4 heads, GELU,
FF 2048), flatten 7x512 -> MLP 512-256-128-64-32 (ReLU after each), distance = Linear(32, 1), actions =
Linear(32, 5 x 4) -> cumsum of (dx, dy) and unit (cos, sin).
NoMaD (context 3, 96x96, 256-d): the same encoders with GroupNorm, 4 pre-norm layers, a goal mask (1 = goal token
masked out of attention and of the mean, the mean rescaled by 5/4), mean-pooled 256-d context; distance =
MLP 256-64-16-1; actions = 8 normalized deltas from a ConditionalUnet1D (down dims 64/128/256, FiLM-free additive
conditioning on [256-d diffusion-step embedding, context]) sampled with a 10-step DDPM (squaredcos_cap_v2 betas,
epsilon prediction, x0 clipped to [-1, 1], fixed-small variance), then unnormalized with ACTION_STATS
(dx in [-2.5, 5], dy in [-4, 4]) and cumsummed into waypoints.
Waypoints of both models are in the robot frame (x forward, y left) in units of the training data's metric waypoint
spacing (one 4 Hz step of the robot's typical motion); deployment scales them by max_v / rate.
"""
import math
import os
from pathlib import Path

# MLX on this M5 GPU runs float32 matmuls in TF32 unless MLX_ENABLE_TF32=0 (measured: 8e-4 relative error on a plain
# matmul, 2-4e-3 on the models' outputs vs the torch reference; with it off the GPU matches torch to ~1e-6, see
# results/r8/nav/parity.json). Full float32 is the runtime setting; it must be set before the first matmul.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
VINT_NPZ = HERE / "data" / "pretrained" / "vint" / "vint_weights.npz"
NOMAD_NPZ = HERE / "data" / "pretrained" / "nomad" / "nomad_weights.npz"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
ACTION_STATS = {"min": np.array([-2.5, -4.0], np.float32), "max": np.array([5.0, 4.0], np.float32)}

# EfficientNet-b0 stages: (repeats, kernel, stride, expand, in, out); se_ratio 0.25 everywhere
B0_STAGES = [(1, 3, 1, 1, 32, 16), (2, 3, 2, 6, 16, 24), (2, 5, 2, 6, 24, 40), (3, 3, 2, 6, 40, 80),
             (3, 5, 1, 6, 80, 112), (4, 5, 2, 6, 112, 192), (1, 3, 1, 6, 192, 320)]


def same_pad(size, k, s):
    """efficientnet_pytorch Conv2dStaticSamePadding: (before, after) for a static input size."""
    o = math.ceil(size / s)
    p = max((o - 1) * s + k - size, 0)
    return p // 2, p - p // 2


def b0_layout(image_size=224):
    """Per block: dict(k, s, expand, cin, cout, pad=(top, bottom, left, right), skip); and the stem padding."""
    stem_pad = same_pad(image_size, 3, 2)
    size = math.ceil(image_size / 2)
    blocks = []
    for r, k, s, e, cin, cout in B0_STAGES:
        for i in range(r):
            stride = s if i == 0 else 1
            c_in = cin if i == 0 else cout
            p = same_pad(size, k, stride)
            # skip: efficientnet_pytorch compares the decoded stride (a list on the stage's first block) with 1, so
            # only repeated blocks skip; every first block has cin != cout in b0 anyway
            blocks.append(dict(k=k, s=stride, expand=e, cin=c_in, cout=cout, pad=(p[0], p[1], p[0], p[1]),
                               skip=(i > 0)))
            size = math.ceil(size / stride)
    return (stem_pad[0], stem_pad[1], stem_pad[0], stem_pad[1]), blocks


# --------------------------------------------------------------------------------------- primitives
def swish(x):
    return x * mx.sigmoid(x)


def mish(x):
    return x * mx.tanh(mx.logaddexp(x, mx.zeros_like(x)))


def gelu(x):
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def conv2d(x, w, stride=1, pad=(0, 0, 0, 0), groups=1, b=None):
    """x NHWC; w MLX layout (O, kh, kw, I/groups); pad = (top, bottom, left, right) zeros."""
    if any(pad):
        x = mx.pad(x, [(0, 0), (pad[0], pad[1]), (pad[2], pad[3]), (0, 0)])
    y = mx.conv2d(x, w, stride=stride, padding=0, groups=groups)
    return y if b is None else y + b


def layer_norm(x, w, b, eps=1e-5):
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean(mx.square(x - mu), axis=-1, keepdims=True)
    return (x - mu) * mx.rsqrt(var + eps) * w + b


def group_norm_nhwc(x, groups, w, b, eps=1e-5):
    n, h, wd, c = x.shape
    g = x.reshape(n, h, wd, groups, c // groups)
    mu = mx.mean(g, axis=(1, 2, 4), keepdims=True)
    var = mx.mean(mx.square(g - mu), axis=(1, 2, 4), keepdims=True)
    g = (g - mu) * mx.rsqrt(var + eps)
    return g.reshape(n, h, wd, c) * w + b


def group_norm_nlc(x, groups, w, b, eps=1e-5):
    n, length, c = x.shape
    g = x.reshape(n, length, groups, c // groups)
    mu = mx.mean(g, axis=(1, 3), keepdims=True)
    var = mx.mean(mx.square(g - mu), axis=(1, 3), keepdims=True)
    g = (g - mu) * mx.rsqrt(var + eps)
    return g.reshape(n, length, c) * w + b


def linear(x, w, b=None):
    y = x @ w.T
    return y if b is None else y + b


# --------------------------------------------------------------------------------------- weights
def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _conv_w(t):
    """torch conv weight (O, I, kh, kw) -> MLX (O, kh, kw, I); conv1d (O, I, k) -> (O, k, I)."""
    if t.ndim == 4:
        return mx.array(np.ascontiguousarray(t.transpose(0, 2, 3, 1)))
    if t.ndim == 3:
        return mx.array(np.ascontiguousarray(t.transpose(0, 2, 1)))
    raise ValueError(t.shape)


class EfficientNetB0:
    """extract_features + global average pool (the models' use of efficientnet_pytorch), eval mode."""

    def __init__(self, sd, prefix, norm="bn"):
        self.norm = norm
        self.stem_pad, self.blocks = b0_layout(224)
        g = lambda name: sd[prefix + name]  # noqa: E731
        self.stem_w = _conv_w(g("_conv_stem.weight"))
        self.n0 = self._norm(sd, prefix + "_bn0")
        self.bw = []
        for i, blk in enumerate(self.blocks):
            p = f"_blocks.{i}."
            d = {}
            if blk["expand"] != 1:
                d["expand_w"] = _conv_w(g(p + "_expand_conv.weight"))
                d["n0"] = self._norm(sd, prefix + p + "_bn0")
            d["dw_w"] = _conv_w(g(p + "_depthwise_conv.weight"))
            d["n1"] = self._norm(sd, prefix + p + "_bn1")
            d["se_r_w"] = mx.array(g(p + "_se_reduce.weight")[:, :, 0, 0])
            d["se_r_b"] = mx.array(g(p + "_se_reduce.bias"))
            d["se_e_w"] = mx.array(g(p + "_se_expand.weight")[:, :, 0, 0])
            d["se_e_b"] = mx.array(g(p + "_se_expand.bias"))
            d["proj_w"] = _conv_w(g(p + "_project_conv.weight"))
            d["n2"] = self._norm(sd, prefix + p + "_bn2")
            self.bw.append(d)
        self.head_w = _conv_w(g("_conv_head.weight"))
        self.n1 = self._norm(sd, prefix + "_bn1")

    def _norm(self, sd, name):
        w, b = sd[name + ".weight"], sd[name + ".bias"]
        if self.norm == "bn":   # fold the running statistics: y = x * scale + shift
            scale = w / np.sqrt(sd[name + ".running_var"] + 1e-3)
            shift = b - sd[name + ".running_mean"] * scale
            return ("bn", mx.array(scale.astype(np.float32)), mx.array(shift.astype(np.float32)))
        return ("gn", len(w) // 16, mx.array(w), mx.array(b))

    @staticmethod
    def _apply(n, x):
        if n[0] == "bn":
            return x * n[1] + n[2]
        return group_norm_nhwc(x, n[1], n[2], n[3])

    def __call__(self, x):
        """x: (N, H, W, C) normalized image(s) -> (N, 1280) pooled features."""
        x = swish(self._apply(self.n0, conv2d(x, self.stem_w, 2, self.stem_pad)))
        for blk, d in zip(self.blocks, self.bw):
            inp = x
            if blk["expand"] != 1:
                x = swish(self._apply(d["n0"], conv2d(x, d["expand_w"])))
            x = swish(self._apply(d["n1"], conv2d(x, d["dw_w"], blk["s"], blk["pad"], groups=x.shape[-1])))
            s = mx.mean(x, axis=(1, 2))
            s = swish(linear(s, d["se_r_w"], d["se_r_b"]))
            s = linear(s, d["se_e_w"], d["se_e_b"])
            x = mx.sigmoid(s)[:, None, None, :] * x
            x = self._apply(d["n2"], conv2d(x, d["proj_w"]))
            if blk["skip"]:
                x = x + inp
        x = swish(self._apply(self.n1, conv2d(x, self.head_w)))
        return mx.mean(x, axis=(1, 2))


class EncoderLayer:
    """torch.nn.TransformerEncoderLayer(batch_first=True, norm_first=True, activation='gelu'), eval mode."""

    def __init__(self, sd, prefix, nhead):
        a = lambda n: mx.array(sd[prefix + n])  # noqa: E731
        self.nhead = nhead
        self.in_w, self.in_b = a("self_attn.in_proj_weight"), a("self_attn.in_proj_bias")
        self.out_w, self.out_b = a("self_attn.out_proj.weight"), a("self_attn.out_proj.bias")
        self.l1_w, self.l1_b, self.l2_w, self.l2_b = a("linear1.weight"), a("linear1.bias"), a("linear2.weight"), a("linear2.bias")
        self.n1 = (a("norm1.weight"), a("norm1.bias"))
        self.n2 = (a("norm2.weight"), a("norm2.bias"))

    def attn(self, x, key_mask=None):
        n, t, e = x.shape
        h = self.nhead
        qkv = linear(x, self.in_w, self.in_b)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(n, t, h, e // h).transpose(0, 2, 1, 3)
        k = k.reshape(n, t, h, e // h).transpose(0, 2, 1, 3)
        v = v.reshape(n, t, h, e // h).transpose(0, 2, 1, 3)
        s = (q * (1.0 / math.sqrt(e // h))) @ k.transpose(0, 1, 3, 2)
        if key_mask is not None:                       # (n, t) True = this key is padding
            s = mx.where(key_mask[:, None, None, :], mx.array(-1e30, s.dtype), s)
        p = mx.softmax(s, axis=-1)
        o = (p @ v).transpose(0, 2, 1, 3).reshape(n, t, e)
        return linear(o, self.out_w, self.out_b)

    def __call__(self, x, key_mask=None):
        x = x + self.attn(layer_norm(x, *self.n1), key_mask)
        y = layer_norm(x, *self.n2)
        return x + linear(gelu(linear(y, self.l1_w, self.l1_b)), self.l2_w, self.l2_b)


def preprocess(images, size):
    """uint8 RGB frames (N, h, w, 3) already cropped to 4:3 and resized to `size` = (W, H) -> normalized NHWC."""
    x = np.asarray(images, np.float32) / 255.0
    assert x.shape[1:3] == (size[1], size[0]), (x.shape, size)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


# --------------------------------------------------------------------------------------- ViNT
class ViNT:
    context = 5
    size = (85, 64)        # (W, H)
    n_way = 5

    def __init__(self, path=VINT_NPZ, sd=None):
        sd = load_npz(path) if sd is None else sd
        self.obs_enc = EfficientNetB0(sd, "obs_encoder.", "bn")
        self.goal_enc = EfficientNetB0(sd, "goal_encoder.", "bn")
        a = lambda n: mx.array(sd[n])  # noqa: E731
        self.c_obs = (a("compress_obs_enc.weight"), a("compress_obs_enc.bias"))
        self.c_goal = (a("compress_goal_enc.weight"), a("compress_goal_enc.bias"))
        self.pos = a("decoder.positional_encoding.pos_enc")[0]        # (7, 512)
        self.layers = [EncoderLayer(sd, f"decoder.sa_decoder.layers.{i}.", 4) for i in range(4)]
        self.out = [(a(f"decoder.output_layers.{i}.weight"), a(f"decoder.output_layers.{i}.bias")) for i in range(5)]
        self.dist = (a("dist_predictor.0.weight"), a("dist_predictor.0.bias"))
        self.act = (a("action_predictor.0.weight"), a("action_predictor.0.bias"))

    def encode_obs(self, x):
        """(N, 64, 85, 3) normalized frames -> (N, 512) tokens (per frame; context-independent, so cacheable)."""
        return linear(self.obs_enc(x), *self.c_obs)

    def encode_goal(self, cur, goal):
        """current frame and goal image (N, 64, 85, 3) each -> (N, 512) obs+goal token."""
        return linear(self.goal_enc(mx.concatenate([cur, goal], axis=-1)), *self.c_goal)

    def head(self, obs_tokens, goal_token):
        """obs_tokens (N, 6, 512) oldest first; goal_token (N, 512) -> distance (N,), waypoints (N, 5, 4)."""
        x = mx.concatenate([obs_tokens, goal_token[:, None, :]], axis=1) + self.pos[None]
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        for w, b in self.out:
            x = mx.maximum(linear(x, w, b), 0)
        dist = linear(x, *self.dist)[:, 0]
        a = linear(x, *self.act).reshape(-1, self.n_way, 4)
        xy = mx.cumsum(a[..., :2], axis=1)
        ang = a[..., 2:]
        ang = ang / mx.maximum(mx.sqrt(mx.sum(ang * ang, axis=-1, keepdims=True)), 1e-12)
        return dist, mx.concatenate([xy, ang], axis=-1)

    def __call__(self, obs, goal):
        """obs (N, 6, 64, 85, 3) oldest first, goal (N, 64, 85, 3) -> (dist, waypoints)."""
        n = obs.shape[0]
        tok = self.encode_obs(obs.reshape(-1, *obs.shape[2:])).reshape(n, self.context + 1, -1)
        return self.head(tok, self.encode_goal(obs[:, -1], goal))


# --------------------------------------------------------------------------------------- NoMaD
def ddpm_schedule(n=10, max_beta=0.999):
    """diffusers DDPMScheduler(num_train_timesteps=n, beta_schedule='squaredcos_cap_v2') in float32."""
    f = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
    betas = np.array([min(1 - f((i + 1) / n) / f(i / n), max_beta) for i in range(n)], np.float32)
    alphas = (1.0 - betas).astype(np.float32)
    return betas, np.cumprod(alphas, dtype=np.float32)


class UNet1D:
    """diffusion_policy ConditionalUnet1D(input_dim=2, global_cond_dim=256, down_dims=[64,128,256], kernel 3,
    8 groups, cond_predict_scale=False), NLC layout."""

    def __init__(self, sd, prefix="noise_pred_net."):
        self.sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        s = self.sd
        a = lambda n: mx.array(s[n])  # noqa: E731
        self.step = (a("diffusion_step_encoder.1.weight"), a("diffusion_step_encoder.1.bias"),
                     a("diffusion_step_encoder.3.weight"), a("diffusion_step_encoder.3.bias"))
        self.down = [(self._res(f"down_modules.{i}.0."), self._res(f"down_modules.{i}.1."),
                      self._conv(f"down_modules.{i}.2.conv") if i < 2 else None) for i in range(3)]
        self.mid = [self._res("mid_modules.0."), self._res("mid_modules.1.")]
        self.up = [(self._res(f"up_modules.{i}.0."), self._res(f"up_modules.{i}.1."),
                    self._convT(f"up_modules.{i}.2.conv")) for i in range(2)]
        self.final_block = self._block("final_conv.0.")
        self.final = self._conv("final_conv.1")

    def _conv(self, name):
        return (_conv_w(self.sd[name + ".weight"]), mx.array(self.sd[name + ".bias"]))

    def _convT(self, name):
        """ConvTranspose1d(C, C, 4, stride 2, padding 1) as zero-insertion + a flipped-kernel conv: torch weight
        (I, O, k) -> conv weight (O, k, I) with k reversed."""
        w = self.sd[name + ".weight"]
        return (mx.array(np.ascontiguousarray(w[:, :, ::-1].transpose(1, 2, 0))), mx.array(self.sd[name + ".bias"]))

    def _block(self, p):
        return (self._conv(p + "block.0"), mx.array(self.sd[p + "block.1.weight"]), mx.array(self.sd[p + "block.1.bias"]))

    def _res(self, p):
        r = {"b0": self._block(p + "blocks.0."), "b1": self._block(p + "blocks.1."),
             "cond": (mx.array(self.sd[p + "cond_encoder.1.weight"]), mx.array(self.sd[p + "cond_encoder.1.bias"]))}
        if p + "residual_conv.weight" in self.sd:
            r["res"] = self._conv(p + "residual_conv")
        return r

    @staticmethod
    def _run_block(blk, x):
        (w, b), gw, gb = blk
        y = mx.conv1d(mx.pad(x, [(0, 0), (1, 1), (0, 0)]), w) + b
        return mish(group_norm_nlc(y, 8, gw, gb))

    def _run_res(self, r, x, cond):
        out = self._run_block(r["b0"], x)
        out = out + linear(mish(cond), *r["cond"])[:, None, :]
        out = self._run_block(r["b1"], out)
        res = x if "res" not in r else mx.conv1d(x, r["res"][0]) + r["res"][1]
        return out + res

    @staticmethod
    def _down(c, x):
        w, b = c
        return mx.conv1d(mx.pad(x, [(0, 0), (1, 1), (0, 0)]), w, stride=2) + b

    @staticmethod
    def _upT(c, x):
        w, b = c
        n, length, ch = x.shape
        z = mx.zeros((n, length, 2, ch), x.dtype)
        z = mx.concatenate([x[:, :, None, :], z[:, :, 1:, :]], axis=2).reshape(n, 2 * length, ch)[:, : 2 * length - 1]
        z = mx.pad(z, [(0, 0), (2, 2), (0, 0)])                 # k - 1 - padding = 2 on both sides
        return mx.conv1d(z, w) + b

    def step_embedding(self, t):
        half = 128
        e = math.log(10000) / (half - 1)
        freqs = mx.exp(mx.arange(half, dtype=mx.float32) * -e)
        arg = t.astype(mx.float32)[:, None] * freqs[None, :]
        emb = mx.concatenate([mx.sin(arg), mx.cos(arg)], axis=-1)
        w1, b1, w2, b2 = self.step
        return linear(mish(linear(emb, w1, b1)), w2, b2)

    def __call__(self, sample, t, global_cond):
        """sample (N, 8, 2), t (N,) int, global_cond (N, 256) -> predicted noise (N, 8, 2)."""
        cond = mx.concatenate([self.step_embedding(t), global_cond], axis=-1)
        x = sample
        h = []
        for r1, r2, down in self.down:
            x = self._run_res(r2, self._run_res(r1, x, cond), cond)
            h.append(x)
            if down is not None:
                x = self._down(down, x)
        for r in self.mid:
            x = self._run_res(r, x, cond)
        for r1, r2, up in self.up:
            x = mx.concatenate([x, h.pop()], axis=-1)
            x = self._run_res(r2, self._run_res(r1, x, cond), cond)
            x = self._upT(up, x)
        x = self._run_block(self.final_block, x)
        return mx.conv1d(x, self.final[0]) + self.final[1]


class NoMaD:
    context = 3
    size = (96, 96)
    n_way = 8
    n_diffusion = 10

    def __init__(self, path=NOMAD_NPZ, sd=None):
        sd = load_npz(path) if sd is None else sd
        p = "vision_encoder."
        self.obs_enc = EfficientNetB0(sd, p + "obs_encoder.", "gn")
        self.goal_enc = EfficientNetB0(sd, p + "goal_encoder.", "gn")
        a = lambda n: mx.array(sd[n])  # noqa: E731
        self.c_obs = (a(p + "compress_obs_enc.weight"), a(p + "compress_obs_enc.bias"))
        self.c_goal = (a(p + "compress_goal_enc.weight"), a(p + "compress_goal_enc.bias"))
        self.pos = a(p + "positional_encoding.pos_enc")[0]           # (5, 256)
        self.layers = [EncoderLayer(sd, f"{p}sa_encoder.layers.{i}.", 4) for i in range(4)]
        self.dist = [(a(f"dist_pred_net.network.{i}.weight"), a(f"dist_pred_net.network.{i}.bias")) for i in (0, 2, 4)]
        self.unet = UNet1D(sd)
        self.betas, self.alphas_cumprod = ddpm_schedule(self.n_diffusion)

    def encode_obs(self, x):
        return linear(self.obs_enc(x), *self.c_obs)

    def encode_goal(self, cur, goal):
        return linear(self.goal_enc(mx.concatenate([cur, goal], axis=-1)), *self.c_goal)

    def context_vector(self, obs_tokens, goal_token, goal_mask):
        """obs_tokens (N, 4, 256), goal_token (N, 256), goal_mask (N,) bool (True = explore / goal masked) -> (N, 256)."""
        x = mx.concatenate([obs_tokens, goal_token[:, None, :]], axis=1) + self.pos[None]
        t = x.shape[1]
        key_mask = mx.concatenate([mx.zeros((x.shape[0], t - 1), mx.bool_), goal_mask[:, None]], axis=1)
        for layer in self.layers:
            x = layer(x, key_mask)
        keep = mx.where(key_mask, 0.0, 1.0)
        scale = mx.where(goal_mask, t / (t - 1), 1.0)[:, None]
        return mx.mean(x * keep[..., None], axis=1) * scale

    def distance(self, ctx):
        x = ctx
        for i, (w, b) in enumerate(self.dist):
            x = linear(x, w, b)
            if i < 2:
                x = mx.maximum(x, 0)
        return x[:, 0]

    def sample_actions(self, ctx, noise):
        """DDPM sampling. ctx (N, 256); noise (n_diffusion + 1, N, 8, 2): [0] the initial sample, [k] the variance
        noise of the step that leaves timestep t = n_diffusion - k (the t = 0 step adds none). Returns the
        normalized deltas (N, 8, 2) in [-1, 1]-ish units."""
        x = noise[0]
        ac = self.alphas_cumprod
        for k, t in enumerate(range(self.n_diffusion - 1, -1, -1)):
            eps = self.unet(x, mx.full((x.shape[0],), t, mx.int32), ctx)
            a_t = float(ac[t])
            a_prev = float(ac[t - 1]) if t > 0 else 1.0
            b_t, b_prev = 1 - a_t, 1 - a_prev
            cur_a = a_t / a_prev
            cur_b = 1 - cur_a
            x0 = mx.clip((x - math.sqrt(b_t) * eps) / math.sqrt(a_t), -1.0, 1.0)
            x = (math.sqrt(a_prev) * cur_b / b_t) * x0 + (math.sqrt(cur_a) * b_prev / b_t) * x
            if t > 0:
                var = max(b_prev / b_t * cur_b, 1e-20)
                x = x + math.sqrt(var) * noise[k + 1]
        return x

    @staticmethod
    def to_waypoints(ndeltas):
        """get_action(): unnormalize deltas with ACTION_STATS, cumsum -> waypoints (N, 8, 2) in step units."""
        d = (np.asarray(ndeltas) + 1) / 2 * (ACTION_STATS["max"] - ACTION_STATS["min"]) + ACTION_STATS["min"]
        return np.cumsum(d, axis=1)
