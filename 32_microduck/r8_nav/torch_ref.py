"""Torch reference modules for the parity test ONLY (never in the runtime path). Written from the official sources:
visualnav-transformer @ dca7981 (train/vint_train/models/vint/{vint,self_attention}.py, models/nomad/{nomad,
nomad_vint}.py, train/train.py), efficientnet_pytorch (model.py / utils.py: EfficientNet.from_name, MBConvBlock,
Conv2dStaticSamePadding, MemoryEfficientSwish), diffusion_policy (conditional_unet1d.py, conv1d_components.py,
positional_embedding.py) and diffusers (DDPMScheduler squaredcos_cap_v2 / epsilon / clip_sample / fixed_small),
none of which is installed here. Attribute names follow the originals so the published state dicts load with
strict=True. The disclosed differences: einops Rearrange is written as reshapes; drop-connect is omitted (it is the
identity in eval mode); only the b0 configuration is built.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

B0_BLOCKS = ["r1_k3_s11_e1_i32_o16_se0.25", "r2_k3_s22_e6_i16_o24_se0.25", "r2_k5_s22_e6_i24_o40_se0.25",
             "r3_k3_s22_e6_i40_o80_se0.25", "r3_k5_s11_e6_i80_o112_se0.25", "r4_k5_s22_e6_i112_o192_se0.25",
             "r1_k3_s11_e6_i192_o320_se0.25"]


# ------------------------------------------------------------------------- efficientnet_pytorch (b0 only)
class Conv2dStaticSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def _out_size(size, stride):
    stride = stride if isinstance(stride, int) else stride[0]
    return [int(math.ceil(size[0] / stride)), int(math.ceil(size[1] / stride))]


class Swish(nn.Module):           # MemoryEfficientSwish forward value
    def forward(self, x):
        return x * torch.sigmoid(x)


def _decode(s):
    o = {}
    for op in s.split("_"):
        import re
        sp = re.split(r"(\d.*)", op)
        if len(sp) >= 2:
            o[sp[0]] = sp[1]
    return dict(num_repeat=int(o["r"]), kernel_size=int(o["k"]), stride=[int(o["s"][0])], expand_ratio=int(o["e"]),
                input_filters=int(o["i"]), output_filters=int(o["o"]), se_ratio=float(o["se"]), id_skip=True)


class MBConvBlock(nn.Module):
    def __init__(self, a, image_size, bn_eps=1e-3, bn_mom=0.01):
        super().__init__()
        self.a = a
        inp, oup = a["input_filters"], a["input_filters"] * a["expand_ratio"]
        if a["expand_ratio"] != 1:
            self._expand_conv = Conv2dStaticSamePadding(inp, oup, 1, bias=False, image_size=image_size)
            self._bn0 = nn.BatchNorm2d(oup, momentum=bn_mom, eps=bn_eps)
        k, s = a["kernel_size"], a["stride"]
        self._depthwise_conv = Conv2dStaticSamePadding(oup, oup, k, stride=s, groups=oup, bias=False, image_size=image_size)
        self._bn1 = nn.BatchNorm2d(oup, momentum=bn_mom, eps=bn_eps)
        image_size = _out_size(image_size, s)
        nsq = max(1, int(a["input_filters"] * a["se_ratio"]))
        self._se_reduce = Conv2dStaticSamePadding(oup, nsq, 1, image_size=(1, 1))
        self._se_expand = Conv2dStaticSamePadding(nsq, oup, 1, image_size=(1, 1))
        self._project_conv = Conv2dStaticSamePadding(oup, a["output_filters"], 1, bias=False, image_size=image_size)
        self._bn2 = nn.BatchNorm2d(a["output_filters"], momentum=bn_mom, eps=bn_eps)
        self._swish = Swish()

    def forward(self, inputs):
        x = inputs
        if self.a["expand_ratio"] != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))
        xs = F.adaptive_avg_pool2d(x, 1)
        xs = self._se_expand(self._swish(self._se_reduce(xs)))
        x = torch.sigmoid(xs) * x
        x = self._bn2(self._project_conv(x))
        if self.a["id_skip"] and self.a["stride"] == 1 and self.a["input_filters"] == self.a["output_filters"]:
            x = x + inputs    # (the decoded stride is a list on a stage's first block, so only repeats skip)
        return x


class EfficientNet(nn.Module):
    """EfficientNet.from_name('efficientnet-b0', in_channels=c): image_size 224 -> static padding."""

    def __init__(self, in_channels=3, image_size=224):
        super().__init__()
        size = [image_size, image_size]
        self._conv_stem = Conv2dStaticSamePadding(in_channels, 32, 3, stride=2, bias=False, image_size=image_size)
        self._bn0 = nn.BatchNorm2d(32, momentum=0.01, eps=1e-3)
        size = _out_size(size, 2)
        self._blocks = nn.ModuleList([])
        for s in B0_BLOCKS:
            a = _decode(s)
            self._blocks.append(MBConvBlock(a, size))
            size = _out_size(size, a["stride"])
            if a["num_repeat"] > 1:
                a = dict(a, input_filters=a["output_filters"], stride=1)
            for _ in range(a["num_repeat"] - 1):
                self._blocks.append(MBConvBlock(a, size))
        self._conv_head = Conv2dStaticSamePadding(320, 1280, 1, bias=False, image_size=size)
        self._bn1 = nn.BatchNorm2d(1280, momentum=0.01, eps=1e-3)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.2)
        self._fc = nn.Linear(1280, 1000)
        self._swish = Swish()

    def extract_features(self, x):
        x = self._swish(self._bn0(self._conv_stem(x)))
        for block in self._blocks:
            x = block(x)
        return self._swish(self._bn1(self._conv_head(x)))


def replace_bn_with_gn(root, features_per_group=16):
    for name, m in list(root.named_modules()):
        if isinstance(m, nn.BatchNorm2d):
            *parent, k = name.split(".")
            pm = root.get_submodule(".".join(parent)) if parent else root
            setattr(pm, k, nn.GroupNorm(m.num_features // features_per_group, m.num_features))
    return root


# ------------------------------------------------------------------------- ViNT
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))

    def forward(self, x):
        return x + self.pos_enc[:, :x.size(1), :]


class MultiLayerDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=(256, 128, 64), nhead=8, num_layers=8, ff_dim_factor=4):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor * embed_dim,
                                                   activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_layers = nn.ModuleList([nn.Linear(seq_len * embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers) - 1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i + 1]))

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        x = x.reshape(x.shape[0], -1)
        for layer in self.output_layers:
            x = F.relu(layer(x))
        return x


class ViNT(nn.Module):
    def __init__(self, context_size=5, len_traj_pred=5, obs_encoding_size=512, nhead=4, num_layers=4, ff=4):
        super().__init__()
        self.context_size, self.len_trajectory_pred, self.num_action_params = context_size, len_traj_pred, 4
        self.obs_encoding_size = obs_encoding_size
        self.obs_encoder = EfficientNet(3)
        self.goal_encoder = EfficientNet(6)
        self.compress_obs_enc = nn.Linear(1280, obs_encoding_size)
        self.compress_goal_enc = nn.Linear(1280, obs_encoding_size)
        self.decoder = MultiLayerDecoder(embed_dim=obs_encoding_size, seq_len=context_size + 2,
                                         output_layers=[256, 128, 64, 32], nhead=nhead, num_layers=num_layers,
                                         ff_dim_factor=ff)
        self.dist_predictor = nn.Sequential(nn.Linear(32, 1))
        self.action_predictor = nn.Sequential(nn.Linear(32, len_traj_pred * 4))

    def forward(self, obs_img, goal_img):
        obsgoal = torch.cat([obs_img[:, 3 * self.context_size:, :, :], goal_img], dim=1)
        g = self.goal_encoder._avg_pooling(self.goal_encoder.extract_features(obsgoal)).flatten(start_dim=1)
        g = self.compress_goal_enc(self.goal_encoder._dropout(g)).unsqueeze(1)
        obs = torch.concat(torch.split(obs_img, 3, dim=1), dim=0)
        o = self.obs_encoder._avg_pooling(self.obs_encoder.extract_features(obs)).flatten(start_dim=1)
        o = self.compress_obs_enc(self.obs_encoder._dropout(o))
        o = torch.transpose(o.reshape((self.context_size + 1, -1, self.obs_encoding_size)), 0, 1)
        final_repr = self.decoder(torch.cat((o, g), dim=1))
        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr).reshape((-1, self.len_trajectory_pred, 4))
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)
        action_pred[:, :, 2:] = F.normalize(action_pred[:, :, 2:].clone(), dim=-1)
        return dist_pred, action_pred


# ------------------------------------------------------------------------- NoMaD
class NoMaD_ViNT(nn.Module):
    def __init__(self, context_size=3, obs_encoding_size=256, nhead=4, num_layers=4, ff=4):
        super().__init__()
        self.obs_encoding_size = self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size
        self.obs_encoder = replace_bn_with_gn(EfficientNet(3))
        self.goal_encoder = replace_bn_with_gn(EfficientNet(6))
        self.compress_obs_enc = nn.Linear(1280, obs_encoding_size)
        self.compress_goal_enc = nn.Linear(1280, obs_encoding_size)
        self.positional_encoding = PositionalEncoding(obs_encoding_size, max_seq_len=context_size + 2)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=obs_encoding_size, nhead=nhead, dim_feedforward=ff * obs_encoding_size,
                                                   activation="gelu", batch_first=True, norm_first=True)
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.goal_mask = torch.zeros((1, context_size + 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True
        self.no_mask = torch.zeros((1, context_size + 2), dtype=torch.bool)
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat([1 - self.no_mask.float(),
                                        (1 - self.goal_mask.float()) * ((context_size + 2) / (context_size + 1))], dim=0)

    def forward(self, obs_img, goal_img, input_goal_mask):
        obsgoal = torch.cat([obs_img[:, 3 * self.context_size:, :, :], goal_img], dim=1)
        g = self.goal_encoder._avg_pooling(self.goal_encoder.extract_features(obsgoal)).flatten(start_dim=1)
        g = self.compress_goal_enc(self.goal_encoder._dropout(g)).unsqueeze(1)
        obs = torch.concat(torch.split(obs_img, 3, dim=1), dim=0)
        o = self.obs_encoder._avg_pooling(self.obs_encoder.extract_features(obs)).flatten(start_dim=1)
        o = self.compress_obs_enc(self.obs_encoder._dropout(o)).unsqueeze(1)
        o = torch.transpose(o.reshape((self.context_size + 1, -1, self.obs_encoding_size)), 0, 1)
        o = torch.cat((o, g), dim=1)
        no_goal_mask = input_goal_mask.long()
        kpm = torch.index_select(self.all_masks, 0, no_goal_mask)
        o = self.positional_encoding(o)
        tok = self.sa_encoder(o, src_key_padding_mask=kpm)
        tok = tok * torch.index_select(self.avg_pool_mask, 0, no_goal_mask).unsqueeze(-1)
        return torch.mean(tok, dim=1)


class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(nn.Linear(embedding_dim, embedding_dim // 4), nn.ReLU(),
                                     nn.Linear(embedding_dim // 4, embedding_dim // 16), nn.ReLU(),
                                     nn.Linear(embedding_dim // 16, 1))

    def forward(self, x):
        return self.network(x.reshape((-1, self.embedding_dim)))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, inp, out, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(inp, out, kernel_size, padding=kernel_size // 2),
                                   nn.GroupNorm(n_groups, out), nn.Mish())

    def forward(self, x):
        return self.block(x)


class _Unsq(nn.Module):           # einops Rearrange('batch t -> batch t 1')
    def forward(self, x):
        return x.unsqueeze(-1)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
                                     Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)])
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels), _Unsq())
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x) + self.cond_encoder(cond)
        return self.blocks[1](out) + self.residual_conv(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim=2, global_cond_dim=256, diffusion_step_embed_dim=256, down_dims=(64, 128, 256),
                 kernel_size=3, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(),
                                                    nn.Linear(dsed * 4, dsed))
        cond_dim = dsed + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
                                          ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups)])
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()]))
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()]))
        self.final_conv = nn.Sequential(Conv1dBlock(start_dim, start_dim, kernel_size), nn.Conv1d(start_dim, input_dim, 1))

    def forward(self, sample, timestep, global_cond):
        sample = sample.permute(0, 2, 1)                  # 'b h t -> b t h'
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.expand(sample.shape[0])
        gf = torch.cat([self.diffusion_step_encoder(timesteps), global_cond], axis=-1)
        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet2(resnet(x, gf), gf)
            h.append(x)
            x = downsample(x)
        for m in self.mid_modules:
            x = m(x, gf)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = upsample(resnet2(resnet(x, gf), gf))
        x = self.final_conv(x)
        return x.permute(0, 2, 1)


class NoMaD(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = replace_bn_with_gn(NoMaD_ViNT())
        self.noise_pred_net = ConditionalUnet1D()
        self.dist_pred_net = DenseNetwork(256)


class DDPM:
    """diffusers DDPMScheduler(num_train_timesteps=10, beta_schedule='squaredcos_cap_v2', clip_sample=True,
    prediction_type='epsilon'), variance_type fixed_small; step() with externally supplied noise."""

    def __init__(self, n=10):
        f = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
        self.betas = torch.tensor([min(1 - f((i + 1) / n) / f(i / n), 0.999) for i in range(n)], dtype=torch.float32)
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)
        self.one = torch.tensor(1.0)
        self.n = n
        self.timesteps = torch.arange(n - 1, -1, -1)     # set_timesteps(n) with n == num_train_timesteps

    def step(self, model_output, t, sample, variance_noise):
        t = int(t)
        prev_t = t - 1
        a_t = self.alphas_cumprod[t]
        a_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        b_t, b_prev = 1 - a_t, 1 - a_prev
        cur_a = a_t / a_prev
        cur_b = 1 - cur_a
        x0 = ((sample - b_t ** 0.5 * model_output) / a_t ** 0.5).clamp(-1, 1)
        prev = (a_prev ** 0.5 * cur_b) / b_t * x0 + cur_a ** 0.5 * b_prev / b_t * sample
        if t > 0:
            var = torch.clamp((1 - a_prev) / (1 - a_t) * cur_b, min=1e-20)
            prev = prev + var ** 0.5 * variance_noise
        return prev
