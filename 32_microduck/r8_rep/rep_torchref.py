"""R8 Gate C: torch REFERENCE modules, used only by the parity test (rep_parity.py). Never in a runtime path.

Each class is transcribed from the official source (read 2026-11-05) so that the official weights load into
it with strict key matching, and so that the MLX ports in rep_ports.py can be checked against it:
  * TorchDINOv2   facebookresearch/dinov2 @ 7764ea0: dinov2/models/vision_transformer.py
                  (prepare_tokens_with_masks, interpolate_pos_encoding with interpolate_offset 0.1 and
                  antialias False = hub defaults), dinov2/layers/{block,attention,layer_scale,mlp}.py.
                  The HF checkpoint's separate query/key/value are concatenated into the official fused qkv.
  * TorchVJEPA21  facebookresearch/vjepa2 @ 204698b: app/vjepa_2_1/models/vision_transformer.py
                  (VisionTransformer.forward, image path), utils/modules.py (rotate_queries_or_keys,
                  RoPEAttention incl. separate_positions + interpolate_rope, Block, MLP),
                  utils/patch_embed.py (PatchEmbed3D); vit_large_rope + the 2.1 hub kwargs
                  (img_temporal_dim_size=1, interpolate_rope=True, modality embedding, GELU MLP).
  * TorchLeWM     lucas-maes/le-wm @ 8edfeb3: module.py (Attention, FeedForward, ConditionalBlock,
                  Transformer, Embedder, MLP, ARPredictor), jepa.py (JEPA.encode / predict); the encoder is
                  HF transformers' ViTModel (add_pooling_layer=False), re-written here with HF's module and
                  parameter names (ViTEmbeddings, ViTSelfAttention, ViTLayer; LN eps 1e-12) so the official
                  state_dict loads with strict=True.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------- DINOv2
class _DAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _DMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class _DBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _DAttention(dim, heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _DMlp(dim, 4 * dim)
        self.ls2 = _LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class TorchDINOv2(nn.Module):
    def __init__(self, dim=384, depth=12, heads=6, patch=14, img=518):
        super().__init__()
        self.patch_size = patch
        self.patch_embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img // patch) ** 2 + 1, dim))
        self.blocks = nn.ModuleList([_DBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.interpolate_offset, self.interpolate_antialias = 0.1, False

    def interpolate_pos_encoding(self, x, w, h):          # official, verbatim logic
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        sx = float(w0 + self.interpolate_offset) / M
        sy = float(h0 + self.interpolate_offset) / M
        patch_pos_embed = F.interpolate(patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2), mode="bicubic",
                                        antialias=self.interpolate_antialias, scale_factor=(sx, sy))
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def forward(self, x):                                   # x (B, 3, H, W) normalised; returns x_norm
        B, nc, w, h = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def load_hf(self, w):
        sd = {"patch_embed.weight": w["embeddings.patch_embeddings.projection.weight"],
              "patch_embed.bias": w["embeddings.patch_embeddings.projection.bias"],
              "cls_token": w["embeddings.cls_token"], "pos_embed": w["embeddings.position_embeddings"],
              "norm.weight": w["layernorm.weight"], "norm.bias": w["layernorm.bias"]}
        for i in range(len(self.blocks)):
            p, q = f"encoder.layer.{i}.", f"blocks.{i}."
            for n in ("weight", "bias"):
                sd[q + "attn.qkv." + n] = torch.cat([w[p + f"attention.attention.{x}.{n}"]
                                                     for x in ("query", "key", "value")], 0)
                sd[q + "attn.proj." + n] = w[p + "attention.output.dense." + n]
                for a in ("norm1", "norm2", "mlp.fc1", "mlp.fc2"):
                    sd[q + a + "." + n] = w[p + a + "." + n]
            sd[q + "ls1.gamma"], sd[q + "ls2.gamma"] = w[p + "layer_scale1.lambda1"], w[p + "layer_scale2.lambda1"]
        self.load_state_dict(sd, strict=True)
        return self


# -------------------------------------------------------------------------------------- V-JEPA 2.1
def rotate_queries_or_keys(x, pos, n_registers=0, has_cls_first=False):    # official, verbatim logic
    B, num_heads, N, D = x.size()
    assert D % 2 == 0
    n_cls = 1 if has_cls_first else 0
    start_ctx, end_ctx = n_cls, N - n_registers
    x_ctx = x[..., start_ctx:end_ctx, :]
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000 ** omega
    freq = torch.einsum("..., f -> ... f", pos, omega)
    emb_sin = freq.sin().repeat_interleave(2, dim=-1)
    emb_cos = freq.cos().repeat_interleave(2, dim=-1)
    y = x_ctx.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x_ctx * emb_cos) + (y * emb_sin)


class _RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads, grid_size, patch_size=16, interpolate_rope=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.d_dim = self.h_dim = self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.interpolate_rope = interpolate_rope
        self.pretrained_grid_size = int(256 / patch_size) if patch_size == 16 else int(252 / patch_size)

    def separate_positions(self, ids, H_patches, W_patches):
        tokens_per_frame, tokens_per_row = int(H_patches * W_patches), W_patches
        frame_ids = ids // tokens_per_frame
        height_ids = (ids - tokens_per_frame * frame_ids) // tokens_per_row
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(self, x, T, H_patches, W_patches):
        B, N, C = x.size()
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
        d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        if self.interpolate_rope:
            h_mask = h_mask * (self.pretrained_grid_size - 1) / (H_patches - 1)
            w_mask = w_mask * (self.pretrained_grid_size - 1) / (W_patches - 1)
        s = 0
        qd = rotate_queries_or_keys(q[..., s:s + self.d_dim], d_mask)
        kd = rotate_queries_or_keys(k[..., s:s + self.d_dim], d_mask)
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s:s + self.h_dim], h_mask)
        kh = rotate_queries_or_keys(k[..., s:s + self.h_dim], h_mask)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s:s + self.w_dim], w_mask)
        kw = rotate_queries_or_keys(k[..., s:s + self.w_dim], w_mask)
        s += self.w_dim
        if s < self.head_dim:
            q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
            k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
        else:
            q, k = torch.cat([qd, qh, qw], dim=-1), torch.cat([kd, kh, kw], dim=-1)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _VJMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _VJBlock(nn.Module):
    def __init__(self, dim, heads, grid_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _RoPEAttention(dim, heads, grid_size)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _VJMlp(dim, 4 * dim)

    def forward(self, x, T, H_patches, W_patches):
        x = x + self.attn(self.norm1(x), T=T, H_patches=H_patches, W_patches=W_patches)
        return x + self.mlp(self.norm2(x))


class _PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=16, tubelet_size=1, in_chans=3, embed_dim=1024):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(tubelet_size, patch_size, patch_size),
                              stride=(tubelet_size, patch_size, patch_size))

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TorchVJEPA21(nn.Module):
    """vit_large_rope, img_size 384, patch 16, image path only (input (B, 3, 1, H, W))."""

    def __init__(self, dim=1024, depth=24, heads=16, patch=16, img=384):
        super().__init__()
        self.patch_size = patch
        self.patch_embed_img = _PatchEmbed3D(patch, 1, 3, dim)
        self.blocks = nn.ModuleList([_VJBlock(dim, heads, img // patch) for _ in range(depth)])
        self.norms_block = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for _ in range(4)])
        self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        _, _, T, H, W = x.shape                       # T == img_temporal_dim_size == 1 -> image mode
        H_patches, W_patches = H // self.patch_size, W // self.patch_size
        x = self.patch_embed_img(x)
        x = x + self.img_mod_embed.repeat(x.shape[0], 1, 1)
        for blk in self.blocks:
            x = blk(x, T=T, H_patches=H_patches, W_patches=W_patches)
        return self.norms_block[-1](x)

    def load_mlx_release(self, w):
        """Encoder weights of the MLX release (MLX layouts) -> torch layouts; video patch embed dropped."""
        sd = {"patch_embed_img.proj.weight": w["patch_embed_img.proj.weight"].permute(0, 4, 1, 2, 3).contiguous(),
              "patch_embed_img.proj.bias": w["patch_embed_img.proj.bias"], "img_mod_embed": w["img_mod_embed"]}
        for k, v in w.items():
            if k.startswith("blocks.") or k.startswith("norms_block."):
                sd[k] = v
        self.load_state_dict(sd, strict=True)
        return self


# ---------------------------------------------------------------------------------- LeWM (official)
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads, self.dropout = heads, dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    def __init__(self, hidden_dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
                                     for _ in range(depth)])

    def forward(self, x, c):                    # input/cond/output projections are Identity (192 = 192)
        for block in self.layers:
            x = block(x, c)
        return self.norm(x)


class ARPredictor(nn.Module):
    def __init__(self, num_frames=3, depth=6, heads=16, mlp_dim=2048, dim=192, dim_head=64, dropout=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        self.dropout = nn.Dropout(0.0)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x, c):
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        return self.transformer(self.dropout(x), c)


class Embedder(nn.Module):
    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=192, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(nn.Linear(smoothed_dim, mlp_scale * emb_dim), nn.SiLU(),
                                   nn.Linear(mlp_scale * emb_dim, emb_dim))

    def forward(self, x):
        x = self.patch_embed(x.float().permute(0, 2, 1)).permute(0, 2, 1)
        return self.embed(x)


class ProjMLP(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=2048, output_dim=192):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


class _HFViTSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.query, self.key, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        sh = lambda t: t.view(B, N, self.heads, C // self.heads).transpose(1, 2)  # noqa: E731
        q, k, v = sh(self.query(x)), sh(self.key(x)), sh(self.value(x))
        a = (q @ k.transpose(-1, -2) / math.sqrt(C // self.heads)).softmax(-1)
        return (a @ v).transpose(1, 2).reshape(B, N, C)


class _HFDense(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.dense = nn.Linear(i, o)


class _HFViTAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attention = _HFViTSelfAttention(dim, heads)
        self.output = _HFDense(dim, dim)

    def forward(self, x):
        return self.output.dense(self.attention(x))


class _HFViTLayer(nn.Module):
    def __init__(self, dim, heads, eps=1e-12):
        super().__init__()
        self.attention = _HFViTAttention(dim, heads)
        self.intermediate = _HFDense(dim, 4 * dim)
        self.output = _HFDense(4 * dim, dim)
        self.layernorm_before = nn.LayerNorm(dim, eps=eps)
        self.layernorm_after = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = self.attention(self.layernorm_before(x)) + x
        h = F.gelu(self.intermediate.dense(self.layernorm_after(x)))
        return self.output.dense(h) + x


class _HFPatch(nn.Module):
    def __init__(self, dim, patch):
        super().__init__()
        self.projection = nn.Conv2d(3, dim, patch, stride=patch)


class _HFEmbeddings(nn.Module):
    def __init__(self, dim, patch, img):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, (img // patch) ** 2 + 1, dim))
        self.patch_embeddings = _HFPatch(dim, patch)

    def forward(self, x):          # 224 px = the trained size: interpolate_pos_encoding returns the table as is
        e = self.patch_embeddings.projection(x).flatten(2).transpose(1, 2)
        e = torch.cat([self.cls_token.expand(e.shape[0], -1, -1), e], 1)
        return e + self.position_embeddings


class _HFEncoder(nn.Module):
    def __init__(self, dim, depth, heads):
        super().__init__()
        self.layer = nn.ModuleList([_HFViTLayer(dim, heads) for _ in range(depth)])


class HFViTModel(nn.Module):
    def __init__(self, dim=192, depth=12, heads=3, patch=14, img=224):
        super().__init__()
        self.embeddings = _HFEmbeddings(dim, patch, img)
        self.encoder = _HFEncoder(dim, depth, heads)
        self.layernorm = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, x):          # -> last_hidden_state
        h = self.embeddings(x)
        for layer in self.encoder.layer:
            h = layer(h)
        return self.layernorm(h)


class TorchLeWM(nn.Module):
    def __init__(self, act_dim=10):
        super().__init__()
        self.encoder = HFViTModel()
        self.predictor = ARPredictor()
        self.action_encoder = Embedder(act_dim, 10, 192)
        self.projector = ProjMLP()
        self.pred_proj = ProjMLP()

    def encode(self, pixels):      # (N, 3, 224, 224) -> cls, emb
        cls = self.encoder(pixels)[:, 0]
        return cls, self.projector(cls)

    def predict(self, emb, act):   # JEPA.predict(emb, action_encoder(action))
        preds = self.predictor(emb, self.action_encoder(act))
        B, T, d = preds.shape
        return self.pred_proj(preds.reshape(B * T, d)).reshape(B, T, d)
