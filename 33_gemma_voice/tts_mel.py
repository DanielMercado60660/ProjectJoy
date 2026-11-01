"""Gemma TTS, continuous version: Gemma 3 270M autoregressively generates project 27's 80-band log-mel
(16 kHz, hop 256 = 62.5 frames/s) in chunks of K frames, rendered by project 28's LJ vocoder.

Sequence: [BOS, text chars ...] [AUDIO_START] [chunk 0] [chunk 1] ...  (position tl+s predicts chunk s)
- Input chunks go through a Tacotron-style prenet whose dropout stays ON at inference: the classic fix for
  autoregressive mel models leaning on their own (imperfect) previous output instead of the text.
- Output: Linear(d -> K*80) normalized mel + a stop logit per step (stop = this is the last chunk).
- Loss: L1 + L2 on valid frames, BCE on stop.
Same output path as project 27 (its mel definition and its vocoder), so the comparison with 27 is like for like.
"""
import json, math, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from gemma import Gemma3, Linear, load_gemma

HERE = os.path.dirname(os.path.abspath(__file__))
N_MELS = 80
_INFO27 = os.path.join(HERE, "..", "27_pocket_voice_foundry", "cache", "info.json")


def mel_stats():
    d = json.load(open(_INFO27))
    return np.array(d["mel_mean"], np.float32), np.array(d["mel_std"], np.float32)


class GemmaMelTTS(nn.Module):
    def __init__(self, backbone: Gemma3, K=4, prenet=256, p_drop=0.5):
        super().__init__()
        self.K, self.p_drop = K, p_drop
        self.backbone = backbone
        d = backbone.c["hidden_size"]
        target = float(mx.std(backbone.embed_tokens.weight[:20000].astype(mx.float32)).item()) * backbone.emb_scale
        self.pre1 = nn.Linear(K * N_MELS, prenet)
        self.pre2 = nn.Linear(prenet, prenet)
        self.pre_out = nn.Linear(prenet, d)
        self.pre_out.weight = mx.random.normal(self.pre_out.weight.shape) * (target / math.sqrt(prenet)) * 1.7
        self.audio_start = mx.random.normal((d,)) * target
        self.mel_out = Linear(d, K * N_MELS, bias=False)
        self.mel_out.weight = mx.zeros((K * N_MELS, d))
        self.stop_out = nn.Linear(d, 1)
        self.backbone.embed_tokens.freeze()
        self.infer_dropout = True

    def prenet(self, x, key=None):
        """x (..., K*80) normalized mel -> (..., d). Dropout always on in training; on at inference if infer_dropout."""
        def drop(h, k):
            if self.p_drop <= 0 or (not self.training and not self.infer_dropout):
                return h
            keep = mx.random.bernoulli(1 - self.p_drop, h.shape, key=k)
            return mx.where(keep, h / (1 - self.p_drop), 0.0)
        k1 = k2 = None
        if key is not None:
            k1, k2 = mx.random.split(key)
        h = drop(nn.relu(self.pre1(x)), k1)
        h = drop(nn.relu(self.pre2(h)), k2)
        return self.pre_out(h)

    def inputs(self, text, tlen, chunks_in):
        """text (B, Lt), tlen (B,), chunks_in (B, La, K*80): chunk s-1 is fed at position tl+s."""
        B, Lt = text.shape
        La = chunks_in.shape[1]
        T = Lt + 1 + La
        te = self.backbone.embed(text)
        ae = self.prenet(chunks_in).astype(te.dtype)
        pos = mx.arange(T)[None, :]
        tl = tlen[:, None]
        bidx = mx.arange(B)[:, None]
        h_text = te[bidx, mx.broadcast_to(mx.clip(pos, 0, Lt - 1), (B, T))]
        h_audio = ae[bidx, mx.clip(pos - tl - 1, 0, La - 1)]
        h = mx.where((pos < tl)[..., None], h_text,
                     mx.where((pos == tl)[..., None], self.audio_start.astype(te.dtype), h_audio))
        return h.astype(self.backbone.compute_dtype)

    def heads(self, h):
        return self.mel_out(h).astype(mx.float32), self.stop_out(h.astype(mx.float32))[..., 0]


def chunkify(mel, K):
    """(F, 80) -> (S, K*80) with S = ceil(F/K), zero-padded, plus a (S, K) frame-valid mask."""
    F = len(mel)
    S = int(math.ceil(F / K))
    pad = np.zeros((S * K, N_MELS), np.float32)
    pad[:F] = mel
    valid = np.zeros(S * K, np.float32)
    valid[:F] = 1
    return pad.reshape(S, K * N_MELS), valid.reshape(S, K)


def make_batch(items, K, noise=0.0, rng=None):
    """items: list of (text_ids, mel_norm (F, 80)). Right-padded; position tl+s predicts chunk s."""
    B = len(items)
    chunks = [chunkify(m, K) for _, m in items]
    Lt = int(math.ceil(max(len(t) for t, _ in items) / 8) * 8)
    S = int(math.ceil(max(len(c) for c, _ in chunks) / 16) * 16)
    T = Lt + S
    text = np.zeros((B, Lt), np.int32)
    tlen = np.zeros(B, np.int32)
    cin = np.zeros((B, S - 1, K * N_MELS), np.float32)
    tgt = np.zeros((B, T, K * N_MELS), np.float32)
    fmask = np.zeros((B, T, K), np.float32)        # which predicted frames are real
    smask = np.zeros((B, T), np.float32)           # which positions have a stop target
    stop = np.zeros((B, T), np.float32)
    for b, ((t, _), (c, v)) in enumerate(zip(items, chunks)):
        n = len(c)
        text[b, :len(t)] = t
        tlen[b] = len(t)
        cin[b, :n - 1] = c[:-1]
        tgt[b, len(t):len(t) + n] = c
        fmask[b, len(t):len(t) + n] = v
        smask[b, len(t):len(t) + n] = 1
        stop[b, len(t) + n - 1] = 1
    if noise > 0:
        cin = cin + rng.normal(0, noise, cin.shape).astype(np.float32)
    return tuple(mx.array(x) for x in (text, tlen, cin, tgt, fmask, smask, stop))


def loss_fn(model, text, tlen, cin, tgt, fmask, smask, stop, stop_weight=1.0, pos_weight=5.0):
    h = model.backbone.hidden(model.inputs(text, tlen, cin))
    mel, st = model.heads(h)
    B, T, _ = mel.shape
    K = model.K
    err = (mel - tgt).reshape(B, T, K, N_MELS)
    m = fmask[..., None]
    n = mx.maximum(m.sum() * N_MELS, 1)
    l1 = (mx.abs(err) * m).sum() / n
    l2 = ((err ** 2) * m).sum() / n
    bce = -(pos_weight * stop * nn.log_sigmoid(st) + (1 - stop) * nn.log_sigmoid(-st))
    ls = (bce * smask).sum() / mx.maximum(smask.sum(), 1)
    return l1 + l2 + stop_weight * ls, (l1, l2, ls)


def build_model(init="gemma", seed=0, K=4, p_drop=0.5):
    mx.random.seed(seed)
    bb, c = load_gemma(dtype=mx.float32)
    if init == "random":
        from mlx.utils import tree_map
        fresh = Gemma3(c)
        fresh.set_dtype(mx.float32)
        fresh.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.02 if p.ndim == 2 else p, fresh.parameters()))
        bb = fresh
    model = GemmaMelTTS(bb, K=K, p_drop=p_drop)
    model.set_dtype(mx.float32)
    model.backbone.embed_tokens.freeze()
    mx.eval(model.parameters())
    return model


def generate(model, text_ids, max_steps, seed=0, stop_threshold=0.5):
    """Greedy (deterministic regression; variation comes only from prenet dropout). Returns normalized mel (F, 80)."""
    bb = model.backbone
    key = mx.random.key(seed)
    caches = [dict() for _ in bb.layers]
    t0 = time.perf_counter()
    h = mx.concatenate([bb.embed(mx.array(text_ids)[None]), model.audio_start.astype(mx.float32)[None, None]], axis=1)
    out = bb.hidden(h, caches, 0)[:, -1:]
    pos = len(text_ids) + 1
    chunks, first, hit_max = [], None, False
    for s in range(max_steps):
        mel, st = model.heads(out)
        c = mel[0, 0]
        mx.eval(c, st)
        chunks.append(np.array(c))
        if first is None:
            first = time.perf_counter() - t0
        if float(mx.sigmoid(st[0, 0]).item()) > stop_threshold:
            break
        key, sub = mx.random.split(key)
        e = model.prenet(c[None, None], key=sub)
        out = bb.hidden(e, caches, pos)
        pos += 1
    else:
        hit_max = True
    mel = np.concatenate(chunks).reshape(-1, N_MELS)
    return mel, {"steps": len(chunks), "frames": len(mel), "hit_max": hit_max, "first_chunk_s": first}
