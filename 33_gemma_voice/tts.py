"""Gemma TTS: Gemma 3 270M backbone + Mimi codec tokens in a delay pattern.

Sequence for one clip (positions are backbone time steps):
    [BOS, text tokens ...] [AUDIO_START] [step 1] [step 2] ... [step S-1]
- Text positions use Gemma's own (frozen) token embeddings, scaled by sqrt(640) as Gemma does.
- Audio step s carries all NQ codebooks at once, delayed: codebook j at step s holds frame s - j
  (MusicGen-style delay). Its input vector is the sum of NQ codebook embeddings of step s-1's tokens.
  Because of the delay, codebook j of a frame is predicted after codebooks < j of the same frame
  have been fed in.
- PAD = 2048 fills the delay corners and marks the end: codebook 0 emitting PAD at step F ends the
  utterance after F frames. S = F + NQ - 1 steps.
- NQ output heads (one Linear d -> NQ*2049) predict step s's tokens from the hidden state at the
  previous position. Gemma's 262K-vocab text head is never computed.
"""
import math, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from gemma import Gemma3, Linear, load_gemma

CB = 2048          # Mimi codebook size
PAD = CB           # pad / end token in every codebook
V = CB + 1
BOS = 2


_CHAR_IDS = {}


def encode_text(tok, text, mode="subword"):
    """BOS + Gemma token ids. mode 'chars' spells the text out: one Gemma token per character (cached per char)."""
    if mode == "chars":
        ids = []
        for ch in text:
            if ch not in _CHAR_IDS:
                _CHAR_IDS[ch] = tok.encode(ch, add_special_tokens=False).ids
            ids.extend(_CHAR_IDS[ch])
    else:
        ids = tok.encode(text, add_special_tokens=False).ids
    return [BOS] + ids


def delay_grid(codes):
    """codes (NQ, F) uint -> (S, NQ) int32 delayed grid, S = F + NQ - 1, PAD where frame s-j is outside [0, F)."""
    nq, F = codes.shape
    S = F + nq - 1
    g = np.full((S, nq), PAD, np.int32)
    for j in range(nq):
        g[j:j + F, j] = codes[j]
    return g


def undelay(grid, F):
    """(S, NQ) delayed grid -> (NQ, F) codes."""
    nq = grid.shape[1]
    return np.stack([grid[j:j + F, j] for j in range(nq)]).astype(np.int64)


class GemmaTTS(nn.Module):
    def __init__(self, backbone: Gemma3, nq: int, n_spk: int = 1):
        super().__init__()
        self.nq = nq
        self.backbone = backbone
        d = backbone.c["hidden_size"]
        # Match the per-dimension scale of Gemma's (scaled) text embeddings, so audio isn't a whisper.
        target = float(mx.std(backbone.embed_tokens.weight[:20000].astype(mx.float32)).item()) * backbone.emb_scale
        self.audio_emb = nn.Embedding(nq * V, d)
        self.audio_emb.weight = mx.random.normal((nq * V, d)) * (target / math.sqrt(nq))
        self.audio_start = mx.random.normal((d,)) * target
        self.heads = Linear(d, nq * V, bias=False)
        self.heads.weight = mx.zeros((nq * V, d))  # uniform first prediction (loss = ln 2049); random init gave ~10.2
        # Speaker embedding added to AUDIO_START and every audio step. Speaker 0 = LJSpeech. Zero init, so checkpoints
        # trained without it (no spk_emb weights) behave exactly as before.
        self.spk_emb = nn.Embedding(max(1, n_spk), d)
        self.spk_emb.weight = mx.zeros((max(1, n_spk), d))
        # T5Gemma-TTS-style options (plain attributes, not weights): progress > 0 replaces integer RoPE positions with
        # normalized progress (text i/(Lt-1)*P, audio step s/(S-1)*P, i.e. PM-RoPE); prefix_lm lets the text prefix
        # attend in both directions.
        self.progress = 0.0
        self.prefix_lm = False

    def positions(self, tlen, S, T):
        """(B,) text lengths, (B,) total audio steps -> (B, T) progress positions for the [text][AUDIO_START][steps]
        layout. Position tl+s predicts step s, so it carries step s's progress; padding is clamped to P."""
        P = self.progress
        j = mx.arange(T)[None, :].astype(mx.float32)
        tl = tlen[:, None].astype(mx.float32)
        Sf = S[:, None].astype(mx.float32)
        text_pos = j / mx.maximum(tl - 1, 1) * P
        aud_pos = mx.minimum((j - tl) / mx.maximum(Sf - 1, 1), 1.0) * P
        return mx.where(j < tl, text_pos, aud_pos)

        self.backbone.embed_tokens.freeze()

    def step_embed(self, grid_tokens):
        """(..., NQ) tokens of one step -> (..., d): sum of the NQ codebook embeddings."""
        return self.audio_emb(grid_tokens + mx.arange(self.nq) * V).sum(-2)

    def inputs(self, text, text_len, grid_in, spk=None):
        """Right-padded batch -> input embeddings (B, T, d).
        text (B, Lt) ids, grid_in (B, La, NQ) = tokens of steps 0..La-1 fed at positions start+1.. (step s-1 at s)."""
        B, Lt = text.shape
        La = grid_in.shape[1]
        T = Lt + 1 + La
        te = self.backbone.embed(text)                                   # (B, Lt, d)
        ae = self.step_embed(grid_in)                                    # (B, La, d)
        # place per-sample: positions [0, tl) text, tl = AUDIO_START, tl+1+s = step s input
        pos = mx.arange(T)[None, :]                                      # (1, T)
        tl = text_len[:, None]
        ti = mx.clip(pos, 0, Lt - 1)
        ai = mx.clip(pos - tl - 1, 0, La - 1)
        bidx = mx.arange(B)[:, None]
        h_text = te[bidx, mx.broadcast_to(ti, (B, T))]
        h_audio = ae[bidx, ai]
        is_text = (pos < tl)[..., None]
        is_start = (pos == tl)[..., None]
        h = mx.where(is_text, h_text, mx.where(is_start, self.audio_start.astype(h_text.dtype), h_audio))
        sv = self.spk_emb(spk if spk is not None else mx.zeros((B,), mx.int32)).astype(h.dtype)    # (B, d)
        h = h + mx.where(is_text, 0.0, sv[:, None, :])
        return h.astype(self.backbone.compute_dtype)

    def logits(self, h_out):
        """(..., d) -> (..., NQ, V)."""
        return self.heads(h_out).reshape(*h_out.shape[:-1], self.nq, V)


def make_batch(items, nq, corrupt=0.0, rng=None, return_spk=False):
    """items: list of (text_ids, grid (S, NQ)). Returns right-padded arrays and the loss targets.
    corrupt: fraction of fed-back (input) audio codes replaced by random codes (training only; targets untouched).
    Position tl+s (AUDIO_START for s=0) predicts grid[s]; the input at tl+1+s is grid[s]."""
    B = len(items)
    Lt = max(len(it[0]) for it in items)
    S = max(len(it[1]) for it in items)
    Lt = int(math.ceil(Lt / 8) * 8)
    S = int(math.ceil(S / 16) * 16)          # few distinct shapes
    text = np.zeros((B, Lt), np.int32)
    tlen = np.zeros(B, np.int32)
    grid_in = np.full((B, S - 1, nq), PAD, np.int32)
    T = Lt + S
    tgt = np.full((B, T, nq), -1, np.int32)
    spk = np.zeros(B, np.int32)
    for b, it in enumerate(items):
        t, g = it[0], it[1]
        spk[b] = it[2] if len(it) > 2 else 0
        text[b, :len(t)] = t
        tlen[b] = len(t)
        grid_in[b, :len(g) - 1] = g[:-1]
        tgt[b, len(t):len(t) + len(g)] = g
    if corrupt > 0:
        hit = (rng.random(grid_in.shape) < corrupt) & (grid_in != PAD)
        grid_in[hit] = rng.integers(0, CB, int(hit.sum()))
    out = (mx.array(text), mx.array(tlen), mx.array(grid_in), mx.array(tgt))
    return out + (mx.array(spk),) if return_spk else out


def loss_fn(model, text, tlen, grid_in, tgt, spk=None, cb_weight=None):
    h = model.inputs(text, tlen, grid_in, spk)
    pos = None
    if model.progress > 0:
        S = (tgt[..., 0] >= 0).sum(1)                                  # true step count per sample
        pos = model.positions(tlen, S, h.shape[1])
    out = model.backbone.hidden(h, pos=pos, prefix_len=tlen if model.prefix_lm else None)
    lg = model.logits(out).astype(mx.float32)                          # (B, T, NQ, V)
    valid = tgt >= 0
    ce = nn.losses.cross_entropy(lg, mx.maximum(tgt, 0), reduction="none") * valid
    per_cb = ce.sum((0, 1)) / mx.maximum(valid.sum((0, 1)), 1)
    w = cb_weight if cb_weight is not None else mx.ones((model.nq,))
    return (per_cb * w).sum() / w.sum(), per_cb


def build_model(nq, init="gemma", seed=0, dtype=mx.float32, n_spk=1, progress=0.0, prefix_lm=False):
    mx.random.seed(seed)
    bb, c = load_gemma(dtype=mx.float32)
    if init == "random":  # same architecture, random weights (ablation)
        from mlx.utils import tree_map
        fresh = Gemma3(c)
        fresh.set_dtype(mx.float32)
        # Gemma-style init: normal(0, 0.02) for matrices; norms keep their zero (= scale 1) init.
        def re(p):
            return mx.random.normal(p.shape) * 0.02 if p.ndim == 2 else p
        fresh.update(tree_map(re, fresh.parameters()))
        bb = fresh
    model = GemmaTTS(bb, nq, n_spk)
    model.progress, model.prefix_lm = float(progress), bool(prefix_lm)
    model.set_dtype(dtype)
    model.backbone.embed_tokens.freeze()
    mx.eval(model.parameters())
    return model


# ----------------------------------------------------------------------------- generation
def generate(model, text_ids, max_frames, temp=0.8, top_k=50, seed=0, min_frames=1, spk=0, est_steps=None):
    """Sample one utterance. Returns (codes (NQ, F), info). Batch 1, KV cache."""
    nq = model.nq
    t_start = time.perf_counter()
    first_audio = None
    key = mx.random.key(seed)
    bb = model.backbone
    caches = [dict() for _ in bb.layers]
    t = mx.array(text_ids)[None]
    sv = model.spk_emb(mx.array([spk]))[0].astype(mx.float32)
    h = mx.concatenate([bb.embed(t), (model.audio_start.astype(mx.float32) + sv)[None, None]], axis=1)
    tl = len(text_ids)
    use_prog = model.progress > 0
    if use_prog:
        assert est_steps, "progress positions need an estimated total step count"
        P = model.progress
        ppos = mx.array([[i / max(tl - 1, 1) * P for i in range(tl)] + [0.0]])
        step_pos = lambda s: mx.array([[min(s / max(est_steps - 1, 1), 1.0) * P]])
    out = bb.hidden(h.astype(bb.embed_tokens.weight.dtype), caches, 0, pos=ppos if use_prog else None,
                    prefix_len=mx.array([tl]) if model.prefix_lm else None)[:, -1]
    pos = tl + 1
    steps, F = [], None
    s = 0
    hit_max = False
    while True:
        lg = model.logits(out)[0].astype(mx.float32)                   # (NQ, V)
        key, sub = mx.random.split(key)
        if top_k:
            kth = mx.sort(lg, axis=-1)[:, -top_k][:, None]
            lg = mx.where(lg < kth, -1e9, lg)
        tok = mx.random.categorical(lg / temp, key=sub)                # (NQ,)
        tok = np.array(tok).astype(np.int32)
        for j in range(nq):
            fr = s - j
            if fr < 0 or (F is not None and fr >= F):
                tok[j] = PAD
            elif tok[j] == PAD and j > 0:   # acoustic codebooks may not end on their own: take best real code
                tok[j] = int(np.argmax(np.array(lg[j, :CB])))
        if F is None:
            if tok[0] == PAD and s >= min_frames:
                F = s
            elif s >= max_frames:
                F, hit_max = s, True
                tok[0] = PAD
            elif tok[0] == PAD:
                tok[0] = int(np.argmax(np.array(lg[0, :CB])))
        steps.append(tok)
        s += 1
        if s == nq and first_audio is None:  # frame 0's last codebook is out: first frame decodable
            first_audio = time.perf_counter() - t_start
        if F is not None and s >= F + nq - 1:
            break
        e = model.step_embed(mx.array(tok)[None, None]) + sv
        out = bb.hidden(e.astype(bb.embed_tokens.weight.dtype), caches, pos, pos=step_pos(s) if use_prog else None)[:, -1]
        pos += 1
    grid = np.stack(steps)
    return undelay(grid, F), {"frames": F, "steps": s, "hit_max": hit_max, "first_audio_s": first_audio}
