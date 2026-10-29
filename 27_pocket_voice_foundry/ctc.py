"""Intelligibility proxy: a small CTC character recognizer trained on *real* training-split log-mels only.

No ASR is installed, so this stands in for one. It hears one speaker (LJ), was trained on ~19 h of her
real speech, and outputs letters, space and apostrophe. Its CER on real held-out mels is the floor any
synthesized system can be compared to. Limits: it shares the training speaker and text domain, it is a
character model with no language model, and it may be more (or less) tolerant of oversmoothed synthetic
mels than a human listener.

  ../bin/gpu_run ../.venv/bin/python ctc.py --max_minutes 12      # -> checkpoints/ctc/ctc.{safetensors,json}
"""
import argparse, json, math, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from model import FFTBlock, masks, pos_enc

ROOT = os.path.dirname(os.path.abspath(__file__))
VOCAB = ["<b>"] + list(" abcdefghijklmnopqrstuvwxyz'")
C2I = {c: i for i, c in enumerate(VOCAB)}


def text_to_ids(t):
    import re
    t = re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))
    t = re.sub(r"\s+", " ", t).strip()
    return [C2I[c] for c in t]


class Recognizer(nn.Module):
    def __init__(self, d=256, layers=4, ff=1024, drop=0.1):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, ff=ff, drop=drop)
        self.d = d
        self.c1 = nn.Conv1d(80, d, 5, padding=2)
        self.c2 = nn.Conv1d(d, d, 5, stride=2, padding=2)
        self.blocks = [FFTBlock(d, 4, ff, 5, drop) for _ in range(layers)]
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, len(VOCAB))

    def __call__(self, x, flen):  # x (B,T,80) normalized -> logprobs (B,T',V), T' lens
        h = nn.gelu(self.c1(x))
        h = nn.gelu(self.c2(h))
        T2 = h.shape[1]
        l2 = (flen + 1) // 2
        m, am = masks(l2, T2)
        h = (h + pos_enc(T2, self.d)[None]) * m
        for b in self.blocks:
            h = b(h, m, am)
        return nn.log_softmax(self.out(self.ln(h)), axis=-1), l2


def ctc_loss(lp, tlen, labels, llen):
    """lp (B,T,V) log-probs; labels (B,L) int; returns mean over batch of -log p(labels)/L."""
    B, T, V = lp.shape
    L = labels.shape[1]
    S = 2 * L + 1
    ext = np.zeros((B, S), np.int32)
    lab = np.array(labels)
    ext[:, 1::2] = lab
    allow2 = np.zeros((B, S), bool)
    allow2[:, 2:] = (ext[:, 2:] != 0) & (ext[:, 2:] != ext[:, :-2])
    emit = mx.take_along_axis(lp, mx.array(ext)[:, None, :].astype(mx.int32) * mx.ones((1, T, 1), mx.int32), axis=2)  # (B,T,S)
    NEG = -1e9
    sidx = np.arange(S)[None]
    a = np.full((B, S), NEG, np.float32)
    a[:, 0] = 0.0
    a[:, 1] = 0.0
    alpha = mx.array(a) + emit[:, 0]
    alpha = mx.where(mx.array(sidx < 2), alpha, NEG)
    al2 = mx.array(allow2)
    tl = np.array(tlen)
    negcol = mx.full((B, 1), NEG)
    for t in range(1, T):
        p1 = mx.concatenate([negcol, alpha[:, :-1]], 1)
        p2 = mx.where(al2, mx.concatenate([negcol, negcol, alpha[:, :-2]], 1), NEG)
        new = mx.logsumexp(mx.stack([alpha, p1, p2], 0), axis=0) + emit[:, t]
        alpha = mx.where(mx.array((t < tl)[:, None]), new, alpha)
    ll = np.array(llen)
    last = mx.array((2 * ll)[:, None].astype(np.int32))
    e1 = mx.take_along_axis(alpha, last, axis=1)[:, 0]
    e2 = mx.take_along_axis(alpha, mx.maximum(last - 1, 0), axis=1)[:, 0]
    nll = -mx.logaddexp(e1, e2)
    return mx.mean(nll / mx.array(np.maximum(ll, 1).astype(np.float32)))


def greedy(lp, l2):
    best = np.array(mx.argmax(lp, -1))
    outs = []
    for b in range(best.shape[0]):
        seq, prev = [], -1
        for k in best[b, : int(l2[b])]:
            if k != prev and k != 0:
                seq.append(VOCAB[k])
            prev = k
        outs.append("".join(seq).strip())
    return outs


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def ref_text(t):
    return "".join(VOCAB[i] for i in text_to_ids(t))


def cer(hyp, ref):
    return edit_distance(hyp, ref) / max(1, len(ref))


def wer_words(hyp, ref):
    h, r = hyp.split(), ref.split()
    return edit_distance(h, r) / max(1, len(r))


def load_recognizer(path):
    cfg = json.load(open(path.replace(".safetensors", ".json")))
    m = Recognizer(**cfg["model"])
    m.load_weights(path)
    m.eval()
    mx.eval(m.parameters())
    return m, cfg


def transcribe(model, cfg, logmels, bs=16):
    """list of (T,80) raw log-mels -> list of strings."""
    mean, std = np.array(cfg["mel_mean"], np.float32), np.array(cfg["mel_std"], np.float32)
    outs = []
    order = np.argsort([len(m) for m in logmels])
    res = {}
    for k in range(0, len(order), bs):
        ii = order[k:k + bs]
        T = max(len(logmels[i]) for i in ii)
        x = np.zeros((len(ii), T, 80), np.float32)
        for b, i in enumerate(ii):
            x[b, : len(logmels[i])] = (logmels[i] - mean) / std
        lp, l2 = model(mx.array(x), mx.array([len(logmels[i]) for i in ii]))
        for i, s in zip(ii, greedy(lp, np.array(l2))):
            res[i] = s
    return [res[i] for i in range(len(logmels))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "ctc", "ctc"))
    ap.add_argument("--log", default=os.path.join(ROOT, "results", "ctc_log.jsonl"))
    ap.add_argument("--max_minutes", type=float, default=12)
    ap.add_argument("--steps", type=int, default=10 ** 9)
    ap.add_argument("--max_frames", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warm", type=int, default=800)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from train import Data, make_batches, lr_at
    info = json.load(open(os.path.join(a.cache, "info.json")))
    mean, std = np.array(info["mel_mean"]), np.array(info["mel_std"])
    tr = Data(os.path.join(a.cache, "train.npz"), mean, std, a.limit)
    va = Data(os.path.join(a.cache, "val.npz"), mean, std, 64)
    labs = [np.array(text_to_ids(t), np.int32) for t in tr.texts]
    mx.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = Recognizer()
    opt = optim.AdamW(learning_rate=a.lr, weight_decay=1e-4)

    def lf(model, x, fl, lab, ll):
        lp, l2 = model(x, fl)
        return ctc_loss(lp, l2, lab, ll)

    vg = nn.value_and_grad(model, lf)
    model.train()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    logf = open(a.log, "w")
    t0 = time.time(); step = 0; epoch = 0; accl = []; best_cer = 1e9

    def save_rec(model, step, vcer):
        model.save_weights(a.out + ".safetensors")
        json.dump({"model": model.cfg, "step": step, "val_cer_64": vcer, "wall_s": time.time() - t0, "mel_mean": info["mel_mean"],
                   "mel_std": info["mel_std"], "vocab": VOCAB, "args": vars(a), "selection": "best val CER (64 val clips, every 500 steps)"},
                  open(a.out + ".json", "w"))

    while step < a.steps and time.time() - t0 < a.max_minutes * 60:
        for bidx in make_batches(tr.lens, a.max_frames, np.random.default_rng(a.seed * 1000 + epoch)):
            if step >= a.steps or time.time() - t0 > a.max_minutes * 60:
                break
            _, _, x, fl = tr.batch(bidx)
            xn = np.array(x)
            for b in range(len(bidx)):  # light SpecAugment: 2 freq masks (<=8 bands), 2 time masks (<=20 frames)
                for _ in range(2):
                    f0 = rng.integers(0, 72); xn[b, :, f0:f0 + rng.integers(0, 9)] = 0
                    t0m = rng.integers(0, max(1, int(fl[b]) - 20)); xn[b, t0m:t0m + rng.integers(0, 21)] = 0
            L = max(len(labs[i]) for i in bidx)
            lab = np.zeros((len(bidx), L), np.int32)
            for b, i in enumerate(bidx):
                lab[b, : len(labs[i])] = labs[i]
            ll = np.array([len(labs[i]) for i in bidx])
            opt.learning_rate = lr_at(step + 1, a.lr, a.warm)
            loss, g = vg(model, mx.array(xn), fl, mx.array(lab), mx.array(ll))
            g, _ = optim.clip_grad_norm(g, 1.0)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state, loss)
            accl.append(float(loss)); step += 1
            if step % 100 == 0:
                rec = {"step": step, "loss": float(np.mean(accl)), "wall_s": time.time() - t0}
                accl = []
                if step % 500 == 0:
                    model.eval()
                    hyp = transcribe(model, {"mel_mean": info["mel_mean"], "mel_std": info["mel_std"]},
                                     [va.mels[va.offs[i]:va.offs[i + 1]].astype(np.float32) for i in range(len(va.ids))])
                    rec["val_cer"] = float(np.mean([cer(h, ref_text(t)) for h, t in zip(hyp, va.texts)]))
                    rec["example"] = hyp[0]
                    if rec["val_cer"] < best_cer:  # keep the val-selected recognizer (a killed run still leaves one)
                        best_cer = rec["val_cer"]; save_rec(model, step, rec["val_cer"])
                    model.train()
                print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
        epoch += 1
    if best_cer == 1e9:  # never validated (tiny smoke runs): save the final weights
        save_rec(model, step, None)
    print("done", step, "best val CER", best_cer if best_cer < 1e9 else "n/a (never validated)", flush=True)


if __name__ == "__main__":
    main()
