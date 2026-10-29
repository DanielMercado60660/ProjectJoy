"""Train the acoustic model with MAS alignment. Checkpoint/resume: weights + optimizer state + step + seeds + config.

  ../bin/gpu_run ../.venv/bin/python train.py --max_minutes 40            # fresh run
  ../bin/gpu_run ../.venv/bin/python train.py --max_minutes 40 --resume   # continue from checkpoints/acoustic/last.*
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from model import Acoustic, loss_fn, align
from text import SYMBOLS, encode

ROOT = os.path.dirname(os.path.abspath(__file__))


class Data:
    def __init__(self, path, mean, std, limit=0):
        z = np.load(path)
        self.ids, self.texts, self.lens, self.offs, self.mels = z["ids"], z["texts"], z["lens"], z["offs"], z["mels"]
        n = limit or len(self.ids)
        self.ids, self.texts, self.lens = self.ids[:n], self.texts[:n], self.lens[:n]
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        self.tok = [np.array(encode(t), np.int32) for t in self.texts]

    def mel(self, i):
        return (self.mels[self.offs[i]:self.offs[i + 1]].astype(np.float32) - self.mean) / self.std

    def batch(self, idxs):
        tl = np.array([len(self.tok[i]) for i in idxs]); fl = np.array([self.lens[i] for i in idxs])
        N = int(np.ceil(tl.max() / 8) * 8); T = int(np.ceil(fl.max() / 16) * 16)
        tok = np.zeros((len(idxs), N), np.int32); x = np.zeros((len(idxs), T, 80), np.float32)
        for b, i in enumerate(idxs):
            tok[b, : tl[b]] = self.tok[i]; x[b, : fl[b]] = self.mel(i)
        return mx.array(tok), mx.array(tl), mx.array(x), mx.array(fl)


def make_batches(lens, max_frames, rng):
    order = np.argsort(lens + rng.random(len(lens)) * 30)  # noisy length sort
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lens[i])
        if cur and m * (len(cur) + 1) > max_frames:
            batches.append(cur); cur, m = [], lens[i]
        cur.append(i); cur_max = m
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def lr_at(step, peak, warm):
    s = max(step, 1)
    return peak * min(s / warm, (warm / s) ** 0.5)


def save(model, opt, meta, prefix):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    model.save_weights(prefix + ".safetensors.tmp.safetensors")
    os.replace(prefix + ".safetensors.tmp.safetensors", prefix + ".safetensors")
    mx.save_safetensors(prefix + ".opt.safetensors", dict(tree_flatten(opt.state)))
    json.dump(meta, open(prefix + ".json", "w"), indent=1)


def evaluate(model, val, idxs, prior_w):
    model.eval()
    tot = np.zeros(3); n = 0; ratio = []
    for k in range(0, len(idxs), 16):
        b = idxs[k:k + 16]
        tok, tl, x, fl = val.batch(b)
        idx, dur = align(model, tok, tl, x, fl, prior_w)
        _, parts = loss_fn(model, tok, tl, x, fl, mx.array(idx), mx.array(dur))
        mx.eval(parts)
        tot += np.array([float(p) for p in parts]) * len(b); n += len(b)
        ratio.append(dur.max(1).mean())
    model.train()
    return dict(zip(["val_mel", "val_prior", "val_dur"], (tot / n).tolist()), val_max_dur=float(np.mean(ratio)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "train_ckpt", "acoustic"))
    ap.add_argument("--log", default=os.path.join(ROOT, "results", "train_log.jsonl"))
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--dec_layers", type=int, default=4)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--max_frames", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warm", type=int, default=2000)
    ap.add_argument("--prior_steps", type=int, default=2000, help="diagonal alignment prior fades out over these steps")
    ap.add_argument("--steps", type=int, default=10 ** 9)
    ap.add_argument("--max_minutes", type=float, default=40)
    ap.add_argument("--save_every_min", type=float, default=4)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--anneal_to", type=float, default=0.0,
                    help="if >0: linearly anneal lr from the schedule's value at the start of this run to this fraction of it, over --max_minutes")
    a = ap.parse_args()

    info = json.load(open(os.path.join(a.cache, "info.json")))
    mean, std = np.array(info["mel_mean"]), np.array(info["mel_std"])
    tr = Data(os.path.join(a.cache, "train.npz"), mean, std, a.limit)
    va = Data(os.path.join(a.cache, "val.npz"), mean, std)
    val_idx = list(range(min(a.n_val, len(va.ids))))

    mx.random.seed(a.seed)
    model = Acoustic(len(SYMBOLS), d=a.d, enc_layers=a.enc_layers, dec_layers=a.dec_layers, ff=a.ff)
    opt = optim.AdamW(learning_rate=a.lr, betas=[0.9, 0.98], weight_decay=1e-6)
    step, epoch, pos, wall_prev, hist = 0, 0, 0, 0.0, []
    if a.resume:
        meta = json.load(open(a.out + ".json"))
        model.load_weights(a.out + ".safetensors")
        opt.state = tree_unflatten(list(mx.load(a.out + ".opt.safetensors").items()))
        step, epoch, pos, wall_prev, hist = meta["step"], meta["epoch"], meta["pos"], meta["wall_s"], meta.get("val_hist", [])
        print(f"resumed at step {step} (epoch {epoch}, batch {pos})", flush=True)
    mx.eval(model.parameters())
    nparams = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"params {nparams / 1e6:.2f} M, train clips {len(tr.ids)}", flush=True)
    model.train()
    vg = nn.value_and_grad(model, loss_fn)

    def meta():
        return {"model": model.cfg, "step": step, "epoch": epoch, "pos": pos, "wall_s": wall_prev + time.time() - t0,
                "seed": a.seed, "args": vars(a), "params": int(nparams), "mel_mean": info["mel_mean"],
                "mel_std": info["mel_std"], "symbols": SYMBOLS, "val_hist": hist}

    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    logf = open(a.log, "a")
    t0 = time.time(); t_save = t0; t_log = t0; s_log = step
    lr_start = lr_at(step + 1, a.lr, a.warm)
    batches = make_batches(tr.lens, a.max_frames, np.random.default_rng(a.seed * 1000 + epoch))
    acc = np.zeros(3); nacc = 0
    while step < a.steps and (time.time() - t0) < a.max_minutes * 60:
        if pos >= len(batches):
            epoch += 1; pos = 0
            batches = make_batches(tr.lens, a.max_frames, np.random.default_rng(a.seed * 1000 + epoch))
        tok, tl, x, fl = tr.batch(batches[pos]); pos += 1
        pw = max(0.0, 1.0 - step / a.prior_steps) if a.prior_steps > 0 else 0.0
        model.eval()
        idx, dur = align(model, tok, tl, x, fl, pw)
        model.train()
        mx.random.seed(a.seed * 100003 + step)
        lr = lr_at(step + 1, a.lr, a.warm)
        if a.anneal_to > 0:
            frac = min(1.0, (time.time() - t0) / (a.max_minutes * 60))
            lr = lr_start * (1 - frac + frac * a.anneal_to)
        opt.learning_rate = lr
        (loss, parts), g = vg(model, tok, tl, x, fl, mx.array(idx), mx.array(dur))
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state, parts)
        step += 1
        acc += np.array([float(p) for p in parts]); nacc += 1
        if step % 100 == 0:
            now = time.time()
            rec = {"step": step, "mel": acc[0] / nacc, "prior": acc[1] / nacc, "dur": acc[2] / nacc,
                   "lr": float(opt.learning_rate), "steps_per_s": (step - s_log) / (now - t_log),
                   "wall_s": wall_prev + now - t0, "prior_w": pw, "peak_gb": mx.get_peak_memory() / 1e9, "max_dur": float(dur.max(1).mean())}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
            acc[:] = 0; nacc = 0; t_log = now; s_log = step
        if step % a.val_every == 0:
            v = evaluate(model, va, val_idx, pw); v["step"] = step; hist.append(v)
            print(json.dumps(v), flush=True); logf.write(json.dumps(v) + "\n"); logf.flush()
            if v["val_mel"] + v["val_dur"] <= min(h["val_mel"] + h["val_dur"] for h in hist):
                save(model, opt, meta(), a.out + "_best")
        if time.time() - t_save > a.save_every_min * 60:
            save(model, opt, meta(), a.out); t_save = time.time()
    save(model, opt, meta(), a.out)
    print(f"done step {step} wall {wall_prev + time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
