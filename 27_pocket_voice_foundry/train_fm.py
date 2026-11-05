"""Train the flow-matching follow-up (model_fm.py) on cache2/ (prep_multi.py). Checkpoint/resume = weights + optimizer
state + step + epoch + batch position + seeds + config; weights-only fp16 snapshots every --snap_every steps (candidates
for the val-side selection in finalize_fm.py / select_fm.py).

  # LJ only (system B)
  ../bin/gpu_run ../.venv/bin/python train_fm.py --sources lj --out runs/fm/lj/ckpt --max_minutes 45
  # multi-speaker pretraining (system C, stage 1): LJ train + LibriTTS-R train speakers + Hi-Fi TTS
  ../bin/gpu_run ../.venv/bin/python train_fm.py --sources lj,libritts,hifi --out runs/fm/multi/ckpt --max_minutes 45
  # LJ fine-tune (system C, stage 2) from a pretrained checkpoint
  ../bin/gpu_run ../.venv/bin/python train_fm.py --sources lj --init runs/fm/multi/ckpt --out runs/fm/ft/ckpt --lr 2e-4 --warm 500
  # continue any run: same command + --resume
Mel normalization always uses 27's LJ train statistics (cache/info.json), so every source lives in LJ's mel space.
"""
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map
from model_fm import FMTTS, loss_fn, align, cap_memory
from text import SYMBOLS, encode
from train import make_batches, lr_at

ROOT = os.path.dirname(os.path.abspath(__file__))


class Source:
    def __init__(self, path):
        z = np.load(os.path.join(path, "index.npz"))
        self.ids, self.texts, self.spk, self.split = z["ids"], z["texts"], z["speakers"], z["splits"]
        self.lens, self.offs = z["lens"], z["offs"]
        self.mels = np.memmap(os.path.join(path, "mels.f16"), np.float16, "r").reshape(-1, 80)
        self.f0 = np.memmap(os.path.join(path, "f0.f16"), np.float16, "r")
        self.f0_stats = {str(k): (float(m), float(s)) for k, m, s in zip(z["spk_names"], z["f0_mean"], z["f0_std"])}


class Data:
    """Union of (source, row) entries for one split. Speaker ids index into `speakers` (fixed at the first run)."""

    def __init__(self, cache, sources, split, speakers, mean, std, limit=0):
        self.src = {s: Source(os.path.join(cache, s)) for s in sources}
        self.items = []
        for s in sources:
            S = self.src[s]
            rows = np.nonzero(S.split == split)[0]
            if limit:
                rows = rows[:limit]
            self.items += [(s, int(i)) for i in rows]
        self.speakers = speakers
        self.spk_id = {k: i for i, k in enumerate(speakers)}
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        self.lens = np.array([self.src[s].lens[i] for s, i in self.items])
        self.tok = [np.array(encode(str(self.src[s].texts[i])), np.int32) for s, i in self.items]

    def batch(self, ids):
        tl = np.array([len(self.tok[k]) for k in ids]); fl = self.lens[ids]
        N = int(np.ceil(tl.max() / 8) * 8); T = int(np.ceil(fl.max() / 16) * 16)
        B = len(ids)
        tok = np.zeros((B, N), np.int32); x = np.zeros((B, T, 80), np.float32)
        lf = np.zeros((B, T), np.float32); vo = np.zeros((B, T), np.float32); spk = np.zeros(B, np.int32)
        for b, k in enumerate(ids):
            s, i = self.items[k]; S = self.src[s]
            a, e = S.offs[i], S.offs[i + 1]
            tok[b, : tl[b]] = self.tok[k]
            x[b, : fl[b]] = (S.mels[a:e].astype(np.float32) - self.mean) / self.std
            f = S.f0[a:e].astype(np.float32); v = f > 0
            m_, s_ = S.f0_stats[str(S.spk[i])]
            lf[b, : fl[b]] = np.where(v, (f - m_) / s_, 0.0); vo[b, : fl[b]] = v
            spk[b] = self.spk_id.get(str(S.spk[i]), 0)
        return tok, tl, x, fl, spk, lf, vo


def save(model, opt, meta, prefix):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    model.save_weights(prefix + ".tmp.safetensors")
    os.replace(prefix + ".tmp.safetensors", prefix + ".safetensors")
    mx.save_safetensors(prefix + ".opt.tmp.safetensors", dict(tree_flatten(opt.state)))
    os.replace(prefix + ".opt.tmp.safetensors", prefix + ".opt.safetensors")
    json.dump(meta, open(prefix + ".json.tmp", "w"), indent=1)
    os.replace(prefix + ".json.tmp", prefix + ".json")


def snapshot(model, meta, prefix):
    """Weights-only fp16 copy (inference candidate)."""
    w = {k: v.astype(mx.float16) for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(prefix + ".safetensors", w)
    json.dump({k: meta[k] for k in ["model", "step", "speakers", "mel_mean", "mel_std", "symbols", "wall_s", "args", "val_hist"]},
              open(prefix + ".json", "w"), indent=1)


def evaluate(model, va, n, prior_w, mm, ms):
    """Val losses on the first n LJ val clips with fixed noise and a fixed grid of flow times (deterministic)."""
    model.eval()
    tot = np.zeros(5); cnt = 0
    for k in range(0, n, 16):
        ids = list(range(k, min(n, k + 16)))
        tok, tl, x, fl, spk, lf, vo = [mx.array(a) for a in va.batch(ids)]
        idx, dur = align(model, tok, tl, spk, x, fl, prior_w)
        B = len(ids)
        x0 = mx.random.normal(x.shape, key=mx.random.key(1234 + k))
        t = mx.array((np.arange(B) + 0.5) / B, dtype=mx.float32)
        _, parts = loss_fn(model, tok, tl, spk, x, fl, mx.array(idx), mx.array(dur), lf, vo, mm, ms, x0, t)
        mx.eval(parts)
        tot += np.array([float(p) for p in parts]) * B; cnt += B
    model.train()
    return dict(zip(["val_cfm", "val_prior", "val_dur", "val_pitch", "val_energy"], (tot / cnt).tolist()))


def main():
    cap_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache2"))
    ap.add_argument("--sources", default="lj")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "fm", "lj", "ckpt"))
    ap.add_argument("--log", default="")
    ap.add_argument("--init", default="", help="start from these weights (fine-tune); speakers table is inherited")
    ap.add_argument("--d", type=int, default=384); ap.add_argument("--enc_layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--enc_k", type=int, default=5)
    ap.add_argument("--dec_d", type=int, default=384); ap.add_argument("--dec_layers", type=int, default=6)
    ap.add_argument("--dec_ff", type=int, default=1536); ap.add_argument("--dec_k", type=int, default=3)
    ap.add_argument("--no_variance", action="store_true")
    ap.add_argument("--max_frames", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warm", type=int, default=2000)
    ap.add_argument("--prior_steps", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=10 ** 9)
    ap.add_argument("--max_minutes", type=float, default=45)
    ap.add_argument("--save_every_min", type=float, default=4)
    ap.add_argument("--snap_every", type=int, default=2000)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--anneal_to", type=float, default=0.0)
    a = ap.parse_args()
    log_path = a.log or a.out + "_log.jsonl"

    info = json.load(open(os.path.join(ROOT, "cache", "info.json")))
    mean, std = np.array(info["mel_mean"], np.float32), np.array(info["mel_std"], np.float32)
    sources = a.sources.split(",")
    if a.resume:
        prev = json.load(open(a.out + ".json")); speakers = prev["speakers"]
    elif a.init:
        prev = json.load(open(a.init + ".json")); speakers = prev["speakers"]
    else:
        speakers = ["lj"]
        for s in sources:
            if s != "lj":
                z = np.load(os.path.join(a.cache, s, "index.npz"))
                speakers += sorted(set(map(str, z["speakers"][z["splits"] == "train"])))
    tr = Data(a.cache, sources, "train", speakers, mean, std, a.limit)
    va = Data(a.cache, ["lj"], "val", speakers, mean, std)
    mm, ms = mx.array(mean), mx.array(std)

    mx.random.seed(a.seed)
    if a.init or a.resume:
        cfg = (prev["model"])
        model = FMTTS(**cfg)
    else:
        model = FMTTS(len(SYMBOLS), len(speakers), d=a.d, enc_layers=a.enc_layers, heads=a.heads, ff=a.ff, enc_k=a.enc_k,
                      dec_d=a.dec_d, dec_layers=a.dec_layers, dec_ff=a.dec_ff, dec_k=a.dec_k, variance=not a.no_variance)
    opt = optim.AdamW(learning_rate=a.lr, betas=[0.9, 0.98], weight_decay=1e-6)
    step, epoch, pos, wall_prev, hist = 0, 0, 0, 0.0, []
    if a.resume:
        model.load_weights(a.out + ".safetensors")
        opt.state = tree_unflatten(list(mx.load(a.out + ".opt.safetensors").items()))
        step, epoch, pos, wall_prev, hist = prev["step"], prev["epoch"], prev["pos"], prev["wall_s"], prev.get("val_hist", [])
        print(f"resumed at step {step} (epoch {epoch}, batch {pos})", flush=True)
    elif a.init:
        w = mx.load(a.init + ".safetensors")
        model.load_weights(list(w.items()))
        print(f"init from {a.init} (step {prev['step']})", flush=True)
    mx.eval(model.parameters())
    nparams = sum(v.size for _, v in tree_flatten(model.parameters()))
    hours = tr.lens.sum() * 256 / 16000 / 3600
    print(f"params {nparams / 1e6:.2f} M, train clips {len(tr.items)} ({hours:.1f} h), speakers {len(speakers)}", flush=True)
    model.train()
    vg = nn.value_and_grad(model, loss_fn)
    prior_offset = prev.get("step", 0) if (a.init and not a.resume) else 0  # fine-tune: alignment prior already faded

    def meta():
        return {"model": model.cfg, "step": step, "epoch": epoch, "pos": pos, "wall_s": wall_prev + time.time() - t0,
                "seed": a.seed, "args": vars(a), "params": int(nparams), "mel_mean": info["mel_mean"], "mel_std": info["mel_std"],
                "symbols": SYMBOLS, "speakers": speakers, "val_hist": hist, "init": a.init}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    logf = open(log_path, "a")
    t0 = time.time(); t_save = t0; t_log = t0; s_log = step
    lr_start = lr_at(step + 1, a.lr, a.warm)
    rng_b = lambda ep: np.random.default_rng(a.seed * 1000 + ep)
    batches = make_batches(tr.lens, a.max_frames, rng_b(epoch))
    pool = ThreadPoolExecutor(1)
    nxt = None
    acc = np.zeros(5); nacc = 0
    while step < a.steps and (time.time() - t0) < a.max_minutes * 60:
        if pos >= len(batches):
            epoch += 1; pos = 0
            batches = make_batches(tr.lens, a.max_frames, rng_b(epoch))
            nxt = None
        cur = nxt.result() if nxt is not None else tr.batch(batches[pos])
        pos += 1
        nxt = pool.submit(tr.batch, batches[pos]) if pos < len(batches) else None
        tok, tl, x, fl, spk, lf, vo = [mx.array(v) for v in cur]
        pw = max(0.0, 1.0 - (step + prior_offset) / a.prior_steps) if a.prior_steps > 0 else 0.0
        model.eval()
        idx, dur = align(model, tok, tl, spk, x, fl, pw)
        model.train()
        key = mx.random.key(a.seed * 100003 + step)
        k1, k2, k3 = mx.random.split(key, 3)
        mx.random.seed(a.seed * 7919 + step)  # dropout
        x0 = mx.random.normal(x.shape, key=k1)
        t = mx.random.uniform(shape=(x.shape[0],), key=k2)
        lr = lr_at(step + 1, a.lr, a.warm)
        if a.anneal_to > 0:
            frac = min(1.0, (time.time() - t0) / (a.max_minutes * 60))
            lr = lr_start * (1 - frac + frac * a.anneal_to)
        opt.learning_rate = lr
        (loss, parts), g = vg(model, tok, tl, spk, x, fl, mx.array(idx), mx.array(dur), lf, vo, mm, ms, x0, t)
        g, gn = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state, parts)
        step += 1
        acc += np.array([float(p) for p in parts]); nacc += 1
        if step % 100 == 0:
            now = time.time()
            rec = {"step": step, **dict(zip(["cfm", "prior", "dur", "pitch", "energy"], (acc / nacc).round(5).tolist())),
                   "gnorm": float(gn), "lr": float(opt.learning_rate), "steps_per_s": round((step - s_log) / (now - t_log), 3),
                   "wall_s": round(wall_prev + now - t0), "prior_w": pw, "epoch": epoch,
                   "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
            acc[:] = 0; nacc = 0; t_log = now; s_log = step
        if step % a.val_every == 0:
            v = evaluate(model, va, min(a.n_val, len(va.items)), 0.0, mm, ms); v["step"] = step; hist.append(v)
            print(json.dumps(v), flush=True); logf.write(json.dumps(v) + "\n"); logf.flush()
        if step % a.snap_every == 0:
            snapshot(model, meta(), a.out + f"_snap{step}")
        if time.time() - t_save > a.save_every_min * 60:
            save(model, opt, meta(), a.out); t_save = time.time()
    save(model, opt, meta(), a.out)
    snapshot(model, meta(), a.out + f"_snap{step}")
    print(f"done step {step} wall {wall_prev + time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
