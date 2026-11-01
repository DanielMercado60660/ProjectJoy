"""Train the continuous Gemma TTS (tts_mel.py) on LJSpeech log-mels (project 27's cache, same splits), select on val.

  ../bin/gpu_run ../.venv/bin/python train_mel.py --out runs/mel1 --max_minutes 45
  ../.venv/bin/python train_mel.py --out runs/mel_smoke --steps 10 --limit 32 --val_every 5     # smoke
Checkpoints and resume as in train.py (runs/<name>/{last,best}/).
"""
import argparse, json, math, os, signal, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gemma import load_tokenizer  # noqa: E402
from tts import encode_text  # noqa: E402
from tts_mel import build_model, make_batch, loss_fn, mel_stats  # noqa: E402
from train import batches_for_epoch, save, load_into  # noqa: E402

CACHE27 = os.path.join(HERE, "..", "27_pocket_voice_foundry", "cache")
NEW = ("pre1", "pre2", "pre_out", "audio_start", "mel_out", "stop_out")


def load_split(split, limit=0, text_mode="chars"):
    """27's cached 16 kHz log-mels for an LJ split, normalized with 27's train stats; text = LJ's normalized column."""
    z = dict(np.load(os.path.join(CACHE27, f"{split}.npz")))     # materialize once (npz access re-reads per key)
    codes = np.load(os.path.join(HERE, "data", "codes32.npz"))
    norm = {str(i): str(t) for i, t in zip(codes["ids"], codes["norm"])}
    mean, std = mel_stats()
    tok = load_tokenizer()
    n = len(z["ids"]) if not limit else min(limit, len(z["ids"]))
    mels = z["mels"]
    items, ids = [], []
    for i in range(n):
        cid = str(z["ids"][i])
        if cid not in norm:
            continue
        m = mels[z["offs"][i]:z["offs"][i + 1]].astype(np.float32)
        items.append((encode_text(tok, norm[cid], text_mode), (m - mean) / std))
        ids.append(cid)
    return items, ids


def val_loss(model, items, K, budget):
    lens = [len(t) + math.ceil(len(m) / K) for t, m in items]
    tot = np.zeros(3); n = 0
    for b in batches_for_epoch(items, budget, 12345, 0, lens):
        batch = make_batch([items[i] for i in b], K)
        _, parts = loss_fn(model, *batch)
        w = len(b)
        tot += np.array([float(p) for p in parts]) * w; n += w
    l1, l2, ls = tot / n
    return float(l1 + l2), {"l1": round(l1, 4), "l2": round(l2, 4), "stop_bce": round(ls, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="gemma", choices=["gemma", "random"])
    ap.add_argument("--K", type=int, default=4, help="mel frames per autoregressive step")
    ap.add_argument("--p_drop", type=float, default=0.5, help="prenet dropout (also at inference)")
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian noise on fed-back mel chunks (normalized units)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_new", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--stop_at", type=int, default=0)
    ap.add_argument("--max_minutes", type=float, default=45)
    ap.add_argument("--budget", type=int, default=12000)
    ap.add_argument("--val_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--val_n", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compute", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--text", default="chars", choices=["chars", "subword"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mem_gb", type=float, default=16, help="MLX memory limit (GB); lab rule <= ~16 GB per process")
    ap.add_argument("--cache_gb", type=float, default=2, help="MLX buffer-cache limit (GB); many sequence shapes grow it")
    a = ap.parse_args()
    mx.set_memory_limit(int(a.mem_gb * 2**30))
    mx.set_cache_limit(int(a.cache_gb * 2**30))
    a.repr = "mel"
    out = os.path.join(HERE, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(out, exist_ok=True)

    train, _ = load_split("train", a.limit, a.text)
    val, _ = load_split("val", a.limit, a.text)
    val = val[: a.val_n]
    model = build_model(a.init, a.seed, a.K, a.p_drop)
    model.backbone.compute_dtype = mx.bfloat16 if a.compute == "bfloat16" else mx.float32
    print(f"train {len(train)} clips, val {len(val)}; trainable "
          f"{sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 1e6:.1f}M", flush=True)

    def sched(peak):
        return optim.join_schedules([optim.linear_schedule(peak * 0.01, peak, a.warmup),
                                     optim.cosine_decay(peak, max(1, a.steps - a.warmup), peak * 0.05)], [a.warmup])
    opt = optim.MultiOptimizer(
        [optim.AdamW(sched(a.lr_new), weight_decay=0.0), optim.AdamW(sched(a.lr), weight_decay=0.01)],
        [lambda path, p: path.split(".")[0] in NEW])
    step, best = 0, float("inf")
    if a.resume and os.path.exists(os.path.join(out, "last", "meta.json")):
        meta = load_into(model, opt, os.path.join(out, "last"))
        step, best = meta["step"], meta["best_val"]
        print(f"resumed at step {step} (best val {best:.4f})", flush=True)

    model.train()
    lg = nn.value_and_grad(model, lambda m, *b: loss_fn(m, *b)[0])
    lens = [len(t) + math.ceil(len(m) / a.K) for t, m in train]
    nrng = np.random.default_rng(a.seed + 11 + step)
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    log = open(os.path.join(out, "train_log.jsonl"), "a")
    t0 = time.time()
    meta = lambda: {"step": step, "best_val": best, "args": vars(a), "wall_s": round(time.time() - t0, 1)}

    def do_val():
        nonlocal best
        model.eval()
        vl, parts = val_loss(model, val, a.K, a.budget)
        model.train()
        rec = {"step": step, "val_loss": round(vl, 4), **parts}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if vl < best:
            best = vl
            save(model, opt, os.path.join(out, "best"), meta())

    if step == 0:
        do_val()
    tok_n, tw = 0, time.time()
    while step < a.steps and not stop["now"]:
        nb = len(batches_for_epoch(train, a.budget, a.seed, 0, lens))
        epoch, k = divmod(step, nb)
        sched_b = batches_for_epoch(train, a.budget, a.seed, epoch, lens)
        for bi in range(k, nb):
            batch = make_batch([train[i] for i in sched_b[bi]], a.K, a.noise, nrng)
            loss, g = lg(model, *batch)
            g, gn = optim.clip_grad_norm(g, 1.0)
            opt.update(model, g)
            mx.eval(loss, gn, model.parameters(), opt.state)
            step += 1
            tok_n += int(batch[0].shape[0] * (batch[0].shape[1] + batch[2].shape[1] + 1))
            if step % a.log_every == 0:
                rec = {"step": step, "epoch": epoch, "loss": round(float(loss), 4), "gnorm": round(float(gn), 3),
                       "tok_s": round(tok_n / (time.time() - tw)), "elapsed_min": round((time.time() - t0) / 60, 2),
                       "peak_gb": round(mx.get_peak_memory() / 2**30, 2)}
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                tok_n, tw = 0, time.time()
            if step % a.val_every == 0:
                do_val()
            if step % a.save_every == 0:
                save(model, opt, os.path.join(out, "last"), meta())
            if step >= a.steps or (a.stop_at and step >= a.stop_at) or stop["now"] or (time.time() - t0) / 60 > a.max_minutes:
                stop["now"] = True
                break
    do_val()
    save(model, opt, os.path.join(out, "last"), meta())
    print(f"done step {step} best val {best:.4f} wall {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
