"""Train Gemma TTS on LJSpeech Mimi tokens (train split), select on val loss.

  ../bin/gpu_run ../.venv/bin/python train.py --out runs/main --max_minutes 40
  ../bin/gpu_run ../.venv/bin/python train.py --out runs/scratch --init random --max_minutes 40   # ablation
  ../.venv/bin/python train.py --out runs/smoke --steps 20 --limit 64                             # smoke

Checkpoints (runs/<name>/{last,best}/): trainable weights, optimizer state, step, config. --resume continues
from last/ with the identical data order (batches are a pure function of seed and epoch).
"""
import argparse, json, math, os, signal, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gemma import load_tokenizer  # noqa: E402
from tts import build_model, delay_grid, make_batch, loss_fn, encode_text  # noqa: E402

NEW = ("audio_emb", "audio_start", "heads")


def load_split(split, nq, limit=0, codes_path=None, text_mode="subword", spk_map=None):
    """Items (text_ids, delayed grid, speaker id). LJ files have no speaker field -> speaker 0. spk_map maps a
    LibriTTS-R speaker string to an id >= 1."""
    z = dict(np.load(codes_path or os.path.join(HERE, "data", "codes32.npz")))  # materialize once
    tok = load_tokenizer()
    idx = [i for i, s in enumerate(z["splits"]) if s == split]
    if limit:
        idx = idx[:limit]
    items, ids = [], []
    for i in idx:
        t = encode_text(tok, str(z["norm"][i]), text_mode)
        c = z["codes"][:nq, z["offs"][i]:z["offs"][i + 1]]
        sp = spk_map[str(z["speaker"][i])] if (spk_map is not None and "speaker" in z) else 0
        items.append((t, delay_grid(c), sp))
        ids.append(str(z["ids"][i]))
    return items, ids


def batches_for_epoch(items, budget, seed, epoch, lens=None):
    """Length-bucketed batches under a padded-token budget; order is a pure function of (seed, epoch)."""
    rng = np.random.default_rng(seed * 1000 + epoch)
    lens = np.array([len(it[0]) + len(it[1]) for it in items]) if lens is None else np.asarray(lens)
    order = np.argsort(lens + rng.uniform(0, 8, len(lens)))  # jitter so buckets differ a little per epoch
    out, cur = [], []
    for i in order:
        L = max([lens[j] for j in cur] + [lens[i]])
        if cur and L * (len(cur) + 1) > budget:
            out.append(cur); cur = []
        cur.append(int(i))
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def save(model, opt, path, meta):
    os.makedirs(path, exist_ok=True)
    mx.save_safetensors(os.path.join(path, "model.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    mx.save_safetensors(os.path.join(path, "optim.safetensors"),
                        {k: v for k, v in tree_flatten(opt.state) if isinstance(v, mx.array)})
    json.dump(meta, open(os.path.join(path, "meta.json"), "w"), indent=1)


def load_into(model, opt, path):
    model.load_weights(list(mx.load(os.path.join(path, "model.safetensors")).items()), strict=False)
    if opt is not None:
        st = mx.load(os.path.join(path, "optim.safetensors"))
        opt.state = tree_unflatten(list(st.items()))
    return json.load(open(os.path.join(path, "meta.json")))


def val_loss(model, items, nq, budget):
    tot, n, per = 0.0, 0, np.zeros(nq)
    for b in batches_for_epoch(items, budget, 12345, 0):
        text, tlen, grid_in, tgt, spk = make_batch([items[i] for i in b], nq, return_spk=True)
        l, pc = loss_fn(model, text, tlen, grid_in, tgt, spk)
        w = int((np.array(tgt[..., 0]) >= 0).sum())
        tot += float(l) * w; per += np.array(pc) * w; n += w
    return tot / n, (per / n).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nq", type=int, default=16)
    ap.add_argument("--init", default="gemma", choices=["gemma", "random"])
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone peak LR")
    ap.add_argument("--lr_new", type=float, default=1e-3, help="audio embeddings / heads peak LR")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--steps", type=int, default=100000, help="cosine horizon / hard stop")
    ap.add_argument("--max_minutes", type=float, default=40)
    ap.add_argument("--budget", type=int, default=12000, help="padded tokens per batch")
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--val_n", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--codes", default=None, help="codes npz (default data/codes32.npz)")
    ap.add_argument("--libritts", default=None, help="comma-separated codes npz files to mix in (train split, with speaker ids), e.g. LibriTTS-R,Hi-Fi TTS")
    ap.add_argument("--progress", type=float, default=0.0, help="PM-RoPE-style progress positions, scale P (0 = off; T5Gemma-TTS uses 2000)")
    ap.add_argument("--prefix_lm", type=int, default=0, help="1 = text prefix attends in both directions")
    ap.add_argument("--n_spk", type=int, default=0, help="speaker-table size override (e.g. to fine-tune a mixed-data checkpoint on LJ only)")
    ap.add_argument("--lj_repeat", type=int, default=2, help="repeat LJ train clips this many times in the mix")
    ap.add_argument("--init_from", default=None, help="checkpoint dir (e.g. runs/pt/best): load its weights, fresh optimizer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--compute", default="float32", choices=["float32", "bfloat16"], help="activation dtype (fp32 master weights)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop_at", type=int, default=0, help="exit after this step (resume test), keeps the LR horizon")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--cb_weight", type=str, default="", help="comma list, e.g. 20,4 (rest 1); default all 1")
    ap.add_argument("--corrupt", type=float, default=0.0, help="fraction of fed-back audio codes randomized")
    ap.add_argument("--text", default="subword", choices=["subword", "chars"])
    ap.add_argument("--select", default="mean", choices=["mean", "cb0"], help="val metric for best/")
    ap.add_argument("--mem_gb", type=float, default=16, help="MLX memory limit (GB); lab rule <= ~16 GB per process")
    ap.add_argument("--cache_gb", type=float, default=2, help="MLX buffer-cache limit (GB); many sequence shapes grow it")
    a = ap.parse_args()
    mx.set_memory_limit(int(a.mem_gb * 2**30))
    mx.set_cache_limit(int(a.cache_gb * 2**30))
    out = os.path.join(HERE, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(out, exist_ok=True)

    train, _ = load_split("train", a.nq, a.limit, a.codes, a.text)
    val, _ = load_split("val", a.nq, a.limit, a.codes, a.text)
    a.n_spk_override, a.n_spk = a.n_spk, 1
    if a.libritts:
        paths = [p for p in a.libritts.split(",") if p]
        spks = set()
        for p in paths:
            spks |= set(map(str, np.load(p)["speaker"]))
        spk_map = {s: i + 1 for i, s in enumerate(sorted(spks))}   # speaker 0 = LJ
        a.n_spk = 1 + len(spk_map)
        extra = []
        for p in paths:
            part, _ = load_split("train", a.nq, a.limit, p, a.text, spk_map)
            print(f"mix: + {os.path.basename(p)} {len(part)} clips", flush=True)
            extra += part
        print(f"mix: LJ {len(train)} clips x{a.lj_repeat} + {len(extra)} extra clips, {a.n_spk} speakers", flush=True)
        train = train * a.lj_repeat + extra
    val = val[: a.val_n]
    dtype = mx.float32 if a.dtype == "float32" else mx.bfloat16
    if a.n_spk_override:
        a.n_spk = a.n_spk_override
    model = build_model(a.nq, a.init, a.seed, dtype, a.n_spk, a.progress, a.prefix_lm)
    model.backbone.compute_dtype = mx.bfloat16 if a.compute == "bfloat16" else mx.float32
    if a.init_from and not (a.resume and os.path.exists(os.path.join(out, "last", "meta.json"))):
        src = os.path.join(HERE, a.init_from) if not os.path.isabs(a.init_from) else a.init_from
        load_into(model, None, src)
        print(f"initialised from {a.init_from} (step {json.load(open(os.path.join(src, 'meta.json')))['step']})", flush=True)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"train {len(train)} clips, val {len(val)}; trainable params {n_train / 1e6:.1f}M", flush=True)

    def sched(peak):
        return optim.join_schedules([optim.linear_schedule(peak * 0.01, peak, a.warmup),
                                     optim.cosine_decay(peak, max(1, a.steps - a.warmup), peak * 0.05)], [a.warmup])
    opt = optim.MultiOptimizer(
        [optim.AdamW(sched(a.lr_new), weight_decay=0.0), optim.AdamW(sched(a.lr), weight_decay=0.01)],
        [lambda path, p: path.split(".")[0] in NEW])
    step, epoch_steps_done, best = 0, 0, float("inf")
    if a.resume and os.path.exists(os.path.join(out, "last", "meta.json")):
        meta = load_into(model, opt, os.path.join(out, "last"))
        step, best = meta["step"], meta["best_val"]
        print(f"resumed at step {step} (best val {best:.4f})", flush=True)

    w = [float(x) for x in a.cb_weight.split(",") if x] if a.cb_weight else []
    cbw = mx.array(w + [1.0] * (a.nq - len(w)))
    lg = nn.value_and_grad(model, lambda m, *b: loss_fn(m, *b, cb_weight=cbw)[0])
    crng = np.random.default_rng(a.seed + 7 + step)

    def train_step(text, tlen, grid_in, tgt, spk):
        loss, g = lg(model, text, tlen, grid_in, tgt, spk)
        g, gn = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        return loss, gn

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    log = open(os.path.join(out, "train_log.jsonl"), "a")
    t0 = time.time()
    tok_window, t_window = 0, time.time()
    meta = lambda: {"step": step, "best_val": best, "args": vars(a), "wall_s": round(time.time() - t0, 1)}
    if step == 0:
        vl0, per0 = val_loss(model, val, a.nq, a.budget)
        rec = {"step": 0, "val_loss": round(vl0, 4), "val_cb": [round(x, 3) for x in per0]}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    while step < a.steps and not stop["now"]:
        epoch_batches = batches_for_epoch(train, a.budget, a.seed, 0)
        nb = len(epoch_batches)
        epoch, k = divmod(step, nb)
        sched_b = batches_for_epoch(train, a.budget, a.seed, epoch)
        for bi in range(k, nb):
            text, tlen, grid_in, tgt, spk = make_batch([train[i] for i in sched_b[bi]], a.nq, a.corrupt, crng, True)
            loss, gn = train_step(text, tlen, grid_in, tgt, spk)
            mx.eval(loss, gn, model.parameters(), opt.state)
            step += 1
            tok_window += int(text.shape[0] * (text.shape[1] + grid_in.shape[1] + 1))
            if step % a.log_every == 0:
                dt = time.time() - t_window
                rec = {"step": step, "epoch": epoch, "loss": round(float(loss), 4), "gnorm": round(float(gn), 3),
                       "tok_s": round(tok_window / dt), "elapsed_min": round((time.time() - t0) / 60, 2),
                       "peak_gb": round(mx.get_peak_memory() / 2**30, 2), "active_gb": round(mx.get_active_memory() / 2**30, 2)}
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                tok_window, t_window = 0, time.time()
            if step % a.val_every == 0:
                vl, per = val_loss(model, val, a.nq, a.budget)
                rec = {"step": step, "val_loss": round(vl, 4), "val_cb": [round(x, 3) for x in per]}
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                crit = per[0] if a.select == "cb0" else vl
                if crit < best:
                    best = crit
                    save(model, opt, os.path.join(out, "best"), meta())
            if step % a.save_every == 0:
                save(model, opt, os.path.join(out, "last"), meta())
            if step >= a.steps or (a.stop_at and step >= a.stop_at) or stop["now"] or (time.time() - t0) / 60 > a.max_minutes:
                stop["now"] = True
                break
    vl, per = val_loss(model, val, a.nq, a.budget)
    rec = {"step": step, "val_loss": round(vl, 4), "val_cb": [round(x, 3) for x in per], "final": True}
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n")
    crit = per[0] if a.select == "cb0" else vl
    if crit < best:
        best = crit
        save(model, opt, os.path.join(out, "best"), meta())
    save(model, opt, os.path.join(out, "last"), meta())
    print(f"done step {step} best val {best:.4f} wall {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
