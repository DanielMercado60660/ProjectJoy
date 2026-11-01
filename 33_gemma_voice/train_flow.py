"""Train the flow-matching Gemma Voice (tts_flow.py) on Mimi latents (mimi_io.py encode-latents).

  ../bin/gpu_run ../.venv/bin/python train_flow.py --out runs/f1 --init_from runs/r5/snap_step16000 \
      --mix lj:4,libritts,hifitts --max_minutes 45
  ../.venv/bin/python train_flow.py --out runs/fsmoke --steps 30 --limit 64 --mix lj        # smoke

Data: each set = a codes npz (texts, splits, speakers, frame offsets) + an aligned latent .npy (same offsets).
--init_from loads matching weights (backbone, speaker table, AUDIO_START) from a codebook checkpoint such as r5; the
latent input projection, flow head and stop head start fresh.
"""
import argparse, json, os, signal, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gemma import load_tokenizer  # noqa: E402
from tts_flow import build_model, make_batch, loss_fn, encode_text, BOS  # noqa: E402
from train import batches_for_epoch, save, load_into  # noqa: E402

NEW = ("in_proj", "audio_start", "head", "stop")
SETS = {"lj": ("data/codes32.npz", "data/lj_lat.npy"),
        "libritts": ("data/libritts_codes32.npz", "data/libritts_lat.npy"),
        "hifitts": ("data/hifitts_codes32.npz", "data/hifitts_lat.npy"),
        "librittsx": ("data/librittsx_meta.npz", "data/librittsx_lat.npy")}   # LibriTTS-R clean-360 + other-500, latents only
EXTRA = ("librittsx",)   # sets added after r5: their new speakers get ids after r5's, so older checkpoints keep theirs


class Store:
    """Concatenation of several latent memmaps behind one global frame offset."""
    def __init__(self):
        self.parts, self.base = [], [0]

    def add(self, path):
        m = np.load(os.path.join(HERE, path), mmap_mode="r")
        self.parts.append(m)
        self.base.append(self.base[-1] + m.shape[0])
        return self.base[-2]

    def __getitem__(self, sl):
        k = int(np.searchsorted(self.base, sl.start, side="right") - 1)
        return self.parts[k][sl.start - self.base[k]:sl.stop - self.base[k]]


def latent_stats(names):
    """Frame-weighted per-dim mean/std over the given sets (from encode-latents' .json)."""
    s1 = s2 = 0.0
    n = 0
    for nm in names:
        info = json.load(open(os.path.join(HERE, SETS[nm][1] + ".json")))
        f, mu, sd = info["frames"], np.array(info["mean"]), np.array(info["std"])
        s1 = s1 + f * mu; s2 = s2 + f * (sd ** 2 + mu ** 2); n += f
    mean = s1 / n
    return mean.astype(np.float32), np.sqrt(s2 / n - mean ** 2).astype(np.float32)


def load_set(name, split, store, tok, spk_map, limit=0):
    codes, lat = SETS[name]
    z = np.load(os.path.join(HERE, codes))
    base = store.add(lat)
    splits, offs, lens, norm = z["splits"], z["offs"], z["lens"], z["norm"]
    spkr = z["speaker"] if "speaker" in z.files else None
    idx = [i for i in range(len(splits)) if splits[i] == split]
    if limit:
        idx = idx[:limit]
    items = []
    for i in idx:
        sp = spk_map[str(spkr[i])] if spkr is not None else 0
        items.append((encode_text(tok, str(norm[i]), "chars"), int(base + offs[i]), int(lens[i]), sp))
    return items, [str(z["ids"][i]) for i in idx]


def speaker_map():
    """Same speaker ids as the codebook runs (r5): LibriTTS-R clean-100 + Hi-Fi speakers sorted, LJ = 0. Speakers from
    EXTRA sets (when their files exist) are appended after those, sorted, so r5/f1/f2 ids never move."""
    spks = set()
    for nm in ("libritts", "hifitts"):
        spks |= set(map(str, np.load(os.path.join(HERE, SETS[nm][0]))["speaker"]))
    m = {s: i + 1 for i, s in enumerate(sorted(spks))}
    for nm in EXTRA:
        if os.path.exists(os.path.join(HERE, SETS[nm][0])):
            for s in sorted(set(map(str, np.load(os.path.join(HERE, SETS[nm][0]))["speaker"])) - set(m)):
                m[s] = len(m) + 1
    return m


def val_loss(model, items, store, mean, std, budget):
    key = mx.random.key(12345)
    tot = fl = st = 0.0
    n = 0
    lens = [len(it[0]) + it[2] + 1 for it in items]
    for b in batches_for_epoch(items, budget, 12345, 0, lens):
        bt = make_batch([items[i] for i in b], store, mean, std)
        key, sub = mx.random.split(key)
        l, f, s = loss_fn(model, *bt, sub)
        w = len(b)
        tot += float(l) * w; fl += float(f) * w; st += float(s) * w; n += w
    return tot / n, fl / n, st / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix", default="lj:4,libritts,hifitts", help="sets with optional repeat, e.g. lj:4,libritts,hifitts")
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--lr", type=float, default=5e-5, help="backbone peak LR")
    ap.add_argument("--lr_new", type=float, default=3e-4, help="in_proj / flow head / stop peak LR")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--max_minutes", type=float, default=45)
    ap.add_argument("--budget", type=int, default=12000)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--snap_every", type=int, default=0, help="keep model-only snapshots every N steps (0 = off)")
    ap.add_argument("--val_n", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--K", type=int, default=4, help="flow noise draws per frame")
    ap.add_argument("--in_noise", type=float, default=0.2, help="max sigma of noise on fed-back latents (normalized units)")
    ap.add_argument("--stop_w", type=float, default=1.0)
    ap.add_argument("--text_drop", type=float, default=0.1, help="fraction of clips trained with text = [BOS] only (CFG)")
    ap.add_argument("--head_w", type=int, default=768)
    ap.add_argument("--head_depth", type=int, default=6)
    ap.add_argument("--progress", type=float, default=0.0)
    ap.add_argument("--prefix_lm", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compute", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--mem_gb", type=float, default=16)
    ap.add_argument("--cache_gb", type=float, default=2)
    a = ap.parse_args()
    mx.set_memory_limit(int(a.mem_gb * 2**30))
    mx.set_cache_limit(int(a.cache_gb * 2**30))
    out = os.path.join(HERE, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(out, exist_ok=True)

    tok = load_tokenizer()
    spk_map = speaker_map()
    a.n_spk = 1 + len(spk_map)
    mix = [(p.split(":")[0], int(p.split(":")[1]) if ":" in p else 1) for p in a.mix.split(",") if p]
    mean, std = latent_stats([nm for nm, _ in mix])
    if a.init_from:   # continuing a flow run: keep the normalisation its weights were trained with
        im = os.path.join(HERE, a.init_from, "meta.json") if not os.path.isabs(a.init_from) else os.path.join(a.init_from, "meta.json")
        ia = json.load(open(im))["args"] if os.path.exists(im) else {}
        if "lat_mean" in ia:
            mean, std = np.array(ia["lat_mean"], np.float32), np.array(ia["lat_std"], np.float32)
            print(f"latent stats from {a.init_from}", flush=True)
    a.lat_mean, a.lat_std = mean.tolist(), std.tolist()
    store = Store()
    train = []
    for nm, rep in mix:
        part, _ = load_set(nm, "train", store, tok, spk_map, a.limit)
        print(f"mix: {nm} {len(part)} clips x{rep}", flush=True)
        train += part * rep
    vstore = Store()
    val, _ = load_set("lj", "val", vstore, tok, spk_map, a.limit)
    val = val[: a.val_n]

    model = build_model(a.seed, a.n_spk, a.progress, a.prefix_lm, a.head_w, a.head_depth)
    model.backbone.compute_dtype = mx.bfloat16 if a.compute == "bfloat16" else mx.float32
    if a.init_from and not (a.resume and os.path.exists(os.path.join(out, "last", "meta.json"))):
        src = os.path.join(HERE, a.init_from) if not os.path.isabs(a.init_from) else a.init_from
        w = mx.load(os.path.join(src, "model.safetensors"))
        mine = dict(tree_flatten(model.parameters()))
        keep = {k: v for k, v in w.items() if k in mine and mine[k].shape == v.shape}
        for k, v in w.items():   # speaker table grew (new sets): copy the old rows, new rows stay zero
            if k == "spk_emb.weight" and k in mine and k not in keep and v.shape[1:] == mine[k].shape[1:] \
                    and v.shape[0] < mine[k].shape[0]:
                keep[k] = mx.concatenate([v.astype(mine[k].dtype), mine[k][v.shape[0]:]], axis=0)
        model.load_weights(list(keep.items()), strict=False)
        print(f"initialised {len(keep)} tensors from {a.init_from} (skipped {len(w) - len(keep)})", flush=True)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    n_head = sum(v.size for _, v in tree_flatten(model.head.parameters()))
    print(f"train {len(train)} clips, val {len(val)}; trainable {n_train / 1e6:.1f}M (flow head {n_head / 1e6:.1f}M)",
          flush=True)

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
        print(f"resumed at step {step}", flush=True)

    vg = nn.value_and_grad(model, lambda m, *b: loss_fn(m, *b, K=a.K, stop_w=a.stop_w)[0])
    rng = np.random.default_rng(a.seed + 7 + step)
    key = mx.random.key(a.seed + 11 + step)
    lens = [len(it[0]) + it[2] + 1 for it in train]

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    log = open(os.path.join(out, "train_log.jsonl"), "a")
    t0 = time.time()
    tok_window, t_window = 0, time.time()
    meta = lambda: {"step": step, "best_val": best, "args": vars(a), "wall_s": round(time.time() - t0, 1)}

    def do_val(final=False):
        vl, vf, vs = val_loss(model, val, vstore, mean, std, a.budget)
        rec = {"step": step, "val_loss": round(vl, 4), "val_flow": round(vf, 4), "val_stop": round(vs, 4)}
        if final:
            rec["final"] = True
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        return vl

    if step == 0:
        do_val()
    nb = len(batches_for_epoch(train, a.budget, a.seed, 0, lens))
    while step < a.steps and not stop["now"]:
        epoch, k = divmod(step, nb)
        sched_b = batches_for_epoch(train, a.budget, a.seed, epoch, lens)
        for bi in range(k, nb):
            its = [train[i] for i in sched_b[bi]]
            if a.text_drop > 0:   # classifier-free guidance: some clips see only [BOS] as text
                its = [((BOS,),) + it[1:] if rng.random() < a.text_drop else it for it in its]
            bt = make_batch(its, store, mean, std, a.in_noise, rng)
            key, sub = mx.random.split(key)
            loss, g = vg(model, *bt, sub)
            g, gn = optim.clip_grad_norm(g, 1.0)
            opt.update(model, g)
            mx.eval(loss, gn, model.parameters(), opt.state)
            step += 1
            tok_window += int(bt[0].shape[0] * (bt[0].shape[1] + bt[2].shape[1] + 1))
            if step % a.log_every == 0:
                dt = time.time() - t_window
                rec = {"step": step, "epoch": epoch, "loss": round(float(loss), 4), "gnorm": round(float(gn), 3),
                       "tok_s": round(tok_window / dt), "elapsed_min": round((time.time() - t0) / 60, 2),
                       "peak_gb": round(mx.get_peak_memory() / 2**30, 2)}
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                tok_window, t_window = 0, time.time()
            if step % a.val_every == 0:
                vl = do_val()
                if vl < best:
                    best = vl
                    save(model, opt, os.path.join(out, "best"), meta())
            if a.snap_every and step % a.snap_every == 0:
                p = os.path.join(out, f"snap_step{step}")
                os.makedirs(p, exist_ok=True)
                mx.save_safetensors(os.path.join(p, "model.safetensors"), dict(tree_flatten(model.trainable_parameters())))
                json.dump(meta(), open(os.path.join(p, "meta.json"), "w"), indent=1)
            if step % a.save_every == 0:
                save(model, opt, os.path.join(out, "last"), meta())
            if step >= a.steps or stop["now"] or (time.time() - t0) / 60 > a.max_minutes:
                stop["now"] = True
                break
    vl = do_val(final=True)
    if vl < best:
        best = vl
        save(model, opt, os.path.join(out, "best"), meta())
    save(model, opt, os.path.join(out, "last"), meta())
    print(f"done step {step} best val {best:.4f} wall {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
