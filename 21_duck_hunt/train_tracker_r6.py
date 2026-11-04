"""R6 learned tracker: DuckNet (same heatmap + sub-cell offset head as train_tracker.py) trained on
domain-randomized scenes (scene_r6.py), optionally with temporal input: the last `nf` frames stacked
on the channel axis (frames before the first are repeated), so the net can use motion like the
classical tracker's background subtraction. Extra per-sample photometric jitter, applied identically
to every frame of a stack. Model selection on DR val scenes only (seeds 110000+); the held-out sets
are never touched here.

  ../bin/gpu_run ../.venv/bin/python train_tracker_r6.py --nf 3 --out checkpoints/r6_tracker_cnn_nf3
"""
import argparse, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import scene as S
import scene_r6 as R
import train_tracker as TT

CELL, G = TT.CELL, TT.G


class DuckNetT(nn.Module):
    def __init__(self, nf=3, c=48):
        super().__init__()
        self.nf = nf
        self.c1 = nn.Conv2d(3 * nf, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c * 3 // 2, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(c * 3 // 2, 2 * c, 3, stride=2, padding=1)
        self.c4 = nn.Conv2d(2 * c, 2 * c, 3, padding=1)
        self.c5 = nn.Conv2d(2 * c, 2 * c, 3, padding=1)
        self.c6 = nn.Conv2d(2 * c, 2 * c, 3, padding=2, dilation=2)   # wider context for "which bird"
        self.head = nn.Conv2d(2 * c, 3, 1)

    def __call__(self, x):
        h = nn.relu(self.c1(x)); h = nn.relu(self.c2(h)); h = nn.relu(self.c3(h))
        h = nn.relu(self.c4(h)) + h; h = nn.relu(self.c5(h)) + h; h = nn.relu(self.c6(h)) + h
        return self.head(h)


def stack_idx(f, nf):
    return np.maximum(f[:, None] - np.arange(nf)[None], 0)            # (B, nf): f, f-1, ...


def make_input(scenes, si, fi, nf):
    """scenes (N,F,H,W,3) uint8 -> (B,H,W,3nf) float in [-0.5, 0.5]"""
    x = scenes[si[:, None], stack_idx(fi, nf)]                          # (B,nf,H,W,3)
    return x.transpose(0, 2, 3, 1, 4).reshape(len(si), S.H, S.W, 3 * nf).astype(np.float32) / 255.0


def jitter(x, rng, nf):
    B = len(x)
    g = rng.uniform(0.5, 1.5, (B, 1, 1, 1)) * np.tile(rng.uniform(0.8, 1.2, (B, 1, 1, 3)), (1, 1, 1, nf))
    x = x * g + rng.uniform(-0.05, 0.05, (B, 1, 1, 1))
    x = x + rng.normal(0, 1, x.shape) * rng.uniform(0, 0.04, (B, 1, 1, 1))
    return np.clip(x, 0, 1).astype(np.float32)


def forward_scene(model, frames, bs=256):
    """frames (F,H,W,3) uint8 of one scene -> raw net output (F,G,G,3)"""
    F = len(frames); outs = []
    for s in range(0, F, bs):
        fi = np.arange(s, min(F, s + bs))
        x = make_input(frames[None], np.zeros(len(fi), int), fi, model.nf)
        outs.append(np.array(model(mx.array(x - 0.5))))
    return np.concatenate(outs)


def predict(model, frames):
    return TT.decode(forward_scene(model, frames))


def peaks(model, frames, k=3):
    """top-k heatmap cells per frame, at least 2 cells apart -> list of [(x, y, p)]"""
    o = forward_scene(model, frames); B = len(o)
    lg = o[..., 0].reshape(B, -1)
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    out = []
    for b in range(B):
        cs = []
        for i in np.argsort(-p[b])[:20]:
            gy, gx = divmod(int(i), G)
            if any(abs(gx - c[3]) < 2 and abs(gy - c[4]) < 2 for c in cs):
                continue
            dx, dy = o[b, gy, gx, 1:]
            cs.append(((gx + 0.5 + dx) * CELL - 0.5, (gy + 0.5 + dy) * CELL - 0.5, p[b, i], gx, gy))
            if len(cs) == k:
                break
        out.append([c[:3] for c in cs])
    return out


def load(path):
    cfg = json.load(open(path + ".json"))["cfg"]
    m = DuckNetT(cfg["nf"], cfg["c"]); m.load_weights(path + ".safetensors"); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nf", type=int, default=3)
    ap.add_argument("--c", type=int, default=48)
    ap.add_argument("--scenes", type=int, default=2000)
    ap.add_argument("--val_scenes", type=int, default=40)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/r6_tracker_cnn_nf3")
    ap.add_argument("--log", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--dr", default="full", choices=["full", "nodist", "haze"],
                    help="nodist: ablation without DR distractors; haze: round-2 DR v2 (DR + global haze), DR v2 val")
    ap.add_argument("--no_cache", action="store_true", help="render val instead of using the cache (smoke test)")
    a = ap.parse_args()
    log = a.log or a.out.replace("checkpoints/", "results/") + "_train_log.jsonl"
    t0 = time.time()
    X, GT, VIS = R.make_set_dr(a.scenes, R.DR_SEED["dr"], fn=dict(full=R.render_dr, nodist=R.render_dr_nodist, haze=R.render_dr2)[a.dr])
    if a.no_cache:
        Xv, GTv, VISv = R.make_set_dr(a.val_scenes, R.DR_SEED["drval"])
    else:
        Xv, GTv, VISv = R.dr2_set("dr2val", a.val_scenes) if a.dr == "haze" else R.dr_set("drval", a.val_scenes)
    print(f"data {X.shape} rendered in {time.time()-t0:.0f}s", flush=True)
    mx.random.seed(a.seed); rng = np.random.default_rng(a.seed)
    model = DuckNetT(a.nf, a.c); mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=optim.cosine_decay(a.lr, a.steps, 1e-5), weight_decay=1e-4)
    lg = nn.value_and_grad(model, TT.loss_fn)
    flog = open(log, "w"); best = 1e9; run = []
    for step in range(1, a.steps + 1):
        si = rng.integers(0, len(X), a.bs); fi = rng.integers(0, S.N_FRAMES, a.bs)
        x = jitter(make_input(X, si, fi, a.nf), rng, a.nf)
        g = GT[si, fi].copy(); w = VIS[si, fi].astype(np.float32)
        flip = rng.random(a.bs) < 0.5
        x[flip] = x[flip, :, ::-1]; g[flip, 0] = S.W - 1 - g[flip, 0]
        idx, off = TT.targets(g)
        l, gr = lg(model, mx.array(x - 0.5), mx.array(idx), mx.array(off), mx.array(w))
        opt.update(model, gr); mx.eval(model.parameters(), opt.state)
        run.append(l.item())
        if step % a.eval_every == 0 or step == a.steps:
            xy = np.stack([predict(model, xv)[0] for xv in Xv])
            err = np.linalg.norm(xy - GTv, axis=-1)[VISv]
            rec = dict(step=step, wall=time.time() - t0, loss=float(np.mean(run[-a.eval_every:])),
                       val_mean_err=float(err.mean()), val_lost=float((err > 5).mean()),
                       val_clipped=float(np.minimum(err, 20).mean()))
            flog.write(json.dumps(rec) + "\n"); flog.flush(); print(rec, flush=True)
            if rec["val_clipped"] < best:
                best = rec["val_clipped"]
                model.save_weights(a.out + ".safetensors")
                json.dump(dict(step=step, cfg=vars(a), val=rec, flight_sha=R.flight_sha()), open(a.out + ".json", "w"))
    print("best val clipped err", best, "wall", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
