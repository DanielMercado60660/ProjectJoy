"""Round 6 trainer: the Vocos-style generator (r6_vocoder.Vocoder) on full LJSpeech LJ001-LJ040 (cache_r6/), two modes
that differ only in the loss, so the adversarial contribution is isolated from data, size, init and step budget:

  --mode gan    Vocos recipe: G = hinge_adv(MPD) + 0.1 hinge_adv(MRD) + FM(MPD) + 0.1 FM(MRD) + 45 x log-mel L1;
                D = hinge(MPD) + 0.1 hinge(MRD). D is new (r6_disc.py), weight-normalised.
  --mode recon  the round-1/3 loss: MR-STFT (SC + log-mag L1 at 512/1024/2048) + 2 x mel L1 (model.vocoder_loss).

Both arms warm-start from --init (default checkpoints/followup_val3k/best, the generator 27 uses), same batch, same crops,
same per-step batch RNG (np.random.default_rng([seed, step])), same G optimiser and LR schedule (cosine over --total_steps).

Chained resumable runs (each <= 45 min wall):
  OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python r6_train.py --mode gan --out checkpoints/r6_runs/gan --total_steps 24000 --max_minutes 43
  (same command again resumes from <out>/state.json: G, D, both optimiser states (incl. their step counters, so the LR
  schedule continues), step, config. Batches depend only on (seed, step), so nothing else needs saving.)
Generator snapshots every --snap_every steps -> <out>/snaps/G_<step>.safetensors (+ .json, loadable by r6_vocoder.load).
Validation (logged only, never test): every --val_every steps on the first --n_val clips of cache_r6/val100.npz.
"""
import argparse, json, os, signal, time
from functools import partial
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from r6_vocoder import Vocoder, HOP
from r6_disc import Discriminators, d_hinge, g_hinge, feat_match, stft_mag
from model import vocoder_loss, _FB
from mel import LOG_FLOOR

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache_r6")
STOP = {"flag": False}


def log_mel_mx(y):
    return mx.log(mx.maximum(stft_mag(y, 1024, HOP) @ _FB, LOG_FLOOR))


_PIDX = {}


def target_stft(y):
    """Complex STFT of the target crop on the generator's own frame grid (n_fft 1024, hop 256, centred, zero-padded)."""
    B, L = y.shape
    n = 1 + L // HOP
    if n not in _PIDX:
        _PIDX[n] = mx.array((np.arange(1024)[None, :] + HOP * np.arange(n)[:, None]).astype(np.int32))
    yp = mx.pad(y, [(0, 0), (512, 512)])
    return mx.fft.rfft(yp[:, _PIDX[n]] * mx.array(np.hanning(1025)[:-1].astype(np.float32)), axis=-1)


def aw(x):
    """anti-wrapping |x - 2 pi round(x / 2 pi)| (APNet)."""
    return mx.abs(x - 2 * np.pi * mx.round(x / (2 * np.pi)))


def phase_loss(ph, y, edge=2, tw=(1.0, 1.0, 1.0)):
    """APNet-style instantaneous-phase + group-delay + instantaneous-frequency anti-wrapping losses between the
    generator's predicted phase and the target's STFT phase, on interior frames (crop edges see zero padding),
    over bins within 60 dB of the crop's peak (phase is meaningless in silence)."""
    S = target_stft(y)[:, edge:-edge]
    p = ph[:, edge:-edge]
    tgt = mx.arctan2(mx.imag(S), mx.real(S))
    mag = mx.abs(S)
    w = mx.stop_gradient((mag > 1e-3 * mx.max(mag, axis=(1, 2), keepdims=True)).astype(mx.float32))
    wm = lambda e, ww: mx.sum(e * ww) / (mx.sum(ww) + 1.0)
    ip = wm(aw(p - tgt), w)
    gd = wm(aw(mx.diff(p, axis=2) - mx.diff(tgt, axis=2)), w[:, :, 1:] * w[:, :, :-1])
    iaf = wm(aw(mx.diff(p, axis=1) - mx.diff(tgt, axis=1)), w[:, 1:] * w[:, :-1])
    return tw[0] * ip + tw[1] * gd + tw[2] * iaf


class Data:
    def __init__(self, frames, cache=CACHE):
        z = np.load(os.path.join(cache, "lj_train_index.npz"))
        self.lens, self.woffs, self.moffs = z["lens"], z["woffs"], z["moffs"]
        self.wav = np.memmap(os.path.join(cache, "lj_train_wav.i16"), np.int16, "r")
        self.mel = np.memmap(os.path.join(cache, "lj_train_mel.f32"), np.float32, "r").reshape(-1, 80)
        self.F = frames
        self.ok = np.nonzero(self.lens > frames + 1)[0]

    def batch(self, B, seed, step):
        rng = np.random.default_rng([seed, step])
        M = np.zeros((B, self.F, 80), np.float32); W = np.zeros((B, (self.F - 1) * HOP), np.float32)
        for b, i in enumerate(rng.choice(self.ok, B)):
            s = int(rng.integers(0, self.lens[i] - self.F))
            M[b] = self.mel[self.moffs[i] + s: self.moffs[i] + s + self.F]
            W[b] = self.wav[self.woffs[i] + s * HOP: self.woffs[i] + (s + self.F - 1) * HOP] / 32768.0
        return mx.array(M), mx.array(W)


def validate(G, n_val, cache=CACHE):
    from metrics import pitch_metrics, mrstft
    z = np.load(os.path.join(cache, "val100.npz"))
    ML, MR, GP = [], [], []
    for i in range(min(n_val, len(z["ids"]))):
        w, m = z[f"wav_{i}"], z[f"mel_{i}"]
        y = G(mx.array(m[None])); mx.eval(y)
        y = np.array(y[0])[:len(w)]
        from mel import log_mel
        ML.append(float(np.mean(np.abs(log_mel(y) - log_mel(w))))); MR.append(mrstft(y, w)); GP.append(pitch_metrics(y, w)["gpe_pct"])
    return {"val_mel_l1": round(float(np.mean(ML)), 4), "val_mrstft": round(float(np.mean(MR)), 4), "val_gpe_pct": round(float(np.mean(GP)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gan", "recon"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=os.path.join(ROOT, "checkpoints", "followup_val3k", "best.safetensors"))
    ap.add_argument("--total_steps", type=int, default=24000, help="LR schedule length and stop step")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--frames", type=int, default=65, help="mel frames per crop -> (frames-1)*256 samples")
    ap.add_argument("--lr_g", type=float, default=2e-4)
    ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--mpd", default="mid")
    ap.add_argument("--mrd_ch", type=int, default=16)
    ap.add_argument("--mrd_coeff", type=float, default=0.1)
    ap.add_argument("--fm_coeff", type=float, default=1.0)
    ap.add_argument("--mel_coeff", type=float, default=45.0)
    ap.add_argument("--phase_w", default="1,1,1", help="weights of the instantaneous-phase, group-delay, inst.-frequency terms")
    ap.add_argument("--mrstft_coeff", type=float, default=0.0, help="gan mode: + coeff x MR-STFT (SC + log-mag at 512/1024/2048)")
    ap.add_argument("--phase_coeff", type=float, default=0.0, help="weight of the APNet-style phase losses (either mode)")
    ap.add_argument("--max_minutes", type=float, default=43)
    ap.add_argument("--save_every_min", type=float, default=5)
    ap.add_argument("--snap_every", type=int, default=2000)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=24)
    ap.add_argument("--stop_at", type=int, default=0, help="stop (and save) at this step; for the resume test")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=CACHE)
    a = ap.parse_args()
    mx.set_memory_limit(14 * 2 ** 30); mx.set_cache_limit(2 * 2 ** 30)
    signal.signal(signal.SIGTERM, lambda *_: STOP.update(flag=True))
    os.makedirs(os.path.join(a.out, "snaps"), exist_ok=True)
    state_path = os.path.join(a.out, "state.json")
    resume = os.path.exists(state_path)
    if resume:
        st = json.load(open(state_path))
        cfg = st["args"]; cfg.update({k: getattr(a, k) for k in ("max_minutes", "stop_at", "no_compile")})
        a = argparse.Namespace(**cfg)
    mx.random.seed(a.seed); np.random.seed(a.seed)

    icfg = json.load(open(a.init.replace(".safetensors", ".json")))["model"]
    G = Vocoder(**icfg); G.load_weights(a.init)
    D = Discriminators(a.mpd, a.mrd_ch) if a.mode == "gan" else None
    sg = optim.cosine_decay(a.lr_g, a.total_steps, a.lr_g * 0.1)
    og = optim.AdamW(learning_rate=sg, betas=[0.8, 0.99], weight_decay=0.01)
    od = optim.AdamW(learning_rate=optim.cosine_decay(a.lr_d, a.total_steps, a.lr_d * 0.1), betas=[0.8, 0.99], weight_decay=0.01) if D else None
    step, wall_prev = 0, 0.0
    if resume:
        G.load_weights(os.path.join(a.out, "G.safetensors"))
        og.state = tree_unflatten(list(mx.load(os.path.join(a.out, "optG.safetensors")).items()))
        if D:
            D.load_weights(os.path.join(a.out, "D.safetensors"))
            od.state = tree_unflatten(list(mx.load(os.path.join(a.out, "optD.safetensors")).items()))
        step, wall_prev = st["step"], st["wall_s"]
        print(f"resumed {a.mode} at step {step}", flush=True)
    else:
        og.init(G.trainable_parameters())
        if D:
            od.init(D.trainable_parameters())
    mx.eval(G.parameters(), og.state)
    if D:
        mx.eval(D.parameters(), od.state)
    npar = lambda m: sum(v.size for _, v in tree_flatten(m.parameters())) / 1e6
    a.cache = getattr(a, "cache", CACHE)
    data = Data(a.frames, a.cache)
    print(f"mode {a.mode}: G {npar(G):.2f} M" + (f", D {npar(D):.2f} M" if D else "") + f", {len(data.ok)} clips, step {step}/{a.total_steps}", flush=True)

    PW = tuple(float(v) for v in getattr(a, "phase_w", "1,1,1").split(","))
    if a.mode == "gan":
        def d_loss(Dm, y, yh):
            mp_r, mr_r = Dm(y); mp_f, mr_f = Dm(yh)
            return d_hinge(mp_r, mp_f) + a.mrd_coeff * d_hinge(mr_r, mr_f)

        def g_loss(Gm, mel, y):
            logmag, ph = Gm.head_out(mel)
            mag = mx.minimum(mx.exp(logmag), 1e2)
            yh = Gm.istft(mag * mx.cos(ph), mag * mx.sin(ph))
            mp_r, mr_r = D(y); mp_f, mr_f = D(yh)
            adv = g_hinge(mp_f) + a.mrd_coeff * g_hinge(mr_f)
            fm = feat_match(mp_r, mp_f) + a.mrd_coeff * feat_match(mr_r, mr_f)
            ml = mx.mean(mx.abs(log_mel_mx(yh) - log_mel_mx(y)))
            pl = phase_loss(ph, y, tw=PW) if a.phase_coeff > 0 else mx.array(0.0)
            tot = adv + a.fm_coeff * fm + a.mel_coeff * ml + a.phase_coeff * pl
            if getattr(a, "mrstft_coeff", 0.0) > 0:
                _, (sc, lmag, _) = vocoder_loss(yh, y)
                tot = tot + a.mrstft_coeff * (sc + lmag)
            return tot, (adv, fm, ml, pl)

        vgd = nn.value_and_grad(D, d_loss)
        vgg = nn.value_and_grad(G, g_loss)
        state = [G.state, D.state, og.state, od.state]

        def train_step(mel, y):
            yh = mx.stop_gradient(G(mel))
            ld, gd = vgd(D, y, yh)
            gd, dn = optim.clip_grad_norm(gd, 100.0)
            od.update(D, gd)
            (lg, parts), gg = vgg(G, mel, y)
            gg, gn = optim.clip_grad_norm(gg, 100.0)
            og.update(G, gg)
            return mx.stack([ld, lg, parts[0], parts[1], parts[2], parts[3], dn, gn])
        names = ["d", "g", "adv", "fm", "mel_l1", "phase", "d_gnorm", "g_gnorm"]
    else:
        def r_loss(Gm, mel, y):
            logmag, ph = Gm.head_out(mel)
            mag = mx.minimum(mx.exp(logmag), 1e2)
            l, (sc, lm, ml) = vocoder_loss(Gm.istft(mag * mx.cos(ph), mag * mx.sin(ph)), y)
            pl = phase_loss(ph, y, tw=PW) if a.phase_coeff > 0 else mx.array(0.0)
            return l + a.phase_coeff * pl, (sc, lm, ml, pl)
        vgg = nn.value_and_grad(G, r_loss)
        state = [G.state, og.state]

        def train_step(mel, y):
            (l, (sc, lm, ml, pl)), gg = vgg(G, mel, y)
            gg, gn = optim.clip_grad_norm(gg, 10.0)
            og.update(G, gg)
            return mx.stack([l, sc, lm, ml, pl, gn])
        names = ["loss", "sc", "logmag", "mel_l1", "phase", "g_gnorm"]
    if not a.no_compile:
        train_step = partial(mx.compile, inputs=state, outputs=state)(train_step)

    def save():
        def put(obj_save, name):
            tmp = os.path.join(a.out, name + ".tmp.safetensors"); obj_save(tmp); os.replace(tmp, os.path.join(a.out, name + ".safetensors"))
        put(G.save_weights, "G"); put(lambda p: mx.save_safetensors(p, dict(tree_flatten(og.state))), "optG")
        if D:
            put(D.save_weights, "D"); put(lambda p: mx.save_safetensors(p, dict(tree_flatten(od.state))), "optD")
        json.dump({"step": step, "wall_s": round(wall_prev + time.time() - t0, 1), "args": vars(a), "model": G.cfg,
                   "disc": D.cfg if D else None}, open(state_path + ".tmp", "w"), indent=1)
        os.replace(state_path + ".tmp", state_path)

    def snap():
        p = os.path.join(a.out, "snaps", f"G_{step}")
        G.save_weights(p + ".safetensors")
        json.dump({"model": G.cfg, "step": step, "mode": a.mode, "init": a.init, "args": vars(a)}, open(p + ".json", "w"), indent=1)

    logf = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time(); t_save = t0; t_log = t0; s_log = step
    acc = np.zeros(len(names)); nacc = 0
    while step < a.total_steps:
        mel, y = data.batch(a.batch, a.seed, step)
        out = train_step(mel, y)
        mx.eval(out, state)
        step += 1
        acc += np.array(out.tolist()); nacc += 1
        if step % 50 == 0 or step == 1:
            now = time.time()
            rec = {"step": step, **{k: round(float(v), 4) for k, v in zip(names, acc / nacc)},
                   "lr_g": float(sg(step)), "steps_per_s": round((step - s_log) / max(now - t_log, 1e-6), 3),
                   "wall_s": round(wall_prev + now - t0), "peak_gb": round(mx.get_peak_memory() / 2 ** 30, 2)}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
            acc[:] = 0; nacc = 0; t_log = now; s_log = step
        if step % a.snap_every == 0 or step == a.total_steps:
            snap()
        if step % a.val_every == 0 or step == a.total_steps:
            tv = time.time(); v = validate(G, a.n_val, a.cache)
            rec = {"step": step, **v, "val_s": round(time.time() - tv, 1)}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
            t_log = time.time(); s_log = step
        if time.time() - t_save > a.save_every_min * 60:
            save(); t_save = time.time()
        if STOP["flag"] or (a.stop_at and step >= a.stop_at) or time.time() - t0 > a.max_minutes * 60:
            break
    save()
    print(f"saved at step {step} ({'SIGTERM' if STOP['flag'] else 'done' if step >= a.total_steps else 'paused'}), "
          f"session {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
