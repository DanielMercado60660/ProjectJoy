"""GAN fine-tune of 28's vocoder (checkpoints/vocoder28, a 7.7 M Vocos-style generator trained without any adversarial
loss) on LJSpeech train (LJ001-040, the same chapters 28 and the acoustic models use).

Generator: vocoder_model.Vocoder, initialized from 28's checkpoint. Discriminators (new, MLX, HiFi-GAN / UnivNet recipes,
narrower than the originals): multi-period (periods 2,3,5,7,11; Conv2d (5,1) stride (3,1), channels 32-64-128-256-256)
and multi-resolution (|STFT| at (512,128), (1024,256), (2048,512); Conv2d (3,9) stride (1,2), 32 channels). Losses: LSGAN;
generator L = adv + 2 * feature matching + 45 * log-mel L1 (HiFi-GAN weights; mel.py's filterbank). Segments: 64 mel
frames = 16,384 samples (1.02 s) cut at random frame offsets from cached real mels + the waveform they were analysed from
(cache2/lj, prep_wav16.py).

  ../bin/gpu_run ../.venv/bin/python train_voc_gan.py --out runs/voc/gan --max_minutes 45      # (+ --resume to continue)
Checkpoint/resume: G, D, both optimizer states, step, epoch, batch position, seed. Generator snapshots (fp32, loadable
with vocoder_model.load_vocoder) every --snap_every steps.
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from model_fm import cap_memory
from vocoder_model import Vocoder, load_vocoder, mx_stft_mag, _FB
from mel import HOP, LOG_FLOOR

ROOT = os.path.dirname(os.path.abspath(__file__))
SEG = 64  # frames per segment -> (SEG) * HOP samples (one extra input frame below)


class PeriodD(nn.Module):
    def __init__(self, p, ch=(32, 64, 128, 256, 256)):
        super().__init__()
        self.p = p
        cin = 1
        self.convs = []
        for i, c in enumerate(ch):
            st = (3, 1) if i < len(ch) - 1 else (1, 1)
            self.convs.append(nn.Conv2d(cin, c, (5, 1), stride=st, padding=(2, 0)))
            cin = c
        self.post = nn.Conv2d(cin, 1, (3, 1), padding=(1, 0))

    def __call__(self, y):
        B, T = y.shape
        if T % self.p:
            y = mx.pad(y, [(0, 0), (0, self.p - T % self.p)])
        x = y.reshape(B, -1, self.p, 1)
        feats = []
        for c in self.convs:
            x = nn.leaky_relu(c(x), 0.1); feats.append(x)
        x = self.post(x); feats.append(x)
        return x.reshape(B, -1), feats


class ResD(nn.Module):
    def __init__(self, n_fft, hop, ch=32):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.convs = [nn.Conv2d(1, ch, (3, 9), padding=(1, 4))] + \
                     [nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4)) for _ in range(3)] + \
                     [nn.Conv2d(ch, ch, (3, 3), padding=(1, 1))]
        self.post = nn.Conv2d(ch, 1, (3, 3), padding=(1, 1))

    def __call__(self, y):
        x = mx.log(mx_stft_mag(y, self.n_fft, self.hop) + 1e-5)[..., None]  # (B, frames, bins, 1)
        feats = []
        for c in self.convs:
            x = nn.leaky_relu(c(x), 0.1); feats.append(x)
        x = self.post(x); feats.append(x)
        return x.reshape(x.shape[0], -1), feats


class Discs(nn.Module):
    def __init__(self):
        super().__init__()
        self.ds = [PeriodD(p) for p in (2, 3, 5, 7, 11)] + [ResD(n, h) for n, h in ((512, 128), (1024, 256), (2048, 512))]

    def __call__(self, y):
        return [d(y) for d in self.ds]


def log_mel_mx(y):
    S = mx_stft_mag(y, 1024, HOP)
    return mx.log(mx.maximum(S @ _FB, LOG_FLOOR))


def d_loss(D, y, y_hat):
    l = 0.0
    for (r, _), (f, _) in zip(D(y), D(y_hat)):
        l = l + mx.mean((1 - r) ** 2) + mx.mean(f ** 2)
    return l


def g_loss(G, D, mel, y):
    y_hat = G(mel)
    real, fake = D(y), D(y_hat)
    adv = fm = 0.0
    for (r, rf), (f, ff) in zip(real, fake):
        adv = adv + mx.mean((1 - f) ** 2)
        for a, b in zip(rf, ff):
            fm = fm + mx.mean(mx.abs(mx.stop_gradient(a) - b))
    ml = mx.mean(mx.abs(log_mel_mx(y_hat) - log_mel_mx(y)))
    return adv + 2.0 * fm + 45.0 * ml, (adv, fm, ml)


class Segments:
    def __init__(self, split="train", limit=0):
        d = os.path.join(ROOT, "cache2", "lj")
        z = np.load(os.path.join(d, "index.npz"))
        self.offs, self.lens = z["offs"], z["lens"]
        self.mels = np.memmap(os.path.join(d, "mels.f16"), np.float16, "r").reshape(-1, 80)
        self.wav = np.memmap(os.path.join(d, "wav16.i16"), np.int16, "r")
        self.woffs = np.load(os.path.join(d, "wav16_offs.npy"))
        self.rows = np.nonzero((z["splits"] == split) & (z["lens"] > SEG + 2))[0]
        if limit:
            self.rows = self.rows[:limit]

    def batch(self, rows, rng):
        mels, ys = [], []
        for i in rows:
            f0 = int(rng.integers(0, self.lens[i] - SEG - 1))
            m = self.mels[self.offs[i] + f0: self.offs[i] + f0 + SEG + 1].astype(np.float32)  # SEG+1 frames -> SEG*HOP samples
            w = self.wav[self.woffs[i] + f0 * HOP: self.woffs[i] + (f0 + SEG) * HOP].astype(np.float32) / 32767.0
            mels.append(m); ys.append(w)
        return np.stack(mels), np.stack(ys)


def main():
    cap_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "voc", "gan"))
    ap.add_argument("--init", default=os.path.join(ROOT, "checkpoints", "vocoder28", "vocoder.safetensors"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--max_minutes", type=float, default=45)
    ap.add_argument("--steps", type=int, default=10 ** 9)
    ap.add_argument("--save_every_min", type=float, default=4)
    ap.add_argument("--snap_every", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    G, gcfg = load_vocoder(a.init)
    D = Discs()
    og = optim.AdamW(learning_rate=a.lr_g, betas=[0.8, 0.99], weight_decay=0.01)
    od = optim.AdamW(learning_rate=a.lr_d, betas=[0.8, 0.99], weight_decay=0.01)
    step, epoch, pos, wall_prev = 0, 0, 0, 0.0
    if a.resume:
        meta = json.load(open(a.out + ".json"))
        G.load_weights(a.out + "_G.safetensors"); D.load_weights(a.out + "_D.safetensors")
        og.state = tree_unflatten(list(mx.load(a.out + "_optG.safetensors").items()))
        od.state = tree_unflatten(list(mx.load(a.out + "_optD.safetensors").items()))
        step, epoch, pos, wall_prev = meta["step"], meta["epoch"], meta["pos"], meta["wall_s"]
        print(f"resumed at step {step}", flush=True)
    mx.eval(G.parameters(), D.parameters())
    nd = sum(v.size for _, v in tree_flatten(D.parameters()))
    data = Segments("train", a.limit)
    print(f"G {sum(v.size for _, v in tree_flatten(G.parameters())) / 1e6:.2f} M, D {nd / 1e6:.2f} M, train clips {len(data.rows)}", flush=True)
    vgd = nn.value_and_grad(D, d_loss)
    vgg = nn.value_and_grad(G, lambda g, mel, y: g_loss(g, D, mel, y))

    def order(ep):
        return np.random.default_rng(a.seed * 1000 + ep).permutation(data.rows)

    def save():
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        for nm, obj in (("_G", G), ("_D", D)):
            obj.save_weights(a.out + nm + ".tmp.safetensors"); os.replace(a.out + nm + ".tmp.safetensors", a.out + nm + ".safetensors")
        for nm, o in (("_optG", og), ("_optD", od)):
            mx.save_safetensors(a.out + nm + ".tmp.safetensors", dict(tree_flatten(o.state)))
            os.replace(a.out + nm + ".tmp.safetensors", a.out + nm + ".safetensors")
        json.dump({"step": step, "epoch": epoch, "pos": pos, "wall_s": wall_prev + time.time() - t0, "args": vars(a),
                   "model": gcfg["model"], "init": a.init}, open(a.out + ".json", "w"), indent=1)

    def snap():
        G.save_weights(a.out + f"_snap{step}.safetensors")
        json.dump({"model": gcfg["model"], "step": step, "init": a.init, "args": vars(a)}, open(a.out + f"_snap{step}.json", "w"))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    logf = open(a.out + "_log.jsonl", "a")
    t0 = time.time(); t_save = t0; t_log = t0; s_log = step
    perm = order(epoch)
    nb = len(perm) // a.batch
    acc = np.zeros(4); nacc = 0
    while step < a.steps and time.time() - t0 < a.max_minutes * 60:
        if pos >= nb:
            epoch += 1; pos = 0; perm = order(epoch)
        rows = perm[pos * a.batch:(pos + 1) * a.batch]; pos += 1
        m, y = data.batch(rows, np.random.default_rng(a.seed * 1000003 + step))
        m, y = mx.array(m), mx.array(y)
        y_hat = mx.stop_gradient(G(m))
        ld, gd = vgd(D, y, y_hat)
        od.update(D, gd)
        mx.eval(D.parameters(), od.state)
        (lg, parts), gg = vgg(G, m, y)
        gg, gn = optim.clip_grad_norm(gg, 100.0)
        og.update(G, gg)
        mx.eval(G.parameters(), og.state, parts, ld)
        step += 1
        acc += np.array([float(ld)] + [float(p) for p in parts]); nacc += 1
        if step % 50 == 0:
            now = time.time()
            rec = {"step": step, **dict(zip(["d", "adv", "fm", "mel_l1"], (acc / nacc).round(4).tolist())), "g_gnorm": round(float(gn), 2),
                   "steps_per_s": round((step - s_log) / (now - t_log), 3), "wall_s": round(wall_prev + now - t0), "epoch": epoch,
                   "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
            acc[:] = 0; nacc = 0; t_log = now; s_log = step
        if step % a.snap_every == 0:
            snap()
        if time.time() - t_save > a.save_every_min * 60:
            save(); t_save = time.time()
    save(); snap()
    print(f"done step {step}", flush=True)


if __name__ == "__main__":
    main()
