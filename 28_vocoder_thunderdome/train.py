"""Train the Vocos-style MLX vocoder on cached LJSpeech mels with MR-STFT + mel L1 loss.
Hard wall-clock cap (--max_minutes) so the run can't exceed the 20-minute budget under load."""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from model import Vocoder, vocoder_loss
from mel import HOP

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_split(path):
    z = np.load(path)
    n = len(z["ids"])
    return [z[f"wav_{i}"] for i in range(n)], [z[f"mel_{i}"] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "cache", "train.npz"))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--max_minutes", type=float, default=18.5)
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "vocoder"))
    ap.add_argument("--log", default=os.path.join(ROOT, "train_log.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.random.seed(a.seed); mx.random.seed(a.seed)
    wavs, mels = load_split(a.data)
    lens = np.array([len(m) for m in mels])
    ok = np.where(lens > a.frames + 1)[0]
    print(f"{len(ok)} usable clips, {sum(len(w) for w in wavs) / 16000 / 60:.1f} min audio")

    model = Vocoder(dim=a.dim, n_blocks=a.blocks)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"params {n_params / 1e6:.2f} M")
    warm = min(300, a.steps // 10)
    sched = optim.join_schedules([optim.linear_schedule(1e-5, a.lr, warm),
                                  optim.cosine_decay(a.lr, a.steps - warm, 1e-5)], [warm])
    opt = optim.AdamW(learning_rate=sched, betas=[0.8, 0.99], weight_decay=0.01)

    def loss_fn(m, mel, wav):
        return vocoder_loss(m(mel), wav)

    vg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    from functools import partial

    @partial(mx.compile, inputs=state, outputs=state)
    def step(mel, wav):
        (loss, parts), g = vg(model, mel, wav)
        g, _ = optim.clip_grad_norm(g, 10.0)
        opt.update(model, g)
        return loss, parts

    def batch():
        M = np.zeros((a.batch, a.frames, 80), np.float32)
        W = np.zeros((a.batch, (a.frames - 1) * HOP), np.float32)
        for b in range(a.batch):
            i = np.random.choice(ok)
            s = np.random.randint(0, lens[i] - a.frames)
            M[b] = mels[i][s:s + a.frames]
            W[b] = wavs[i][s * HOP:(s + a.frames - 1) * HOP]
        return mx.array(M), mx.array(W)

    def save(stepno, tag=""):
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        model.save_weights(a.out + ".safetensors")
        json.dump({"model": model.cfg, "step": stepno, "args": vars(a), "params": int(n_params),
                   "wall_s": round(time.time() - t0, 1)}, open(a.out + ".json", "w"), indent=1)

    logf = open(a.log, "w")
    t0 = time.time(); ema = None
    for it in range(1, a.steps + 1):
        mel, wav = batch()
        loss, (sc, lm, ml) = step(mel, wav)
        mx.eval(loss, model.parameters(), opt.state)
        l = loss.item(); ema = l if ema is None else 0.98 * ema + 0.02 * l
        el = time.time() - t0
        if it % 50 == 0 or it == 1:
            rec = {"step": it, "loss": round(l, 4), "ema": round(ema, 4), "sc": round(sc.item(), 4),
                   "logmag": round(lm.item(), 4), "mel_l1": round(ml.item(), 4), "lr": float(sched(it)),
                   "elapsed_s": round(el, 1)}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(rec, flush=True)
        if it % 1000 == 0:
            save(it)
        if el > a.max_minutes * 60:
            print(f"wall-clock cap hit at step {it}")
            break
    save(it)
    print(f"done: {it} steps in {(time.time() - t0) / 60:.1f} min -> {a.out}.safetensors")


if __name__ == "__main__":
    main()
