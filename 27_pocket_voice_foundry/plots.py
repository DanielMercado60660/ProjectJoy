"""Training curves + an alignment picture.
  ../.venv/bin/python plots.py    # -> media/train_curve.png, media/alignment.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))


def curves():
    tr, va = [], []
    for l in open(os.path.join(ROOT, "results", "train_log.jsonl")):
        r = json.loads(l)
        (va if "val_mel" in r else tr).append(r)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    s = [r["step"] for r in tr]
    for k in ("mel", "prior", "dur"):
        ax[0].plot(s, [r[k] for r in tr], label=f"train {k}", lw=1)
    vs = [r["step"] for r in va]
    for k in ("val_mel", "val_prior", "val_dur"):
        ax[0].plot(vs, [r[k] for r in va], "o-", ms=3, label=k)
    ax[0].set_xlabel("step"); ax[0].set_yscale("log"); ax[0].legend(fontsize=7); ax[0].set_title("acoustic model losses")
    cl = os.path.join(ROOT, "results", "ctc_log.jsonl")
    if os.path.exists(cl):
        rr = [json.loads(l) for l in open(cl)]
        ax[1].plot([r["step"] for r in rr], [r["loss"] for r in rr], lw=1, label="train CTC loss / char")
        v = [r for r in rr if "val_cer" in r]
        ax2 = ax[1].twinx()
        ax2.plot([r["step"] for r in v], [100 * r["val_cer"] for r in v], "o-", color="C3", ms=3, label="val CER %")
        ax2.set_ylabel("val CER %"); ax[1].set_xlabel("step"); ax[1].set_title("CTC recognizer (intelligibility proxy)")
        ax[1].legend(fontsize=7, loc="upper center"); ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "media", "train_curve.png"), dpi=80); plt.close(fig)


def alignment():
    import mlx.core as mx
    from synth import TTS
    from model import align
    from text import encode
    tts = TTS()
    z = np.load(os.path.join(ROOT, "cache", "val.npz"))
    i = 3
    m = z["mels"][z["offs"][i]:z["offs"][i + 1]].astype(np.float32)
    text = str(z["texts"][i]); tok = encode(text)
    x = (m - tts.mean) / tts.std
    _, mu, _ = tts.model.encode(mx.array(np.array(tok, np.int32)[None]), mx.array([len(tok)]))
    ll = np.array(tts.model.log_lik(mu, mx.array(x[None])))[0]
    idx, dur = align(tts.model, mx.array(np.array(tok, np.int32)[None]), mx.array([len(tok)]), mx.array(x[None]), mx.array([len(x)]))
    _, dpred = tts.mel_from_tokens(tok)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    ax[0].imshow(m.T, origin="lower", aspect="auto", cmap="magma", vmin=-11.5, vmax=2); ax[0].set_ylabel("mel band")
    ax[0].set_title(f"val clip {z['ids'][i]}: \"{text[:80]}\"", fontsize=9)
    ax[1].imshow(np.maximum(ll, np.percentile(ll, 5)), origin="lower", aspect="auto", cmap="viridis")
    ax[1].plot(np.arange(len(idx[0][:len(x)])), idx[0][:len(x)], "r-", lw=1, label="MAS path (training target)")
    ax[1].plot(np.cumsum(dpred) - dpred / 2, np.arange(len(dpred)), "w.", ms=3, label="predicted durations (centres)")
    ax[1].set_ylabel("character index"); ax[1].set_xlabel("frame (16 ms)"); ax[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "media", "alignment.png"), dpi=80); plt.close(fig)


if __name__ == "__main__":
    curves()
    alignment()
