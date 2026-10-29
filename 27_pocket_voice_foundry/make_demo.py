"""Build the listening demo from the saved checkpoints (re-synthesizes everything; needs no eval outputs).

  media/heldout1_<system>.wav    held-out test sentence 1 x 7 systems, loudness-matched to -23 LUFS (all 4 sentences are in demo.html)
  media/hard<k>_<system>.wav     3 hand-written hard sentences x {tts, baseline} (neural vocoder)
  media/mels.png                 log-mel strip: recording / TTS / baseline for held-out sentence 1
  media/demo.html                offline blind page: per sentence the systems get shuffled letters, revealed on click
  ../.venv/bin/python make_demo.py
"""
import base64, json, os, subprocess, tempfile
import numpy as np
import soundfile as sf
from scipy.signal import lfilter
from mel import SR, load_ljspeech_16k, log_mel
from synth import TTS, BaselineTTS, NeuralVocoder, gl, write_wav
from text import normalize, encode

ROOT = os.path.dirname(os.path.abspath(__file__))
LJ = os.path.join(ROOT, "..", "data", "ljspeech", "LJSpeech-1.1", "wavs")
TARGET_LUFS = -23.0
SYSTEMS = [("rec", "Real recording (16 kHz)"), ("gt_neural", "Real mel -> 28's neural vocoder (upper bound)"),
           ("gt_gl", "Real mel -> Griffin-Lim 64 (upper bound for GL)"), ("tts_neural", "Learned TTS -> neural vocoder"),
           ("tts_gl", "Learned TTS -> Griffin-Lim 64"), ("baseline_neural", "Baseline (template concat) -> neural vocoder"),
           ("baseline_gl", "Baseline (template concat) -> Griffin-Lim 64")]
HARD = [0, 3, 9]  # indices into hard_sentences.txt


# --- loudness (BS.1770-4, mono): copied from ../28_vocoder_thunderdome/make_demo.py (pyloudnorm isn't installed) ---
def _biquad(kind, fc, Q, G=0.0, fs=SR):
    A = 10 ** (G / 40); w0 = 2 * np.pi * fc / fs; al = np.sin(w0) / (2 * Q); c = np.cos(w0)
    if kind == "shelf":
        b = [A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * c),
             A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al)]
        a = [(A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * c),
             (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al]
    else:
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]; a = [1 + al, -2 * c, 1 - al]
    return np.array(b) / a[0], np.array(a) / a[0]


def lufs(x):
    b1, a1 = _biquad("shelf", 1681.974450955533, 0.7071752369554196, 3.999843853973347)
    b2, a2 = _biquad("hp", 38.13547087602444, 0.5003270373238773)
    y = lfilter(b2, a2, lfilter(b1, a1, x.astype(np.float64)))
    blk, step = int(0.4 * SR), int(0.1 * SR)
    z = np.array([np.mean(y[i:i + blk] ** 2) for i in range(0, len(y) - blk + 1, step)])
    l = -0.691 + 10 * np.log10(z + 1e-12)
    z = z[l > -70]
    rel = -0.691 + 10 * np.log10(np.mean(z)) - 10
    z = z[-0.691 + 10 * np.log10(z) > rel]
    return float(-0.691 + 10 * np.log10(np.mean(z)))
# --- end copy ---


def level(y):
    g = 10 ** ((TARGET_LUFS - lufs(y)) / 20)
    y = y * g
    pk = np.max(np.abs(y))
    return y * (0.95 / pk) if pk > 0.95 else y


def mp3_b64(y):
    with tempfile.TemporaryDirectory() as d:
        w, m = os.path.join(d, "a.wav"), os.path.join(d, "a.mp3")
        sf.write(w, y, SR, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", w, "-codec:a", "libmp3lame", "-b:a", "96k", m], check=True)
        return base64.b64encode(open(m, "rb").read()).decode()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--media", default=os.path.join(ROOT, "media"))
    ap.add_argument("--n_held", type=int, default=4)
    ap.add_argument("--tts", default=None)
    ap.add_argument("--baseline", default=None)
    a = ap.parse_args()
    os.makedirs(a.media, exist_ok=True)
    z = np.load(os.path.join(a.cache, "test.npz"))
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(z["ids"]), size=min(100, len(z["ids"])), replace=False))  # same subset as eval.py
    sel = [i for i in sel if 2.5 * SR / 256 <= z["lens"][i] <= 5.5 * SR / 256][: a.n_held]
    tts = TTS(a.tts) if a.tts else TTS()
    bl, voc = (BaselineTTS(a.baseline) if a.baseline else BaselineTTS()), NeuralVocoder()
    items = []
    for k, i in enumerate(sel):
        cid, text = str(z["ids"][i]), str(z["texts"][i])
        rec = load_ljspeech_16k(os.path.join(LJ, cid + ".wav"))
        gm = log_mel(rec)
        tm, _ = tts.mel_from_tokens(encode(text)); bm, _ = bl.mel_from_tokens(encode(text))
        ys = {"rec": rec, "gt_neural": voc(gm), "gt_gl": gl(gm), "tts_neural": voc(tm), "tts_gl": gl(tm), "baseline_neural": voc(bm),
              "baseline_gl": gl(bm)}
        if k == 0:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
            for axi, (nm, m) in zip(ax, [("real recording", gm), ("learned TTS", tm), ("baseline", bm)]):
                axi.imshow(m.T, origin="lower", aspect="auto", vmin=-11.5, vmax=2, cmap="magma"); axi.set_ylabel(nm)
            ax[0].set_title(f"held-out {cid}: \"{text[:70]}\"", fontsize=9); ax[-1].set_xlabel("frame (16 ms)")
            fig.tight_layout(); fig.savefig(os.path.join(a.media, "mels.png"), dpi=80); plt.close(fig)
        items.append({"kind": "held-out test sentence", "id": cid, "text": text, "ys": ys})
    hard = [l.split("||")[0].strip() for l in open(os.path.join(ROOT, "hard_sentences.txt")) if l.strip() and not l.startswith("#")]
    for k in HARD:
        t = hard[k]
        tm = tts(t); bm = bl(t)
        items.append({"kind": "hand-written hard sentence (no recording exists)", "id": f"hard{k}", "text": t + "  [normalized: " + normalize(t) + "]",
                      "ys": {"tts_neural": voc(tm), "baseline_neural": voc(bm)}})
    names = dict(SYSTEMS)
    page = []
    prng = np.random.default_rng(7)
    for k, it in enumerate(items):
        keys = list(it["ys"])
        perm = prng.permutation(len(keys))
        opts = []
        for j, p in enumerate(perm):
            key = keys[p]
            y = level(it["ys"][key].astype(np.float64))
            tag = f"heldout{k + 1}" if it["kind"].startswith("held") else it["id"]
            if k == 0 or not it["kind"].startswith("held"):  # WAV copies for group 1 + hard sentences only (size)
                write_wav(os.path.join(a.media, f"{tag}_{key}.wav"), y)
            opts.append({"letter": "ABCDEFG"[j], "lab": base64.b64encode(names[key].encode()).decode(), "audio": mp3_b64(y)})  # label hidden until reveal
        page.append({"kind": it["kind"], "id": it["id"], "text": it["text"], "opts": opts})
    html = open(os.path.join(ROOT, "demo_template.html")).read().replace("__DATA__", json.dumps(page))
    open(os.path.join(a.media, "demo.html"), "w").write(html)
    print("wrote", len(page), "groups to", os.path.join(a.media, "demo.html"))


if __name__ == "__main__":
    main()
