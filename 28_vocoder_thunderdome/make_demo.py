"""Build the listening demo from results/wavs (written by eval.py):
  media/<id>_{gt,gl64,neural}.wav  loudness-matched to -23 LUFS (BS.1770 K-weighted, gated; implemented
                                   here because pyloudnorm isn't installed), same gain cap per clip
  media/spec_<id>.png              spectrograms GT / Griffin-Lim / neural, plus a zoom on a plosive onset
  media/demo.html                  offline blind A/B/C page, labels shuffled per clip, reveal button
"""
import base64, io, json, os
import numpy as np
import soundfile as sf
from scipy.signal import lfilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mel import SR, stft
from metrics import hf_env_db, gt_onsets

ROOT = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(ROOT, "media")
TARGET_LUFS = -23.0
METHODS = [("gt", "Ground truth"), ("gl64", "Griffin-Lim (64 it)"), ("neural", "Neural (MLX, 400 clips)")]


def _biquad(kind, fc, Q, G=0.0, fs=SR):
    A = 10 ** (G / 40); w0 = 2 * np.pi * fc / fs; al = np.sin(w0) / (2 * Q); c = np.cos(w0)
    if kind == "shelf":
        b = [A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * c),
             A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al)]
        a = [(A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * c),
             (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al]
    else:  # high-pass
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]; a = [1 + al, -2 * c, 1 - al]
    return np.array(b) / a[0], np.array(a) / a[0]


def lufs(x):
    """ITU-R BS.1770-4 integrated loudness (mono), filter params as in pyloudnorm."""
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


def spec_db(x):
    S = np.abs(stft(x, 1024, 128))
    return 20 * np.log10(S.T + 1e-5)


def main():
    global MEDIA
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--media", default=MEDIA)
    ap.add_argument("--ablation", default=os.path.join(ROOT, "results_3k"),
                    help="results dir of the 3000-clip ablation; added as a 4th blind option if present")
    a = ap.parse_args()
    global METHODS
    abl = os.path.join(a.ablation, "metrics.json")
    if os.path.exists(abl):
        METHODS = METHODS + [("neural3k", "Neural, 3000-clip ablation")]
        res_abl = json.load(open(abl))
    MEDIA = a.media
    os.makedirs(MEDIA, exist_ok=True)
    res = json.load(open(os.path.join(a.results, "metrics.json")))
    split = json.load(open(os.path.join(a.cache, "split.json")))["test"]
    z = np.load(os.path.join(a.cache, "test.npz"))
    ids = list(z["ids"])
    # 4 plosive-heavy clips + 2 others, fixed order
    trans = [d for d in split if d["transient"]][:4]
    other = [d for d in split if not d["transient"]][:2]
    if not other:
        other = [d for d in split if d not in trans][:2]
    clips = trans + other
    pc = {(r["method"], r["id"]): r for r in res["per_clip"]}
    if any(m == "neural3k" for m, _ in METHODS):
        pc.update({("neural3k", r["id"]): r for r in res_abl["per_clip"] if r["method"] == "neural"})
        res["summary"]["neural3k"] = res_abl["summary"]["neural"]
    rng = np.random.default_rng(28)
    # Balanced blind order (review fix): a plain per-clip permutation with seed 28 put ground truth in
    # slot A on 5 of 6 clips. Spread the ground-truth slot across A-D, then shuffle the rest.
    nm = len(METHODS)
    gt_slots = rng.permutation(np.resize(rng.permutation(nm), len(clips))).tolist()
    page = []
    for ci, d in enumerate(clips):
        cid = d["id"]
        x = z[f"wav_{ids.index(cid)}"]
        sig = {"gt": x}
        for m, _ in METHODS[1:]:
            src = os.path.join(a.ablation, "wavs", f"{cid}_neural.wav") if m == "neural3k" else \
                os.path.join(a.results, "wavs", f"{cid}_{m}.wav")
            sig[m], _ = sf.read(src, dtype="float32")
        gains = {m: 10 ** ((TARGET_LUFS - lufs(s)) / 20) for m, s in sig.items()}
        peak = max(np.abs(s * gains[m]).max() for m, s in sig.items())
        cap = min(1.0, 0.95 / peak)  # same extra gain for all three so they stay matched
        entries = []
        for m, label in METHODS:
            y = sig[m] * gains[m] * cap
            path = os.path.join(MEDIA, f"{cid}_{m}.wav")
            sf.write(path, y, SR, subtype="PCM_16")
            buf = io.BytesIO(); sf.write(buf, y, SR, subtype="PCM_16", format="WAV")
            entries.append({"label": label, "file": os.path.basename(path), "lufs": round(lufs(y), 2),
                            "b64": base64.b64encode(buf.getvalue()).decode(),
                            "mel_l1": round(pc[(m, cid)]["mel_l1"], 3) if m != "gt" else 0.0,
                            "mrstft": round(pc[(m, cid)]["mrstft"], 3) if m != "gt" else 0.0})
        rest = (1 + rng.permutation(nm - 1)).tolist()  # entries[0] is ground truth
        order = rest[:gt_slots[ci]] + [0] + rest[gt_slots[ci]:]
        page.append({"id": cid, "text": d["text"], "transient": d["transient"],
                     "entries": [entries[k] for k in order]})
        # spectrogram figure: full clip rows + zoom on the strongest GT onset
        env = hf_env_db(x); ons = gt_onsets(env)
        t0 = ons[int(np.argmax([env[t] for t in ons]))] / 1000 if ons else 1.0
        fig, ax = plt.subplots(len(METHODS), 2, figsize=(14, 2.7 * len(METHODS)), gridspec_kw={"width_ratios": [3, 1]})
        for r, (m, label) in enumerate(METHODS):
            Sd = spec_db(sig[m]); vmax = spec_db(x).max()
            ext = [0, len(sig[m]) / SR, 0, SR / 2000]
            ax[r, 0].imshow(Sd, origin="lower", aspect="auto", extent=ext, vmin=vmax - 80, vmax=vmax, cmap="magma")
            ax[r, 0].axvline(t0, color="cyan", lw=0.6, ls="--")
            ax[r, 0].set_ylabel(f"{label}\nkHz")
            lo, hi = max(0, int((t0 - 0.08) * SR)), int((t0 + 0.08) * SR)
            seg = sig[m][lo:hi]
            Sz = 20 * np.log10(np.abs(stft(seg, 256, 16)).T + 1e-5)
            ax[r, 1].imshow(Sz, origin="lower", aspect="auto", extent=[lo / SR, hi / SR, 0, 8], vmin=vmax - 80,
                            vmax=vmax - 10, cmap="magma")
            ax[r, 1].axvline(t0, color="cyan", lw=0.6, ls="--")
        ax[0, 0].set_title(f"{cid}: \"{d['text'][:90]}\"", fontsize=9)
        ax[0, 1].set_title("zoom: strongest plosive/onset ±80 ms (n_fft 256)", fontsize=9)
        ax[-1, 0].set_xlabel("s"); ax[-1, 1].set_xlabel("s")
        fig.tight_layout(); fig.savefig(os.path.join(MEDIA, f"spec_{cid}.png"), dpi=90); plt.close(fig)
        from PIL import Image
        jb = io.BytesIO(); Image.open(os.path.join(MEDIA, f"spec_{cid}.png")).convert("RGB").save(jb, "JPEG", quality=72)
        page[-1]["spec_b64"] = base64.b64encode(jb.getvalue()).decode()
        print(cid, [e["label"] for e in page[-1]["entries"]])
    s = res["summary"]
    tbl = "".join(f"<tr><td>{n}</td><td>{s[n]['mrstft']:.3f}</td><td>{s[n]['mel_l1']:.3f}</td>"
                  f"<td>{s[n]['f0_rmse_cents']:.1f}</td><td>{s[n]['transient_pre_onset_excess_db']:+.2f}</td></tr>"
                  for n in ["gl64", "neural", "neural3k"] if n in s)
    html = open(os.path.join(ROOT, "demo_template.html")).read()
    html = html.replace("__DATA__", json.dumps(page)).replace("__TABLE__", tbl).replace("__N__", str(res["n_clips"]))
    open(os.path.join(MEDIA, "demo.html"), "w").write(html)
    print("wrote", os.path.join(MEDIA, "demo.html"), f"{os.path.getsize(os.path.join(MEDIA, 'demo.html')) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
