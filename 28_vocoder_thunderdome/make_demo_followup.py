"""Follow-up (round 3) blind listening page: ground truth, Griffin-Lim 64, the main neural vocoder (400 clips) and
the validation-selected 3000-clip model (train_val.py), on the same 6 held-out clips as media/demo.html.
Separate page; the existing media/demo.html and media/*.wav are not touched.
  media/followup_val3k/<id>_{gt,gl64,neural,val3k}.wav   -23 LUFS (BS.1770, make_demo.lufs), shared peak cap
  media/followup_val3k/spec_<id>.png                     spectrograms
  media/followup_val3k/demo.html                          offline, ground-truth slot balanced across A-D (seeded)
Usage: ../.venv/bin/python make_demo_followup.py --new results/followup_val3k/best
"""
import argparse, base64, io, json, os
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from mel import SR, stft
from metrics import hf_env_db, gt_onsets
from make_demo import lufs, spec_db, TARGET_LUFS

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--new", default=os.path.join(ROOT, "results", "followup_val3k", "best"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--media", default=os.path.join(ROOT, "media", "followup_val3k"))
    a = ap.parse_args()
    METHODS = [("gt", "Ground truth"), ("gl64", "Griffin-Lim (64 it)"), ("neural", "Neural (main, 400 clips)"),
               ("val3k", "Neural (3000 clips, val-selected)")]
    os.makedirs(a.media, exist_ok=True)
    res = json.load(open(os.path.join(a.results, "metrics.json")))
    rnew = json.load(open(os.path.join(a.new, "metrics.json")))
    pc = {(r["method"], r["id"]): r for r in res["per_clip"]}
    pc.update({("val3k", r["id"]): r for r in rnew["per_clip"] if r["method"] == "neural"})
    summ = dict(res["summary"]); summ["val3k"] = rnew["summary"]["neural"]
    split = json.load(open(os.path.join(a.cache, "split.json")))["test"]
    z = np.load(os.path.join(a.cache, "test.npz")); ids = list(z["ids"])
    clips = [d for d in split if d["transient"]][:4] + [d for d in split if not d["transient"]][:2]
    rng = np.random.default_rng(2803)
    nm = len(METHODS)
    gt_slots = rng.permutation(np.resize(rng.permutation(nm), len(clips))).tolist()
    page = []
    for ci, d in enumerate(clips):
        cid = d["id"]; x = z[f"wav_{ids.index(cid)}"]
        src = {"gl64": os.path.join(a.results, "wavs", f"{cid}_gl64.wav"),
               "neural": os.path.join(a.results, "wavs", f"{cid}_neural.wav"),
               "val3k": os.path.join(a.new, "wavs", f"{cid}_neural.wav")}
        sig = {"gt": x, **{m: sf.read(p, dtype="float32")[0][:len(x)] for m, p in src.items()}}
        gains = {m: 10 ** ((TARGET_LUFS - lufs(s)) / 20) for m, s in sig.items()}
        cap = min(1.0, 0.95 / max(np.abs(s * gains[m]).max() for m, s in sig.items()))
        entries = []
        for m, label in METHODS:
            y = sig[m] * gains[m] * cap
            sf.write(os.path.join(a.media, f"{cid}_{m}.wav"), y, SR, subtype="PCM_16")
            buf = io.BytesIO(); sf.write(buf, y, SR, subtype="PCM_16", format="WAV")
            entries.append({"label": label, "file": f"{cid}_{m}.wav", "lufs": round(lufs(y), 2),
                            "b64": base64.b64encode(buf.getvalue()).decode(),
                            "mel_l1": round(pc[(m, cid)]["mel_l1"], 3) if m != "gt" else 0.0,
                            "mrstft": round(pc[(m, cid)]["mrstft"], 3) if m != "gt" else 0.0})
        rest = (1 + rng.permutation(nm - 1)).tolist()
        order = rest[:gt_slots[ci]] + [0] + rest[gt_slots[ci]:]
        page.append({"id": cid, "text": d["text"], "transient": d["transient"], "entries": [entries[k] for k in order]})
        env = hf_env_db(x); ons = gt_onsets(env)
        t0 = ons[int(np.argmax([env[t] for t in ons]))] / 1000 if ons else 1.0
        fig, ax = plt.subplots(nm, 2, figsize=(14, 2.7 * nm), gridspec_kw={"width_ratios": [3, 1]})
        vmax = spec_db(x).max()
        for r, (m, label) in enumerate(METHODS):
            ax[r, 0].imshow(spec_db(sig[m]), origin="lower", aspect="auto", extent=[0, len(sig[m]) / SR, 0, SR / 2000],
                            vmin=vmax - 80, vmax=vmax, cmap="magma")
            ax[r, 0].axvline(t0, color="cyan", lw=0.6, ls="--"); ax[r, 0].set_ylabel(f"{label}\nkHz")
            lo, hi = max(0, int((t0 - 0.08) * SR)), int((t0 + 0.08) * SR)
            Sz = 20 * np.log10(np.abs(stft(sig[m][lo:hi], 256, 16)).T + 1e-5)
            ax[r, 1].imshow(Sz, origin="lower", aspect="auto", extent=[lo / SR, hi / SR, 0, 8], vmin=vmax - 80,
                            vmax=vmax - 10, cmap="magma")
            ax[r, 1].axvline(t0, color="cyan", lw=0.6, ls="--")
        ax[0, 0].set_title(f"{cid}: \"{d['text'][:90]}\"", fontsize=9)
        ax[0, 1].set_title("zoom: strongest plosive/onset ±80 ms (n_fft 256)", fontsize=9)
        fig.tight_layout(); png = os.path.join(a.media, f"spec_{cid}.png"); fig.savefig(png, dpi=90); plt.close(fig)
        jb = io.BytesIO(); Image.open(png).convert("RGB").save(jb, "JPEG", quality=72)
        page[-1]["spec_b64"] = base64.b64encode(jb.getvalue()).decode()
        print(cid, [e["label"] for e in page[-1]["entries"]])
    names = {"gl64": "Griffin-Lim 64", "neural": "neural, 400 clips (main)", "val3k": "neural, 3000 clips, val-selected"}
    tbl = "".join(f"<tr><td>{names[n]}</td><td>{s['gpe_pct']:.1f}</td><td>{s['f0_rmse_cents']:.1f}</td>"
                  f"<td>{s['transient_pre_onset_excess_db']:+.2f}</td><td>{s['mrstft']:.3f}</td><td>{s['mel_l1']:.3f}</td></tr>"
                  for n, s in ((n, summ[n]) for n in names))
    html = open(os.path.join(ROOT, "demo_followup_template.html")).read()
    html = html.replace("__DATA__", json.dumps(page)).replace("__TABLE__", tbl).replace("__N__", str(res["n_clips"]))
    out = os.path.join(a.media, "demo.html"); open(out, "w").write(html)
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
