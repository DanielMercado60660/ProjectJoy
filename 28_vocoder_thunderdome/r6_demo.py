"""Round 6 listening page: media/demo_r6.html (self-contained, offline) + media/r6/<set>_<id>_<system>.wav.

  ../.venv/bin/python r6_demo.py --systems gt,gl64,old,recon,gan --clips test27:6,ltr_heldout:2
Per clip, every version is loudness-matched to -23 LUFS (BS.1770, make_demo.lufs) with one shared peak cap, the order
is shuffled (seed 606) with the recording's slot balanced across positions, and a reveal button shows labels, the
clip's UTMOS / GPE / mel L1 and spectrograms. Clips are picked deterministically: evenly spaced through each set's
id list (no cherry-picking on scores). Reads results_r6/wavs/<set>/<system>/<id>.wav (r6_eval.py) and the per-clip
scores in results_r6/{mos,metrics}/<set>/<system>.json.
"""
import argparse, base64, io, json, os
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from make_demo import lufs, spec_db
from mel import SR

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results_r6")
LABELS = {"gt": "Recording", "gl64": "Griffin–Lim 64", "old": "Old neural (round 3, what 27 uses)",
          "recon": "New, no GAN (equal steps, same data)", "gan": "New GAN vocoder (round 6)"}


def per_clip(set_, sys_):
    d = {}
    for kind in ("mos", "metrics"):
        p = os.path.join(R, kind, set_, f"{sys_}.json")
        if os.path.exists(p):
            for r in json.load(open(p))["clips" if kind == "mos" else "per_clip"]:
                d.setdefault(r["id"], {}).update({"utmos": r["mos"]} if kind == "mos" else r)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="gt,gl64,old,recon,gan")
    ap.add_argument("--clips", default="test27:6,ltr_heldout:2")
    ap.add_argument("--names", default="", help="sys=dirname overrides, e.g. gan=gan_s20000")
    ap.add_argument("--out", default=os.path.join(ROOT, "media", "demo_r6.html"))
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    systems = a.systems.split(",")
    dirs = {s: s for s in systems}
    dirs.update(dict(kv.split("=", 1) for kv in a.names.split(",") if kv))
    mdir = os.path.join(ROOT, "media", "r6"); os.makedirs(mdir, exist_ok=True)
    clips = []
    for spec in a.clips.split(","):
        set_, n = spec.split(":")
        ids = sorted(os.path.basename(f)[:-4] for f in os.listdir(os.path.join(R, "wavs", set_, "gt")))
        clips += [(set_, ids[int(round(i))]) for i in np.linspace(0, len(ids) - 1, int(n) + 2)[1:-1]]
    rng = np.random.default_rng(606)
    nm = len(systems)
    gt_slots = rng.permutation(np.resize(rng.permutation(nm), len(clips))).tolist()
    page = []
    for ci, (set_, cid) in enumerate(clips):
        sig = {s: sf.read(os.path.join(R, "wavs", set_, dirs[s], cid + ".wav"), dtype="float32")[0] for s in systems}
        gains = {s: 10 ** ((-23.0 - lufs(x)) / 20) for s, x in sig.items()}
        cap = min(1.0, 0.95 / max(np.abs(x * gains[s]).max() for s, x in sig.items()))
        entries = []
        for s in systems:
            y = sig[s] * gains[s] * cap
            sf.write(os.path.join(mdir, f"{set_}_{cid}_{s}.wav"), y, SR, subtype="PCM_16")
            buf = io.BytesIO(); sf.write(buf, y, SR, subtype="PCM_16", format="WAV")
            sc = per_clip(set_, dirs[s]).get(cid, {})
            entries.append({"label": LABELS.get(s, s), "sys": s, "lufs": round(lufs(y), 2), "b64": base64.b64encode(buf.getvalue()).decode(),
                            "utmos": round(sc.get("utmos", float("nan")), 2), "gpe": round(sc.get("gpe_pct", 0.0), 1),
                            "mel_l1": round(sc.get("mel_l1", 0.0), 3)})
        rest = [k for k in rng.permutation(nm).tolist() if k != 0]
        order = rest[:gt_slots[ci]] + [0] + rest[gt_slots[ci]:]
        fig, ax = plt.subplots(nm, 1, figsize=(12, 1.9 * nm), sharex=True)
        vmax = spec_db(sig[systems[0]]).max()
        for r, s in enumerate(systems):
            ax[r].imshow(spec_db(sig[s]), origin="lower", aspect="auto", extent=[0, len(sig[s]) / SR, 0, 8], vmin=vmax - 80, vmax=vmax, cmap="magma")
            ax[r].set_ylabel(f"{LABELS.get(s, s)[:22]}\nkHz", fontsize=8)
        ax[-1].set_xlabel("s"); fig.tight_layout()
        jb = io.BytesIO(); fig.savefig(jb, format="png", dpi=80); plt.close(fig); jb.seek(0)
        jj = io.BytesIO(); Image.open(jb).convert("RGB").save(jj, "JPEG", quality=70)
        page.append({"id": cid, "set": set_, "entries": [entries[k] for k in order], "spec_b64": base64.b64encode(jj.getvalue()).decode()})
        print(set_, cid, [e["sys"] for e in page[-1]["entries"]], flush=True)
    html = open(os.path.join(ROOT, "r6_demo_template.html")).read()
    html = html.replace("__DATA__", json.dumps(page)).replace("__NOTE__", a.note).replace("__SLOTS__", "ABCDEFG"[:nm])
    open(a.out, "w").write(html)
    print("wrote", a.out, f"{os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
