"""Follow-up data cache: 27's exact log-mel (mel.py: 16 kHz, 80 bands) + frame-level log-F0 (pitch.py) for three sources.

  NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python prep_multi.py --sources lj,libritts,hifi

Sources and keep rules (texts go through 27's text.normalize, so every source uses the same 37 symbols):
  lj        LJSpeech 1.1, 27's chapter split and keep rule (train LJ001-040, val LJ041-044, test LJ045-050; 1-10 s after
            truncating to a multiple of HOP; normalized text > 2 chars). Same clips as 27's cache/.
  libritts  LibriTTS-R train-clean-100 (24 kHz), 1-12 s, *.normalized.txt; 6 whole speakers held out as 'val'
            (random.Random(0).sample over sorted speaker ids, the same rule and seed as ../33_gemma_voice/mimi_io.py).
  hifi      Hi-Fi TTS v0, the *_manifest_{clean,other}_train.json files only, 1-12 s by manifest duration,
            text_normalized; speakers 'hifi<id>'.
Output per source, cache2/<src>/: mels.f16 (raw fp16, frames x 80, read with np.memmap), f0.f16 (raw fp16 log-F0 per
frame, 0 = unvoiced), index.npz (ids, texts, speakers, splits, lens, offs, per-speaker voiced log-F0 mean/std).
Streams to disk in order (imap), so RAM stays small however many hours there are. A source with index.npz is skipped.
"""
import argparse, glob, json, os, random, time
from math import gcd
from multiprocessing import Pool
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from mel import SR, HOP, log_mel
from pitch import f0
from text import normalize

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "data")


def rows_lj():
    lj = os.path.join(DATA, "ljspeech", "LJSpeech-1.1")
    out = []
    for l in open(os.path.join(lj, "metadata.csv"), encoding="utf-8"):
        r = l.rstrip("\n").split("|")
        ch = int(r[0][2:5])
        split = "train" if ch <= 40 else ("val" if ch <= 44 else "test")
        out.append(dict(id=r[0], path=os.path.join(lj, "wavs", r[0] + ".wav"), text=normalize(r[2]), spk="lj",
                        split=split, lo=1.0, hi=10.0))
    return out


def rows_libritts():
    root = os.path.join(DATA, "libritts_r", "extracted")
    wavs = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    spks = sorted({os.path.basename(w).split("_")[0] for w in wavs})
    val_spk = set(random.Random(0).sample(spks, min(6, max(0, len(spks) - 1))))
    out = []
    for w in wavs:
        t = w[:-4] + ".normalized.txt"
        if not os.path.exists(t):
            continue
        uid = os.path.basename(w)[:-4]
        spk = uid.split("_")[0]
        out.append(dict(id=uid, path=w, text=normalize(open(t, encoding="utf-8").read().strip()), spk="lt" + spk,
                        split="val" if spk in val_spk else "train", lo=1.0, hi=12.0))
    return out


def rows_hifi():
    root = os.path.join(DATA, "hifitts", "extracted", "hi_fi_tts_v0")
    out = []
    for f in sorted(glob.glob(os.path.join(root, "*_manifest_*_train.json"))):
        spk = os.path.basename(f).split("_")[0]
        for i, l in enumerate(open(f)):
            d = json.loads(l)
            if 1.0 <= d["duration"] <= 12.0:
                out.append(dict(id=f"hifi{spk}_{os.path.basename(f).split('_')[2]}_{i}", path=os.path.join(root, d["audio_filepath"]),
                                text=normalize(d["text_normalized"]), spk="hifi" + spk, split="train", lo=0.0, hi=1e9))
    return out


def _work(r):
    try:
        x, sr = sf.read(r["path"], dtype="float32")
    except Exception as e:  # unreadable file -> dropped and counted
        return None
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    x = x[: (len(x) // HOP) * HOP]
    if not (r["lo"] * SR <= len(x) <= r["hi"] * SR) or len(r["text"]) <= 2:
        return None
    m = log_mel(x).astype(np.float16)
    lf, _ = f0(x)
    return m, lf.astype(np.float16)


def build(src, rows, out, workers):
    d = os.path.join(out, src)
    if os.path.exists(os.path.join(d, "index.npz")):
        print(f"{src}: index exists, skipping", flush=True)
        return
    os.makedirs(d, exist_ok=True)
    fm, ff = open(os.path.join(d, "mels.f16"), "wb"), open(os.path.join(d, "f0.f16"), "wb")
    keep, lens, dropped = [], [], 0
    st = {}
    t0 = time.time()
    with Pool(workers) as p:
        for i, res in enumerate(p.imap(_work, rows, chunksize=16)):
            if res is None:
                dropped += 1
                continue
            m, lf = res
            fm.write(m.tobytes()); ff.write(lf.tobytes())
            keep.append(i); lens.append(len(m))
            v = lf[lf > 0].astype(np.float64)
            s = st.setdefault(rows[i]["spk"], [0.0, 0.0, 0])
            s[0] += v.sum(); s[1] += (v * v).sum(); s[2] += len(v)
            if (i + 1) % 5000 == 0:
                print(f"{src} {i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    fm.close(); ff.close()
    lens = np.array(lens, np.int64)
    spk_names = sorted(st)
    mu = np.array([st[k][0] / max(1, st[k][2]) for k in spk_names])
    sd = np.array([np.sqrt(max(1e-6, st[k][1] / max(1, st[k][2]) - (st[k][0] / max(1, st[k][2])) ** 2)) for k in spk_names])
    sel = [rows[i] for i in keep]
    np.savez(os.path.join(d, "index.npz"), ids=np.array([r["id"] for r in sel]), texts=np.array([r["text"] for r in sel]),
             speakers=np.array([r["spk"] for r in sel]), splits=np.array([r["split"] for r in sel]), lens=lens,
             offs=np.concatenate([[0], np.cumsum(lens)]), spk_names=np.array(spk_names), f0_mean=mu, f0_std=sd)
    info = {"clips": len(sel), "dropped": dropped, "frames": int(lens.sum()), "hours": round(lens.sum() * HOP / SR / 3600, 3),
            "speakers": len(spk_names), "wall_s": round(time.time() - t0, 1),
            "by_split": {s: int(sum(1 for r in sel if r["split"] == s)) for s in sorted({r["split"] for r in sel})},
            "hours_by_split": {s: round(float(lens[[r["split"] == s for r in sel]].sum()) * HOP / SR / 3600, 3)
                               for s in sorted({r["split"] for r in sel})}}
    json.dump(info, open(os.path.join(d, "info.json"), "w"), indent=1)
    print(src, json.dumps(info), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="lj,libritts,hifi")
    ap.add_argument("--out", default=os.path.join(ROOT, "cache2"))
    ap.add_argument("--limit", type=int, default=0, help="rows per source (smoke)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for src in a.sources.split(","):
        rows = {"lj": rows_lj, "libritts": rows_libritts, "hifi": rows_hifi}[src]()
        if a.limit:
            rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]
        print(f"{src}: {len(rows)} candidate rows", flush=True)
        build(src, rows, a.out, a.workers)


if __name__ == "__main__":
    main()
