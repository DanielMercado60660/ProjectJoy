"""Round 6 data: the full LJSpeech training chapters plus every evaluation set, on the one fixed mel (mel.py).

  OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python r6_prepare.py          # -> cache_r6/ (~3 GB, ~1-2 min)

Splits (LJSpeech is one speaker, so LJ splits are by chapter):
  lj_train   LJ001-LJ040, every clip (1-10 s in practice), whole hops. Memory-mapped: wav int16 + log-mel float32.
  val100     27's 100 validation ids (LJ041-LJ044; read from 27's results/fm/lists/val_real.json). Checkpoint selection only.
  test27     27's 100 test ids (LJ045-LJ050; from 27's results/fm/lists/test_real.json). Scored once at the end.
  test28     28's own 24 held-out clips are cache/test.npz (prepare_data.py); not rebuilt here.
  ltr_heldout  LibriTTS-R train-clean-100, the 6 speakers 27/33 hold out (random.Random(0).sample of sorted speaker ids),
             10 clips each (2-10 s, seed 0). Never trained on (round 6 trains on LJ only), so every speaker here is unseen.
  hifi_unseen  Hi-Fi TTS (CC BY 4.0), 5 test-manifest clips (2-10 s) per each of its 10 speakers; also never trained on.
Every clip: mono, resampled to 16 kHz with resample_poly, trimmed to whole hops (256), mel = mel.log_mel.
"""
import glob, json, os, random, time
from math import gcd
from multiprocessing import Pool
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from mel import log_mel, SR, HOP

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(ROOT, "..", "data"))
LJ = os.path.join(DATA, "ljspeech", "LJSpeech-1.1")
LISTS27 = os.path.abspath(os.path.join(ROOT, "..", "27_pocket_voice_foundry", "results", "fm", "lists"))
OUT = os.path.join(ROOT, "cache_r6")


def load16(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x[: (len(x) // HOP) * HOP]


def work(item):
    cid, path = item
    x = load16(path)
    return cid, x, log_mel(x)


def save_set(name, items, meta=None):
    np.savez(os.path.join(OUT, f"{name}.npz"), ids=np.array([c for c, _, _ in items]),
             **{f"wav_{i}": x for i, (_, x, _) in enumerate(items)},
             **{f"mel_{i}": m for i, (_, _, m) in enumerate(items)})
    info = {"n": len(items), "minutes": round(sum(len(x) for _, x, _ in items) / SR / 60, 2), "ids": [c for c, _, _ in items]}
    if meta:
        info.update(meta)
    json.dump(info, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(name, info["n"], "clips", info["minutes"], "min", flush=True)


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n_train", type=int, default=0, help="0 = every LJ001-040 clip; >0 = first N (smoke tests)")
    ap.add_argument("--sets", default="val100,test27,ltr_heldout,hifi_unseen")
    ap.add_argument("--n_eval", type=int, default=0, help="0 = full eval sets; >0 = first N clips of each (smoke tests)")
    a = ap.parse_args()
    OUT = a.out
    sets = a.sets.split(",")
    cut = lambda L: L[:a.n_eval] if a.n_eval else L
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rows = [l.rstrip("\n").split("|") for l in open(os.path.join(LJ, "metadata.csv"), encoding="utf-8")]
    ids = sorted(r[0] for r in rows)
    wav = lambda c: os.path.join(LJ, "wavs", c + ".wav")
    pool = Pool(4)

    # --- eval sets ---
    val_ids = [d["id"] for d in json.load(open(os.path.join(LISTS27, "val_real.json")))]
    test_ids = [d["id"] for d in json.load(open(os.path.join(LISTS27, "test_real.json")))]
    assert all(41 <= int(c[2:5]) <= 44 for c in val_ids) and all(45 <= int(c[2:5]) <= 50 for c in test_ids)
    if "val100" in sets:
        save_set("val100", pool.map(work, [(c, wav(c)) for c in cut(val_ids)]), {"source": "27 results/fm/lists/val_real.json"})
    if "test27" in sets:
        save_set("test27", pool.map(work, [(c, wav(c)) for c in cut(test_ids)]), {"source": "27 results/fm/lists/test_real.json"})

    if "ltr_heldout" not in sets:
        return train(pool, ids, wav, a, t0)
    ltr = sorted(glob.glob(os.path.join(DATA, "libritts_r", "extracted", "**", "*.wav"), recursive=True))
    spks = sorted({os.path.basename(w).split("_")[0] for w in ltr})
    held = sorted(random.Random(0).sample(spks, 6))
    rng = np.random.default_rng(0); items = []
    for s in held:
        ws = [w for w in ltr if os.path.basename(w).split("_")[0] == s]
        rng.shuffle(ws); n = 0
        for w in ws:
            if n >= 10:
                break
            d = sf.info(w).duration
            if 2.0 <= d <= 10.0:
                items.append(("ltr" + os.path.basename(w)[:-4], w)); n += 1
    save_set("ltr_heldout", pool.map(work, cut(items)), {"speakers": held, "n_speakers_total": len(spks),
                                                     "rule": "random.Random(0).sample(sorted speaker ids, 6), as 27/33"})

    hroot = os.path.join(DATA, "hifitts", "extracted", "hi_fi_tts_v0")
    items = []
    for f in sorted(glob.glob(os.path.join(hroot, "*_manifest_*_test.json"))):
        spk = os.path.basename(f).split("_")[0]
        if any(i[0].startswith(f"hifi{spk}_") for i in items):
            continue  # one manifest per speaker
        L = [json.loads(l) for l in open(f)]
        L = [d for d in L if 2.0 <= d["duration"] <= 10.0]
        rng.shuffle(L)
        items += [(f"hifi{spk}_{os.path.basename(d['audio_filepath'])[:-5]}", os.path.join(hroot, d["audio_filepath"])) for d in L[:5]]
    save_set("hifi_unseen", pool.map(work, items), {"speakers": sorted({i[0].split('_')[0] for i in items})})
    train(pool, ids, wav, a, t0)


def train(pool, ids, wav, a, t0):
    # --- LJ train, memory-mapped ---
    tr = [c for c in ids if 1 <= int(c[2:5]) <= 40][: a.n_train or None]
    W, M, lens, kept = [], [], [], []
    for cid, x, m in pool.imap(work, [(c, wav(c)) for c in tr], chunksize=16):
        if len(x) < 1.0 * SR:
            continue
        W.append(np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)); M.append(m.astype(np.float32))
        lens.append(len(m)); kept.append(cid)
    lens = np.array(lens)
    woffs = np.concatenate([[0], np.cumsum([len(w) for w in W])[:-1]])
    moffs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    np.concatenate(W).tofile(os.path.join(OUT, "lj_train_wav.i16"))
    np.concatenate(M).tofile(os.path.join(OUT, "lj_train_mel.f32"))
    np.savez(os.path.join(OUT, "lj_train_index.npz"), ids=np.array(kept), lens=lens, woffs=woffs, moffs=moffs)
    hours = sum(len(w) for w in W) / SR / 3600
    json.dump({"n": len(kept), "hours": round(hours, 2), "chapters": "LJ001-LJ040", "dropped_short": len(tr) - len(kept),
               "wall_s": round(time.time() - t0, 1)}, open(os.path.join(OUT, "lj_train.json"), "w"), indent=1)
    print(f"lj_train {len(kept)} clips {hours:.2f} h; total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
