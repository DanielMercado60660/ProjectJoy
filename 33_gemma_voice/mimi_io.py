"""Mimi codec I/O. The ONLY script that runs under ../.venv-moshi (moshi_mlx 0.3.0, MLX 0.26.5).

Everything else in this folder uses the shared ../.venv and calls this script through a subprocess.
Mimi: 24 kHz mono, 12.5 frames/s, up to 32 residual codebooks of 2048 entries (codebook 0 is the
semantic one). Weights: kyutai/moshiko-mlx-q8 / tokenizer-e351c8d8-checkpoint125.safetensors (CC-BY-4.0).

  ../.venv-moshi/bin/python mimi_io.py encode --out cache/codes32.npz         # all LJSpeech clips, nq=32
  ../.venv-moshi/bin/python mimi_io.py decode --codes x.npz --nq 8 --out dir   # codes -> 24 kHz wavs
  ../.venv-moshi/bin/python mimi_io.py roundtrip --ids ids.txt --nq 8 --out dir
  ../.venv-moshi/bin/python mimi_io.py encode-libritts --root ../data/libritts_r/extracted --out data/libritts_codes32.npz
"""
import argparse, json, os, re, sys, time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import mlx.core as mx
from huggingface_hub import hf_hub_download
from moshi_mlx.models.mimi import Mimi, mimi_202407

LJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ljspeech", "LJSpeech-1.1")
SR = 24000
MAX_NQ = 32
# MLX keeps freed GPU buffers for reuse; with thousands of different clip lengths that cache grew without bound
# (Hi-Fi TTS encode: system free memory 78% -> 36% in ~10 min). Cap it, and clear it periodically.
(getattr(mx, "set_cache_limit", None) or mx.metal.set_cache_limit)(2 << 30)
_clear = getattr(mx, "clear_cache", None) or mx.metal.clear_cache


def weights_path():
    return hf_hub_download("kyutai/moshiko-mlx-q8", "tokenizer-e351c8d8-checkpoint125.safetensors")


def load_mimi(nq):
    """Mimi with nq codebooks. The checkpoint holds 32; drop rvq_rest layers >= nq-1 for smaller nq."""
    orig_load = mx.load
    pat = re.compile(r"rvq_rest\.vq\.layers\.(\d+)\.")

    def filtered(f):
        return {k: v for k, v in orig_load(f).items() if not ((m := pat.search(k)) and int(m.group(1)) >= nq - 1)}

    mx.load = filtered
    try:
        m = Mimi(mimi_202407(nq))
        m.load_pytorch_weights(weights_path(), strict=True)
    finally:
        mx.load = orig_load
    return m


def split_rows():
    """27's split and keep rule: train LJ001-040, val LJ041-044, test LJ045-050; 1-10 s; text len > 2."""
    rows = [l.rstrip("\n").split("|") for l in open(os.path.join(LJ, "metadata.csv"), encoding="utf-8")]
    out = []
    for cid, raw, norm in rows:
        ch = int(cid[2:5])
        split = "train" if ch <= 40 else ("val" if ch <= 44 else "test")
        out.append((cid, split, raw, norm))
    return out


def load_wav24(cid):
    x, sr = sf.read(os.path.join(LJ, "wavs", cid + ".wav"), dtype="float32")
    assert sr == 22050, sr
    return x, resample_poly(x, 160, 147).astype(np.float32)  # 22050 -> 24000


def encode_one(m, x24):
    m.reset_all()
    c = m.encode(mx.array(x24)[None, None])
    mx.eval(c)
    out = np.array(c[0]).astype(np.uint16)  # (nq, frames); max code 2047
    del c
    return out


def decode_one(m, codes):
    m.reset_all()
    y = m.decode(mx.array(codes.astype(np.uint32))[None])
    mx.eval(y)
    return np.array(y[0, 0], dtype=np.float32)


def cmd_encode(a):
    m = load_mimi(MAX_NQ)
    rows = split_rows()
    if a.limit:
        rows = [r for r in rows if r[1] == "train"][: a.limit] + [r for r in rows if r[1] == "val"][: a.limit] + \
               [r for r in rows if r[1] == "test"][: a.limit]
    ids, splits, raws, norms, codes, nsamp = [], [], [], [], [], []
    t0 = time.time()
    for i, (cid, split, raw, norm) in enumerate(rows):
        x22, x24 = load_wav24(cid)
        dur = len(x22) / 22050
        if not (1.0 <= dur <= 10.0) or len(norm) <= 2:
            continue
        codes.append(encode_one(m, x24))
        ids.append(cid); splits.append(split); raws.append(raw); norms.append(norm); nsamp.append(len(x24))
        if i % 1000 == 0:
            print(f"{i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    lens = np.array([c.shape[1] for c in codes])
    offs = np.concatenate([[0], np.cumsum(lens)])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, ids=np.array(ids), splits=np.array(splits), raw=np.array(raws), norm=np.array(norms),
             lens=lens, offs=offs, nsamp=np.array(nsamp), codes=np.concatenate(codes, axis=1))
    info = {s: int(sum(1 for x in splits if x == s)) for s in ("train", "val", "test")}
    info.update(frames=int(lens.sum()), hours=round(float(lens.sum()) / 12.5 / 3600, 3), wall_s=round(time.time() - t0, 1))
    print(json.dumps(info), flush=True)
    json.dump(info, open(a.out.replace(".npz", ".json"), "w"))


def cmd_encode_libritts(a):
    """LibriTTS-R (24 kHz, CC BY 4.0) -> same npz layout as `encode`. Keeps 1-12 s clips with normalized text.
    Split: a few whole speakers (seed 0) are 'val' for pretraining selection; the rest 'train'. No LJ data inside."""
    import glob, random
    wavs = sorted(glob.glob(os.path.join(a.root, "**", "*.wav"), recursive=True))
    if a.limit:
        wavs = wavs[: a.limit]
    spks = sorted({os.path.basename(w).split("_")[0] for w in wavs})
    val_spk = set(random.Random(0).sample(spks, min(a.val_speakers, max(0, len(spks) - 1))))
    m = load_mimi(MAX_NQ)
    ids, splits, norms, spk_l, codes, nsamp = [], [], [], [], [], []
    t0 = time.time()
    for i, w in enumerate(wavs):
        tpath = w[:-4] + ".normalized.txt"
        if not os.path.exists(tpath):
            continue
        text = open(tpath, encoding="utf-8").read().strip()
        x, sr = sf.read(w, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            from math import gcd
            g = gcd(SR, sr)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        dur = len(x) / SR
        if not (1.0 <= dur <= 12.0) or len(text) <= 2:
            continue
        uid = os.path.basename(w)[:-4]
        spk = uid.split("_")[0]
        codes.append(encode_one(m, x))
        ids.append(uid); splits.append("val" if spk in val_spk else "train"); norms.append(text)
        spk_l.append(spk); nsamp.append(len(x))
        if i % 2000 == 0:
            print(f"{i}/{len(wavs)} {time.time() - t0:.0f}s", flush=True)
    lens = np.array([c.shape[1] for c in codes])
    offs = np.concatenate([[0], np.cumsum(lens)])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, ids=np.array(ids), splits=np.array(splits), raw=np.array(norms), norm=np.array(norms),
             speaker=np.array(spk_l), lens=lens, offs=offs, nsamp=np.array(nsamp), codes=np.concatenate(codes, axis=1))
    info = {s: int(sum(1 for x in splits if x == s)) for s in ("train", "val")}
    info.update(speakers=len(spks), val_speakers=sorted(val_spk), frames=int(lens.sum()),
                hours=round(float(lens.sum()) / 12.5 / 3600, 3), wall_s=round(time.time() - t0, 1))
    print(json.dumps(info), flush=True)
    json.dump(info, open(a.out.replace(".npz", ".json"), "w"))


def cmd_encode_hifitts(a):
    """Hi-Fi TTS (NVIDIA, CC BY 4.0; 44.1 kHz FLAC, 10 speakers) -> same npz layout. Uses the train manifests only
    (clean + other), text_normalized (cased, punctuated), keeps 1-12 s clips. --max_hours caps hours per speaker."""
    import glob
    root = os.path.join(a.root, "hi_fi_tts_v0")
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*_manifest_*_train.json"))):
        spk = os.path.basename(f).split("_")[0]
        for l in open(f):
            d = json.loads(l)
            if 1.0 <= d["duration"] <= 12.0 and len(d["text_normalized"]) > 2:
                rows.append((spk, d))
    if a.max_hours:
        keep, tot = [], {}
        for spk, d in rows:
            if tot.get(spk, 0) + d["duration"] <= a.max_hours * 3600:
                keep.append((spk, d)); tot[spk] = tot.get(spk, 0) + d["duration"]
        rows = keep
    if a.limit:
        rows = rows[: a.limit]
    m = load_mimi(MAX_NQ)
    shard_dir = a.out.replace(".npz", "_shards")
    os.makedirs(shard_dir, exist_ok=True)
    SH = 20000
    t0 = time.time()
    for k in range(0, len(rows), SH):
        path = os.path.join(shard_dir, f"shard_{k // SH:03d}.npz")
        if os.path.exists(path):
            continue                                  # resume: finished shard
        ids, norms, spk_l, codes, nsamp = [], [], [], [], []
        for i in range(k, min(k + SH, len(rows))):
            spk, d = rows[i]
            x, sr = sf.read(os.path.join(root, d["audio_filepath"]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            from math import gcd
            g = gcd(SR, sr)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
            codes.append(encode_one(m, x))
            ids.append(f"hifi_{spk}_{i}"); norms.append(d["text_normalized"]); spk_l.append("hifi" + spk); nsamp.append(len(x))
            if i % 500 == 0:
                _clear()
            if i % 5000 == 0:
                print(f"{i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
        lens = np.array([c.shape[1] for c in codes])
        np.savez(path + ".tmp.npz", ids=np.array(ids), norm=np.array(norms), speaker=np.array(spk_l), lens=lens,
                 nsamp=np.array(nsamp), codes=np.concatenate(codes, axis=1))
        os.replace(path + ".tmp.npz", path)
        print(f"shard {k // SH} saved ({len(ids)} clips)", flush=True)
    # merge shards -> one npz in the usual layout
    parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz"))) if ".tmp" not in f]
    cat = lambda key: np.concatenate([p[key] for p in parts], axis=-1 if key == "codes" else 0)
    lens = cat("lens")
    offs = np.concatenate([[0], np.cumsum(lens)])
    ids = cat("ids")
    np.savez(a.out, ids=ids, splits=np.array(["train"] * len(ids)), raw=cat("norm"), norm=cat("norm"),
             speaker=cat("speaker"), lens=lens, offs=offs, nsamp=cat("nsamp"), codes=cat("codes"))
    info = {"train": int(len(ids)), "speakers": sorted(set(map(str, cat("speaker")))), "frames": int(lens.sum()),
            "hours": round(float(lens.sum()) / 12.5 / 3600, 3), "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(info), flush=True)
    json.dump(info, open(a.out.replace(".npz", ".json"), "w"))


def cmd_decode(a):
    """--codes npz with keys names (N,), lens (N,), codes (nq_any, total_frames) -> <out>/<name>.wav."""
    z = np.load(a.codes)
    m = load_mimi(a.nq)
    os.makedirs(a.out, exist_ok=True)
    offs = np.concatenate([[0], np.cumsum(z["lens"])])
    t = 0.0
    for i, name in enumerate(z["names"]):
        c = z["codes"][: a.nq, offs[i]:offs[i + 1]]
        t0 = time.perf_counter()
        y = decode_one(m, c) if c.shape[1] > 0 else np.zeros(0, np.float32)
        t += time.perf_counter() - t0
        sf.write(os.path.join(a.out, f"{name}.wav"), y, SR)
    print(json.dumps({"decoded": int(len(z["names"])), "decode_s": round(t, 3)}), flush=True)


def cmd_roundtrip(a):
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    m32 = load_mimi(MAX_NQ)
    ms = {nq: load_mimi(nq) for nq in a.nq}
    for nq in a.nq:
        os.makedirs(os.path.join(a.out, f"nq{nq}"), exist_ok=True)
    checked = False
    for cid in ids:
        _, x24 = load_wav24(cid)
        c32 = encode_one(m32, x24)
        for nq, m in ms.items():
            if not checked:  # RVQ prefix property: encoding with nq codebooks == first nq of 32
                assert np.array_equal(encode_one(m, x24), c32[:nq]), "nq-prefix mismatch"
            sf.write(os.path.join(a.out, f"nq{nq}", f"{cid}.wav"), decode_one(m, c32[:nq]), SR)
        checked = True
    print(json.dumps({"clips": len(ids), "nq": a.nq}), flush=True)


# ---- continuous latents (flow-matching target): the 512-dim encoder output just before the quantizer, 12.5 frames/s.
def latent_one(m, x24, space="z"):
    """24 kHz mono -> (frames, 512) float32. space 'z': Mimi's encode() stopped before quantizer.encode.
    space 'c': the quantizer's projected space [rvq_first.input_proj(z) (256, semantic); rvq_rest.input_proj(z) (256)],
    i.e. what the codebooks quantize. The flow-matching target (decodes like 'proj' mode, codebook lookup skipped)."""
    m.reset_all()
    m.encoder.reset_state()
    for c in m.encoder_cache:
        c.reset()
    h = m.encoder(mx.array(x24)[None, None])
    h = m.encoder_transformer(h, cache=m.encoder_cache)[0]
    z = m.downsample(h)                                   # (1, 512, frames)
    if space == "c":
        q = m.quantizer
        z = mx.concatenate([q.rvq_first.input_proj(z), q.rvq_rest.input_proj(z)], axis=1)
    mx.eval(z)
    return np.array(z[0].T, dtype=np.float32)


def decode_latent(m, z, mode="proj"):
    """(frames, 512) latent -> 24 kHz wav. mode 'raw': hand z straight to the decoder side (fails: UTMOS 1.5). mode
    'proj': z through the quantizer's own input/output projections with the codebook lookup skipped (rvq_first +
    rvq_rest, as in decode()). mode 'c': a latent already in the projected space (latent_one space 'c')."""
    m.reset_all()
    m.decoder.reset_state()
    for c in m.decoder_cache:
        c.reset()
    xs = mx.array(z.T.astype(np.float32))[None]           # (1, 512, frames)
    q = m.quantizer
    if mode == "proj":
        xs = q.rvq_first.output_proj(q.rvq_first.input_proj(xs)) + q.rvq_rest.output_proj(q.rvq_rest.input_proj(xs))
    elif mode == "c":
        xs = q.rvq_first.output_proj(xs[:, :256]) + q.rvq_rest.output_proj(xs[:, 256:])
    xs = m.upsample(xs)
    xs = m.decoder_transformer(xs, cache=m.decoder_cache)[0]
    y = m.decoder(xs)
    mx.eval(y)
    return np.array(y[0, 0], dtype=np.float32)


def cmd_latent_roundtrip(a):
    """LJ ids -> latent -> wav for each mode in --modes (out/<mode>/<id>.wav). Quality ceiling of the flow target."""
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    m = load_mimi(MAX_NQ)
    for mode in a.modes:
        os.makedirs(os.path.join(a.out, mode), exist_ok=True)
    stats = []
    for cid in ids:
        _, x24 = load_wav24(cid)
        z = latent_one(m, x24)
        stats.append((z.mean(0), z.std(0)))
        for mode in a.modes:
            sf.write(os.path.join(a.out, mode, f"{cid}.wav"), decode_latent(m, z, mode), SR)
    mu = np.mean([s[0] for s in stats], 0); sd = np.mean([s[1] for s in stats], 0)
    print(json.dumps({"clips": len(ids), "modes": a.modes, "dim": int(mu.shape[0]), "mean_abs_mu": float(np.abs(mu).mean()),
                      "sd_median": float(np.median(sd)), "sd_min": float(sd.min()), "sd_max": float(sd.max())}), flush=True)


def _hifi_paths(root):
    """Rebuild cmd_encode_hifitts's row order (max_hours 0, limit 0) -> list of audio paths; id hifi_<spk>_<i> = row i."""
    import glob
    root = os.path.join(root, "hi_fi_tts_v0")
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*_manifest_*_train.json"))):
        spk = os.path.basename(f).split("_")[0]
        for l in open(f):
            d = json.loads(l)
            if 1.0 <= d["duration"] <= 12.0 and len(d["text_normalized"]) > 2:
                rows.append((spk, os.path.join(root, d["audio_filepath"])))
    return rows


def cmd_encode_latents(a):
    """Latents (space 'c') for every clip of an existing codes npz, in the same order: <out> = .npy float16 (total_frames, 512),
    row-aligned with that npz's offs (asserts frames == lens per clip). <out>.json records shape and source."""
    import glob
    z = np.load(a.codes)
    ids, lens = [str(i) for i in z["ids"]], z["lens"]
    if a.set == "lj":
        path_of = lambda i, cid: os.path.join(LJ, "wavs", cid + ".wav")
    elif a.set == "libritts":
        index = {os.path.basename(w)[:-4]: w for w in glob.glob(os.path.join(a.root, "**", "*.wav"), recursive=True)}
        path_of = lambda i, cid: index[cid]
    else:
        rows = _hifi_paths(a.root)
        def path_of(i, cid):
            spk, k = cid.split("_")[1], int(cid.split("_")[2])
            assert rows[k][0] == spk, (cid, rows[k][0])
            return rows[k][1]
    m = load_mimi(MAX_NQ)
    n = len(ids) if not a.limit else a.limit
    total = int(lens[:n].sum())
    out = np.lib.format.open_memmap(a.out, mode="w+", dtype=np.float16, shape=(total, 512))
    off, t0, s1, s2 = 0, time.time(), np.zeros(512), np.zeros(512)
    for i in range(n):
        x, sr = sf.read(path_of(i, ids[i]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            from math import gcd
            g = gcd(SR, sr)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        lat = latent_one(m, x, "c")
        assert lat.shape[0] == lens[i], (ids[i], lat.shape, lens[i])
        out[off:off + lat.shape[0]] = lat.astype(np.float16)
        s1 += lat.sum(0); s2 += (lat.astype(np.float64) ** 2).sum(0)
        off += lat.shape[0]
        if i % 500 == 0:
            _clear()
        if i % 5000 == 0:
            print(f"{i}/{n} {time.time() - t0:.0f}s", flush=True)
    out.flush()
    mean = s1 / off; std = np.sqrt(np.maximum(s2 / off - mean ** 2, 1e-12))
    info = {"codes": os.path.basename(a.codes), "set": a.set, "space": "c", "clips": n, "frames": off, "dim": 512,
            "wall_s": round(time.time() - t0, 1), "mean": mean.tolist(), "std": std.tolist()}
    json.dump(info, open(a.out + ".json", "w"))
    print(json.dumps({k: v for k, v in info.items() if k not in ("mean", "std")}), flush=True)


def cmd_encode_latents_dirs(a):
    """LibriTTS-R-style folders (<uid>.wav + <uid>.normalized.txt, uid = <spk>_...) -> latents only, no codes:
    <out>.npy fp16 (total_frames, 512) in space 'c', plus <meta> npz with the fields train_flow.load_set reads
    (ids, splits = all 'train', norm, raw, speaker, lens, offs, nsamp). Keeps 1-12 s clips with text > 2 chars.
    Frames per clip = ceil(samples_24k / 1920) (exact for Mimi; asserted per clip)."""
    import glob
    rows, bad = [], []
    for root in a.roots:
        for w in sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)):
            t = w[:-4] + ".normalized.txt"
            if not os.path.exists(t):
                continue
            try:
                inf = sf.info(w)
                text = open(t, encoding="utf-8").read().strip()
            except Exception:
                bad.append(w)
                continue
            n24 = int(round(inf.frames * SR / inf.samplerate))
            if 1.0 <= n24 / SR <= 12.0 and len(text) > 2:
                rows.append((w, text, n24))
    if a.limit:
        rows = rows[: a.limit]
    lens = np.array([-(-n // 1920) for _, _, n in rows])
    offs = np.concatenate([[0], np.cumsum(lens)])
    ids = [os.path.basename(w)[:-4] for w, _, _ in rows]
    spk = [i.split("_")[0] for i in ids]
    print(json.dumps({"clips": len(rows), "speakers": len(set(spk)), "hours": round(float(lens.sum()) / 12.5 / 3600, 1),
                      "unreadable": len(bad), "unreadable_first": bad[:5]}),
          flush=True)
    m = load_mimi(MAX_NQ)
    out = np.lib.format.open_memmap(a.out, mode="w+", dtype=np.float16, shape=(int(lens.sum()), 512))
    t0, s1, s2 = time.time(), np.zeros(512), np.zeros(512)
    for i, (w, _, n24) in enumerate(rows):
        x, sr = sf.read(w, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            from math import gcd
            g = gcd(SR, sr)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        lat = latent_one(m, x, "c")
        assert lat.shape[0] == lens[i], (ids[i], lat.shape, lens[i])
        out[offs[i]:offs[i + 1]] = lat.astype(np.float16)
        s1 += lat.sum(0); s2 += (lat.astype(np.float64) ** 2).sum(0)
        if i % 500 == 0:
            _clear()
        if i % 5000 == 0:
            print(f"{i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    out.flush()
    np.savez(a.meta, ids=np.array(ids), splits=np.array(["train"] * len(ids)), norm=np.array([r[1] for r in rows]),
             raw=np.array([r[1] for r in rows]), speaker=np.array(spk), lens=lens, offs=offs,
             nsamp=np.array([r[2] for r in rows]))
    off = int(lens.sum())
    mean = s1 / off; std = np.sqrt(np.maximum(s2 / off - mean ** 2, 1e-12))
    info = {"meta": os.path.basename(a.meta), "roots": a.roots, "space": "c", "clips": len(ids), "frames": off, "dim": 512,
            "speakers": len(set(spk)), "wall_s": round(time.time() - t0, 1), "mean": mean.tolist(), "std": std.tolist()}
    json.dump(info, open(a.out + ".json", "w"))
    print(json.dumps({k: v for k, v in info.items() if k not in ("mean", "std")}), flush=True)


def cmd_decode_latents(a):
    """--lat npz with names (N,), lens (N,), lat (total_frames, 512) -> <out>/<name>.wav (generated latents)."""
    z = np.load(a.lat)
    m = load_mimi(MAX_NQ)
    os.makedirs(a.out, exist_ok=True)
    offs = np.concatenate([[0], np.cumsum(z["lens"])])
    t = 0.0
    for i, name in enumerate(z["names"]):
        lat = z["lat"][offs[i]:offs[i + 1]]
        t0 = time.perf_counter()
        y = decode_latent(m, lat, a.mode) if lat.shape[0] > 0 else np.zeros(0, np.float32)
        t += time.perf_counter() - t0
        sf.write(os.path.join(a.out, f"{name}.wav"), y, SR)
    print(json.dumps({"decoded": int(len(z["names"])), "decode_s": round(t, 3)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode"); e.add_argument("--out", required=True); e.add_argument("--limit", type=int, default=0)
    d = sub.add_parser("decode"); d.add_argument("--codes", required=True); d.add_argument("--nq", type=int, required=True)
    d.add_argument("--out", required=True)
    l = sub.add_parser("encode-libritts"); l.add_argument("--root", required=True); l.add_argument("--out", required=True)
    l.add_argument("--limit", type=int, default=0); l.add_argument("--val_speakers", type=int, default=6)
    h = sub.add_parser("encode-hifitts"); h.add_argument("--root", required=True); h.add_argument("--out", required=True)
    h.add_argument("--limit", type=int, default=0); h.add_argument("--max_hours", type=float, default=0)
    r = sub.add_parser("roundtrip"); r.add_argument("--ids", required=True); r.add_argument("--out", required=True)
    r.add_argument("--nq", type=int, nargs="+", default=[8, 16, 32])
    lr = sub.add_parser("latent-roundtrip"); lr.add_argument("--ids", required=True); lr.add_argument("--out", required=True)
    lr.add_argument("--modes", nargs="+", default=["raw", "proj"])
    el = sub.add_parser("encode-latents"); el.add_argument("--codes", required=True); el.add_argument("--out", required=True)
    el.add_argument("--set", required=True, choices=["lj", "libritts", "hifitts"]); el.add_argument("--root", default="")
    el.add_argument("--limit", type=int, default=0)
    ed = sub.add_parser("encode-latents-dirs"); ed.add_argument("--roots", nargs="+", required=True)
    ed.add_argument("--out", required=True); ed.add_argument("--meta", required=True); ed.add_argument("--limit", type=int, default=0)
    dl = sub.add_parser("decode-latents"); dl.add_argument("--lat", required=True); dl.add_argument("--out", required=True)
    dl.add_argument("--mode", default="c")
    a = ap.parse_args()
    {"encode": cmd_encode, "encode-libritts": cmd_encode_libritts, "encode-hifitts": cmd_encode_hifitts, "decode": cmd_decode,
     "roundtrip": cmd_roundtrip, "latent-roundtrip": cmd_latent_roundtrip, "encode-latents": cmd_encode_latents,
     "encode-latents-dirs": cmd_encode_latents_dirs, "decode-latents": cmd_decode_latents}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
