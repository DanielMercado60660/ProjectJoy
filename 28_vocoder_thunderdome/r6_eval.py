"""Round 6 evaluation: vocode an eval set from its real mels, write the wavs, and score objective metrics per clip.

  OMP_NUM_THREADS=4 ../.venv/bin/python r6_eval.py --set test27 --system gl64
  OMP_NUM_THREADS=4 ../.venv/bin/python r6_eval.py --set val100 --system gan_s8000 --ckpt checkpoints/r6_runs/gan/snaps/G_8000.safetensors
Sets: val100, test27, ltr_heldout, hifi_unseen (cache_r6/<set>.npz) and test28 (28's 24 held-out clips, cache/test.npz).
Systems: gt (the 16 kHz recording itself), gl64 (NNLS + fast Griffin-Lim, 64 it, griffin_lim.py), or any generator
checkpoint (--ckpt; loaded with r6_vocoder.load, which also reads the round-1/3 checkpoints).
Writes results_r6/wavs/<set>/<system>/<id>.wav (float) and results_r6/metrics/<set>/<system>.json:
mel L1, MR-STFT (metrics.py), YIN pitch (GPE, f0 RMSE, VDE, periodicity), synthesis time.
"""
import argparse, json, os, time
import numpy as np
import soundfile as sf
from metrics import mrstft, mel_l1, pitch_metrics
from mel import SR

ROOT = os.path.dirname(os.path.abspath(__file__))


def set_path(name, cache=os.path.join(ROOT, "cache_r6")):
    return os.path.join(ROOT, "cache", "test.npz") if name == "test28" else os.path.join(cache, f"{name}.npz")


def system_fn(name, ckpt):
    if name == "gt":
        return None
    if name.startswith("gl"):
        from griffin_lim import griffin_lim
        n = int(name[2:])
        return lambda m: griffin_lim(m, n_iter=n, seed=0)
    from r6_vocoder import load, vocode
    G = load(ckpt)
    return lambda m: vocode(G, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_metrics", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_r6"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache_r6"))
    a = ap.parse_args()
    z = np.load(set_path(a.set, a.cache))
    ids = list(z["ids"])[: a.limit or None]
    fn = system_fn(a.system, a.ckpt)
    wdir = os.path.join(a.out, "wavs", a.set, a.system)
    os.makedirs(wdir, exist_ok=True)
    if fn is not None:
        fn(z["mel_0"][:64])  # warm-up
    rows = []
    for i, cid in enumerate(ids):
        x, m = z[f"wav_{i}"], z[f"mel_{i}"]
        t = time.perf_counter()
        y = x.copy() if fn is None else fn(m)[:len(x)]
        dt = time.perf_counter() - t
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        sf.write(os.path.join(wdir, f"{cid}.wav"), y.astype(np.float32), SR, subtype="FLOAT")
        r = {"id": cid, "seconds": len(x) / SR, "synth_s": dt}
        if not a.no_metrics:
            r.update({"mrstft": mrstft(y, x), "mel_l1": mel_l1(y, x), **pitch_metrics(y, x)})
        rows.append(r)
    keys = [k for k in rows[0] if k not in ("id", "seconds", "synth_s")]
    summ = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summ["rtf_under_load"] = float(sum(r["synth_s"] for r in rows) / sum(r["seconds"] for r in rows))
    out = {"set": a.set, "system": a.system, "ckpt": a.ckpt, "n": len(rows), "summary": summ, "per_clip": rows}
    mdir = os.path.join(a.out, "metrics", a.set)
    os.makedirs(mdir, exist_ok=True)
    json.dump(out, open(os.path.join(mdir, f"{a.system}.json"), "w"), indent=1)
    print(a.set, a.system, json.dumps({k: round(v, 4) for k, v in summ.items()}), flush=True)


if __name__ == "__main__":
    main()
