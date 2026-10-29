"""Follow-up (round 3): MR-STFT per resolution (spectral convergence + log-magnitude L1 at each of 512/128,
1024/256, 2048/512) from saved eval WAVs vs the ground-truth test clips, so the reviewer's per-resolution table
is traceable to a file. Caveat: the 1024/256 term is Griffin-Lim's own consistency objective, and all three
terms are in the neural vocoder's training loss; only pitch/voicing/periodicity/transients are neutral.
Usage: ../.venv/bin/python mrstft_per_res.py --out results/followup_val3k/mrstft_per_res.json \
           gl64=results/wavs/{id}_gl64.wav neural400=results/wavs/{id}_neural.wav ...
"""
import argparse, json, os
import numpy as np
import soundfile as sf
from mel import stft
from metrics import RES

ROOT = os.path.dirname(os.path.abspath(__file__))


def per_res(y, x):
    out = {}
    for n, h in RES:
        w = np.hanning(n + 1)[:-1].astype(np.float32)
        a, b = np.abs(stft(y, n, h, w)), np.abs(stft(x, n, h, w))
        out[f"{n}/{h}"] = float(np.linalg.norm(b - a) / (np.linalg.norm(b) + 1e-9)
                                + np.mean(np.abs(np.log(a + 1e-5) - np.log(b + 1e-5))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "cache", "test.npz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("methods", nargs="+", help="name=path pattern with {id}")
    a = ap.parse_args()
    z = np.load(a.data); ids = list(z["ids"])
    res = {}
    for spec in a.methods:
        name, pat = spec.split("=", 1)
        rows = []
        for i, cid in enumerate(ids):
            x = z[f"wav_{i}"]
            y, _ = sf.read(os.path.join(ROOT, pat.format(id=cid)), dtype="float32")
            rows.append(per_res(y[:len(x)], x))
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        m["mean"] = float(np.mean([m[k] for k in rows[0]]))
        m["mean_without_1024"] = float(np.mean([m["512/128"], m["2048/512"]]))
        res[name] = m
        print(name, {k: round(v, 3) for k, v in m.items()})
    json.dump({"n_clips": len(ids), "caveat": "1024/256 is GL's objective; all terms are in the neural training loss",
               "per_method": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
