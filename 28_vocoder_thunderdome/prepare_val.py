"""Follow-up (round 3): build a validation split from LJSpeech chapters LJ041-LJ044, which are disjoint from
both the training chapters (LJ001-LJ040) and the held-out test chapters (LJ045-LJ050). Same 16 kHz log-mel,
same 2-7 s length filter as the test set. Used only for checkpoint selection in train_val.py.
Usage: ../.venv/bin/python prepare_val.py --n_val 32 --out cache_val
"""
import argparse, json, os
import numpy as np
from mel import load_ljspeech_16k, log_mel, SR

ROOT = os.path.dirname(os.path.abspath(__file__))
LJ = os.path.join(ROOT, "..", "data", "ljspeech", "LJSpeech-1.1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_val", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(ROOT, "cache_val"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = [l.rstrip("\n").split("|") for l in open(os.path.join(LJ, "metadata.csv"), encoding="utf-8")]
    meta = {r[0]: r[2] for r in rows}
    ids = [c for c in meta if 41 <= int(c[2:5]) <= 44]
    rng.shuffle(ids)
    out = []
    for cid in ids:
        if len(out) >= a.n_val:
            break
        x = load_ljspeech_16k(os.path.join(LJ, "wavs", cid + ".wav"))
        if not (2.0 * SR <= len(x) <= 7.0 * SR):
            continue
        x = x[: (len(x) // 256) * 256]
        out.append((cid, x, log_mel(x)))
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "val.npz"), ids=np.array([c for c, _, _ in out]),
             **{f"wav_{i}": x for i, (_, x, _) in enumerate(out)},
             **{f"mel_{i}": m for i, (_, _, m) in enumerate(out)})
    json.dump({"val": [{"id": c, "text": meta[c], "seconds": round(len(x) / SR, 2)} for c, x, _ in out]},
              open(os.path.join(a.out, "val_split.json"), "w"), indent=1)
    print(f"val {len(out)} clips ({sum(len(x) for _, x, _ in out) / SR / 60:.1f} min) from chapters LJ041-LJ044")


if __name__ == "__main__":
    main()
