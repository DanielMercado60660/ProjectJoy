"""Cache 16 kHz log-mels (28's mel definition) + normalized text for LJSpeech, split by chapter.

Split (same convention as ../28_vocoder_thunderdome): train = chapters LJ001-LJ040, val = LJ041-LJ044
(baseline tuning, checkpoint choice), test = LJ045-LJ050 (held out: never trained or tuned on).
Train clips longer than --max_s seconds are dropped (batching); val/test keep 1-10 s clips.
Output: cache/{train,val,test}.npz with concatenated fp16 mels + offsets, ids, texts.
"""
import argparse, json, os
from multiprocessing import Pool
import numpy as np
from mel import load_ljspeech_16k, log_mel, SR, HOP
from text import normalize

ROOT = os.path.dirname(os.path.abspath(__file__))
LJ = os.path.join(ROOT, "..", "data", "ljspeech", "LJSpeech-1.1")


def _work(cid):
    x = load_ljspeech_16k(os.path.join(LJ, "wavs", cid + ".wav"))
    x = x[: (len(x) // HOP) * HOP]
    return cid, len(x), log_mel(x).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--max_s", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0, help="per split (smoke)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rows = [l.rstrip("\n").split("|") for l in open(os.path.join(LJ, "metadata.csv"), encoding="utf-8")]
    chap = lambda cid: int(cid[2:5])
    splits = {"train": [r for r in rows if chap(r[0]) <= 40],
              "val": [r for r in rows if 41 <= chap(r[0]) <= 44],
              "test": [r for r in rows if chap(r[0]) >= 45]}
    os.makedirs(a.out, exist_ok=True)
    info = {}
    for name, rs in splits.items():
        if a.limit:
            rs = rs[: a.limit]
        with Pool(a.workers) as p:
            res = p.map(_work, [r[0] for r in rs], chunksize=16)
        texts = {r[0]: normalize(r[2]) for r in rs}
        keep = [(c, n, m) for c, n, m in res if 1.0 * SR <= n <= a.max_s * SR and len(texts[c]) > 2]
        lens = np.array([len(m) for _, _, m in keep])
        offs = np.concatenate([[0], np.cumsum(lens)])
        np.savez(os.path.join(a.out, f"{name}.npz"), ids=np.array([c for c, _, _ in keep]),
                 texts=np.array([texts[c] for c, _, _ in keep]), lens=lens, offs=offs,
                 mels=np.concatenate([m for _, _, m in keep]), raw=np.array([dict((r[0], r[1]) for r in rs)[c] for c, _, _ in keep]))
        info[name] = {"clips": len(keep), "dropped": len(rs) - len(keep), "hours": round(lens.sum() * HOP / SR / 3600, 3)}
        print(name, info[name], flush=True)
    m = np.load(os.path.join(a.out, "train.npz"))["mels"].astype(np.float32)
    info["mel_mean"], info["mel_std"] = m.mean(0).tolist(), m.std(0).tolist()
    json.dump(info, open(os.path.join(a.out, "info.json"), "w"))


if __name__ == "__main__":
    main()
