"""Pick a few hundred LJSpeech clips, resample to 16 kHz, compute log-mels, cache as .npz.

Split: LJSpeech is a single speaker, so the held-out split is by *text/chapter*: train clips come
from chapters LJ001-LJ040, held-out clips from chapters LJ045-LJ050 (no sentence overlap possible).
Held-out clips whose transcript is plosive-heavy (word-initial p/t/k/b/d/g) are tagged "transient".
"""
import argparse, json, os, re
import numpy as np
from mel import load_ljspeech_16k, log_mel, SR

ROOT = os.path.dirname(os.path.abspath(__file__))
LJ = os.path.join(ROOT, "..", "data", "ljspeech", "LJSpeech-1.1")


def plosive_score(text):
    words = re.findall(r"[a-z]+", text.lower())
    return sum(1 for w in words if w[0] in "ptkbdg") / max(1, len(words))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = [l.rstrip("\n").split("|") for l in open(os.path.join(LJ, "metadata.csv"), encoding="utf-8")]
    meta = {r[0]: r[2] for r in rows}
    chap = lambda cid: int(cid[2:5])
    train_ids = [c for c in meta if 1 <= chap(c) <= 40]
    test_ids = [c for c in meta if 45 <= chap(c) <= 50]
    rng.shuffle(train_ids); rng.shuffle(test_ids)
    os.makedirs(a.out, exist_ok=True)

    def collect(ids, n, min_s, max_s):
        out = []
        for cid in ids:
            if len(out) >= n:
                break
            x = load_ljspeech_16k(os.path.join(LJ, "wavs", cid + ".wav"))
            if not (min_s * SR <= len(x) <= max_s * SR):
                continue
            x = x[: (len(x) // 256) * 256]  # whole hops
            out.append((cid, x, log_mel(x)))
        return out

    tr = collect(train_ids, a.n_train, 1.0, 10.0)
    te = collect(test_ids, a.n_test, 2.0, 7.0)
    for name, items in (("train", tr), ("test", te)):
        np.savez(os.path.join(a.out, f"{name}.npz"),
                 ids=np.array([c for c, _, _ in items]),
                 **{f"wav_{i}": x for i, (_, x, _) in enumerate(items)},
                 **{f"mel_{i}": m for i, (_, _, m) in enumerate(items)})
    info = {"train": [c for c, _, _ in tr],
            "test": [{"id": c, "text": meta[c], "seconds": round(len(x) / SR, 2),
                      "plosive_score": round(plosive_score(meta[c]), 3)} for c, x, _ in te]}
    ranked = sorted(info["test"], key=lambda d: -d["plosive_score"])
    for d in info["test"]:
        d["transient"] = d["id"] in {r["id"] for r in ranked[: max(4, len(ranked) // 3)]}
    json.dump(info, open(os.path.join(a.out, "split.json"), "w"), indent=1)
    print(f"train {len(tr)} clips ({sum(len(x) for _, x, _ in tr) / SR / 60:.1f} min), "
          f"test {len(te)} clips ({sum(len(x) for _, x, _ in te) / SR / 60:.1f} min)")


if __name__ == "__main__":
    main()
