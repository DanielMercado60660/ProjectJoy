"""
Pretraining data: download ClimbMix parquet shards, tokenize them into flat uint16 token files,
and serve (x, y) batches.

Upstream nanochat migrated from FineWeb-Edu to `karpathy/climbmix-400b-shuffle` in March 2026;
we follow upstream. Like upstream, the LAST shard (shard_06542) is the held-out validation split.

Usage:
    python data.py --download 8          # shards 0..7 + val shard
    python data.py --tokenize            # -> data/tokens/{train_XXXXX,val}.npy (needs tokenizer)
"""
import argparse
import glob
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "data", "climbmix")
TOK_DIR = os.path.join(HERE, "data", "tokens")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
VAL_SHARD = 6542


def list_shards(split):
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "shard_*.parquet")))
    val = [p for p in paths if p.endswith(f"shard_{VAL_SHARD:05d}.parquet")]
    return val if split == "val" else [p for p in paths if p not in val]


def iter_docs(path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        for t in pf.read_row_group(rg).column("text").to_pylist():
            yield t


def download(n):
    os.makedirs(RAW_DIR, exist_ok=True)
    for i in [VAL_SHARD] + list(range(n)):
        fn = os.path.join(RAW_DIR, f"shard_{i:05d}.parquet")
        if os.path.exists(fn):
            continue
        print("downloading", fn, flush=True)
        subprocess.check_call(["curl", "-sL", "--retry", "3", "-o", fn + ".tmp", f"{BASE_URL}/shard_{i:05d}.parquet"])
        os.rename(fn + ".tmp", fn)


def tokenize_all(val_docs=20000, min_index=0):
    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    from tokenizer import ChatTokenizer
    ct = ChatTokenizer.load()
    os.makedirs(TOK_DIR, exist_ok=True)
    jobs = [(p, os.path.join(TOK_DIR, "val.npy"), val_docs) for p in list_shards("val")]
    jobs += [(p, os.path.join(TOK_DIR, "train_" + os.path.basename(p).split("_")[1].split(".")[0] + ".npy"), None)
             for p in list_shards("train") if int(os.path.basename(p)[6:11]) >= min_index]
    for src, dst, limit in jobs:
        if os.path.exists(dst):
            continue
        chunks, docs = [], []
        for i, d in enumerate(iter_docs(src)):
            if limit is not None and i >= limit:
                break
            docs.append(d)
            if len(docs) == 4096:
                chunks.extend(np.asarray(x, dtype=np.uint16) for x in ct.encode_batch(docs))
                docs = []
        if docs:
            chunks.extend(np.asarray(x, dtype=np.uint16) for x in ct.encode_batch(docs))
        arr = np.concatenate(chunks)
        np.save(dst + ".tmp.npy", arr)
        os.rename(dst + ".tmp.npy", dst)
        print(f"{os.path.basename(dst)}: {len(arr)/1e6:.1f}M tokens", flush=True)


class TokenStream:
    """Random-window loader over the concatenated token stream (documents separated by BOS).

    Simplification vs upstream: upstream packs BOS-aligned documents with a best-fit packer so
    every row starts at a document boundary. We take uniformly random windows of T+1 tokens.
    Deterministic given (seed, step), which makes resume exact without saving loader state.
    """

    def __init__(self, split, seed=0):
        if split == "val":
            files = [os.path.join(TOK_DIR, "val.npy")]
        else:
            files = sorted(f for f in glob.glob(os.path.join(TOK_DIR, "train_*.npy")) if ".tmp" not in f)
        assert files, f"no token files for {split}; run `python data.py --tokenize`"
        self.arrays = [np.load(f, mmap_mode="r") for f in files]
        self.sizes = np.array([len(a) for a in self.arrays], dtype=np.int64)
        self.total = int(self.sizes.sum())
        self.seed = seed

    def batch(self, step, B, T):
        rng = np.random.default_rng((self.seed, step))
        probs = self.sizes / self.total
        which = rng.choice(len(self.arrays), size=B, p=probs)
        out = np.empty((B, T + 1), dtype=np.int32)
        for i, w in enumerate(which):
            a = self.arrays[w]
            s = rng.integers(0, len(a) - T - 1)
            out[i] = a[s:s + T + 1]
        return out[:, :-1], out[:, 1:]

    def sequential_batches(self, B, T, n_batches):
        """Fixed, non-overlapping windows from the start of the stream (used for val loss)."""
        a = self.arrays[0]
        for k in range(n_batches):
            s = k * B * (T + 1)
            chunk = np.asarray(a[s:s + B * (T + 1)], dtype=np.int32).reshape(B, T + 1)
            yield chunk[:, :-1], chunk[:, 1:]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", type=int, default=0)
    ap.add_argument("--tokenize", action="store_true")
    ap.add_argument("--min-index", type=int, default=0, help="only tokenize train shards with index >= this")
    a = ap.parse_args()
    if a.download:
        download(a.download)
    if a.tokenize:
        tokenize_all(min_index=a.min_index)
