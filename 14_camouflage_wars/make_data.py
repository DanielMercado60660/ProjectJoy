"""Generate the fixed dataset splits into data/ (deterministic seeds, 4 worker processes)."""
import argparse, os, pickle, time
import numpy as np
from scenes import TRAIN_FAMILIES, HELDOUT_FAMILIES, make_split_parallel

SPLITS = {  # name: (seed, families)
    "train": (1, TRAIN_FAMILIES),
    "val": (2, TRAIN_FAMILIES),
    "test_id": (3, TRAIN_FAMILIES),      # seen families, new scenes
    "test_ood": (4, HELDOUT_FAMILIES),   # unseen families for background AND creature skins
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_train", type=int, default=24000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, (seed, fams) in SPLITS.items():
        n = a.n_train if name == "train" else (a.n_eval // 2 if name == "val" else a.n_eval)
        t = time.time()
        X, Y, M = make_split_parallel(seed, n, fams, workers=a.workers, chunk=max(1, min(500, n // a.workers + 1)))
        np.savez(os.path.join(a.out, f"{name}.npz"), X=X, Y=Y)
        with open(os.path.join(a.out, f"{name}_meta.pkl"), "wb") as f:
            pickle.dump(M, f)
        print(f"{name}: {n} scenes, families={fams}, {time.time()-t:.1f}s, fg frac {(Y>0).mean():.3f}")
