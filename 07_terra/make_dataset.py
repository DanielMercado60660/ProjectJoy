"""Generate training/validation/test data for the learned heuristic.

Each map seed gives static features; each (map, goal) gives an exact cost-to-go field
(scipy Dijkstra on the shared cost graph) as the label.
Splits by seed range: train [0, n_train), val [500000, ...), test maps are generated on the
fly in eval_heuristic.py from seeds >= 900000 (never seen here).

  ../.venv/bin/python make_dataset.py --n_train 2000 --n_val 100 --goals 4
"""
import argparse, os, time
from multiprocessing import Pool
import numpy as np
from terra import terrain, learn

SIZE = 96


def one_map(args):
    seed, goals = args
    T = terrain.generate(seed, SIZE)
    st = learn.static_features(T["elev"], T["kind"])
    M = learn.sparse_graph(T["elev"], T["kind"])
    rng = np.random.default_rng(seed + 7)
    passable = np.argwhere(T["kind"] != terrain.WATER)
    out = []
    for _ in range(goals):
        gr, gc = passable[rng.integers(len(passable))]
        ctg = learn.cost_to_go(T["elev"], T["kind"], (gr, gc), M)
        t, mask = learn.target_logratio(ctg, gr, gc)
        out.append((gr, gc, t.astype(np.float16), mask))
    return st.astype(np.float16), out


def build(seeds, goals, path):
    t0 = time.time()
    with Pool(4) as pool:
        res = pool.map(one_map, [(s, goals) for s in seeds], chunksize=8)
    static = np.stack([r[0] for r in res])
    gi, goal_rc, tgt, mask = [], [], [], []
    for i, (_, outs) in enumerate(res):
        for gr, gc, t, m in outs:
            gi.append(i); goal_rc.append((gr, gc)); tgt.append(t); mask.append(m)
    np.savez(path, static=static, map_index=np.array(gi), goal=np.array(goal_rc),
             target=np.stack(tgt), mask=np.stack(mask), seeds=np.array(seeds))
    print(f"{path}: {len(seeds)} maps, {len(gi)} samples, {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--goals", type=int, default=4)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    build(list(range(500000, 500000 + a.n_val)), a.goals, f"{a.out}/val.npz")
    build(list(range(a.n_train)), a.goals, f"{a.out}/train.npz")
