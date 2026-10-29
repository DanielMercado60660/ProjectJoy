"""Reviewer check: stub_lines + 'edge fill' — a stub much longer than a road width is a road running along the hole
edge, so continue it 1-2 px into the hole (dilate the long stub into the hole). Tuned on val only.
Writes results/review/edgefill.json."""
import itertools, json
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import binary_dilation
from evaluate import sets
from metrics import score, composite, stubs, K8
from review_check import stub_lines2, cf


def edgefill(known, hole, minlen=6, k=1, **p):
    out = stub_lines2(known, hole, **p).astype(bool)
    lab, n = stubs(known, hole); hb = hole.astype(bool)
    for i in range(n):
        s = lab == i + 1
        if s.sum() >= minlen and k > 0:
            out |= binary_dilation(s, K8, iterations=k) & hb
    return out.astype(np.uint8)


def rows(params, ims, holes):
    return [score(im, h, composite(im * (1 - h), h, edgefill(im * (1 - h), h, **params))) for im, h in zip(ims, holes)]


def job(a):
    p, ims, holes = a; r = rows(p, ims, holes)
    return p, cf(r), float(np.mean([x["partition_ok"] for x in r]))


if __name__ == "__main__":
    D = sets(300, 500); v = D["val"]
    grid = [dict(minlen=m, k=k, w_ang=w, thr=t, ext=1.0, lw=3, match="greedy") for m, k, w, t in
            itertools.product([3, 4, 5, 6, 8], [0, 1, 2, 3, 4], [4.0, 8.0, 16.0], [3.0, 4.5, 6.0])]
    with Pool(4) as pool:
        res = pool.map(job, [(p, v["ims"], v["holes"]) for p in grid])
    res.sort(key=lambda r: r[1], reverse=True)
    print("val top", res[:5], flush=True)
    best = res[0][0]; out = dict(val_grid=[dict(params=p, cf=c, partition_exact=pe) for p, c, pe in res], best=best, test={})
    for s in ["test_id", "test_bighole", "test_delaunay"]:
        r = rows(best, D[s]["ims"], D[s]["holes"])
        out["test"][s] = {"0.8/0.8": cf(r), "0.7/0.7": cf(r, .7, .7), "0.9/0.9": cf(r, .9, .9), "partition_exact": float(np.mean([x["partition_ok"] for x in r]))}
        print(s, out["test"][s], flush=True)
    json.dump(out, open("results/review/edgefill.json", "w"), indent=1)
