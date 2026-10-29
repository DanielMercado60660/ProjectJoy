"""Reviewer checks (REVIEW.md): (1) retune stub_lines on val with a wider thr grid and an optimal (max-weight)
matching option; (2) re-score saved restorations under alternative faithfulness thresholds; (3) count maps with a
road running along the hole edge (a stub much longer than a road width) and C&F on them per method.
Writes results/review/review.json. Does not touch the builder's results."""
import itertools, json, time
from multiprocessing import Pool
import numpy as np
import networkx as nx
from PIL import Image, ImageDraw
from evaluate import sets
from metrics import score, composite, stubs
from baselines import stub_geometry, _draw


def stub_lines2(known, hole, w_ang=4.0, thr=3.0, ext=1.0, lw=3, match="greedy"):
    hole_b = hole.astype(bool); ys, xs = np.nonzero(hole_b)
    hs = max(np.ptp(ys), np.ptp(xs)) + 1
    geo = stub_geometry(known, hole); n = len(geo); cands = []
    for i in range(n):
        for j in range(i + 1, n):
            ci, di, _ = geo[i]; cj, dj, _ = geo[j]
            v = cj - ci; L = np.linalg.norm(v)
            if L < 1.5: continue
            u = v / L; cost = L / hs + w_ang * ((1 - np.dot(di, u)) + (1 - np.dot(dj, -u)))
            if cost < thr: cands.append((cost, i, j))
    if match == "greedy":
        cands.sort(); used = set(); pairs = []
        for c, i, j in cands:
            if i in used or j in used: continue
            used.add(i); used.add(j); pairs.append((i, j))
    else:  # optimal: min total cost matching among pairs with cost < thr (max-weight on thr - cost)
        G = nx.Graph(); G.add_weighted_edges_from([(i, j, thr - c) for c, i, j in cands])
        pairs = list(nx.max_weight_matching(G)); used = {k for p in pairs for k in p}
    im = Image.new("L", known.shape[::-1], 0); dr = ImageDraw.Draw(im)
    for i, j in pairs: _draw(dr, geo[i][0], geo[j][0], lw)
    out = np.asarray(im).copy() > 127
    for i in range(n):
        if i in used or ext <= 0: continue
        c, d, _ = geo[i]; maxlen = ext * hs; t = 0.0; end = c.copy()
        while t < maxlen:
            t += 0.5; p = c + t * d; y, x = int(round(p[0])), int(round(p[1]))
            if not (0 <= y < hole.shape[0] and 0 <= x < hole.shape[1]) or (not hole_b[y, x] and t > 2): break
            end = p
            if t > 3 and out[y, x]: break
        im2 = Image.fromarray((out * 255).astype(np.uint8)); dr2 = ImageDraw.Draw(im2); _draw(dr2, c, end, lw)
        out = np.asarray(im2) > 127
    return (out & hole_b).astype(np.uint8)


def rows_for(params, ims, holes):
    return [score(im, h, composite(im * (1 - h), h, stub_lines2(im * (1 - h), h, **params))) for im, h in zip(ims, holes)]


def cf(rows, tp=0.8, tr=0.8, mask=None):
    ok = np.array([r["partition_ok"] for r in rows]); p = np.array([r["relaxed_p"] for r in rows]); q = np.array([r["relaxed_r"] for r in rows])
    v = ok & (p >= tp) & (q >= tr)
    return float(v[mask].mean()) if mask is not None else float(v.mean())


def _job(a):
    params, ims, holes = a
    return params, cf(rows_for(params, ims, holes)), float(np.mean([r["partition_ok"] for r in rows_for(params, ims, holes)])) if False else None


def _job2(a):
    params, ims, holes = a
    r = rows_for(params, ims, holes)
    return params, cf(r), float(np.mean([x["partition_ok"] for x in r]))


def edge_road_mask(ims, holes, minlen=6):
    """map has a stub with >= minlen ring pixels (roads are 2-3 px wide, so a crossing stub is <=~4-5 px)."""
    out = []
    for im, h in zip(ims, holes):
        lab, n = stubs(im, h)
        out.append(any((lab == i + 1).sum() >= minlen for i in range(n)))
    return np.array(out)


if __name__ == "__main__":
    t0 = time.time(); D = sets(300, 500); v = D["val"]
    grid = [dict(w_ang=w, thr=t, ext=e, lw=l, match=m) for w, t, e, l, m in
            itertools.product([2.0, 4.0, 8.0], [3.0, 4.5, 6.0, 10.0, 1e9], [1.0], [3], ["greedy", "optimal"])]
    with Pool(4) as pool:
        res = pool.map(_job2, [(p, v["ims"], v["holes"]) for p in grid])
    res.sort(key=lambda r: r[1], reverse=True)
    print("val top:", res[:6], flush=True)
    best = res[0][0]
    best_by = {m: max([r for r in res if r[0]["match"] == m], key=lambda r: r[1])[0] for m in ("greedy", "optimal")}
    out = dict(val_grid=[dict(params=p, cf=c, partition_exact=pe) for p, c, pe in res], best=best, best_by_match=best_by, test={}, alt_thresholds={}, edge_road={})
    thr_alts = [(0.8, 0.8), (0.7, 0.7), (0.9, 0.9), (0.6, 0.6)]
    for s in ["val", "test_id", "test_bighole", "test_delaunay"]:
        d = D[s]; z = np.load(f"results/samples_{s}.npz")
        er = edge_road_mask(d["ims"], d["holes"])
        R = {m: [score(im, h, r) for im, h, r in zip(d["ims"], d["holes"], z[f"rest_{m}"])] for m in ("fill_all", "diffusion", "stub_lines", "unet")}
        for m, bp in best_by.items():
            R[f"stub_lines_retuned_{m}"] = rows_for(bp, d["ims"], d["holes"])
        out["alt_thresholds"][s] = {m: {f"{a}/{b}": cf(rr, a, b) for a, b in thr_alts} | {"partition_exact": float(np.mean([x['partition_ok'] for x in rr]))} for m, rr in R.items()}
        out["edge_road"][s] = dict(n_maps=int(er.sum()), frac=float(er.mean()),
                                   cf_on_edge={m: cf(rr, mask=er) for m, rr in R.items()}, cf_off_edge={m: cf(rr, mask=~er) for m, rr in R.items()})
        print(s, json.dumps(out["alt_thresholds"][s]), json.dumps(out["edge_road"][s]), flush=True)
    out["seconds"] = time.time() - t0
    json.dump(out, open("results/review/review.json", "w"), indent=1)
