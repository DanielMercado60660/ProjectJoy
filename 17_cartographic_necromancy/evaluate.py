"""Tune baselines + U-Net threshold on validation, then score everything on three held-out test sets.

  test_id       : grid style, new seeds, holes 12-24 px (training distribution)
  test_bighole  : grid style, new seeds, holes 28-34 px (held-out hole size)
  test_delaunay : random planar (Delaunay) style, new seeds, holes 12-24 px (held-out map style)
Tuning objective for every method: correct_and_faithful = exact stub partition AND relaxed (1-px tolerance) precision and recall in the hole both >= 0.8.
plausible_but_wrong = faithful pixels (both >= 0.8) but wrong stub partition.
Writes results/eval.json and results/samples_<set>.npz (per-sample stats + restorations for demo.py).
"""
import argparse, itertools, json, os, time
from multiprocessing import Pool
import numpy as np
from maps import build_set, VAL0, TEST0
from metrics import score, aggregate, composite
from baselines import stub_lines, diffusion

OBJ = "correct_and_faithful"


def sets(n_val, n_test):
    return {
        "val": build_set(range(VAL0, VAL0 + n_val), "grid", (12, 24), hole_seed=11),
        "test_id": build_set(range(TEST0, TEST0 + n_test), "grid", (12, 24), hole_seed=21),
        "test_bighole": build_set(range(TEST0 + 10000, TEST0 + 10000 + n_test), "grid", (28, 34), hole_seed=22),
        "test_delaunay": build_set(range(TEST0 + 20000, TEST0 + 20000 + n_test), "delaunay", (12, 24), hole_seed=23),
    }


def run_method(name, params, ims, holes, prob=None):
    rows, outs = [], []
    for k, (im, h) in enumerate(zip(ims, holes)):
        known = im * (1 - h)
        if name == "fill_all": p = h
        elif name == "diffusion": p = diffusion(known, h, **params)
        elif name == "stub_lines": p = stub_lines(known, h, **params)
        elif name == "unet": p = (prob[k] > params["thr"]).astype(np.uint8)
        r = composite(known, h, p)
        rows.append(score(im, h, r)); outs.append(r)
    return rows, np.stack(outs)


def _tune_job(args):
    name, params, ims, holes = args
    rows, _ = run_method(name, params, ims, holes)
    return params, aggregate(rows)


GRIDS = {
    "diffusion": [dict(iters=i, thr=t) for i, t in itertools.product([30, 100, 400, 1500], [0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4])],
    "stub_lines": [dict(w_ang=w, thr=t, ext=e, lw=l) for w, t, e, l in
                   itertools.product([0.0, 1.0, 2.0, 4.0, 8.0], [0.0, 0.75, 1.5, 3.0], [0.0, 1.0, 1.5, 2.5], [2, 3, 4])],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/unet")
    ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    D = sets(a.n_val, a.n_test)
    from train import load_model, predict_prob
    model, cfg = load_model(a.ckpt)
    probs = {k: predict_prob(model, d["ims"], d["holes"]) for k, d in D.items()}
    v = D["val"]
    best, tuning = {"fill_all": {}}, {}
    with Pool(4) as pool:
        for name, grid in GRIDS.items():
            res = pool.map(_tune_job, [(name, p, v["ims"], v["holes"]) for p in grid])
            res.sort(key=lambda r: (r[1][OBJ], r[1]["stub_correct"]), reverse=True)
            best[name] = res[0][0]
            tuning[name] = [dict(params=p, **{k: s[k] for k in (OBJ, "pair_f1", "stub_correct", "hole_iou")}) for p, s in res]
            print(name, "best", res[0][0], {k: round(res[0][1][k], 3) for k in (OBJ, "pair_f1", "hole_iou")}, flush=True)
    ures = []
    for thr in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        rows, _ = run_method("unet", dict(thr=thr), v["ims"], v["holes"], probs["val"])
        ures.append((dict(thr=thr), aggregate(rows)))
    ures.sort(key=lambda r: (r[1][OBJ], r[1]["stub_correct"]), reverse=True)
    best["unet"] = ures[0][0]
    tuning["unet"] = [dict(params=p, **{k: s[k] for k in (OBJ, "pair_f1", "stub_correct", "hole_iou")}) for p, s in ures]
    print("unet best", ures[0][0], {k: round(ures[0][1][k], 3) for k in (OBJ, "pair_f1", "hole_iou")}, flush=True)

    out = dict(best_params=best, tuning_val=tuning, ckpt=a.ckpt, ckpt_cfg=cfg, results={})
    for sname, d in D.items():
        out["results"][sname] = {}
        save = dict(ims=d["ims"], holes=d["holes"], prob_unet=probs[sname].astype(np.float16))
        for m in ["fill_all", "diffusion", "stub_lines", "unet"]:
            rows, restored = run_method(m, best[m], d["ims"], d["holes"], probs[sname])
            out["results"][sname][m] = aggregate(rows)
            save[f"rest_{m}"] = restored
            save[f"ok_{m}"] = np.array([r["partition_ok"] for r in rows])
            save[f"iou_{m}"] = np.array([r["iou"] for r in rows]); save[f"rf1_{m}"] = np.array([r["relaxed_f1"] for r in rows])
        np.savez_compressed(f"{a.out}/samples_{sname}.npz", **save)
        print(sname, {m: round(r[OBJ], 3) for m, r in out["results"][sname].items()}, flush=True)
    out["eval_seconds"] = time.time() - t0
    json.dump(out, open(f"{a.out}/eval.json", "w"), indent=1)
    # compact table
    keys = ["correct_and_faithful", "partition_exact", "stub_correct", "pair_recall", "pair_precision", "pair_f1",
            "continuity", "hole_iou", "relaxed_f1", "plausible_but_wrong", "dangling_per_map"]
    lines = ["| set | method | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 2)]
    for sname, rr in out["results"].items():
        for m, r in rr.items():
            lines.append(f"| {sname} | {m} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + " |")
    open(f"{a.out}/eval_table.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
