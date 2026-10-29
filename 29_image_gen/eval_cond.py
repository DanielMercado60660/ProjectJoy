"""Attribute accuracy of the conditional flow model: ask for every one of the 1296 attribute combos,
read the attributes back with the evaluator classifier.

    ../bin/gpu_run ../.venv/bin/python eval_cond.py runs/flow_cond
Writes results/cond_eval.json and runs/flow_cond/combos.npz (k samples per combo, used by demo.html)
(also reports CFD/valid/memorization of the conditional samples).
"""
import json, os, sys, time

import numpy as np
import mlx.core as mx

import data as D
import evaluate as E
import train_gen as TG
from train_classifier import predict


def main(run, steps=20, k=2, scales=(0.0, 1.0, 3.0)):
    model, g, cfg = TG.load_generator(run)
    S = E.real_stats(32)
    combos = np.array(np.meshgrid(*[np.arange(n) for n in D.N_CLASSES], indexing="ij")).reshape(6, -1).T  # (1296,6) in combo-id order
    assert (D.combo_id(combos) == np.arange(len(combos))).all()
    out = {"steps": steps, "sampler": "euler", "by_guidance": {}}
    best = None
    for w in scales:
        cond = mx.array(np.repeat(combos, k, axis=0).astype(np.int32))
        t0 = time.time()
        x = TG.to_uint8(TG.sample(model, g, len(cond), cfg, mx.random.key(11), steps=steps, cond=cond, guidance=w))
        dt = time.time() - t0
        probs, _ = predict(S["clf"], x)
        pred = np.stack([p.argmax(1) for p in probs], 1)
        want = np.asarray(cond)
        per = {a: float((pred[:, j] == want[:, j]).mean()) for j, a in enumerate(D.ATTR_NAMES)}
        m = E.strip(E.metrics(x))
        r = {"per_attr_acc": per, "all6_acc": float((pred == want).all(1).mean()), "n": len(x),
             "sec_per_1k": dt / len(x) * 1000, "nfe_per_sample": steps * (2 if w > 0 else 1),
             "valid": m["valid"], "cfd": m["cfd"], "mem_rate": m["mem_rate"], "div_feat": m["div_feat"]}
        out["by_guidance"][str(w)] = r
        print(w, json.dumps(r), flush=True)
        if best is None or r["all6_acc"] > best[0]:
            best = (r["all6_acc"], w, x, pred)
    # chance-level reference: per attribute 1/k, all six = 1/1296
    out["chance_all6"] = 1 / D.N_COMBOS
    out["best_guidance"] = best[1]
    os.makedirs(run, exist_ok=True)  # clean clone: runs/ is not committed
    np.savez_compressed(os.path.join(run, "combos.npz"), x=best[2], pred=best[3], k=k)
    json.dump(out, open("results/cond_eval.json", "w"), indent=1)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1])
