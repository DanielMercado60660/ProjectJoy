"""Train the evaluator: attribute classifier on real train data, report held-out test accuracy.

    ../bin/gpu_run ../.venv/bin/python train_classifier.py
Writes checkpoints/classifier_{res}.safetensors and results/classifier_{res}.json
"""
import argparse, json, os, time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import data as D
import models as M


def loss_fn(m, x, y):
    logits, _ = m(x)
    return sum(nn.losses.cross_entropy(l, y[:, j]).mean() for j, l in enumerate(M.split_logits(logits)))


def predict(m, x_uint8, batch=1000):
    """Returns per-attribute probabilities list (6 arrays (N,k)) and features (N,256)."""
    probs = [[] for _ in D.N_CLASSES]; feats = []
    for i in range(0, len(x_uint8), batch):
        x = mx.array(x_uint8[i:i + batch]).astype(mx.float32) / 127.5 - 1
        logits, f = m(x)
        for j, l in enumerate(M.split_logits(logits)):
            probs[j].append(np.asarray(mx.softmax(l, axis=-1)))
        feats.append(np.asarray(f))
    return [np.concatenate(p) for p in probs], np.concatenate(feats)


def load_classifier(res=32):
    m = M.AttrClassifier()
    d = os.environ.get("MONSTER_CKPT_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    m.load_weights(os.path.join(d, f"classifier_{res}.safetensors"))
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--min_epochs", type=int, default=20, help="5 epochs proved too few (eyes 86%% acc)")
    a = ap.parse_args()
    a.epochs = max(a.epochs, a.min_epochs)
    mx.random.seed(0)
    xtr, ytr, xte, yte = D.load(a.res)
    m = M.AttrClassifier()
    steps = a.epochs * len(xtr) // a.batch
    opt = optim.AdamW(optim.cosine_decay(2e-3, steps), weight_decay=1e-4)
    vg = nn.value_and_grad(m, loss_fn)
    rng = np.random.default_rng(0)
    t0 = time.time()
    xtr_mx, ytr_mx = mx.array(xtr), mx.array(ytr.astype(np.int32))
    for ep in range(a.epochs):
        perm = rng.permutation(len(xtr))
        tot = 0
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            idx = mx.array(perm[i:i + a.batch])
            x = xtr_mx[idx].astype(mx.float32) / 127.5 - 1
            # light augmentation: brightness/contrast jitter + gaussian noise, so the evaluator is
            # less brittle on slightly-off generated images
            k1, k2 = mx.random.split(mx.random.key(ep * 100000 + i))
            x = x + 0.03 * mx.random.normal(x.shape, key=k1)
            l, g = vg(m, x, ytr_mx[idx]); opt.update(m, g); mx.eval(m.parameters(), opt.state, l)
            tot += float(l)
        m.eval()
        probs, _ = predict(m, xte)
        acc = [float((p.argmax(1) == yte[:, j]).mean()) for j, p in enumerate(probs)]
        m.train()
        print(f"epoch {ep} loss {tot / (len(perm) // a.batch):.4f} test acc {np.round(acc, 4).tolist()} {time.time() - t0:.0f}s", flush=True)
    m.eval()
    os.makedirs(a.ckpt_dir, exist_ok=True); os.makedirs(a.results_dir, exist_ok=True)
    m.save_weights(os.path.join(a.ckpt_dir, f"classifier_{a.res}.safetensors"))
    probs, _ = predict(m, xte)
    pred = np.stack([p.argmax(1) for p in probs], 1)
    conf = np.stack([p.max(1) for p in probs], 1)
    res = {"per_attr_test_acc": dict(zip(D.ATTR_NAMES, [float((pred[:, j] == yte[:, j]).mean()) for j in range(6)])),
           "all_attrs_correct": float((pred == yte).all(1).mean()),
           "frac_all_conf_gt_0.9": float((conf > 0.9).all(1).mean()),
           "train_sec": time.time() - t0, "epochs": a.epochs}
    json.dump(res, open(os.path.join(a.results_dir, f"classifier_{a.res}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
