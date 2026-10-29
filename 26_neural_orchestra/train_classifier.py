"""Train the separate style classifier used to score conditional control.

  ../.venv/bin/python train_classifier.py      -> checkpoints/classifier/

Trained on the train split only; reports accuracy on val and test (real patterns).
"""
import argparse
import json
import os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import data
from model import StyleClassifier, to_input, classify_numpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/drums.npz")
    ap.add_argument("--out", default="checkpoints/classifier")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = data.load(a.data)
    tr, va, te = (data.subset(d, k) for k in ("train", "val", "test"))
    X, Y = to_input(tr["hits"], tr["vel"], tr["off"]), tr["style"]
    mx.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    clf = StyleClassifier(a.hidden)
    opt = optim.Adam(learning_rate=1e-3)

    def loss_fn(m, x, y):
        return nn.losses.cross_entropy(m(x), y).mean()

    lg = nn.value_and_grad(clf, loss_fn)
    for ep in range(a.epochs):
        clf.train()
        perm = rng.permutation(len(X))
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            loss, g = lg(clf, mx.array(X[idx]), mx.array(Y[idx]))
            opt.update(clf, g)
            mx.eval(clf.parameters(), opt.state)
        if (ep + 1) % 10 == 0:
            clf.eval()
            acc = (classify_numpy(clf, va["hits"], va["vel"], va["off"]).argmax(1) == va["style"]).mean()
            print(f"epoch {ep + 1} loss {float(loss):.4f} val acc {acc:.4f}", flush=True)
    clf.eval()
    res = {}
    for name, sp in (("val", va), ("test", te)):
        p = classify_numpy(clf, sp["hits"], sp["vel"], sp["off"])
        res[f"{name}_acc"] = float((p.argmax(1) == sp["style"]).mean())
    # hits-only probe: does the classifier still work if velocity/timing are flattened?
    p = classify_numpy(clf, te["hits"], te["hits"] * 0.8, np.zeros_like(te["off"]))
    res["test_acc_hits_only_input"] = float((p.argmax(1) == te["style"]).mean())
    print(res)
    os.makedirs(a.out, exist_ok=True)
    clf.save_weights(os.path.join(a.out, "classifier.safetensors"))
    with open(os.path.join(a.out, "classifier_config.json"), "w") as f:
        json.dump({"hidden": a.hidden, **res}, f, indent=1)


if __name__ == "__main__":
    main()
