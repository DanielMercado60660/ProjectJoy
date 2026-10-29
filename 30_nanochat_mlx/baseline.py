"""
Non-language baseline policy: a small MLP classifier over structured state features
(the same information the language model sees as text), trained on the same expert traces
(same episodes, same number of examples) to predict one of the 28 actions.

    ../.venv/bin/python baseline.py --episodes 8000 --out checkpoints/baseline_mlp
"""
import argparse
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from sandbox import ACTION_ID, ACTIONS, TRAIN_SEED0, expert_action, make_env

HERE = os.path.dirname(os.path.abspath(__file__))


class MLP(nn.Module):
    def __init__(self, din, hidden=256, nout=len(ACTIONS)):
        super().__init__()
        self.l1 = nn.Linear(din, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, nout)

    def __call__(self, x):
        return self.l3(nn.relu(self.l2(nn.relu(self.l1(x)))))


def build(n_episodes, seed0):
    X, Y = [], []
    for s in range(seed0, seed0 + n_episodes):
        env = make_env(s, "train")
        while not env.success():
            a = expert_action(env)
            X.append(env.features())
            Y.append(ACTION_ID[a])
            env.step(a)
    return np.stack(X), np.array(Y, dtype=np.int32)


def load(path):
    meta = json.load(open(os.path.join(path, "meta.json")))
    m = MLP(meta["din"], meta["hidden"])
    m.load_weights(os.path.join(path, "model.safetensors"))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="checkpoints/baseline_mlp")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    mx.random.seed(a.seed)
    X, Y = build(a.episodes, TRAIN_SEED0)
    Xv, Yv = build(300, TRAIN_SEED0 + 5_000_000)
    print("train examples", X.shape, "val", Xv.shape)
    model = MLP(X.shape[1], a.hidden)
    mx.eval(model.parameters())
    opt = optim.AdamW(1e-3, weight_decay=1e-4)
    lf = lambda m, x, y: nn.losses.cross_entropy(m(x), y, reduction="mean")
    lg = nn.value_and_grad(model, lf)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            loss, g = lg(model, mx.array(X[idx]), mx.array(Y[idx]))
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
        acc = float((mx.argmax(model(mx.array(Xv)), axis=-1) == mx.array(Yv)).mean())
        if ep % 5 == 4 or ep == a.epochs - 1:
            print(f"epoch {ep+1} loss {loss.item():.4f} val action acc {acc:.4f}")
    out = os.path.join(HERE, a.out)
    os.makedirs(out, exist_ok=True)
    model.save_weights(os.path.join(out, "model.safetensors"))
    json.dump({"din": int(X.shape[1]), "hidden": a.hidden, "episodes": a.episodes, "examples": int(len(X)),
               "val_action_acc": acc, "train_time_s": time.time() - t0,
               "params": sum(v.size for _, v in __import__("mlx.utils").utils.tree_flatten(model.parameters()))},
              open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
