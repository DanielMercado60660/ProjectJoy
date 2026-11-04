"""Runner-side controls for the WSP result (Claude). Numpy only, seconds. No env steps.

What did Astra's construction owe to (a) the routing idea vs (b) the closed-form regression on synthetic inputs?
All variants reuse the routing skeleton every attempt converged on (shared layer 1 with per-task bias columns,
a +-1 task unit, 64 fwd / 63 bwd layer-2 units gated by a saturating offset, output-bias cancellation).

  ablate_route_only       NO fitting at all. Layer-1 obs rows = average of the aligned specialist rows; layer-2 units
                          = each specialist's 64 / 63 units with the largest outgoing |a3| column norm, ORIGINAL a2 rows
                          and a3 columns; one layer-1 unit dropped (smallest sum_t ||a3_t a2_t[:, i]||).
  ablate_route_refit_out  same, but the output layer of each bank is refit by one ridge solve to the specialist's
                          output logits on standard-normal synthetic inputs (layer 2 untouched).

  ../.venv/bin/python wsp/ablations.py     -> wsp/baselines/ablate_*_s<seed>.npz
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K = 12.0


def load(seed, name):
    z = np.load(os.path.join(HERE, "data", f"s{seed}", f"{name}.npz"))
    return {k: z[k].astype(np.float64) for k in z.files}


def specialist_parts(p, task, X):
    """Specialist forward on synthetic obs X (N,17) under its own label: h1, h2, logits."""
    oh = np.zeros((len(X), 2))
    oh[:, task] = 1
    x = np.concatenate([X, oh], 1)
    h1 = np.tanh(x @ p["a1.weight"].T + p["a1.bias"])
    h2 = np.tanh(h1 @ p["a2.weight"].T + p["a2.bias"])
    return h1, h2, h2 @ p["a3.weight"].T + p["a3.bias"]


def build(seed, refit_out):
    F, B = load(seed, "fwd"), load(seed, "bwd")
    imp = sum(np.linalg.norm(P["a3.weight"] @ P["a2.weight"], axis=0) for P in (F, B))
    drop = int(np.argmin(imp))
    keep1 = [i for i in range(128) if i != drop]                     # 127 shared layer-1 units
    W1 = np.zeros((128, 19))
    W1[:127, :17] = 0.5 * (F["a1.weight"][keep1, :17] + B["a1.weight"][keep1, :17])
    W1[:127, 17] = F["a1.bias"][keep1] + F["a1.weight"][keep1, 17]  # fwd effective bias
    W1[:127, 18] = B["a1.bias"][keep1] + B["a1.weight"][keep1, 18]  # bwd effective bias
    W1[127, 17], W1[127, 18] = K, -K                                 # s = +1 fwd, -1 bwd
    b1 = np.zeros(128)
    W2, b2 = np.zeros((128, 128)), np.zeros(128)
    sel = {}
    rows = {"fwd": range(0, 64), "bwd": range(64, 127)}
    rng = np.random.default_rng(1000 + seed)
    X = np.clip(rng.standard_normal((20000, 17)), -10, 10)
    fits = {}
    for task, P, sign in (("fwd", F, 1.0), ("bwd", B, -1.0)):
        n = len(rows[task])
        sel[task] = np.argsort(-np.linalg.norm(P["a3.weight"], axis=0))[:n]
        for r, j in zip(rows[task], sel[task]):
            w = P["a2.weight"][j, keep1]
            b = P["a2.bias"][j]
            M = np.abs(w).sum() + abs(b) + K
            W2[r, :127] = w
            # active: offset 0; inactive: -M (saturates to -1)
            W2[r, 127] = sign * M / 2
            b2[r] = b - M / 2
        C = P["a3.weight"][:, sel[task]]
        d = P["a3.bias"].copy()
        if refit_out:   # one ridge solve: specialist logits ~ merged active-bank features, synthetic inputs
            ti = 0 if task == "fwd" else 1
            _, _, L = specialist_parts(P, ti, X)
            oh = np.zeros((len(X), 2))
            oh[:, ti] = 1
            h1 = np.tanh(np.concatenate([X, oh], 1) @ W1.T + b1)
            h2 = np.tanh(h1 @ W2.T + b2)[:, list(rows[task])]
            A = np.concatenate([h2, np.ones((len(X), 1))], 1)
            reg = 1e-4 * len(X) * np.eye(A.shape[1])
            reg[-1, -1] = 0
            sol = np.linalg.solve(A.T @ A + reg, A.T @ L)
            C, d = sol[:-1].T, sol[-1]
        fits[task] = (C, d)
    W2[127, 127], b2[127] = K, 0.0                                   # output-layer task sign
    W3, b3 = np.zeros((6, 128)), np.zeros(6)
    (Cf, df), (Cb, db) = fits["fwd"], fits["bwd"]
    W3[:, 0:64], W3[:, 64:127] = Cf, Cb
    Sf, Sb = Cf.sum(1), Cb.sum(1)      # inactive bank outputs -1 -> contributes -S
    b3[:] = (df + db + Sf + Sb) / 2
    W3[:, 127] = (df + Sb - db - Sf) / 2
    out = {"a1.weight": W1, "a1.bias": b1, "a2.weight": W2, "a2.bias": b2, "a3.weight": W3, "a3.bias": b3,
           "log_std": F["log_std"]}
    return {k: v.astype(np.float32) for k, v in out.items()}


def check(p, seed):
    """Algebra check on synthetic inputs: merged == routed compressed nets (gates exact)."""
    rng = np.random.default_rng(5)
    X = np.clip(rng.standard_normal((2000, 17)) * 2, -10, 10)
    for t in (0, 1):
        oh = np.zeros((len(X), 2))
        oh[:, t] = 1
        h1 = np.tanh(np.concatenate([X, oh], 1) @ p["a1.weight"].T + p["a1.bias"])
        z2 = h1 @ p["a2.weight"].T + p["a2.bias"]
        off = slice(64, 127) if t == 0 else slice(0, 64)
        assert np.all(z2[:, off] <= -11.9), (seed, t, z2[:, off].max())


def main():
    os.makedirs(os.path.join(HERE, "baselines"), exist_ok=True)
    for seed in range(4):
        for name, ro in (("ablate_route_only", False), ("ablate_route_refit_out", True)):
            p = build(seed, ro)
            check(p, seed)
            np.savez(os.path.join(HERE, "baselines", f"{name}_s{seed}.npz"), **p)
    print("ok")


if __name__ == "__main__":
    main()
