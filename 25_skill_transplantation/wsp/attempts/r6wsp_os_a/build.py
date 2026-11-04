"""ONE-SHOT, deterministic, closed-form conditional policy compression.

Reads only wsp/data/s*/{theta0,fwd,bwd}.npz. No environment, autodiff,
optimization loop, rollout data, or candidate search is used.
"""

import os

# Set before importing numerical libraries; coexist with other Joy Lab jobs.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh, qr, solve


HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
N_FIT = 32768
N_CHECK = 8192
SHAPES = {
    "a1.weight": (128, 19), "a1.bias": (128,),
    "a2.weight": (128, 128), "a2.bias": (128,),
    "a3.weight": (6, 128), "a3.bias": (6,), "log_std": (6,),
}


def synthetic(p, task, n, seed):
    """Independent algebraic samples, never states obtained from a simulator.

    70% moderate normalized coordinates with narrow physical pitch and a
    directional velocity prior; 20% broad independent coordinates; 10%
    small raw-coordinate perturbations about zero. Clip at the task's bound.
    """
    rng = np.random.default_rng(seed)
    mean = p["norm_mean"]
    std = np.sqrt(p["norm_var"] + 1e-8)
    n_run, n_broad = int(0.70 * n), int(0.20 * n)
    n_zero = n - n_run - n_broad
    speed = 6.0 if task == 0 else -8.5

    run = rng.normal(size=(n_run, 17))
    scale = np.exp(rng.uniform(np.log(0.65), np.log(1.55), (n_run, 1)))
    run *= scale
    run[:, 0] *= 0.60
    run[:, 1] = (rng.normal(0.0, 0.25, n_run) - mean[1]) / std[1]
    velocity = rng.normal(speed, 2.5, n_run)
    slow = rng.random(n_run) < 0.20
    velocity[slow] = rng.uniform(0.0, abs(speed), slow.sum()) * np.sign(speed)
    run[:, 8] = (velocity - mean[8]) / std[8]
    run[:, 9:11] *= 0.80

    broad = rng.normal(0.0, 1.5, (n_broad, 17))
    broad[:, 8] = (rng.normal(speed * 0.5, 5.0, n_broad) - mean[8]) / std[8]

    raw = np.empty((n_zero, 17))
    raw[:, :8] = rng.uniform(-0.1, 0.1, (n_zero, 8))
    raw[:, 8:] = rng.normal(0.0, 0.1, (n_zero, 9))
    zero = (raw - mean) / std
    return np.clip(np.concatenate((run, broad, zero)), -10, 10).astype(np.float32)


def teacher(p, obs, task):
    h1 = np.tanh(obs @ p["a1.weight"][:, :17].T
                 + (p["a1.bias"] + p["a1.weight"][:, 17 + task]))
    z2 = h1 @ p["a2.weight"].T + p["a2.bias"]
    h2 = np.tanh(z2)
    logits = h2 @ p["a3.weight"].T + p["a3.bias"]
    return z2, h2, logits


def forward(p, obs, task, return_hidden=False):
    x = np.zeros((len(obs), 19), np.float32)
    x[:, :17] = obs
    x[:, 17 + task] = 1
    h1 = np.tanh(x @ p["a1.weight"].T + p["a1.bias"])
    h2 = np.tanh(h1 @ p["a2.weight"].T + p["a2.bias"])
    a = np.tanh(h2 @ p["a3.weight"].T + p["a3.bias"])
    return (a, h1, h2) if return_hidden else a


def affine_fit(x, y, weight, ridge=1e-6):
    """One centered weighted ridge least-squares solve, intercept unpenalized.

    Return coefficient (features, outputs) and intercept. This is a direct
    linear solve, with no updates, reweighting iterations, or loss search.
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    w = np.asarray(weight, np.float64)
    w = w / w.sum()
    xm = w @ x
    ym = w @ y
    cov = x.T @ (w[:, None] * x) - np.outer(xm, xm)
    rhs = x.T @ (w[:, None] * y) - np.outer(xm, ym)
    penalty = ridge * max(float(np.trace(cov) / len(cov)), 1e-12)
    cov.flat[:: len(cov) + 1] += penalty
    coef = solve(cov, rhs, assume_a="pos", check_finite=False)
    return coef, ym - xm @ coef


def select_second(h2, logits, p, count):
    """Output-weighted pivoted QR of an activation covariance square root.

    QR selects a diverse subset of existing teacher neurons. Its column
    scaling includes readout magnitude and average output sensitivity.
    """
    h = h2.astype(np.float64)
    h -= h.mean(axis=0)
    cov = h.T @ h / len(h)
    value, vector = eigh(cov, check_finite=False)
    factor = np.sqrt(np.maximum(value, 0))[:, None] * vector.T
    sensitivity = np.mean(0.1 + 0.9 * (1 - np.tanh(logits.astype(np.float64)) ** 2) ** 2,
                          axis=0)
    importance = np.sqrt(sensitivity @ (p["a3.weight"].astype(np.float64) ** 2))
    _, _, order = qr(factor * importance[None, :], pivoting=True,
                     mode="economic", check_finite=False)
    return np.asarray(order[:count], dtype=np.int64)


def compress_branch(p, task, obs, common_w, common_bias, count):
    z2, h2, logits = teacher(p, obs, task)
    selected = select_second(h2, logits, p, count)
    basis = np.tanh(obs @ common_w.T + common_bias).astype(np.float64)
    rows, intercept = [], []
    for j in selected:
        # Each target has its own fixed, teacher-derived weight. Emphasize
        # the tanh transition while keeping all samples in the solve.
        w = 0.1 + 0.9 * (1 - h2[:, j].astype(np.float64) ** 2) ** 2
        c, b = affine_fit(basis, z2[:, j:j + 1], w)
        rows.append(c[:, 0])
        intercept.append(b[0])
    w2 = np.asarray(rows, np.float32)
    b2 = np.asarray(intercept, np.float32)
    student_h2 = np.tanh(basis.astype(np.float32) @ w2.T + b2)

    readout, offset = [], []
    for action in range(6):
        derivative = 1 - np.tanh(logits[:, action].astype(np.float64)) ** 2
        c, b = affine_fit(student_h2, logits[:, action:action + 1],
                          0.1 + 0.9 * derivative ** 2)
        readout.append(c[:, 0])
        offset.append(b[0])
    return w2, b2, np.asarray(readout), np.asarray(offset), selected


def load_seed(seed):
    networks = {}
    source_hashes = {}
    for name in ("theta0", "fwd", "bwd"):
        path = DATA / f"s{seed}" / f"{name}.npz"
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as src:
            networks[name] = {k: src[k].copy() for k in src.files}
        for key, shape in SHAPES.items():
            assert networks[name][key].shape == shape
    for name in ("fwd", "bwd"):
        for key in ("norm_mean", "norm_var"):
            assert np.array_equal(networks[name][key], networks["theta0"][key])
    return networks, source_hashes


def build_seed(seed):
    start = time.monotonic()
    nets, hashes = load_seed(seed)
    experts = [nets["fwd"], nets["bwd"]]
    inputs = [synthetic(p, t, N_FIT, 62026000 + 100 * seed + t)
              for t, p in enumerate(experts)]

    # Preserve aligned neuron identities. Free one weak path for a task bit.
    saliency = np.zeros(128)
    for p in experts:
        out_norm2 = np.sum(p["a3.weight"].astype(np.float64) ** 2, axis=0)
        saliency += out_norm2 @ (p["a2.weight"].astype(np.float64) ** 2)
    dropped = int(np.argmin(saliency))
    keep = np.delete(np.arange(128), dropped)
    common_w = ((experts[0]["a1.weight"][keep, :17].astype(np.float64)
                 + experts[1]["a1.weight"][keep, :17]) * 0.5).astype(np.float32)
    task_biases = []
    for t, p in enumerate(experts):
        center = inputs[t].astype(np.float64).mean(axis=0)
        bias = p["a1.bias"][keep].astype(np.float64) + p["a1.weight"][keep, 17 + t]
        bias += (p["a1.weight"][keep, :17].astype(np.float64) - common_w) @ center
        task_biases.append(bias.astype(np.float32))

    branches = [compress_branch(p, t, inputs[t], common_w, task_biases[t], count)
                for t, (p, count) in enumerate(zip(experts, (64, 63)))]
    merged = {k: np.zeros(shape, np.float32) for k, shape in SHAPES.items()}
    merged["log_std"] = nets["theta0"]["log_std"].astype(np.float32).copy()
    merged["a1.weight"][:127, :17] = common_w
    merged["a1.weight"][:127, 17] = task_biases[0]
    merged["a1.weight"][:127, 18] = task_biases[1]
    merged["a1.weight"][127, 17:] = (20, -20)

    gate_bounds = []
    for task, (begin, end) in enumerate(((0, 64), (64, 127))):
        w2, b2, readout, offset, selected = branches[task]
        # For any bounded hidden vector, inactive preactivation >= 12.
        gate = ((np.abs(w2.astype(np.float64)).sum(axis=1)
                 + np.abs(b2.astype(np.float64)) + 12.0) / 2).astype(np.float32)
        merged["a2.weight"][begin:end, :127] = w2
        merged["a2.weight"][begin:end, 127] = gate * (-1 if task == 0 else 1)
        merged["a2.bias"][begin:end] = b2 + gate
        merged["a3.weight"][:, begin:end] = readout.astype(np.float32)
        lower = (merged["a2.bias"][begin:end].astype(np.float64) + gate
                 - np.abs(w2.astype(np.float64)).sum(axis=1))
        gate_bounds.append(float(lower.min()))

    merged["a2.weight"][127, 127] = 20
    cf = merged["a3.weight"][:, :64].astype(np.float64).sum(axis=1)
    cb = merged["a3.weight"][:, 64:127].astype(np.float64).sum(axis=1)
    df, db = branches[0][3], branches[1][3]
    merged["a3.bias"] = ((df + db - cf - cb) * 0.5).astype(np.float32)
    merged["a3.weight"][:, 127] = ((df - db + cf - cb) * 0.5).astype(np.float32)

    path = HERE / f"merged_s{seed}.npz"
    for k, shape in SHAPES.items():
        assert merged[k].dtype == np.float32 and merged[k].shape == shape
        assert np.isfinite(merged[k]).all()
    np.savez(path, **merged)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(SHAPES)
        reloaded = {k: archive[k] for k in archive.files}
        for k in SHAPES:
            assert np.array_equal(reloaded[k], merged[k])

    report = {
        "seed": seed, "source_sha256": hashes, "fit_samples_per_task": N_FIT,
        "check_samples_per_task": N_CHECK, "dropped_first_neuron": dropped,
        "shared_first_neurons": keep.tolist(),
        "fwd_second_neurons": branches[0][4].tolist(),
        "bwd_second_neurons": branches[1][4].tolist(),
        "inactive_preactivation_lower_bounds": gate_bounds,
        "synthetic_only": {},
    }
    for task, p in enumerate(experts):
        obs = synthetic(p, task, N_CHECK, 62029000 + 100 * seed + task)
        _, _, logits = teacher(p, obs, task)
        reference = np.tanh(logits)
        action, h1, h2 = forward(reloaded, obs, task, return_hidden=True)
        off = h2[:, 64:127] if task == 0 else h2[:, :64]
        assert np.array_equal(off, np.ones_like(off))
        assert np.all(h1[:, 127] == (1 if task == 0 else -1))
        assert np.all(h2[:, 127] == (1 if task == 0 else -1))
        assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1
        error = action.astype(np.float64) - reference
        report["synthetic_only"][("fwd", "bwd")[task]] = {
            "action_rmse": float(np.sqrt(np.mean(error ** 2))),
            "action_mae": float(np.mean(np.abs(error))),
            "per_action_rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
            "p95_absolute_action_error": float(np.percentile(np.abs(error), 95)),
            "max_inactive_deviation_from_one": float(np.max(np.abs(off - 1))),
            "regime_rmse": {
                name: float(np.sqrt(np.mean(error[region] ** 2)))
                for name, region in (
                    ("directional", slice(0, int(0.70 * N_CHECK))),
                    ("broad", slice(int(0.70 * N_CHECK), int(0.70 * N_CHECK) + int(0.20 * N_CHECK))),
                    ("near_raw_zero", slice(int(0.70 * N_CHECK) + int(0.20 * N_CHECK), None)),
                )
            },
        }
    # Stress the exact routing property over the entire observation clipping box.
    rng = np.random.default_rng(62031000 + seed)
    corners = rng.choice(np.array([-10, 10], np.float32), size=(1024, 17))
    for task in (0, 1):
        _, _, hidden = forward(reloaded, corners, task, return_hidden=True)
        off = hidden[:, 64:127] if task == 0 else hidden[:, :64]
        assert np.all(off == 1)
    report["elapsed_seconds"] = time.monotonic() - start
    (HERE / f"diagnostics_s{seed}.json").write_text(json.dumps(report, indent=2) + "\n")
    brief = {"seed": seed, "seconds": report["elapsed_seconds"],
             "synthetic_rmse": {k: v["action_rmse"] for k, v in report["synthetic_only"].items()}}
    print(json.dumps(brief), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(4), default=None)
    args = parser.parse_args()
    for seed in range(4) if args.seed is None else (args.seed,):
        build_seed(seed)


if __name__ == "__main__":
    main()
