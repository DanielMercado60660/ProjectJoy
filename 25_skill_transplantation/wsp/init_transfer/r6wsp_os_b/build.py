#!/usr/bin/env python3
"""One-shot algebraic policy packing. No environments, optimization loops, or training.

Run from any directory with the ProjectJoy Python interpreter. All generated
files stay beside this script. Only wsp/data/{s0,s1,s2,s3}/*.npz are read.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "4"

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import qr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data_init"
N_FIT = 49152
N_BASIS = 8192
N_CHECK = 16384
TASKS = ("fwd", "bwd")
COUNTS = (64, 63)
SHAPES = {"a1.weight": (128, 19), "a1.bias": (128,),
          "a2.weight": (128, 128), "a2.bias": (128,),
          "a3.weight": (6, 128), "a3.bias": (6,), "log_std": (6,)}


def synthetic(p, seed, task, n, stream, stress=False):
    """Independent fabricated observations, never states from a simulator.

    Most coordinates are multiscale normalized Gaussians. A task-conditioned
    velocity mixture covers startup, intermediate speeds, and cruising. Root
    pitch is biased toward raw zero, with a broad Gaussian component retained.
    The stress distribution is fixed in advance, broader, and velocity-agnostic.
    """
    rng = np.random.default_rng(np.random.SeedSequence([62031, seed, task, stream]))
    scale = rng.choice(np.array([0.45, 0.85, 1.25, 1.9]), n,
                       p=[0.12, 0.38, 0.35, 0.15])
    x = rng.normal(size=(n, 17)) * scale[:, None]
    mean = p["norm_mean"].astype(np.float64)
    std = np.sqrt(p["norm_var"].astype(np.float64) + 1e-8)
    upright = rng.random(n) < 0.8
    x[upright, 1] = (rng.normal(0.0, 0.48, upright.sum()) - mean[1]) / std[1]
    # This is a chosen input distribution, not inferred trajectories or resets.
    mixture = rng.random(n)
    sign = 1.0 if task == 0 else -1.0
    speed = rng.normal(5.5 if task == 0 else 8.0, 2.2, n)
    speed[mixture < 0.18] = rng.normal(0.0, 1.2, (mixture < 0.18).sum())
    middle = (mixture >= 0.18) & (mixture < 0.4)
    speed[middle] = rng.uniform(0.0, 8.0 if task == 0 else 11.0, middle.sum())
    x[:, 8] = (sign * speed - mean[8]) / std[8]
    if stress:
        x *= 1.5
        x[:, 8] = rng.uniform(-7.5, 7.5, n)
    return np.clip(x, -10.0, 10.0).astype(np.float32)


def specialist(p, obs, task):
    h1 = np.tanh(obs @ p["a1.weight"][:, :17].T
                 + p["a1.bias"] + p["a1.weight"][:, 17 + task])
    z2 = h1 @ p["a2.weight"].T + p["a2.bias"]
    h2 = np.tanh(z2)
    logits = h2 @ p["a3.weight"].T + p["a3.bias"]
    return h1, z2, logits


def forward(p, obs, task, hidden=False):
    x = np.zeros((len(obs), 19), np.float32)
    x[:, :17] = obs
    x[:, 17 + task] = 1.0
    h1 = np.tanh(x @ p["a1.weight"].T + p["a1.bias"])
    h2 = np.tanh(h1 @ p["a2.weight"].T + p["a2.bias"])
    logits = h2 @ p["a3.weight"].T + p["a3.bias"]
    if hidden:
        return h1, h2, logits
    return np.tanh(logits)


def shared_basis(ps, xs):
    """128 paired, shared directions, each via one 19-variable linear solve.

    Regress the specialists' *preactivations* onto observations and two task
    indicators. Fixed teacher tanh slopes weight the regression. There is no
    refitting, parameter search, or loss-minimization iteration.
    """
    design = np.zeros((2 * N_BASIS, 19), np.float64)
    targets = []
    for task, (p, x) in enumerate(zip(ps, xs)):
        s = slice(task * N_BASIS, (task + 1) * N_BASIS)
        design[s, :17] = x[:N_BASIS]
        design[s, 17 + task] = 1.0
        targets.append(x[:N_BASIS].astype(np.float64)
                       @ p["a1.weight"][:, :17].astype(np.float64).T
                       + p["a1.bias"] + p["a1.weight"][:, 17 + task])
    targets = np.concatenate(targets)
    coefficients = np.empty((128, 19), np.float64)
    for j in range(128):
        z = targets[:, j]
        weights = (0.2 + 0.8 * (1.0 - np.tanh(z) ** 2)) ** 2
        gram = design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights * z)
        coefficients[j] = np.linalg.solve(gram + 1e-8 * np.eye(19), rhs)

    # Drop one redundant shared feature using output-weighted pivoted QR.
    h = np.tanh(design @ coefficients.T)
    h -= h.mean(axis=0, keepdims=True)
    influence = np.zeros(128, np.float64)
    for p in ps:
        output_size = np.linalg.norm(p["a3.weight"], axis=0)
        influence += np.sum((p["a2.weight"] * output_size[:, None]) ** 2, axis=0)
    _, _, piv = qr(h * np.sqrt(influence)[None, :], mode="economic", pivoting=True)
    keep = np.sort(piv[:127])
    return coefficients[keep].astype(np.float32), keep


def fit_second_layer(shared, x, target, p, task, keep):
    h = np.tanh(x @ shared[:, :17].T + shared[:, 17 + task])
    design = np.column_stack((h.astype(np.float64), np.ones(len(h))))
    # Small fixed Tikhonov anchor to the corresponding specialist coefficients.
    prior = np.vstack((p["a2.weight"][:, keep].T, p["a2.bias"][None, :]))
    reg = np.full(128, len(x) * 2e-5)
    reg[-1] = 1e-8
    coefficients = np.linalg.solve(design.T @ design + np.diag(reg),
                                    design.T @ target.astype(np.float64)
                                    + reg[:, None] * prior)
    approx = np.tanh(design @ coefficients)
    return coefficients, approx


def select_second_layer(activation, p, count):
    # Deterministic feature pruning; QR sees only synthetic activation columns.
    h = activation[:N_BASIS].copy()
    h -= h.mean(axis=0, keepdims=True)
    importance = np.linalg.norm(p["a3.weight"], axis=0)
    _, _, piv = qr(h * importance[None, :], mode="economic", pivoting=True)
    return np.sort(piv[:count])


def output_fit(h, logits, prior):
    design = np.column_stack((h.astype(np.float64), np.ones(len(h))))
    target = logits.astype(np.float64)
    coefficients = np.empty((design.shape[1], 6), np.float64)
    for k in range(6):
        # Fixed teacher-only weights approximate squared action-space error.
        weight = np.maximum(0.1, 1.0 - np.tanh(target[:, k]) ** 2) ** 2
        reg = np.full(design.shape[1], len(h) * 1e-6)
        reg[-1] = 1e-8
        gram = design.T @ (weight[:, None] * design) + np.diag(reg)
        rhs = design.T @ (weight * target[:, k]) + reg * prior[:, k]
        coefficients[:, k] = np.linalg.solve(gram, rhs)
    return coefficients[:-1].T, coefficients[-1]


def metrics(actual, target):
    difference = actual.astype(np.float64) - target.astype(np.float64)
    return {"action_rmse": float(np.sqrt(np.mean(difference ** 2))),
            "action_mae": float(np.mean(np.abs(difference))),
            "absolute_error_p95": float(np.quantile(np.abs(difference), 0.95)),
            "absolute_error_max": float(np.max(np.abs(difference))),
            "per_action_rmse": np.sqrt(np.mean(difference ** 2, axis=0)).tolist()}


def build_seed(seed):
    ps = []
    source_hashes = {}
    for task in TASKS:
        path = DATA / f"s{seed}" / f"{task}.npz"
        source_hashes[task] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as source:
            ps.append({k: source[k].copy() for k in source.files})
    assert np.array_equal(ps[0]["norm_mean"], ps[1]["norm_mean"])
    assert np.array_equal(ps[0]["norm_var"], ps[1]["norm_var"])
    xs = [synthetic(p, seed, t, N_FIT, 0) for t, p in enumerate(ps)]
    teacher = [specialist(p, x, t) for t, (p, x) in enumerate(zip(ps, xs))]
    shared, keep = shared_basis(ps, xs)
    result = {k: np.zeros(shape, np.float32) for k, shape in SHAPES.items()}
    result["log_std"] = ps[0]["log_std"].astype(np.float32).copy()
    result["a1.weight"][:127] = shared
    result["a1.weight"][127, 17:] = (12.0, -12.0)
    result["a2.weight"][127, 127] = 12.0
    banks = []
    selections = []
    gates = []
    start = 0
    for task, (p, x, count) in enumerate(zip(ps, xs, COUNTS)):
        coefficients, activation = fit_second_layer(shared, x, teacher[task][1], p, task, keep)
        selected = select_second_layer(activation, p, count)
        selections.append(selected.tolist())
        bank = np.arange(start, start + count)
        banks.append(bank)
        w = coefficients[:-1, selected].T
        b = coefficients[-1, selected]
        margin = np.sum(np.abs(w), axis=1) + np.abs(b) + 12.0
        result["a2.weight"][bank, :127] = w
        result["a2.weight"][bank, 127] = (1.0 if task == 0 else -1.0) * margin / 2.0
        result["a2.bias"][bank] = b - margin / 2.0
        gates.append({"inactive_preactivation_upper_bound": -12.0,
                      "gate_offset_min": float(margin.min()),
                      "gate_offset_max": float(margin.max())})
        start += count

    fits = []
    for task, (p, x, bank) in enumerate(zip(ps, xs, banks)):
        _, h2, _ = forward(result, x, task, hidden=True)
        prior = np.vstack((p["a3.weight"][:, selections[task]].T, p["a3.bias"][None, :]))
        fits.append(output_fit(h2[:, bank], teacher[task][2], prior))
    cf, df = fits[0]
    cb, db = fits[1]
    result["a3.weight"][:, banks[0]] = cf
    result["a3.weight"][:, banks[1]] = cb
    offset_f = df + cb.sum(axis=1)
    offset_b = db + cf.sum(axis=1)
    result["a3.bias"] = ((offset_f + offset_b) / 2.0).astype(np.float32)
    result["a3.weight"][:, 127] = (offset_f - offset_b) / 2.0

    for key, shape in SHAPES.items():
        assert result[key].shape == shape, (key, result[key].shape)
        assert result[key].dtype == np.float32, key
        assert np.isfinite(result[key]).all(), key
    path = HERE / f"merged_s{seed}.npz"
    np.savez(path, **result)
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == set(SHAPES)
        for key in SHAPES:
            assert np.array_equal(saved[key], result[key])

    report = {"seed": seed, "source_sha256": source_hashes,
              "first_layer_specialist_indices": keep.tolist(),
              "second_layer_specialist_indices": selections, "gates": gates,
              "synthetic_checks": {}}
    for task, (p, bank) in enumerate(zip(ps, banks)):
        report["synthetic_checks"][TASKS[task]] = {}
        for distribution, stress in (("independent_matched", False), ("broader_stress", True)):
            x = synthetic(p, seed, task, N_CHECK, 1 if not stress else 2, stress=stress)
            _, _, target_logits = specialist(p, x, task)
            _, h2, actual_logits = forward(result, x, task, hidden=True)
            inactive = banks[1 - task]
            assert np.array_equal(h2[:, inactive], -np.ones_like(h2[:, inactive]))
            assert np.all(h2[:, 127] == (1.0 if task == 0 else -1.0))
            actual = np.tanh(actual_logits)
            assert np.isfinite(actual).all() and np.max(np.abs(actual)) <= 1.0
            m = metrics(actual, np.tanh(target_logits))
            m["inactive_bank_max_deviation_from_minus_one"] = float(np.max(np.abs(h2[:, inactive] + 1.0)))
            report["synthetic_checks"][TASKS[task]][distribution] = m
    report["output_weight_max_abs"] = float(np.max(np.abs(result["a3.weight"])))
    report["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return report


def main():
    started = time.monotonic()
    reports = []
    for seed in range(4):
        report = build_seed(seed)
        reports.append(report)
        brief = {task: {dist: round(m["action_rmse"], 6) for dist, m in checks.items()}
                 for task, checks in report["synthetic_checks"].items()}
        print(f"seed {seed}: synthetic action RMSE {brief}", flush=True)
    summary = {"method": "paired shared features, QR pruning, closed-form regression, saturated task routing",
               "numpy_version": np.__version__, "fit_samples_per_task": N_FIT,
               "basis_samples_per_task": N_BASIS, "check_samples_per_task_per_distribution": N_CHECK,
               "environment_calls": 0, "gradient_steps": 0, "seeds": reports}
    (HERE / "diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Built and checked four networks in {time.monotonic() - started:.2f} seconds.", flush=True)


if __name__ == "__main__":
    main()
