"""Misspecified classical dynamics: finite-duration spring/damper contacts.

This family cannot implement the simulator's instantaneous velocity exchange and
overlap projection. Stiffness is finite (80..32000); there is no hard-impulse or
projection limit in the search. Known walls/gravity use the same integrator.
Only NumPy is needed. No simulator parameters are fitted from test trajectories.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import sim
from r5_core import Budget, BudgetExpired, SCALE, digest, sample_windows, pos_error, write_json

BOUNDS = dict(stiffness=(80., 32000.), damping_ratio=(0., 1.2), contact_radius=(.04, .09))


def validate_parameters(parameters):
    if set(parameters) != set(BOUNDS):
        raise ValueError("soft-contact parameters must be stiffness, damping_ratio, contact_radius")
    for k, (lo, hi) in BOUNDS.items():
        if not np.isfinite(parameters[k]) or not lo <= parameters[k] <= hi:
            raise ValueError(f"{k} outside finite soft-contact family")


def soft_step(state, stiffness, damping_ratio, contact_radius):
    """Spring force + non-attractive damping, simultaneous equal/opposite pair forces.

    Unit masses. c = zeta*sqrt(2*k) gives relative-coordinate damping ratio zeta.
    There is no instantaneous collision impulse or position separation correction.
    The effective radius belongs only to contacts; the known wall radius stays fixed.
    """
    s = np.asarray(state, float).reshape(-1, 3, 4)
    p, v = s[..., :2].copy(), s[..., 2:].copy()
    h = sim.FRAME_DT / sim.SUBSTEPS
    damping = damping_ratio * np.sqrt(2 * stiffness)
    for _ in range(sim.SUBSTEPS):
        v[..., 1] -= sim.G * h
        p += v * h
        lo, hi = p < sim.R, p > 1 - sim.R
        v = np.where((lo & (v < 0)) | (hi & (v > 0)), -sim.E_WALL * v, v)
        p = np.where(lo, 2 * sim.R - p, p)
        p = np.where(hi, 2 * (1 - sim.R) - p, p)
        force = np.zeros_like(v)
        # All forces read the same p/v; unlike hard sequential contact resolution.
        for i, j in ((0, 1), (0, 2), (1, 2)):
            delta = p[:, j] - p[:, i]
            distance = np.sqrt(np.sum(delta ** 2, axis=-1) + 1e-24)
            normal = delta / distance[:, None]
            rate = np.sum((v[:, j] - v[:, i]) * normal, axis=-1)
            overlap = 2 * contact_radius - distance
            magnitude = np.where(overlap > 0, np.maximum(stiffness * overlap - damping * rate, 0), 0)
            f = magnitude[:, None] * normal
            force[:, i] -= f
            force[:, j] += f
        v += h * force
    return np.concatenate([p, v], axis=-1).reshape(state.shape)


def soft_rollout(init, horizon, parameters, *, final=False, budget=None):
    validate_parameters(parameters)
    s = np.asarray(init, float)
    out = [s]
    for _ in range(horizon):
        if budget:
            budget.check()
        s = soft_step(s, **parameters)
        if not final:
            out.append(s)
    return s if final else np.stack(out, axis=1)


def proposals(count, seed=53011):
    # Upper end still resolves an undamped contact across ~6 substeps; finite
    # contact duration is preserved without artificially stopping at a soft law.
    grid = itertools.product((80., 600., 4400., 32000.), (0., .15, .5, 1.), (.045, .06, .08))
    result = [dict(stiffness=k, damping_ratio=z, contact_radius=r) for k, z, r in grid]
    rng = np.random.default_rng(seed)
    while len(result) < count:
        result.append(dict(stiffness=float(np.exp(rng.uniform(*np.log(BOUNDS["stiffness"])))),
                           damping_ratio=float(rng.uniform(0, 1.2)), contact_radius=float(rng.uniform(.04, .09))))
    return result[:count]


def fit_soft(a, budget):
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    data = Path(a.data)
    tr, val = np.load(data / "train.npz"), np.load(data / "val.npz")
    if a.fit_horizon >= tr["states"].shape[1] or a.horizon >= val["states"].shape[1]:
        raise ValueError("requested horizon exceeds dataset")
    win = sample_windows(tr["states"], tr["flags"], a.samples, a.fit_horizon, np.random.default_rng(53012))
    truth = val["states"][:a.val_scenes, :a.horizon + 1]
    trials, complete = [], False

    def assess(parameters, stage):
        prediction = soft_rollout(win[:, 0], a.fit_horizon, parameters, budget=budget)
        loss = float(np.mean(((prediction[:, 1:] - win[:, 1:]) / SCALE) ** 2))
        row = dict(parameters=parameters, stage=stage, train_loss=loss if np.isfinite(loss) else None)
        trials.append(row)

    try:
        for p in proposals(a.proposals):
            assess(p, "coarse_or_random")
        centres = sorted([r for r in trials if r["train_loss"] is not None], key=lambda r: r["train_loss"])[:8]
        if not centres:
            raise RuntimeError("all soft-contact fits were nonfinite")
        rng = np.random.default_rng(53013)
        for i in range(a.refine):
            best = centres[i % len(centres)]["parameters"]
            p = dict(stiffness=float(np.clip(best["stiffness"] * np.exp(rng.normal(0, .3)), *BOUNDS["stiffness"])),
                     damping_ratio=float(np.clip(best["damping_ratio"] + rng.normal(0, .1), *BOUNDS["damping_ratio"])),
                     contact_radius=float(np.clip(best["contact_radius"] + rng.normal(0, .005), *BOUNDS["contact_radius"])))
            assess(p, "local_refinement")
        shortlist = sorted([r for r in trials if r["train_loss"] is not None], key=lambda r: r["train_loss"])[:a.shortlist]
        for row in shortlist:
            prediction = soft_rollout(truth[:, 0], a.horizon, row["parameters"], budget=budget)
            error = pos_error(prediction, truth)[:, 1:]
            row["val_mean_error"] = float(error.mean()) if np.isfinite(error).all() else None
            row["val_final_mean_error"] = float(error[:, -1].mean()) if np.isfinite(error).all() else None
        complete = True
    except BudgetExpired:
        pass
    validated = [r for r in trials if r.get("val_mean_error") is not None]
    selected = min(validated, key=lambda r: r["val_mean_error"]) if validated else None
    write_json(a.out, dict(complete=complete, family="finite_spring_damper_v1", oracle=False,
                           contains_exact_hard_law=False, ranges=BOUNDS, selected=selected, trials=trials,
                           train_sha256=digest(data / "train.npz"), val_sha256=digest(data / "val.npz"),
                           requested_trials=a.proposals + a.refine, shortlist=a.shortlist,
                           train_samples=a.samples, fit_horizon=a.fit_horizon, val_scenes=len(truth), horizon=a.horizon,
                           validation_objective="mean per-ball position error over frames 1..horizon, natural validation scenes",
                           seconds=budget.elapsed))
    print(f"SOFT_FIT_{'OK' if complete else 'PARTIAL'} trials={len(trials)} selected={selected} seconds={budget.elapsed:.2f}", flush=True)
    if not complete or selected is None:
        raise RuntimeError("soft-contact fit incomplete; retain receipt and rerun to a new path with enough budget")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--proposals", type=int, default=192)
    ap.add_argument("--refine", type=int, default=64)
    ap.add_argument("--shortlist", type=int, default=32)
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--val-scenes", type=int, default=128)
    ap.add_argument("--fit-horizon", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--max-minutes", type=float, required=True)
    a = ap.parse_args()
    if min(a.proposals, a.shortlist, a.samples, a.val_scenes, a.fit_horizon, a.horizon) < 1 or a.refine < 0:
        ap.error("counts must be positive (refine may be zero)")
    fit_soft(a, Budget(a.max_minutes))


if __name__ == "__main__":
    main()
