"""Planner mechanics; NumPy only. Predictor never receives ground-truth outcomes."""
import numpy as np
from r5_core import candidates, cem_proposal


def score_endpoints(endpoints, target, beta=0.0):
    """endpoints: (members, candidates, 12). Risk is in box-distance units."""
    assert endpoints.ndim == 3 and endpoints.shape[-1] == 12
    positions = endpoints[..., :2]
    mean = positions.mean(axis=0)
    disagreement = np.sqrt(((positions - mean[None]) ** 2).sum(axis=-1).mean(axis=0))
    distance = np.linalg.norm(mean - target, axis=-1)
    score = distance + beta * disagreement
    valid = np.isfinite(endpoints).all(axis=(0, 2)) & np.isfinite(score)
    return np.where(valid, score, np.inf), distance, disagreement, mean


def choose(velocities, endpoints, target, beta=0.0):
    score, distance, uncertainty, mean = score_endpoints(endpoints, target, beta)
    i = int(np.argmin(score))
    valid = bool(np.isfinite(score[i]))
    return dict(velocity=velocities[i].tolist(),
                dream=float(distance[i]) if valid else None,
                disagreement=float(uncertainty[i]) if valid else None,
                predicted_position=mean[i].tolist() if valid else None,
                score=float(score[i]) if valid else None,
                invalid_candidates=int((~np.isfinite(score)).sum()),
                all_invalid=bool(not np.isfinite(score).any()))


def random_search(predict, target, task_id, efforts, beta=0.0, seed=51902):
    v = candidates(task_id, max(efforts), seed)
    endpoints = predict(v)
    return {str(n): choose(v[:n], endpoints[:, :n], target, beta) for n in efforts}


def cem_search(predict, target, task_id, iterations, population=64, beta=0.0, seed=51902):
    """Incumbent retained across iterations; each iteration uses population new shots.

    Common random numbers across methods, but model-dependent elites/proposals.
    The first population is exactly the random-shooting prefix.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, task_id, 789]))
    velocity = candidates(task_id, population, seed)
    all_v, all_end, result = [], [], {}
    for it in range(1, max(iterations) + 1):
        endpoints = predict(velocity)
        all_v.append(velocity)
        all_end.append(endpoints)
        if it in iterations:
            result[str(it)] = choose(np.concatenate(all_v), np.concatenate(all_end, axis=1), target, beta)
        score = score_endpoints(endpoints, target, beta)[0]
        velocity = cem_proposal(velocity, score, population, rng)
    return result
