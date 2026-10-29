"""Round 5: probe for a placement task. NumPy/SciPy only; no neural imports.

The true simulator is used only by observation/scoring code. Policies always assume
unit actuator gain. Planning is finite-quadrature, one-step EIG or task VOI.
"""
from dataclasses import dataclass
import time

import numpy as np

from physics import (DESIGNS, G, MU_HI, MU_LO, MU_RATIO, NT, SIGMA, VIEW,
                     loglik_frames, positions, rest_distance, world1_positions)

BUDGETS = (0, 1, 2, 4)
METHODS = ("eig", "task_voi", "lsq", "tobit")
FAMILIES = ("matched", "gain_shift")
ZONE_WIDTH = 0.05
# Same final action menu for everyone, including both oracles.
FINAL_ACTIONS = np.array([(f, t) for f in range(1, 16)
                          for t in np.linspace(0.1, 1.0, 181)])
PROTOCOLS = ("fixed8", "fixed12", "ladder") + tuple(
    f"adaptive_{f}_{d}" for f in (10, 12, 15) for d in (35, 65, 90, "target"))
AIM_OFFSETS = (-0.5, 0.0, 0.5)  # multiples of the target half-width


class TimeLimit(Exception):
    """Discard the current incomplete work unit and resume it deterministically."""


class Guard:
    def __init__(self, max_minutes):
        if not 0 < max_minutes <= 45:
            raise ValueError("--max-minutes must be in (0, 45]")
        self.started = time.monotonic()
        self.deadline = self.started + 60 * max_minutes

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeLimit()


def rng_for(seed, world, stream, step=0):
    """Counter-addressed independent PCG64 streams; no resume-dependent RNG state."""
    return np.random.default_rng(np.random.SeedSequence([seed, world, stream, step]))


def make_world(seed, index, legacy_mu=False):
    rng = rng_for(seed, index, 1)
    mu = rng.uniform(MU_LO, MU_HI)
    if legacy_mu:
        # Earlier run_active.py used default_rng([seed, world]).uniform(...).
        # Reuse those friction values only; new task, noise and targets remain R5.
        mu = np.random.default_rng([2026, index]).uniform(MU_LO, MU_HI)
    target = rng.uniform(0.25, 0.85)
    gain = 1 + rng.choice([-1, 1]) * rng.uniform(0.06, 0.12)
    noise = rng_for(seed, index, 2).standard_normal((max(BUDGETS), NT))
    return float(mu), float(target), float(gain), noise


def true_observation(action, mu, gain, noise):
    x = positions(*action, mu, MU_RATIO * mu, k=gain)
    y = x + SIGMA * noise
    cens = y > VIEW
    return np.where(cens, np.nan, y), cens


def true_stop(action, mu, gain=1.0):
    return rest_distance(action[..., 0], action[..., 1], mu, MU_RATIO * mu, k=gain)


def cdf_at(x, grid, weights):
    """Mid-cell CDF, with explicit support endpoints; batched weights allowed.

    Midpoint grids make a uniform prior exactly uniform. This avoids the old
    right-edge CDF's half-cell interval shift. Quantile particles are an additional
    approximation during planning, never the final inference grid.
    """
    x = np.asarray(x)
    nodes = np.r_[MU_LO, grid, MU_HI]
    cdf = np.concatenate((np.zeros(weights.shape[:-1] + (1,)),
                          np.cumsum(weights, axis=-1) - weights / 2,
                          np.ones(weights.shape[:-1] + (1,))), axis=-1)
    hi = np.clip(np.searchsorted(nodes, x, side="right"), 1, len(nodes) - 1)
    frac = np.clip((x - nodes[hi - 1]) / (nodes[hi] - nodes[hi - 1]), 0, 1)
    return cdf[..., hi - 1] * (1 - frac) + cdf[..., hi] * frac


def quantiles(grid, weights, q):
    c = np.r_[0, np.cumsum(weights) - weights / 2, 1]
    return np.interp(q, c, np.r_[MU_LO, grid, MU_HI])


def success_bounds(actions, target):
    """Invert analytic stopping distance to a friction interval for every action.

    Positive target zone excludes the stationary branch. Static threshold is a
    strict inequality, whose boundary has zero mass under the continuous prior.
    """
    f, t = actions.T
    low_d, high_d = target - ZONE_WIDTH / 2, target + ZONE_WIDTH / 2
    lower = f * f * t * t / (G * (2 * high_d + f * t * t))
    upper = np.minimum(f * f * t * t / (G * (2 * low_d + f * t * t)),
                       f / (MU_RATIO * G))
    return lower, upper


class TerminalTask:
    def __init__(self, target, actions=FINAL_ACTIONS):
        self.target = float(target)
        self.actions = actions
        self.lower, self.upper = success_bounds(actions, target)

    def probabilities(self, grid, weights):
        return np.maximum(0, cdf_at(self.upper, grid, weights)
                          - cdf_at(self.lower, grid, weights))

    def bayes_action(self, post):
        p = self.probabilities(post.grid, post.weights)
        # Stable argmax on a common menu; no knowledge of truth in tie-breaking.
        i = int(np.argmax(p))
        return self.actions[i], float(p[i])

    def point_action(self, mu, offset=0.0, gain=1.0):
        distances = true_stop(self.actions, mu, gain)
        aim = self.target + offset * ZONE_WIDTH / 2
        return self.actions[int(np.argmin(np.abs(distances - aim)))]

    def probability_of(self, action, post):
        lo, hi = success_bounds(np.array([action]), self.target)
        return float(np.maximum(0, cdf_at(hi, post.grid, post.weights)
                                - cdf_at(lo, post.grid, post.weights))[0])


class Belief:
    def __init__(self, grid_size=6001):
        assert grid_size >= 101
        self.grid = MU_LO + (np.arange(grid_size) + 0.5) * (MU_HI - MU_LO) / grid_size
        self.logp = np.zeros(grid_size)
        self.sse = np.zeros(grid_size)

    @property
    def weights(self):
        w = np.exp(self.logp - self.logp.max())
        return w / w.sum()

    def update(self, action, y, cens):
        x = world1_positions(*action, self.grid)
        self.logp += loglik_frames(y, cens, x)
        self.logp -= self.logp.max()
        self.sse += np.where(cens, 0, (np.nan_to_num(y) - x) ** 2).sum(-1)

    def estimate(self, method):
        if method in ("eig", "task_voi", "blind"):
            return float(self.weights @ self.grid)
        objective = self.sse if method == "lsq" else -self.logp
        tied = np.flatnonzero(np.isclose(objective, objective.min(), rtol=0, atol=1e-10))
        # Unidentified flat fits choose the prior centre, not a range endpoint.
        return float(self.grid[tied[np.argmin(abs(self.grid[tied] - (MU_LO + MU_HI) / 2))]])

    def stats(self):
        w = self.weights
        mean = float(w @ self.grid)
        sd = float(np.sqrt(w @ (self.grid - mean) ** 2))
        q = quantiles(self.grid, w, [0.05, 0.25, 0.75, 0.95])
        return dict(posterior_mean=mean, posterior_sd=sd,
                    ci50=[float(q[1]), float(q[2])], ci90=[float(q[0]), float(q[3])],
                    grid_step_over_sd=float((MU_HI - MU_LO) / len(w) / max(sd, 1e-300)))


@dataclass(frozen=True)
class PlannerConfig:
    grid_size: int = 6001
    particles: int = 64
    fantasies: int = 16
    design_stride: int = 1

    def validate(self):
        assert self.grid_size >= 101 and self.particles >= 4 and self.fantasies >= 2
        assert self.design_stride >= 1


def design_scores(post, task, rng, cfg, guard=None):
    """Both scores use identical candidate/simulation/inference budgets.

    EIG = E_y KL[p(theta|y) || p(theta)] in the quantile-particle model.
    Task VOI = E_y max_a P(success|y,a) - max_a P(success|a).
    The next observation comes from the SAME discrete planning model. Stratified
    sampled generating particles + common noise reduce ranking variance. This is
    a myopic criterion at every budget, not a full multi-probe tree search.
    """
    k, n = cfg.particles, cfg.fantasies
    particles = quantiles(post.grid, post.weights, (np.arange(k) + 0.5) / k)
    source = np.minimum(((np.arange(n) + rng.random(n)) / n * k).astype(int), k - 1)
    eps = rng.standard_normal((n, NT))
    designs = DESIGNS[::cfg.design_stride]
    prior_value = float(task.probabilities(particles, np.full(k, 1 / k)).max())
    info, value = [], []
    for start in range(0, len(designs), 5):
        if guard:
            guard.check()
        d = designs[start:start + 5]
        x = world1_positions(d[:, 0, None], d[:, 1, None], particles[None, :])
        y = x[:, source, :] + SIGMA * eps
        cens = y > VIEW
        ll = loglik_frames(y[:, :, None, :], cens[:, :, None, :], x[:, None, :, :])
        logw = ll - ll.max(-1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(-1, keepdims=True)
        info.extend((np.log(k) + (w * np.log(np.maximum(w, 1e-300))).sum(-1)).mean(-1))
        value.extend(task.probabilities(particles, w).max(-1).mean(-1) - prior_value)
    return designs, np.asarray(info), np.asarray(value)


def baseline_probe(protocol, mu_hat, step, target):
    if protocol == "fixed8":
        return np.array([8.0, 0.3])
    if protocol == "fixed12":
        return np.array([12.0, 0.2])
    if protocol == "ladder":
        return np.array([float((6, 9, 12, 15)[step]), 0.3])
    _, force, distance = protocol.split("_")
    choices = DESIGNS[DESIGNS[:, 0] == int(force)]
    stops = true_stop(choices, mu_hat)
    aim = target if distance == "target" else int(distance) / 100
    return choices[int(np.argmin(abs(stops - aim)))]


def trace_observation(action, y, cens, selected_score=None):
    return dict(action=action.tolist(), y=[None if c else float(v) for v, c in zip(y, cens)],
                cens=cens.tolist(), selected_score=selected_score)


def score_record(index, family, method, budget, mu, gain, task, post, action,
                 estimate, history, selection_seconds):
    distance = float(true_stop(action, mu, gain))
    error = distance - task.target
    stats = post.stats() if post is not None else {}
    return dict(world=index, family=family, method=method, budget=budget,
                attempts=budget + 1, resets=budget, mu_true=mu, gain_true=gain,
                target=task.target, action=action.tolist(), stop=distance,
                success=bool(abs(error) <= ZONE_WIDTH / 2), placement_error=error,
                mu_estimate=estimate, mu_error=estimate - mu,
                predicted_success=task.probability_of(action, post) if post is not None else None,
                selection_seconds=selection_seconds, history=list(history), **stats)


def evaluate_world(seed, index, cfg, tuning, guard=None, legacy_mu=False):
    mu, target, shifted_gain, noise = make_world(seed, index, legacy_mu)
    task = TerminalTask(target)
    rows = []
    for family in FAMILIES:
        gain = 1.0 if family == "matched" else shifted_gain
        prior = Belief(cfg.grid_size)
        blind_action, _ = task.bayes_action(prior)
        rows.append(score_record(index, family, "blind", 0, mu, gain, task, prior,
                                 blind_action, prior.estimate("blind"), [], 0))
        # mu_oracle knows only mu, so it is NOT a ceiling under gain mismatch.
        for method, known_gain in (("mu_oracle", 1.0), ("world_oracle", gain)):
            action = task.point_action(mu, gain=known_gain)
            rows.append(score_record(index, family, method, 0, mu, gain, task, None,
                                     action, mu, [], 0))
        for method in METHODS:
            post, history, selection_seconds = Belief(cfg.grid_size), [], 0.0
            for budget in range(max(BUDGETS) + 1):
                if guard:
                    guard.check()
                if budget in BUDGETS:
                    estimate = post.estimate(method)
                    action = (task.bayes_action(post)[0] if method in ("eig", "task_voi") else
                              task.point_action(estimate, tuning[method]["offset"]))
                    rows.append(score_record(index, family, method, budget, mu, gain, task,
                                             post, action, estimate, history, selection_seconds))
                if budget == max(BUDGETS):
                    break
                start = time.monotonic()
                score = None
                if method in ("eig", "task_voi"):
                    # Shared planning random numbers across the two policies and families.
                    designs, info, value = design_scores(post, task, rng_for(seed, index, 3, budget), cfg, guard)
                    scores = info if method == "eig" else value
                    chosen = int(np.argmax(scores))
                    action, score = designs[chosen], float(scores[chosen])
                else:
                    action = baseline_probe(tuning[method]["protocol"], post.estimate(method), budget, target)
                selection_seconds += time.monotonic() - start
                y, cens = true_observation(action, mu, gain, noise[budget])
                history.append(trace_observation(action, y, cens, score))
                post.update(action, y, cens)
    return rows


def tune_world(seed, index, grid_size, guard=None):
    """Tune ONLY on unit-gain validation worlds, all three nonzero budgets equally."""
    mu, target, _, noise = make_world(seed, index)
    task, out = TerminalTask(target), []
    for method in ("lsq", "tobit"):
        for protocol in PROTOCOLS:
            post = Belief(grid_size)
            successes = np.zeros(len(AIM_OFFSETS))
            errors = np.zeros(len(AIM_OFFSETS))
            for step in range(max(BUDGETS)):
                if guard:
                    guard.check()
                action = baseline_probe(protocol, post.estimate(method), step, target)
                y, cens = true_observation(action, mu, 1.0, noise[step])
                post.update(action, y, cens)
                if step + 1 in BUDGETS:
                    for j, offset in enumerate(AIM_OFFSETS):
                        final = task.point_action(post.estimate(method), offset)
                        error = abs(float(true_stop(final, mu)) - target)
                        successes[j] += error <= ZONE_WIDTH / 2
                        errors[j] += error
            for offset, success, error in zip(AIM_OFFSETS, successes, errors):
                out.append(dict(method=method, protocol=protocol, offset=offset,
                                successes=int(success), abs_error_sum=float(error), trials=3))
    return out
