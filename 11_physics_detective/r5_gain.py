"""Turn 2's single model fix: marginalize an unknown, constant actuator gain.

Probe histories are supplied by frozen R5 policies. This module changes only
inference and the scored action; it does not secretly obtain extra observations.
Both a joint Bayesian controller and a joint Tobit MLE use the same physics/data.
"""
from dataclasses import dataclass
import time

import numpy as np

from physics import G, MU_HI, MU_LO, MU_RATIO, loglik_frames, positions
from r5_core import (BUDGETS, FINAL_ACTIONS, Belief, TerminalTask, score_record)

OWNERS = ("eig", "lsq")
GAIN_HALF_WIDTHS = (.15, .25)
NEW_METHODS = ("gain_bayes", "gain_mle", "same_probes_unit_bayes")


@dataclass(frozen=True)
class GainConfig:
    mu_size: int = 2401
    gain_size: int = 641

    def validate(self):
        if self.mu_size < 101 or self.gain_size < 11:
            raise ValueError("Gain grid needs >=101 mu cells and >=11 gain cells")


def marginal_quantile(grid, weights, q, lo, hi):
    return np.interp(q, np.r_[0, np.cumsum(weights) - weights / 2, 1], np.r_[lo, grid, hi])


class GainBelief:
    def __init__(self, cfg, half_width):
        cfg.validate()
        if not 0 < half_width < 1:
            raise ValueError("Gain prior half-width must be between zero and one")
        self.mu = MU_LO + (np.arange(cfg.mu_size) + .5) * (MU_HI - MU_LO) / cfg.mu_size
        self.gain_lo, self.gain_hi = 1 - half_width, 1 + half_width
        self.gain = self.gain_lo + (np.arange(cfg.gain_size) + .5) * (2 * half_width) / cfg.gain_size
        self.logp = np.zeros((cfg.gain_size, cfg.mu_size))

    @property
    def weights(self):
        w = np.exp(self.logp - self.logp.max())
        return w / w.sum()

    def update(self, action, y, cens):
        prediction = positions(*action, self.mu[None, :], MU_RATIO * self.mu[None, :], k=self.gain[:, None])
        assert prediction.shape == self.logp.shape + (len(y),)
        self.logp += loglik_frames(y, cens, prediction)
        self.logp -= self.logp.max()

    def mle(self):
        indices = np.argwhere(np.isclose(self.logp, self.logp.max(), rtol=0, atol=1e-10))
        # Empty/flat likelihood: deterministic centre, never an endpoint accident.
        cost = ((self.mu[indices[:, 1]] - (MU_LO + MU_HI) / 2) / (MU_HI - MU_LO)) ** 2
        cost += ((self.gain[indices[:, 0]] - 1) / (self.gain_hi - self.gain_lo)) ** 2
        g, m = indices[int(np.argmin(cost))]
        return float(self.mu[m]), float(self.gain[g])

    def stats(self):
        w = self.weights
        wm, wg = w.sum(0), w.sum(1)
        mm, mg = float(wm @ self.mu), float(wg @ self.gain)
        sm = float(np.sqrt(wm @ (self.mu - mm) ** 2))
        sg = float(np.sqrt(wg @ (self.gain - mg) ** 2))
        qm = marginal_quantile(self.mu, wm, [.05, .25, .75, .95], MU_LO, MU_HI)
        qg = marginal_quantile(self.gain, wg, [.05, .25, .75, .95], self.gain_lo, self.gain_hi)
        dm, dg = self.mu[1] - self.mu[0], self.gain[1] - self.gain[0]
        # Marginals can hide a thin diagonal ridge. Also measure conditional grid
        # resolution, weighted by the mass of the conditioning coordinate.
        cond_m = w / np.maximum(wg[:, None], 1e-300)
        cond_g = w / np.maximum(wm[None, :], 1e-300)
        cond_m_mean = cond_m @ self.mu
        cond_g_mean = self.gain @ cond_g
        cond_m_sd = np.sqrt(np.maximum(0, (cond_m * (self.mu[None, :] - cond_m_mean[:, None]) ** 2).sum(1)))
        cond_g_sd = np.sqrt(np.maximum(0, (cond_g * (self.gain[:, None] - cond_g_mean[None, :]) ** 2).sum(0)))
        return dict(posterior_mean=mm, posterior_sd=sm, ci50=[float(qm[1]), float(qm[2])],
                    ci90=[float(qm[0]), float(qm[3])], grid_step_over_sd=float(dm / max(sm, 1e-300)),
                    gain_posterior_mean=mg, gain_posterior_sd=sg,
                    gain_ci50=[float(qg[1]), float(qg[2])], gain_ci90=[float(qg[0]), float(qg[3])],
                    gain_grid_step_over_sd=float(dg / max(sg, 1e-300)),
                    mu_conditional_underresolved_mass=float(wg @ (cond_m_sd < 2 * dm)),
                    gain_conditional_underresolved_mass=float(wm @ (cond_g_sd < 2 * dg)),
                    joint_max_cell_mass=float(w.max()))


class GainTask:
    """Integrate the stopping-success interval over the joint (mu,gain) grid.

    This uses a CDF along mu for every gain cell. It never constructs an
    actions x mu x gain tensor. The marginal gain mass is preserved at endpoints.
    """
    def __init__(self, target, post, actions=FINAL_ACTIONS):
        self.target, self.actions = target, actions
        force = post.gain[:, None] * actions[None, :, 0]
        duration = actions[None, :, 1]
        lower = force ** 2 * duration ** 2 / (G * (2 * (target + .025) + force * duration ** 2))
        upper = np.minimum(force ** 2 * duration ** 2 / (G * (2 * (target - .025) + force * duration ** 2)),
                           force / (MU_RATIO * G))
        nodes = np.r_[MU_LO, post.mu, MU_HI]
        self.interpolation = []
        for bound in (lower, upper):
            hi = np.clip(np.searchsorted(nodes, bound, side="right"), 1, len(nodes) - 1)
            frac = np.clip((bound - nodes[hi - 1]) / (nodes[hi] - nodes[hi - 1]), 0, 1)
            self.interpolation.append((hi, frac))

    def probabilities(self, post):
        w = post.weights
        cdf = np.concatenate((np.zeros((len(post.gain), 1)), np.cumsum(w, axis=1) - w / 2,
                              w.sum(1, keepdims=True)), axis=1)
        endpoints = []
        for hi, frac in self.interpolation:
            endpoints.append(np.take_along_axis(cdf, hi - 1, axis=1) * (1 - frac)
                             + np.take_along_axis(cdf, hi, axis=1) * frac)
        return np.clip(np.maximum(0, endpoints[1] - endpoints[0]).sum(0), 0, 1)


def history_arrays(history):
    return (np.asarray(history["action"]),
            np.array([np.nan if v is None else v for v in history["y"]]),
            np.asarray(history["cens"]))


def corrected_history(rows, owner, half_width, cfg, methods=("gain_bayes", "gain_mle"), guard=None):
    """Score both corrected estimators on exactly one existing probe sequence."""
    chosen = {r["budget"]: r for r in rows if r["method"] == owner}
    assert set(chosen) == set(BUDGETS)
    final_row = chosen[max(BUDGETS)]
    post = GainBelief(cfg, half_width)
    task = GainTask(final_row["target"], post)
    point_task = TerminalTask(final_row["target"])
    out, compute = [], 0.0
    for budget in range(max(BUDGETS) + 1):
        if guard:
            guard.check()
        if budget:
            start = time.monotonic()
            post.update(*history_arrays(final_row["history"][budget - 1]))
            compute += time.monotonic() - start
        if budget not in BUDGETS:
            continue
        start = time.monotonic()
        stats = post.stats()
        probabilities = task.probabilities(post)
        mu_mle, gain_mle = post.mle()
        compute += time.monotonic() - start
        for method in methods:
            if method == "gain_bayes":
                i = int(np.argmax(probabilities))
                action = task.actions[i]
                estimate, gain_estimate = stats["posterior_mean"], stats["gain_posterior_mean"]
            else:
                action = point_task.point_action(mu_mle, gain=gain_mle)
                i = int(np.flatnonzero(np.all(task.actions == action, axis=1))[0])
                estimate, gain_estimate = mu_mle, gain_mle
            source = chosen[budget]
            record = score_record(source["world"], source["family"], method, budget,
                                  source["mu_true"], source["gain_true"], point_task, None,
                                  action, estimate, source["history"], source["selection_seconds"])
            record.update(stats, predicted_success=float(probabilities[i]), gain_estimate=gain_estimate,
                          gain_error=gain_estimate - source["gain_true"], probe_owner=owner,
                          gain_prior_half_width=half_width, correction_seconds=compute)
            out.append(record)
    return out


def unit_control_history(rows, owner, grid_size, guard=None):
    chosen = {r["budget"]: r for r in rows if r["method"] == owner}
    final = chosen[max(BUDGETS)]
    post, task, out = Belief(grid_size), TerminalTask(final["target"]), []
    for budget in range(max(BUDGETS) + 1):
        if guard:
            guard.check()
        if budget:
            post.update(*history_arrays(final["history"][budget - 1]))
        if budget in BUDGETS:
            source = chosen[budget]
            record = score_record(source["world"], source["family"], "same_probes_unit_bayes", budget,
                                  source["mu_true"], source["gain_true"], task, post, task.bayes_action(post)[0],
                                  post.estimate("eig"), source["history"], source["selection_seconds"])
            record["probe_owner"] = owner
            out.append(record)
    return out


def followup_world(original_rows, selected, cfg, unit_grid_size, guard=None):
    out = list(original_rows)
    for family in ("matched", "gain_shift"):
        rows = [r for r in original_rows if r["family"] == family]
        # If both selected the same configuration, compute the posterior once.
        configs = {}
        for method in ("gain_bayes", "gain_mle"):
            setting = selected[method]
            configs.setdefault((setting["owner"], setting["half_width"]), []).append(method)
        for (owner, width), methods in configs.items():
            out.extend(corrected_history(rows, owner, width, cfg, methods, guard))
        out.extend(unit_control_history(rows, selected["gain_bayes"]["owner"], unit_grid_size, guard))
    return out
