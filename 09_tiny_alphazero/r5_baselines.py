"""Validation-tuned rule grid and full-information classical logistic baseline."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from r5_core import calibration, group_masks, learner_features


def macro_nll(y, p, values, group):
    masks = group_masks(values, group, y)
    # All rows, every ply group, and hard draws. Empty groups are explicit elsewhere.
    names = ("all", "early", "middle", "late", "drawn")
    return float(np.mean([calibration(y[masks[k]], p[masks[k]])["nll"]
                          for k in names if masks[k].any()]))


def fit_baselines(x, y, vx, vy, values, group, deadline):
    lx, lv = learner_features(x), learner_features(vx)
    mean, std = lx.mean(0), np.maximum(lx.std(0), .05)
    X, V = (lx-mean)/std, (lv-mean)/std
    rows, best = [], None
    # Same input and labels as the MLP; convex optimization to convergence, not
    # an artificially short training schedule. All regularizers use full val.
    for ridge in (.0001, .001, .01, .1, 1., 10.):
        if time.monotonic() >= deadline:
            raise TimeoutError("Baseline tuning unfinished; do not evaluate")
        def fg(w):
            z = X @ w[:-1] + w[-1]
            residual = expit(z) - y
            loss = np.mean(np.logaddexp(0, z)-y*z) + ridge * np.sum(w[:-1]**2)/2
            grad = np.r_[X.T @ residual/len(y) + ridge*w[:-1], residual.mean()]
            return loss, grad
        def guard(_):
            if time.monotonic() >= deadline:
                raise TimeoutError("Classical optimizer deadline; no partial selection")
        fit = minimize(fg, np.zeros(x.shape[1]+1), jac=True, method="L-BFGS-B",
                       callback=guard, options=dict(maxiter=400, ftol=1e-10))
        p = expit(V @ fit.x[:-1]+fit.x[-1])
        score = macro_nll(vy, p, values, group)
        row = dict(kind="logistic", ridge=ridge, score=score, converged=bool(fit.success),
                   iterations=int(fit.nit))
        rows.append(row)
        if best is None or score < best[0]:
            best = (score, dict(weights=fit.x.tolist(), mean=mean.tolist(), std=std.tolist(), **row))
    # 4 AB depths x 3 uncertainty signals x 4 weights x 4 biases x 3 scales
    # = 576 rule candidates. Includes strong AB8 and tactical threats.
    best_rule = None
    for depth in range(4):
        if time.monotonic() >= deadline:
            raise TimeoutError("Rule grid unfinished; no partial selection")
        for signal in range(3):
            for weight in (0., .5, 1., 2.):
                raw = vx[:, 7+depth] + weight*vx[:, signal] + .25*vx[:, 6]
                for bias in (-3., -2., -1., 0.):
                    for scale in (.5, 1., 2.):
                        p = expit(scale*raw+bias)
                        score = macro_nll(vy, p, values, group)
                        row = dict(kind="rule", depth=depth, signal=signal, weight=weight,
                                   bias=bias, scale=scale, score=score)
                        rows.append(row)
                        if best_rule is None or score < best_rule[0]:
                            best_rule = score, row
    return dict(logistic=best[1], rule=best_rule[1], trials=rows,
                tuning="full validation: overall + early/middle/late + drawn NLL")


def scores(name, x, baselines, learned):
    if name == "learned":
        return learned(x)
    if name in ("entropy", "uncertainty", "disagreement"):
        return x[:, ("entropy", "uncertainty", "disagreement").index(name)]
    if name == "rule":
        c = baselines["rule"]
        return expit(c["scale"]*(x[:, 7+c["depth"]]+c["weight"]*x[:, c["signal"]]+.25*x[:, 6])+c["bias"])
    if name == "logistic":
        c = baselines["logistic"]
        w = np.asarray(c["weights"])
        return expit(((learner_features(x)-np.asarray(c["mean"]))/np.asarray(c["std"])) @ w[:-1]+w[-1])
    raise ValueError(name)
