"""Reviewer checks (cross-review, not part of the builder's pipeline).

    ../.venv/bin/python review_checks.py      # ~1 min, writes results/review_checks.json

1. Stronger scripted attacker: aim at the open side of the goal mouth (away from the defender's y)
   instead of the goal centre, where the goalie stands. The aim fraction is tuned on TUNING starts
   (seed 77, training distribution, never used by eval.py), then the pick is evaluated on the held-out
   start sets of eval.py (seeds 1001-1003), 500 matches per cell, same metrics.
2. CPU-fp32 vs GPU log-prob mismatch: on 256 mid-match states per checkpoint, compare the policy mean
   and the log-prob of sampled actions computed on mx.cpu vs mx.gpu, and the resulting approx-KL
   (k3 estimator) between "old" (CPU rollout) and "new" (GPU, same weights) log-probs.
"""
import json
import numpy as np
import mlx.core as mx
from env import Arena, W, GOAL_HALF
import scripted
from eval import run_episodes, summarize


def make_aim_attacker(aim):
    def pol(a):
        oy = a.pos[:, 1, 1]
        side = -np.sign(oy + 1e-6)                    # open side = opposite the defender's y
        goal = np.stack([np.full(a.n, W), side * aim * GOAL_HALF], 1)
        return scripted._behind_ball(a.pos[:, 0], a.th[:, 0], a.ball, goal)
    return pol


def tune():
    out = {}
    for aim in [0.0, 0.3, 0.5, 0.65, 0.8, 0.9]:
        res, _ = run_episodes(make_aim_attacker(aim), "goalie", "heldout", 400, seed=77)
        out[aim] = summarize(res)["goal"]
        print(f"tune aim {aim:.2f}: goal {out[aim]:.3f}", flush=True)
    return out


def tf32_probe(ckpt, n=256, warm=60):
    from ppo import load_ckpt, make_policy, logprob
    model, norm, _ = load_ckpt(ckpt)
    pol = make_policy(model, norm)
    a = Arena(n, seed=55, opponent="goalie")
    for _ in range(warm):                            # mid-match states
        a.step(np.clip(pol(a), -1, 1), scripted.goalie(a))
    x = mx.array(norm(a.obs()))
    with mx.stream(mx.cpu):
        mu_c, _ = model(x); mx.eval(mu_c)
    with mx.stream(mx.gpu):
        mu_g, _ = model(x); mx.eval(mu_g)
    rng = np.random.default_rng(0)
    std = np.exp(np.array(model.log_std))
    act = mx.array((np.array(mu_c) + rng.standard_normal((n, 2)) * std).astype(np.float32))
    with mx.stream(mx.cpu):
        lp_c = logprob(mu_c, model.log_std, act); mx.eval(lp_c)
    with mx.stream(mx.gpu):
        lp_g = logprob(mu_g, model.log_std, act); mx.eval(lp_g)
    d = np.array(lp_g) - np.array(lp_c)
    r = np.exp(d)
    return dict(mean_abs_mu=float(np.abs(np.array(mu_c)).mean()), std=std.tolist(),
                max_abs_dmu=float(np.abs(np.array(mu_c) - np.array(mu_g)).max()),
                mean_abs_dlogp=float(np.abs(d).mean()), approx_kl_k3=float(np.mean((r - 1) - d)),
                frac_ratio_outside_clip=float(np.mean(np.abs(r - 1) > 0.2)))


def main():
    out = {"tune_goal_rate_vs_goalie_seed77": tune()}
    best = max(out["tune_goal_rate_vs_goalie_seed77"], key=out["tune_goal_rate_vs_goalie_seed77"].get)
    out["picked_aim"] = best
    pol = make_aim_attacker(best)
    for opp in ["goalie", "keeper", "chaser"]:
        for s in ["heldout", "corner", "kickoff"]:
            m = summarize(run_episodes(pol, opp, s, 500)[0])
            out[f"aim_attacker|{opp}|{s}"] = m
            print(f"aim {best} {opp:7s} {s:8s} goal {m['goal']:.3f} conc {m['conceded']:.3f} own {m['own_goal']:.3f}",
                  flush=True)
    for ck in ["checkpoints/run2_snapshot_10min.npz", "checkpoints/ppo.npz", "checkpoints/run3_ent0_anneal.npz",
               "checkpoints/run4_tanh_relp.npz"]:
        out[f"tf32|{ck}"] = tf32_probe(ck)
        print(ck, out[f"tf32|{ck}"], flush=True)
    json.dump(out, open("results/review_checks.json", "w"), indent=1)
    print("wrote results/review_checks.json")


if __name__ == "__main__":
    main()
