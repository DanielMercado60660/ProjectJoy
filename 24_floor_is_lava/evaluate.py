"""Held-out evaluation: tuned CPG baseline vs residual PPO policy on the SAME held-out layouts.

Held-out layouts use seeds HELDOUT_SEED_BASE + i (training only ever samples seeds < HELDOUT_SEED_BASE).
Initial-state noise uses fixed seeds too, identical for both methods.

Metrics per method: completion, lava touches, flips/collapses, stalls/timeouts, progress fraction,
energy per metre (sum |tau * qdot| dt / distance, completed episodes), cost of transport, and a push
test (lateral force on the trunk for 0.1 s when it passes the start of stone 3).

Usage: ../bin/gpu_run ../.venv/bin/python evaluate.py --policy checkpoints/policy.npz --n 300 --out results/eval_main.json
"""
import argparse, json, os, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_task, _pol, _cfg, _flat = None, None, None, False


def _init(cpg, policy_path, planner=False, gap_max=None, flat=False, mask=False):
    global _task, _pol, _cfg, _flat
    _flat = flat
    import task as T, ppo, lava_env as E
    _cfg = dict(E.LAYOUT_CFG)
    if gap_max is not None:
        _cfg["gap"] = (_cfg["gap"][0], gap_max)
    _task = T.LavaTask(cpg, planner=planner)
    _pol = ppo.make_policy_fn(policy_path, mask)[0] if policy_path else None


def _run(args):
    i, use_policy, push_mag = args
    import task as T, lava_env as E
    seed = E.HELDOUT_SEED_BASE + i
    L = E.make_layout(seed, _cfg, flat=_flat)
    push = None
    if push_mag > 0:
        sign = 1.0 if i % 2 == 0 else -1.0
        push = (L[3, 0], np.array([0.0, sign * push_mag, 0.0]), 0.1)
    r = T.run_episode(_task, L, policy=_pol if use_policy else None,
                      rng=np.random.default_rng(seed + 50_000), init_noise=0.05, push=push)
    gaps = L[1:, 0] - L[:-1, 1]
    r.update(i=i, max_gap=float(gaps.max()), mean_gap=float(gaps.mean()), mass=_task.env.mass)
    # "fell after push": failure within 2 s of the push onset
    if push is not None and r["pushed"]:
        r["fail_after_push_2s"] = bool((r["lava"] or r["flip"] or r["collapsed"]) and r["t"] - r["push_t"] <= 2.0)
    return r


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    bs = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summarize(rs, push=False):
    n = len(rs)
    s = {"n": n}
    for key in ["success", "lava", "flip", "collapsed", "stall", "timeout"]:
        k = sum(bool(r[key]) for r in rs)
        s[key] = wilson(k, n)
    s["falls"] = wilson(sum(bool(r["lava"] or r["flip"] or r["collapsed"]) for r in rs), n)
    s["frac"] = boot_ci([r["frac"] for r in rs])
    ok = [r for r in rs if r["success"]]
    s["energy_per_m_completed"] = boot_ci([r["energy_total"] / r["progress"] for r in ok])
    s["cost_of_transport_completed"] = boot_ci([r["energy_total"] / (r["mass"] * 9.81 * r["progress"]) for r in ok])
    s["time_completed_s"] = boot_ci([r["t"] for r in ok])
    s["speed_completed_mps"] = boot_ci([r["progress"] / r["t"] for r in ok])
    s["airborne_frac"] = boot_ci([r["airborne_frac"] for r in rs])       # control steps with no foot contact
    s["max_trunk_height_above_stone"] = boot_ci([r["max_trunk_height"] for r in rs])
    s["stance_foot_slip_mps"] = boot_ci([r["stance_slip_mps"] for r in rs])  # xy speed of feet in contact
    if push:
        pr = [r for r in rs if r["pushed"]]
        s["n_reached_push"] = len(pr)
        s["success_given_pushed"] = wilson(sum(r["success"] for r in pr), len(pr))
        s["fail_within_2s_of_push"] = wilson(sum(r.get("fail_after_push_2s", False) for r in pr), len(pr))
    # completion by largest gap in the layout
    bins = [0.05, 0.10, 0.15, 0.20]
    s["success_by_max_gap"] = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = [r for r in rs if lo <= r["max_gap"] < hi]
        s["success_by_max_gap"][f"{lo:.2f}-{hi:.2f}"] = wilson(sum(r["success"] for r in sub), len(sub)) + (len(sub),)
    return s


def run_method(cpg, policy_path, n, push_mag, workers, planner=False, gap_max=None, flat=False, mask=False):
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cpg, policy_path, planner, gap_max, flat, mask)) as pool:
        return pool.map(_run, [(i, policy_path is not None, push_mag) for i in range(n)], chunksize=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="checkpoints/policy.npz")
    ap.add_argument("--cpg", default="checkpoints/cpg_params.json")
    ap.add_argument("--planner", default="checkpoints/planner_params.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--push", type=float, default=40.0, help="lateral push force in N for 0.1 s")
    ap.add_argument("--gap_max", type=float, default=None, help="override max gap (default: training max 0.20)")
    ap.add_argument("--mask_stones", action="store_true", help="zero the policy's stone observations at test time")
    ap.add_argument("--baselines", type=int, default=1, help="0: evaluate only the policy")
    ap.add_argument("--flat", action="store_true", help="flat walkway (no gaps): isolates push recovery from footholds")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/eval_main.json")
    a = ap.parse_args()
    os.chdir(HERE)
    import task as T
    import ppo
    cpg_base = json.load(open(a.cpg))["params"] if os.path.exists(a.cpg) else dict(T.DEFAULT_CPG)
    out = {"n_layouts": a.n, "gap_max": a.gap_max or 0.20, "flat": a.flat, "mask_stones": a.mask_stones, "policy": a.policy, "heldout_seeds": "10_000_000 + i, i < n", "push_N": a.push, "push_s": 0.1}
    t0 = time.time()
    methods = [("cpg_baseline", cpg_base, None, False)] if a.baselines else []
    if a.baselines and a.planner and os.path.exists(a.planner):
        methods.append(("cpg_planner_baseline", json.load(open(a.planner))["params"], None, True))
    if a.policy and os.path.exists(a.policy):
        cfg = ppo.load_policy(a.policy)[0]
        methods.append(("residual_ppo", cfg["cpg_params"], a.policy, False))
        if cfg["cpg_params"] != cpg_base:
            out["note"] = "policy was trained on a different CPG than --cpg; baseline uses --cpg"
    raw = {}
    for name, cpg, pol, plan in methods:
        mk = a.mask_stones and pol is not None
        rs = run_method(cpg, pol, a.n, 0.0, a.workers, plan, a.gap_max, a.flat, mk)
        rp = run_method(cpg, pol, a.n, a.push, a.workers, plan, a.gap_max, a.flat, mk) if a.push > 0 else []
        raw[name] = dict(clean=rs, push=rp)
        out[name] = dict(clean=summarize(rs), push=summarize(rp, push=True) if rp else None)
        print(f"{name}: success {out[name]['clean']['success']}  push-success {out[name]['push']['success'] if rp else None} "
              f"({time.time()-t0:.0f}s)", flush=True)
    if "residual_ppo" in raw:
        b_ = np.array([r["success"] for r in raw["residual_ppo"]["clean"]], float)
        for base in [k for k in raw if k != "residual_ppo"]:
            a_ = np.array([r["success"] for r in raw[base]["clean"]], float)
            out[f"paired_success_diff_ppo_minus_{base}"] = boot_ci(b_ - a_)
            out[f"discordant_vs_{base}"] = dict(ppo_only=int(((b_ == 1) & (a_ == 0)).sum()),
                                                base_only=int(((a_ == 1) & (b_ == 0)).sum()))
            # energy / speed on layouts BOTH completed (removes the easy-layout selection bias)
            both = [(p, q) for p, q in zip(raw["residual_ppo"]["clean"], raw[base]["clean"]) if p["success"] and q["success"]]
            em = lambda r: r["energy_total"] / r["progress"]
            out[f"both_completed_vs_{base}"] = dict(
                n=len(both),
                ppo_energy_per_m=boot_ci([em(p) for p, q in both]), base_energy_per_m=boot_ci([em(q) for p, q in both]),
                diff_energy_per_m_ppo_minus_base=boot_ci([em(p) - em(q) for p, q in both]),
                ppo_speed=boot_ci([p["progress"] / p["t"] for p, q in both]),
                base_speed=boot_ci([q["progress"] / q["t"] for p, q in both]))
    out["wall_s"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    json.dump(raw, open(a.out.replace(".json", "_episodes.json"), "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(json.dumps({k: v for k, v in out.items() if k not in raw}, indent=1, default=float))


if __name__ == "__main__":
    main()
