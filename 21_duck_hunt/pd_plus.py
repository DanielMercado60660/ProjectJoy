"""Reviewer-added stronger flight baseline ("PD+", minimum-time-ish guidance). Not used by PPO.

Why: the builder's PD (`flight.pd_action`) has three structural handicaps:
  1. With the tuned w_next = 0 its mid-course speed ramp is d / r_slow, i.e. it decelerates
     toward ZERO at every waypoint centre (and at the last one, where nothing requires slowing).
     "Keep speed through the waypoint" was only reachable together with "bend toward the next leg".
  2. It returns acc / AMAX unbounded; env.step clips each component to [-1, 1] BEFORE the norm
     clip, so whenever the PD saturates (60% of steps) the thrust direction is rotated and gravity
     compensation can be chopped. PPO's tanh output never hits the component clip.
  3. It aims at the ring centre instead of planning to graze the ring.

PD+ (same observations, same thrust limit, same actuator, same unobserved wind):
  - aim point on the current ring, at margin*WP_R from the centre along the bisector of the
    directions centre->duck and centre->next waypoint (the shortest-path touching point for a
    ring between two points); for the last waypoint, the ring point nearest the duck
  - desired speed = min(vmax, sqrt(v_wp^2 + 2*a_brk*dist_to_aim)), v_wp interpolated between
    v_straight and v_turn by the turn angle at the waypoint; no slow-down at the last waypoint
  - same velocity loop as the PD (along/cross gains, vertical gain, gravity + drag feed-forward)
  - direction-preserving saturation: the command is scaled to |a| <= AMAX before env.step

Tuned with the SAME budget and protocol as tune_pd.py: 2000 random configs on training courses
C[:128] + 800 local refinement steps on C (256), wind seed 11; selection by flight.score.

  ../bin/gpu_run ../.venv/bin/python pd_plus.py --tune    # -> checkpoints/pd_plus_params.json, results/pd_plus_tuning.json
  ../.venv/bin/python pd_plus.py --eval                   # -> results/pd_plus_eval.json
"""
import json, sys, time
import numpy as np
import flight as F

RANGES = dict(vmax=(4, 24), a_brk=(2, 25), v_straight=(2, 20), v_turn=(0.5, 12), margin=(0.0, 0.95),
              kv=(1, 14), kperp=(1, 20), kz=(0.5, 3.0), drag_comp=(0, 1.5))
DEFAULT = dict(vmax=12.0, a_brk=8.0, v_straight=10.0, v_turn=4.0, margin=0.6, kv=4.0, kperp=4.0,
               kz=1.5, drag_comp=1.0)


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def norm_clip(acc):
    """Scale an acceleration command (m/s^2) to |a| <= AMAX keeping its direction; return a/AMAX."""
    n = np.linalg.norm(acc, axis=1, keepdims=True)
    return acc / np.maximum(n, F.AMAX)


def pd_plus_action(env, prm):
    cur, nxt, last = env.targets()
    to_p = _unit(env.p - cur)
    to_n = _unit(nxt - cur)
    bis = _unit(to_p + to_n)
    bis = np.where(last[:, None], to_p, bis)
    aim = cur + prm["margin"] * F.WP_R * bis
    e = aim - env.p
    d = np.linalg.norm(e, axis=1, keepdims=True)
    u = e / np.maximum(d, 1e-6)
    # turn angle at the waypoint: incoming direction u, outgoing direction aim->next
    out = _unit(nxt - aim)
    cos_t = np.clip((u * out).sum(1, keepdims=True), -1, 1)
    frac = (1 - cos_t) / 2                                  # 0 straight, 1 full reversal
    v_wp = prm["v_straight"] * (1 - frac) + prm["v_turn"] * frac
    v_wp = np.where(last[:, None], prm["vmax"], v_wp)       # completion only needs the ring crossed
    speed = np.minimum(prm["vmax"], np.sqrt(v_wp ** 2 + 2 * prm["a_brk"] * d))
    vdes = speed * u
    ev = vdes - env.v
    along = (ev * u).sum(1, keepdims=True) * u
    acc = prm["kv"] * along + prm["kperp"] * (ev - along)
    acc[:, 1] *= prm["kz"]
    acc[:, 1] += F.G
    acc += prm["drag_comp"] * F.DRAG * np.linalg.norm(env.v, axis=1, keepdims=True) * env.v
    return norm_clip(acc)


def pd_normclip_action(env, prm):
    """The builder's tuned PD, unchanged except for direction-preserving saturation."""
    return norm_clip(F.pd_action(env, prm) * F.AMAX)


def sample(rng):
    return {k: float(rng.uniform(*v)) for k, v in RANGES.items()}


def perturb(rng, p, scale):
    return {k: float(np.clip(p[k] + rng.normal() * scale * (hi - lo), lo, hi)) for k, (lo, hi) in RANGES.items()}


def evaluate(prm, courses, seed=11):
    r = F.run_episodes(courses, lambda e, o: pd_plus_action(e, prm), seed=seed)
    return F.score(r), F.summarize(r)


def tune(n_random=2000, n_local=800):
    rng = np.random.default_rng(0)
    C = F.train_courses(256); Ca = C[:128]
    t0 = time.time()
    best = (evaluate(DEFAULT, Ca)[0], dict(DEFAULT))
    for i in range(n_random):
        p = sample(rng)
        s, _ = evaluate(p, Ca)
        if s > best[0]:
            best = (s, p)
    print(f"random best {best[0]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    cur = best[1]; cur_s, _ = evaluate(cur, C)
    for i in range(n_local):
        p = perturb(rng, cur, 0.15 * (1 - i / max(n_local, 1)) + 0.02)
        s, _ = evaluate(p, C)
        if s > cur_s:
            cur_s, cur = s, p
    s, summ = evaluate(cur, C)
    print(f"local best {s:.4f} {summ} ({time.time()-t0:.0f}s)", flush=True)
    edges = [k for k, (lo, hi) in RANGES.items() if min(cur[k] - lo, hi - cur[k]) < 0.02 * (hi - lo)]
    out = dict(params=cur, train_score=s, train_summary=summ, n_configs=1 + n_random + n_local,
               ranges=RANGES, params_at_range_edge=edges, wall_s=time.time() - t0,
               env_steps_upper_bound=(1 + n_random) * 128 * F.MAX_STEPS + n_local * 256 * F.MAX_STEPS)
    json.dump(cur, open("checkpoints/pd_plus_params.json", "w"), indent=1)
    json.dump(out, open("results/pd_plus_tuning.json", "w"), indent=1)


def evaluate_heldout(n=500):
    import train_flight as TF
    model = TF.load("checkpoints/flight_ppo")
    prm_pd = json.load(open("checkpoints/pd_params.json"))
    prm_plus = json.load(open("checkpoints/pd_plus_params.json"))
    pols = {"ppo": TF.policy_fn(model),
            "pd_tuned": lambda e, o: F.pd_action(e, prm_pd),
            "pd_tuned_normclip": lambda e, o: pd_normclip_action(e, prm_pd),
            "pd_plus": lambda e, o: pd_plus_action(e, prm_plus)}
    C = F.heldout_courses(n)
    res = {}
    for cond, wind in [("heldout_wind2", 2.0), ("heldout_wind4_stress", 4.0), ("heldout_calm", 0.0)]:
        res[cond] = {}; T = {}
        for name, fn in pols.items():
            r = F.run_episodes(C, fn, seed=99, wind_sigma=wind)   # same courses + wind seed as eval_flight.py
            res[cond][name] = F.summarize(r); T[name] = r["t"]
        for b in ["pd_tuned", "pd_plus"]:
            both = ~np.isnan(T["ppo"]) & ~np.isnan(T[b])
            res[cond][f"paired_ppo_vs_{b}"] = dict(
                n=int(both.sum()), ppo_mean=float(T["ppo"][both].mean()), base_mean=float(T[b][both].mean()),
                ppo_faster_frac=float((T["ppo"][both] < T[b][both]).mean()),
                tie_frac=float((T["ppo"][both] == T[b][both]).mean()))
            dd = T["ppo"][both] - T[b][both]
            brng = np.random.default_rng(0)
            bs = [brng.choice(dd, len(dd)).mean() for _ in range(5000)]
            res[cond][f"paired_ppo_vs_{b}"].update(
                mean_diff_ppo_minus_base=float(dd.mean()), ci95_lo=float(np.percentile(bs, 2.5)),
                ci95_hi=float(np.percentile(bs, 97.5)), base_faster_frac=float((dd > 0).mean()))
        print(cond, json.dumps(res[cond], indent=1), flush=True)
    Ct = F.train_courses(256)
    res["train_courses_wind2"] = {k: F.summarize(F.run_episodes(Ct, fn, seed=11)) for k, fn in pols.items()}
    json.dump(res, open("results/pd_plus_eval.json", "w"), indent=1)
    return res


def style_and_figure():
    """Pass style (as analyze_flight.py) + a 4-course held-out plot of PPO vs PD+."""
    import analyze_flight as AF, train_flight as TF
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = F.heldout_courses(500)
    prm = json.load(open("checkpoints/pd_plus_params.json"))
    fn = lambda e, o: pd_plus_action(e, prm)
    out = dict(pd_plus=AF.run(C, fn))
    print(json.dumps(out, indent=1)); json.dump(out, open("results/pd_plus_style.json", "w"), indent=1)
    ppo = TF.policy_fn(TF.load("checkpoints/flight_ppo"))
    rp = F.run_episodes(C[:4], ppo, seed=99, record=True)
    rb = F.run_episodes(C[:4], fn, seed=99, record=True)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    for i, ax in enumerate(axs):
        W = C[i]
        ax.plot(rb["traj"][i, :, 0], rb["traj"][i, :, 1], color="#228833", lw=1.5,
                label=f"PD+ ({rb['t'][i]:.2f}s)" if rb["complete"][i] else "PD+ (fail)")
        ax.plot(rp["traj"][i, :, 0], rp["traj"][i, :, 1], color="#cc6677", lw=1.5,
                label=f"PPO ({rp['t'][i]:.2f}s)" if rp["complete"][i] else "PPO (fail)")
        for k in range(1, F.N_WP + 1):
            ax.add_patch(plt.Circle(W[k], F.WP_R, fill=False, color="k"))
            ax.text(W[k, 0] + 1, W[k, 1] + 0.6, str(k), fontsize=8)
        ax.plot(*W[0], "ks", ms=5)
        ax.set_aspect("equal"); ax.set_xlim(-5, 35); ax.set_ylim(0, 18); ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"held-out course {i}", fontsize=9)
    plt.tight_layout(); plt.savefig("media/flight_heldout_pdplus.png", dpi=110)


if __name__ == "__main__":
    if "--style" in sys.argv:
        style_and_figure()
    if "--quick" in sys.argv:
        C = F.train_courses(64)
        for prm in [DEFAULT]:
            print(F.summarize(F.run_episodes(C, lambda e, o: pd_plus_action(e, prm), seed=11)))
        prm_pd = json.load(open("checkpoints/pd_params.json"))
        print("pd", F.summarize(F.run_episodes(C, lambda e, o: F.pd_action(e, prm_pd), seed=11)))
        print("pd normclip", F.summarize(F.run_episodes(C, lambda e, o: pd_normclip_action(e, prm_pd), seed=11)))
    if "--tune" in sys.argv:
        tune()
    if "--eval" in sys.argv:
        evaluate_heldout()
