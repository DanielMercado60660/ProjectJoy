"""Tune the PD/pursuit baseline on TRAINING courses only (random search + local refinement).
Writes checkpoints/pd_params.json and results/pd_tuning.json."""
import json, sys, time
import numpy as np
import flight as F

RANGES = dict(vmax=(3, 22), kv=(1, 14), r_slow=(0.3, 8), w_next=(0, 2.5), r_blend=(0.3, 6),
              drag_comp=(0, 1.5), kz=(0.5, 3.0), kperp=(1, 20))


def sample(rng):
    return {k: float(rng.uniform(*v)) for k, v in RANGES.items()}


def perturb(rng, p, scale):
    q = {}
    for k, (lo, hi) in RANGES.items():
        q[k] = float(np.clip(p[k] + rng.normal() * scale * (hi - lo), lo, hi))
    return q


def evaluate(prm, courses, seed=11):
    r = F.run_episodes(courses, lambda e, o: F.pd_action(e, prm), seed=seed)
    return F.score(r), F.summarize(r)


def main(n_random=2000, n_local=800, quick=False):
    rng = np.random.default_rng(0)
    C = F.train_courses(256)
    Ca = C[:128]
    t0 = time.time()
    hist = []
    best = (evaluate(F.PD_DEFAULT, Ca)[0], dict(F.PD_DEFAULT, kperp=F.PD_DEFAULT["kv"]))
    hist.append(best[0])
    for i in range(n_random):
        p = sample(rng)
        s, _ = evaluate(p, Ca)
        hist.append(s)
        if s > best[0]:
            best = (s, p)
    print(f"random search best {best[0]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    # local refinement on all 256 training courses
    cur_s, _ = evaluate(best[1], C)
    cur = best[1]
    for i in range(n_local):
        scale = 0.15 * (1 - i / max(n_local, 1)) + 0.02
        p = perturb(rng, cur, scale)
        s, _ = evaluate(p, C)
        if s > cur_s:
            cur_s, cur = s, p
    s, summ = evaluate(cur, C)
    print(f"local refinement best {s:.4f}: {summ} ({time.time()-t0:.0f}s)", flush=True)
    out = dict(params=cur, train_score=s, train_summary=summ, n_configs=1 + n_random + n_local,
               ranges=RANGES, wall_s=time.time() - t0,
               env_steps_upper_bound=(1 + n_random) * 128 * F.MAX_STEPS + n_local * 256 * F.MAX_STEPS)
    if not quick:
        json.dump(cur, open("checkpoints/pd_params.json", "w"), indent=1)
        json.dump(out, open("results/pd_tuning.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    if "--quick" in sys.argv:
        print(main(5, 5, quick=True))
    else:
        main()
