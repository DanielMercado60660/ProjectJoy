"""R6 flight: give the PD+ baseline its best fair shot before the final comparison.

  wide  : fresh tune, SAME protocol and budget as pd_plus.py --tune (2000 random on training
          courses C[:128] + 800 local on C, 2801 configs), but with widened ranges
          (the reviewer's v_turn sat at the bottom of its range).
  cont  : 2801 more local-refinement configs on C, starting from the better of (reviewer PD+,
          wide), i.e. PD+ with 2x the budget; this is the matched baseline for anything that
          builds on the tuned PD+ (a residual policy) and so already spent PD+'s budget.

Selection on training courses only (flight.score, wind seed 11), exactly like pd_plus.py.
  ../bin/gpu_run ../.venv/bin/python pd_plus_r6.py wide
  ../bin/gpu_run ../.venv/bin/python pd_plus_r6.py cont
"""
import json, sys, time
import numpy as np
import flight as F
import pd_plus as P

WIDE = dict(vmax=(4, 30), a_brk=(2, 40), v_straight=(0, 25), v_turn=(0.0, 12), margin=(0.0, 0.99),
            kv=(1, 25), kperp=(1, 40), kz=(0.5, 4.0), drag_comp=(0, 2.0))


def sample(rng):
    return {k: float(rng.uniform(*v)) for k, v in WIDE.items()}


def perturb(rng, p, scale):
    return {k: float(np.clip(p[k] + rng.normal() * scale * (hi - lo), lo, hi)) for k, (lo, hi) in WIDE.items()}


def local(rng, cur, C, n_local, log):
    cur_s, _ = P.evaluate(cur, C)
    for i in range(n_local):
        p = perturb(rng, cur, 0.15 * (1 - i / max(n_local, 1)) + 0.02)
        s, _ = P.evaluate(p, C)
        if s > cur_s:
            cur_s, cur = s, p
            log.append((i, s))
    return cur, cur_s


def finish(name, cur, n_configs, steps_bound, t0, extra):
    C = F.train_courses(256)
    s, summ = P.evaluate(cur, C)
    edges = [k for k, (lo, hi) in WIDE.items() if min(cur[k] - lo, hi - cur[k]) < 0.02 * (hi - lo)]
    out = dict(params=cur, train_score=s, train_summary=summ, n_configs=n_configs, ranges=WIDE,
               params_at_range_edge=edges, env_steps_upper_bound=steps_bound, wall_s=time.time() - t0, **extra)
    print(name, json.dumps(out, indent=1), flush=True)
    json.dump(cur, open(f"checkpoints/pd_plus_r6_{name}_params.json", "w"), indent=1)
    json.dump(out, open(f"results/r6_pd_plus_{name}_tuning.json", "w"), indent=1)


def wide(n_random=2000, n_local=800):
    rng = np.random.default_rng(100)
    C = F.train_courses(256); Ca = C[:128]
    t0 = time.time()
    best = (P.evaluate(P.DEFAULT, Ca)[0], dict(P.DEFAULT))
    for i in range(n_random):
        p = sample(rng)
        s, _ = P.evaluate(p, Ca)
        if s > best[0]:
            best = (s, p)
    log = []
    cur, _ = local(rng, best[1], C, n_local, log)
    finish("wide", cur, 1 + n_random + n_local,
           (1 + n_random) * 128 * F.MAX_STEPS + n_local * 256 * F.MAX_STEPS, t0, dict(improvements=log))


def cont(n_local=2801):
    rng = np.random.default_rng(101)
    C = F.train_courses(256)
    t0 = time.time()
    cands = {"reviewer": json.load(open("checkpoints/pd_plus_params.json")),
             "wide": json.load(open("checkpoints/pd_plus_r6_wide_params.json"))}
    sc = {k: P.evaluate(v, C)[0] for k, v in cands.items()}
    start = max(sc, key=sc.get)
    log = []
    cur, _ = local(rng, dict(cands[start]), C, n_local, log)
    finish("cont", cur, n_local, n_local * 256 * F.MAX_STEPS, t0,
           dict(start_from=start, start_scores=sc, improvements=log,
                note="on top of the start point's own 2801-config budget, i.e. 2x PD+ budget in total"))


if __name__ == "__main__":
    {"wide": wide, "cont": cont}[sys.argv[1]]()
