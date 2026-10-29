"""Round-3 follow-up: does evolution still beat a well-trimmed hand design once span has a structural cost?

Physics: `sim_struct.py` (wing root bending-strength limit on the original 3 mm foam sheet; the wing
folds and the flight ends when |L_wing| * b/(3 pi) * (1 - m_wing/m) > sigma * c * t^2 / 6).
The original model `sim.py` is untouched and still selectable.

Levels of allowable stress sigma, fixed BEFORE any struct search was run:
  0.25 MPa  HEADLINE (0.5 MPa foam flexural strength / safety factor 2)
  0.50 MPa  sensitivity (no safety factor; a pre-run probe showed it never binds on the training launches
            for the old CMA-ES winner, max load 0.59 of capacity)
  0.125 MPa sensitivity (weak / damp sheet)

Per level, all on the 6 training launches (sim.training_conditions(seed=0)), RK4 dt = 5 ms:
  cma      CMA-ES (search.cma_es), SEEDS x BUDGET evals
  random   random search, same budget                       <- matched-compute baseline
  hand     textbook glider (tail 25 % of wing area, 18 cm arm, as the shipped baseline) whose span, chord,
           ballast, ballast position and both incidences are picked from a grid scored on the TRAINING
           mean (5 spans x 3 chords x 6 ballasts x 3 positions x 3 x 5 incidences = 4050 designs, i.e.
           about one CMA seed's budget)                     <- hand baseline, re-trimmed under new physics
Then: the same 64 held-out launches (sim.heldout_conditions(seed=12345)), dt-halving and still-air energy
checks, stability eigenvalues, wing-break counts, and the backwards-glide loophole check.

Usage:  NUMBA_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python struct_followup.py [--budget 4000 --seeds 5]
Writes checkpoints/struct/search_s<sigma>.npz, checkpoints/struct/designs.json, results/struct/eval.json,
results/struct/stdout.txt (tee yourself).
"""
import argparse
import itertools
import json
import os
import time

import numpy as np

import sim
import sim_struct as ss
from search import DT_SOUND, TMAX, cma_es, random_search
from evaluate import REL_TOL, ABS_TOL, E_TOL, BASELINE, eig_summary

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = [0.25e6, 0.5e6, 0.125e6]          # headline first
HAND_TAIL = (0.25, 0.18)


class StructFitness:
    def __init__(self, sigma):
        self.conds, self.gusts = sim.training_conditions(seed=0)
        self.sigma = sigma
        self.n_evals = 0

    def __call__(self, genes):
        genes = np.atleast_2d(genes)
        clipped = np.clip(genes, 0.0, 1.0)
        stats = ss.evaluate_batch(clipped, self.conds, self.gusts, DT_SOUND, TMAX, 0, self.sigma)
        self.n_evals += len(genes)
        fit = stats[:, :, 0].mean(1) - 100.0 * ((genes - clipped) ** 2).sum(1)
        return fit, stats


def hand_grid():
    rows = []
    for span, chord, mb, xb, iw, it in itertools.product(
            [0.30, 0.35, 0.40, 0.45, 0.50], [0.04, 0.06, 0.08], np.linspace(0, 0.02, 6),
            [0.0, 0.03, 0.06], [0.0, 2.0, 4.0], np.linspace(-6, 2, 5)):
        rows.append([span, chord, HAND_TAIL[0], HAND_TAIL[1], mb, xb, iw, it])
    return np.array(rows)


def sanity(genes, conds, gusts, sigma, dt=DT_SOUND):
    g = np.atleast_2d(genes)
    s1 = ss.evaluate_batch(g, conds, gusts, dt, TMAX, 0, sigma)
    s2 = ss.evaluate_batch(g, conds, gusts, dt / 2, TMAX, 0, sigma)
    r1, r2 = s1[:, :, 0], s2[:, :, 0]
    dev = np.abs(r1 - r2)
    ok = dev <= np.maximum(REL_TOL * np.abs(r2), ABS_TOL)
    c0 = conds.copy(); c0[:, 4] = 0.0
    s0 = ss.evaluate_batch(g, c0, np.zeros_like(gusts), dt, TMAX, 0, sigma)
    return dict(dt_pass=ok.all(1), n_dt_fail=(~ok).sum(1),
                max_rel_dev=(dev / np.maximum(np.abs(r2), 1e-9)).max(1), max_abs_dev=dev.max(1),
                break_mismatch=((s1[:, :, 2] == -2) != (s2[:, :, 2] == -2)).sum(1),
                energy_pass=(s0[:, :, 3] <= E_TOL).all(1), max_step_dE=s0[:, :, 3].max(1))


def report(name, genes, sigma, tc, tg, hc, hg, cma_ho=None):
    g = np.atleast_2d(genes)
    nc, ng = sim.still_air()
    nom = ss.evaluate_batch(g, nc, ng, DT_SOUND, TMAX, 0, sigma)[0, 0]
    tr = ss.evaluate_batch(g, tc, tg, DT_SOUND, TMAX, 0, sigma)[0]
    ho = ss.evaluate_batch(g, hc, hg, DT_SOUND, TMAX, 0, sigma)[0]
    ho_orig = sim.evaluate_batch(g, hc, hg, DT_SOUND, TMAX, 0)[0]
    st, sh = sanity(g, tc, tg, sigma), sanity(g, hc, hg, sigma)
    r = ho[:, 0]
    out = dict(
        name=name, genes=[float(x) for x in genes], design=ss.describe(genes, sigma),
        still_air_m=float(nom[0]), still_air_status=float(nom[2]), train_mean_m=float(tr[:, 0].mean()),
        train_breaks=int((tr[:, 2] == -2).sum()), train_max_load=float(tr[:, 10].max()),
        heldout=dict(mean=float(r.mean()), std=float(r.std()), p10=float(np.percentile(r, 10)),
                     median=float(np.median(r)), worst=float(r.min()),
                     breaks=int((ho[:, 2] == -2).sum()), n_negative=int((r < 0).sum()),
                     max_load=float(ho[:, 10].max()),
                     mean_original_physics=float(ho_orig[:, 0].mean()),
                     negative_original_physics=int((ho_orig[:, 0] < 0).sum())),
        sanity=dict(dt_pass_train=bool(st["dt_pass"][0]), dt_pass_heldout=bool(sh["dt_pass"][0]),
                    n_dt_fail_heldout=int(sh["n_dt_fail"][0]),
                    max_rel_dev_heldout=float(sh["max_rel_dev"][0]),
                    max_abs_dev_heldout_m=float(sh["max_abs_dev"][0]),
                    break_mismatch_heldout=int(sh["break_mismatch"][0]),
                    energy_pass=bool(st["energy_pass"][0] and sh["energy_pass"][0]),
                    max_step_dE_rel=float(max(st["max_step_dE"][0], sh["max_step_dE"][0]))),
        stability=eig_summary(genes),
        heldout_ranges=[float(x) for x in r],
    )
    if cma_ho is not None:
        out["cma_wins_vs_this"] = int((cma_ho > r).sum())
    return out


def line(rep):
    h, s, d = rep["heldout"], rep["sanity"], rep["design"]
    return (f"{rep['name']:14s} span {d['span_m']*100:4.1f}cm chord {d['chord_m']*100:4.1f}cm AR {d['aspect_ratio']:4.1f} "
            f"m {d['mass_g']:4.1f}g nsafe {d['safe_load_factor']:4.2f} | still {rep['still_air_m']:6.1f} "
            f"train {rep['train_mean_m']:6.1f} (brk {rep['train_breaks']}) held {h['mean']:6.1f} p10 {h['p10']:6.1f} "
            f"worst {h['worst']:6.1f} brk {h['breaks']:2d} neg {h['n_negative']} | dt {s['dt_pass_train'] and s['dt_pass_heldout']} "
            f"({s['n_dt_fail_heldout']} fail, maxdev {s['max_rel_dev_heldout']:.1e}) E {s['energy_pass']}"
            + (f" | CMA wins {rep['cma_wins_vs_this']}/64" if "cma_wins_vs_this" in rep else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--levels", default=",".join(str(x) for x in LEVELS))
    ap.add_argument("--n-heldout", type=int, default=64)
    ap.add_argument("--grid-stride", type=int, default=1, help="subsample the hand grid (smoke only)")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()
    ck_dir = os.path.join(args.outdir, "checkpoints", "struct")
    res_dir = os.path.join(args.outdir, "results", "struct")
    os.makedirs(ck_dir, exist_ok=True); os.makedirs(res_dir, exist_ok=True)
    t0 = time.time()
    tc, tg = sim.training_conditions(seed=0)
    hc, hg = sim.heldout_conditions(n=args.n_heldout)
    old = json.load(open(os.path.join(HERE, "checkpoints", "designs.json")))
    probe = json.load(open(os.path.join(HERE, "results", "review_baseline_probe.json")))
    grid_u = hand_grid()[::args.grid_stride]
    grid_g = sim.phys_units_to_genes(grid_u)
    out = dict(args=vars(args), physics=dict(T_SHEET=ss.T_SHEET, AREAL=sim.AREAL, levels_Pa=[float(x) for x in args.levels.split(",")],
                                             headline_Pa=0.25e6), levels={}, timing_s={})
    designs = {}
    for sigma in [float(x) for x in args.levels.split(",")]:
        tl = time.time()
        key = f"{sigma/1e6:g}MPa"
        f = StructFitness(sigma)
        f(np.full((2, 8), 0.5)); f.n_evals = 0
        cma = [cma_es(f, budget=args.budget, seed=s) for s in range(args.seeds)]
        n_cma = f.n_evals
        rnd = [random_search(f, budget=args.budget, seed=s) for s in range(args.seeds)]
        n_rnd = f.n_evals - n_cma
        fit, _ = f(grid_g)
        n_hand = len(grid_g)
        hand = grid_g[int(np.argmax(fit))]
        print(f"[{key}] cma per seed " + " ".join(f"{r['best_f']:.1f}" for r in cma)
              + " | random " + " ".join(f"{r['best_f']:.1f}" for r in rnd)
              + f" | hand grid best {fit.max():.1f} ({n_hand} designs)  {time.time()-tl:.0f}s", flush=True)
        np.savez(os.path.join(ck_dir, f"search_s{sigma/1e6:g}.npz"),
                 cma_best_x=np.array([r["best_x"] for r in cma]), cma_best_f=np.array([r["best_f"] for r in cma]),
                 cma_hist=np.array([r["hist"][:, 1] for r in cma]), cma_hist_evals=cma[0]["hist"][:, 0],
                 cma_mean=np.array([r["mean"] for r in cma]), cma_sigma=np.array([r["sigma"] for r in cma]),
                 cma_C=np.array([r["C"] for r in cma]),
                 random_best_x=np.array([r["best_x"] for r in rnd]), random_best_f=np.array([r["best_f"] for r in rnd]),
                 hand_grid_units=grid_u, hand_grid_fit=fit, hand_best=hand)
        ib = int(np.argmax([r["best_f"] for r in cma])); cma_x = cma[ib]["best_x"]
        rb = int(np.argmax([r["best_f"] for r in rnd])); rnd_x = rnd[rb]["best_x"]
        cma_ho = ss.evaluate_batch(cma_x[None], hc, hg, DT_SOUND, TMAX, 0, sigma)[0, :, 0]
        L = dict(evals=dict(cma_per_seed=args.budget, random_per_seed=args.budget, seeds=args.seeds,
                            cma_total=n_cma, random_total=n_rnd, hand_grid=n_hand), designs={})
        cands = {"cma_best": cma_x, "random_best": rnd_x, "hand_retrim": hand,
                 "old_baseline": BASELINE, "old_cma_best": np.array(old["cma_best"]["genes"]),
                 "old_mapel_best": np.array(old["mapel_best"]["genes"])}
        for k in ("0.5_0.04_still-trim", "0.5_0.05_still-trim", "0.5_0.06_still-trim"):
            cands["review_" + k.replace("_still-trim", "")] = sim.phys_units_to_genes(probe[k]["units"])
        for name, g in cands.items():
            rep = report(name, g, sigma, tc, tg, hc, hg, cma_ho if name != "cma_best" else None)
            L["designs"][name] = rep
            print(f"[{key}] " + line(rep), flush=True)
        # per-seed held-out means (seed spread, not just best seed)
        for arm, runs in (("cma", cma), ("random", rnd)):
            X = np.array([r["best_x"] for r in runs])
            ho = ss.evaluate_batch(X, hc, hg, DT_SOUND, TMAX, 0, sigma)
            u = sim.genes_to_phys_units(X)
            L[f"{arm}_seeds"] = dict(train_fit=[float(r["best_f"]) for r in runs],
                                     heldout_mean=[float(v) for v in ho[:, :, 0].mean(1)],
                                     heldout_breaks=[int(v) for v in (ho[:, :, 2] == -2).sum(1)],
                                     span_cm=[float(v) for v in 100 * u[:, 0]],
                                     chord_cm=[float(v) for v in 100 * u[:, 1]])
            print(f"[{key}] {arm} seeds: held-out means " + " ".join(f"{v:.1f}" for v in L[f"{arm}_seeds"]["heldout_mean"])
                  + " | spans " + " ".join(f"{v:.1f}" for v in L[f"{arm}_seeds"]["span_cm"]), flush=True)
        # hand-grid landscape: best training fitness per span (does the hand designer's span optimum move?)
        L["hand_best_per_span"] = {f"{s*100:.0f}cm": float(fit[grid_u[:, 0] == s].max()) for s in np.unique(grid_u[:, 0])}
        print(f"[{key}] hand best train per span: {L['hand_best_per_span']}", flush=True)
        # backwards-glide loophole: old MAP-Elites flying wing on held-out launch #10
        if args.n_heldout > 10:
            s10 = ss.fly(cands["old_mapel_best"], hc[10], hg[10], sigma=sigma)[0]
            s10o = sim.fly(cands["old_mapel_best"], hc[10], hg[10])[0]
            L["loophole_launch10"] = dict(struct_range=float(s10[0]), struct_status=float(s10[2]),
                                          struct_time=float(s10[1]), original_range=float(s10o[0]))
            print(f"[{key}] old mapel_best on held-out #10: original {s10o[0]:.1f} m, struct {s10[0]:.1f} m "
                  f"status {s10[2]:.0f} t {s10[1]:.1f}s", flush=True)
        L["time_s"] = time.time() - tl
        out["levels"][key] = L
        designs[key] = {n: dict(genes=v["genes"], design=v["design"]) for n, v in L["designs"].items()
                        if n in ("cma_best", "random_best", "hand_retrim")}
        out["timing_s"]["total"] = time.time() - t0
        out["uptime"] = os.popen("uptime").read().strip()
        with open(os.path.join(res_dir, "eval.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        with open(os.path.join(ck_dir, "designs.json"), "w") as fh:
            json.dump(designs, fh, indent=1)
    print(f"done {time.time()-t0:.0f}s | {out['uptime']}")


if __name__ == "__main__":
    main()
