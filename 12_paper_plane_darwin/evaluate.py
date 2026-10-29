"""Audit and compare designs from checkpoints/search.npz against the hand-designed baseline.

For every candidate: still-air nominal range, training-mean range, physical sanity checks
(dt-halving and per-step energy), held-out robustness (64 unseen launches/gusts/model errors),
stability (static margin, linearised glide modes, pitch-rate RMS), and archive diversity.

Usage:  NUMBA_NUM_THREADS=4 ../.venv/bin/python evaluate.py [--ckpt checkpoints/search.npz]
Writes results/eval.json and checkpoints/designs.json.
"""
import argparse
import json
import os

import numpy as np

import sim
from search import DT_SOUND, DT_CHEAP, TMAX, ME_BINS, ME_RANGE

HERE = os.path.dirname(os.path.abspath(__file__))
REL_TOL = 0.005      # dt-halving: ranges must agree within 0.5 % (or 5 cm)
ABS_TOL = 0.05
E_TOL = 1e-9         # energy: no step may gain more than 1e-9 of launch energy in still air

# Hand-designed baseline: classic rule-of-thumb glider (AR 5, tail volume ~0.5, CG ~30 % chord,
# 3 deg decalage).  Ballast and tail incidence picked from 18 hand-trim test flights in still air
# (like test-throwing a real glider); nothing searched automatically.
BASELINE_UNITS = [0.30, 0.06, 0.25, 0.18, 0.006, 0.06, 2.0, -1.0]
BASELINE = sim.phys_units_to_genes(BASELINE_UNITS)


def sanity(genes, conds, gusts, dt=DT_SOUND):
    """Halve dt (twice) and compare ranges; check energy never increases in still air."""
    g = np.atleast_2d(genes)
    s1 = sim.evaluate_batch(g, conds, gusts, dt, TMAX, 0)
    s2 = sim.evaluate_batch(g, conds, gusts, dt / 2, TMAX, 0)
    r1, r2 = s1[:, :, 0], s2[:, :, 0]
    dev = np.abs(r1 - r2)
    ok_dt = dev <= np.maximum(REL_TOL * np.abs(r2), ABS_TOL)
    # still-air copies of the same launches (wind & gusts off) for the energy audit
    c0 = conds.copy(); c0[:, 4] = 0.0
    s0 = sim.evaluate_batch(g, c0, np.zeros_like(gusts), dt, TMAX, 0)
    return dict(dt_pass=ok_dt.all(1), max_rel_dev=(dev / np.maximum(np.abs(r2), 1e-9)).max(1),
                max_abs_dev=dev.max(1), energy_pass=(s0[:, :, 3] <= E_TOL).all(1),
                max_step_dE=s0[:, :, 3].max(1), n_gain_steps=s0[:, :, 9].sum(1))


def eig_summary(genes):
    eig, st, conv = sim.glide_eigen(genes)
    if eig is None:
        return dict(converged=False)
    re_max = float(eig.real.max())
    osc = eig[np.abs(eig.imag) > 1e-6]
    modes = []
    for e in osc[osc.imag > 0]:
        wn = abs(e)
        modes.append(dict(wn=float(wn), zeta=float(-e.real / wn), period_s=float(2 * np.pi / e.imag)))
    modes.sort(key=lambda m: m["wn"])
    return dict(converged=conv, max_real=re_max, stable=re_max < 0, modes=modes,
                glide_ratio=float(-st[2] / st[3]) if st[3] < 0 else None,
                glide_speed=float(np.hypot(st[2], st[3])), sink_rate=float(-st[3]),
                eigs=[[float(e.real), float(e.imag)] for e in eig])


def full_report(name, genes, tc, tg, hc, hg):
    g = np.atleast_2d(genes)
    nom_c, nom_g = sim.still_air()
    nom = sim.evaluate_batch(g, nom_c, nom_g, DT_SOUND, TMAX, 0)[0, 0]
    tr = sim.evaluate_batch(g, tc, tg, DT_SOUND, TMAX, 0)[0]
    ho = sim.evaluate_batch(g, hc, hg, DT_SOUND, TMAX, 0)[0]
    tr_cheap = sim.evaluate_batch(g, tc, tg, DT_CHEAP, TMAX, 1)[0]
    san_train = sanity(g, tc, tg)
    san_ho = sanity(g, hc, hg)
    r = ho[:, 0]
    return dict(
        name=name, genes=[float(x) for x in genes], design=sim.describe(genes),
        nominal_range_m=float(nom[0]), nominal_time_s=float(nom[1]),
        nominal_pitch_rate_rms=float(nom[6]), nominal_peak_climb_m=float(nom[5]),
        train_mean_range_m=float(tr[:, 0].mean()),
        train_mean_range_cheap_sim_m=float(tr_cheap[:, 0].mean()),
        cheap_sim_max_step_dE=float(tr_cheap[:, 3].max()),
        heldout=dict(mean=float(r.mean()), std=float(r.std()), p10=float(np.percentile(r, 10)),
                     worst=float(r.min()), median=float(np.median(r)),
                     landed_frac=float((ho[:, 2] == 1).mean()),
                     mean_pitch_rate_rms=float(ho[:, 6].mean())),
        sanity=dict(train_dt_pass=bool(san_train["dt_pass"][0]), heldout_dt_pass=bool(san_ho["dt_pass"][0]),
                    max_rel_dev_heldout=float(san_ho["max_rel_dev"][0]),
                    max_abs_dev_heldout_m=float(san_ho["max_abs_dev"][0]),
                    energy_pass=bool(san_train["energy_pass"][0] and san_ho["energy_pass"][0]),
                    max_step_dE_rel=float(max(san_train["max_step_dE"][0], san_ho["max_step_dE"][0])),
                    energy_gain_steps=int(san_train["n_gain_steps"][0] + san_ho["n_gain_steps"][0])),
        stability=eig_summary(genes),
        heldout_ranges=[float(x) for x in r],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "search.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "eval.json"))
    ap.add_argument("--n-heldout", type=int, default=64)
    ap.add_argument("--designs", default=os.path.join(HERE, "checkpoints", "designs.json"))
    args = ap.parse_args()
    ck = np.load(args.ckpt)
    tc, tg = sim.training_conditions(seed=0)
    hc, hg = sim.heldout_conditions(n=args.n_heldout)

    cands = {"baseline_hand": BASELINE}
    i = int(np.argmax(ck["cma_best_f"])); cands["cma_best"] = ck["cma_best_x"][i]
    i = int(np.argmax(ck["random_best_f"])); cands["random_best"] = ck["random_best_x"][i]
    fit, xg = ck["me_fit"], ck["me_x"]
    ok = np.isfinite(fit)
    ij = np.unravel_index(np.argmax(np.where(ok, fit, -np.inf)), fit.shape)
    cands["mapel_best"] = xg[ij]
    i = int(np.argmax(ck["cheap_best_f"])); cands["cheap_sim_best"] = ck["cheap_best_x"][i]

    out = {"tolerances": dict(dt_rel=REL_TOL, dt_abs_m=ABS_TOL, energy_step_rel=E_TOL,
                              dt_search=DT_SOUND, dt_check=DT_SOUND / 2, dt_cheap=DT_CHEAP),
           "candidates": {}}
    for name, g in cands.items():
        rep = full_report(name, g, tc, tg, hc, hg)
        out["candidates"][name] = rep
        h = rep["heldout"]; s = rep["sanity"]
        print(f"{name:15s} nominal {rep['nominal_range_m']:6.1f}  train {rep['train_mean_range_m']:6.1f} "
              f"(cheap-sim {rep['train_mean_range_cheap_sim_m']:6.1f})  heldout mean {h['mean']:6.1f} "
              f"p10 {h['p10']:6.1f} worst {h['worst']:6.1f} | dt-pass {s['train_dt_pass'] and s['heldout_dt_pass']} "
              f"maxdev {s['max_rel_dev_heldout']:.2e} energy-pass {s['energy_pass']} | "
              f"SM {rep['design']['static_margin']:.2f} stable {rep['stability'].get('stable')}")

    # per-seed spread for CMA / random / cheap (training fitness, search sim) and held-out mean
    for arm in ("cma", "random", "cheap"):
        X = ck[f"{arm}_best_x"]
        ho = sim.evaluate_batch(X, hc, hg, DT_SOUND, TMAX, 0)[:, :, 0].mean(1)
        trs = sim.evaluate_batch(X, tc, tg, DT_SOUND, TMAX, 0)[:, :, 0].mean(1)
        san = sanity(X, tc, tg)
        out[f"{arm}_seeds"] = dict(search_fitness=[float(v) for v in ck[f"{arm}_best_f"]],
                                   train_mean_sound_sim=[float(v) for v in trs],
                                   heldout_mean=[float(v) for v in ho],
                                   dt_pass=[bool(v) for v in san["dt_pass"]],
                                   energy_pass=[bool(v) for v in san["energy_pass"]])

    # archive: coverage, QD score, sanity of every elite, diversity in design space
    E = xg[ok]
    Ef = fit[ok]
    san = sanity(E, tc, tg)
    units = sim.genes_to_phys_units(E)
    out["archive"] = dict(
        cells_filled=int(ok.sum()), cells_total=int(ok.size), coverage=float(ok.mean()),
        qd_score=float(np.maximum(Ef, 0).sum()), best=float(Ef.max()),
        n_elites_beating_baseline_train=int((Ef > out["candidates"]["baseline_hand"]["train_mean_range_m"]).sum()),
        elite_dt_pass_frac=float(san["dt_pass"].mean()), elite_energy_pass_frac=float(san["energy_pass"].mean()),
        mean_pairwise_gene_dist=float(np.mean(np.linalg.norm(E[:, None] - E[None], axis=-1))),
        gene_std=[float(v) for v in E.std(0)],
        span_range_m=[float(units[:, 0].min()), float(units[:, 0].max())],
        bins=list(ME_BINS), ranges=[list(r) for r in ME_RANGE],
    )
    # a few hand-picked "species" from the archive for the demo: top-3 cells by fitness from
    # distinct descriptor regions (longest flight time >= 80 % of best, smoothest, most porpoising)
    good = Ef > 0.8 * Ef.max()
    idx = np.argwhere(ok)
    sp = {}
    if good.any():
        gi = np.where(good)[0]
        sp["archive_floater"] = int(gi[np.argmax(idx[gi, 0])])
        sp["archive_smooth"] = int(gi[np.argmin(idx[gi, 1])])
        sp["archive_porpoise"] = int(gi[np.argmax(idx[gi, 1])])
    for name, k in sp.items():
        rep = full_report(name, E[k], tc, tg, hc, hg)
        rep["cell"] = [int(v) for v in idx[k]]
        out["candidates"][name] = rep
        print(f"{name:15s} cell {idx[k]} train {rep['train_mean_range_m']:.1f} nominal t {rep['nominal_time_s']:.1f}s "
              f"qrms {rep['nominal_pitch_rate_rms']:.3f} heldout {rep['heldout']['mean']:.1f} SM {rep['design']['static_margin']:.2f}")
    print("archive", {k: v for k, v in out["archive"].items() if k not in ("gene_std",)})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    designs = {k: dict(genes=v["genes"], design=v["design"]) for k, v in out["candidates"].items()}
    with open(args.designs, "w") as fh:
        json.dump(designs, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
