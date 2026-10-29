"""Review audit for the struct follow-up (RESULTS.md section 12).  Re-evaluates, unchanged, EVERY design in
results/struct/eval.json (winners, baselines, old designs, reviewer probes) and every per-seed CMA-ES / random-search winner in search_s<sigma>.npz.

1. Release-state gap: sim_struct.simulate never checks the load on the release state.  The release-state load
   ratio is computed directly; any launch with ratio > 1 is re-scored 0 m ("corrected": the wing folds at
   release), whether or not the run later broke.  Same rule for every design, both dt.
2. dt-halving failures: for every launch that fails the 0.5 % / 5 cm halving test, print the status and peak
   load ratio at both dt, to see whether the failure is a near-threshold break flip.

Writes results/struct/release_audit_all.txt.  ~20 s.
"""
import json
import os

import numpy as np

import sim
import sim_struct as ss
from search import DT_SOUND, TMAX
from evaluate import REL_TOL, ABS_TOL

HERE = os.path.dirname(os.path.abspath(__file__))


def release_load(G, conds, gusts, sigma):
    """Load ratio on the release state itself (the state sim_struct.simulate never checks)."""
    L = np.zeros((len(G), len(conds)))
    for i, g in enumerate(G):
        P = sim.decode(np.asarray(g, dtype=np.float64))
        m_cap, k = ss.capacity(P, sigma)
        for j, c in enumerate(conds):
            h0, v0, g0, a0, U, clam, cd0m, q0 = c
            st = np.array([0.0, h0, v0 * np.cos(g0), v0 * np.sin(g0), g0 + a0, q0])
            L[i, j] = k * ss._wing_lift(st, P, U, gusts[j], clam, cd0m) / m_cap
    return L


def corrected(s, rl):
    r = s[:, :, 0].copy()
    gap = rl > 1.0                 # wing already over its limit at release: folds there, 0 m
    r[gap] = 0.0
    return r, gap


def main():
    ev = json.load(open(os.path.join(HERE, "results", "struct", "eval.json")))
    designs = {k: L["designs"] for k, L in ev["levels"].items()}
    tc, tg = sim.training_conditions(seed=0)
    hc, hg = sim.heldout_conditions(n=64)
    out = []
    for key, D in designs.items():
        sigma = float(key.replace("MPa", "")) * 1e6
        z = np.load(os.path.join(HERE, "checkpoints", "struct", f"search_s{sigma/1e6:g}.npz"))
        names = list(D) + [f"cma_seed{i}" for i in range(5)] + [f"rnd_seed{i}" for i in range(5)]
        G = np.array([D[n]["genes"] for n in D] + list(z["cma_best_x"]) + list(z["random_best_x"]))
        G = np.clip(G, 0, 1)
        out.append(f"== {key} ==   name | train raw -> corr (gap) | held-out raw -> corr (gap, breaks incl. gap)")
        st = ss.evaluate_batch(G, tc, tg, DT_SOUND, TMAX, 0, sigma)
        sh = ss.evaluate_batch(G, hc, hg, DT_SOUND, TMAX, 0, sigma)
        sh2 = ss.evaluate_batch(G, hc, hg, DT_SOUND / 2, TMAX, 0, sigma)
        st2 = ss.evaluate_batch(G, tc, tg, DT_SOUND / 2, TMAX, 0, sigma)
        rt, gt = corrected(st, release_load(G, tc, tg, sigma))
        rh, gh = corrected(sh, release_load(G, hc, hg, sigma))
        rh2, _ = corrected(sh2, release_load(G, hc, hg, sigma))
        rt2, _ = corrected(st2, release_load(G, tc, tg, sigma))
        for i, n in enumerate(names):
            brk = int(((sh[i, :, 2] == -2) | gh[i]).sum())
            out.append(f"{n:16s} | {st[i,:,0].mean():6.1f} -> {rt[i].mean():6.1f} ({gt[i].sum()}) | "
                       f"{sh[i,:,0].mean():6.1f} -> {rh[i].mean():6.1f} ({gh[i].sum()}, {brk})")
        for lab, a, b in (("train", st, st2), ("held", sh, sh2)):
            dev = np.abs(a[:, :, 0] - b[:, :, 0])
            bad = dev > np.maximum(REL_TOL * np.abs(b[:, :, 0]), ABS_TOL)
            out.append(f"  {lab}: {bad.sum()} raw dt-halving failures; release-exceed among them: "
                       f"{int((bad & (gt if lab == 'train' else gh)).sum())}")
            ca, cb = (rt, rt2) if lab == "train" else (rh, rh2)
            badc = np.abs(ca - cb) > np.maximum(REL_TOL * np.abs(cb), ABS_TOL)
            out.append(f"  {lab}: dt-halving failures after release correction: {int(badc.sum())}"
                       + "".join(f" [{names[i]} #{k}: {ca[i,k]:.2f} vs {cb[i,k]:.2f}]" for i, k in zip(*np.nonzero(badc))))
            for i, k in zip(*np.nonzero(bad)):
                out.append(f"  dt-fail {lab} {names[i]:16s} launch {k:2d}: dt {a[i,k,0]:7.2f} m st {a[i,k,2]:+.0f} "
                           f"load {a[i,k,10]:.3f} | dt/2 {b[i,k,0]:7.2f} m st {b[i,k,2]:+.0f} load {b[i,k,10]:.3f}")
        cs = [f"cma_seed{i}" for i in range(5)]
        rs = [f"rnd_seed{i}" for i in range(5)]
        ix = {n: i for i, n in enumerate(names)}
        out.append(f"  5-seed held-out mean corrected: CMA {np.mean([rh[ix[n]].mean() for n in cs]):.1f} "
                   f"random {np.mean([rh[ix[n]].mean() for n in rs]):.1f}")
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(HERE, "results", "struct", "release_audit_all.txt"), "w") as fh:
        fh.write(__doc__ + "\n" + txt + "\n")


if __name__ == "__main__":
    main()
