"""Reviewer probe: is the hand baseline a strawman?  Keeps its rule-of-thumb tail (25 % area, 18 cm arm)
and re-trims ballast, ballast position and incidences on a 972-point grid scored on the STILL-AIR nominal
throw only (a hand-trim analogue; "train-trim" = picked on the 6 training launches instead), for the
shipped 30 cm planform and for the same shape stretched to the 50 cm span bound.  Held-out launches are
only read for reporting.  ~4 min on 4 threads under load.
Usage: NUMBA_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python review/baseline_probe.py results/review_baseline_probe.json
"""
import sys, itertools, json, numpy as np
import os; HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import sim
from search import DT_SOUND, TMAX
from evaluate import BASELINE_UNITS
tc, tg = sim.training_conditions(seed=0); hc, hg = sim.heldout_conditions(64)
nc, ng = sim.still_air()
def ev(units):
    g = np.atleast_2d(sim.phys_units_to_genes(np.array(units)))
    return (sim.evaluate_batch(g, nc, ng, DT_SOUND, TMAX, 0)[:, 0, 0],
            sim.evaluate_batch(g, tc, tg, DT_SOUND, TMAX, 0)[:, :, 0].mean(1),
            sim.evaluate_batch(g, hc, hg, DT_SOUND, TMAX, 0)[:, :, 0])
b = ev([BASELINE_UNITS])
print("baseline as shipped: still %.1f train %.1f held %.1f" % (b[0][0], b[1][0], b[2][0].mean()))
cma = np.array(json.load(open(os.path.join(HERE, "checkpoints", "designs.json")))["cma_best"]["genes"])
cmah = sim.evaluate_batch(cma[None], hc, hg, DT_SOUND, TMAX, 0)[0, :, 0]
out = {}
# same rule-of-thumb shape, span stretched; re-trim ballast/pos/incidences on STILL AIR only (hand-trim analogue)
for span, chord in [(0.30, 0.06), (0.50, 0.06), (0.50, 0.05), (0.50, 0.04)]:
    grid = []
    for mb, xb, iw, it in itertools.product(np.linspace(0, 0.02, 9), [0.0, 0.03, 0.06, 0.10], [0, 2, 4], np.linspace(-6, 2, 9)):
        grid.append([span, chord, 0.25, 0.18, mb, xb, iw, it])
    grid = np.array(grid)
    s, t, h = ev(grid)
    i = int(np.argmax(s))       # chosen by still-air nominal flight only
    j = int(np.argmax(t))       # chosen by training mean
    for tag, k in (("still-trim", i), ("train-trim", j)):
        wins = int((cmah > h[k]).sum())
        print(f"span {span} chord {chord} AR {span/chord:.1f} {tag}: units {np.round(grid[k],3).tolist()} still {s[k]:.1f} train {t[k]:.1f} held mean {h[k].mean():.1f} p10 {np.percentile(h[k],10):.1f} worst {h[k].min():.1f}  CMA wins {wins}/64  (grid {len(grid)})")
        out[f"{span}_{chord}_{tag}"] = dict(units=grid[k].tolist(), still=float(s[k]), train=float(t[k]), held_mean=float(h[k].mean()), held_p10=float(np.percentile(h[k],10)), held_worst=float(h[k].min()), cma_wins=wins, n_trim_flights=len(grid))
json.dump(out, open(sys.argv[1], "w"), indent=1)
