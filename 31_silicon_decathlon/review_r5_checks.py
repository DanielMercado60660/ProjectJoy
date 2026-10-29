"""Reviewer checks for round 5 (read-only, NumPy only, no MLX / Core ML, < 2 s).

Reads results/r5/{nominal,stress,old_secondary}/episode_*.json and prints the extra
numbers quoted in RESULTS.md's "Reviewer checks" paragraphs:
  1. model numerics vs latency: per-frame tracker estimate error at frame boundaries
     (no time weighting) and mean |estimate difference| between engines;
  2. how often each engine has the lowest tracker pursuit error in matched conditions;
  3. per-condition agreement of age rankings with round 3 (p50/p95/p99, full and active);
  4. how much of the age -> error R^2 a bare design indicator (delay x policy) explains;
  5. background-load bookkeeping per condition (1-min load average, matmuls/s);
  6. coverage cost of dropping stale frames.
Usage: ../.venv/bin/python review_r5_checks.py | tee results/r5/review_r5_checks.txt
"""
from collections import Counter, defaultdict
import glob
import json

import numpy as np

ENG = ["numpy_cpu", "coreml_cpu", "coreml_ane", "mlx_gpu"]
R3 = {"car": ["numpy_cpu", "coreml_cpu", "coreml_ane", "mlx_gpu"],
      "tracker": ["coreml_cpu", "coreml_ane", "numpy_cpu", "mlx_gpu"]}
E = {run: [json.load(open(f)) for f in sorted(glob.glob(f"results/r5/{run}/episode_*.json"))]
     for run in ("nominal", "stress", "old_secondary")}


def key(d, skip):
    return tuple((k, str(v)) for k, v in sorted(d["job"].items()) if k != skip)


def loso_r2(X, y, seed):
    pred = np.zeros(len(y))
    for s in set(seed):
        tr = seed != s
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        pred[~tr] = X[~tr] @ b
    return 1-((y-pred)**2).sum()/((y-y.mean())**2).sum()


def onehot(c):
    return np.column_stack([(c == v).astype(float) for v in sorted(set(c))])


print("## 1. Tracker numerics vs latency (zero injected delay)")
for run in ("nominal", "stress", "old_secondary"):
    eps = [d for d in E[run] if d["job"]["event"] == "tracker" and d["job"]["delay_ms"] == 0]
    by, rows = defaultdict(dict), defaultdict(list)
    for d in eps:
        pres = np.array(d["geometry"]["present"])
        app = np.array([a["applied_s"] for a in d["actions"]])
        fr = np.array([a["frame"] for a in d["actions"]])
        est, inforce, errs = [], [], []
        for t, truth, _cursor, e in d["trace"]:
            k = fr[app <= t]
            inforce.append(k.max() if len(k) else -1)
            est.append(e)
            errs.append(np.linalg.norm(np.array(e)-np.array(truth), axis=1)[pres].mean())
        by[key(d, "engine")][d["job"]["engine"]] = (np.array(est), np.array(inforce))
        rows[d["job"]["engine"]].append((np.mean(errs[1:]), d["task"]["pursuit_error_px"],
                                         d["task"]["tracking_error_px"]))
    print(f"### {run}")
    for e in ENG:
        v = np.array(rows[e])
        print(f"{e:11s} n={len(v):2d}  frame-boundary estimate error {v[:, 0].mean():.4f} px | "
              f"time-weighted tracking {v[:, 2].mean():.4f} | pursuit {v[:, 1].mean():.4f}")
    for i, a in enumerate(ENG):
        for b in ENG[i+1:]:
            diffs, same = [], []
            for eng in by.values():
                (ea, fa), (eb, fb) = eng[a], eng[b]
                m = min(len(ea), len(eb))
                ok = fa[:m] == fb[:m]
                same.append(ok.mean())
                diffs.append(np.linalg.norm(ea[:m]-eb[:m], axis=2)[ok].mean())
            print(f"  |est {a} - {b}| = {np.mean(diffs):.4f} px (same frame in force {np.mean(same):.3f})")

print("\n## 2. Lowest tracker pursuit error per matched condition")
for name, src, flt in (("nominal", "nominal", lambda d: True),
                       ("stress 0 ms", "stress", lambda d: d["job"]["delay_ms"] == 0),
                       ("stress 50 ms", "stress", lambda d: d["job"]["delay_ms"] == 50),
                       ("old_secondary", "old_secondary", lambda d: True)):
    g = defaultdict(dict)
    for d in E[src]:
        if d["job"]["event"] == "tracker" and flt(d):
            g[key(d, "engine")][d["job"]["engine"]] = d["task"]["pursuit_error_px"]
    wins = Counter(min(v, key=v.get) for v in g.values())
    beats = {e: sum(v["coreml_cpu"] < v[e] for v in g.values()) for e in ENG if e != "coreml_cpu"}
    print(f"{name:14s} conditions={len(g):2d} lowest={dict(wins)} coreml_cpu beats={beats}")

print("\n## 3. Nominal age rankings vs round-3 isolated order, per matched condition (8 each)")
for ev in ("car", "tracker"):
    for active in (False, True):
        g = defaultdict(dict)
        for d in E["nominal"]:
            if d["job"]["event"] != ev:
                continue
            a = [x["age_ms"] for x in d["actions"] if x["task_active"] or not active]
            g[key(d, "engine")][d["job"]["engine"]] = [np.percentile(a, p) for p in (50, 95, 99)]
        for i, p in enumerate(("p50", "p95", "p99")):
            exact = sum(sorted(v, key=lambda e: v[e][i]) == R3[ev] for v in g.values())
            mlx_last = sum(max(v, key=lambda e: v[e][i]) == "mlx_gpu" for v in g.values())
            np_first = sum(min(v, key=lambda e: v[e][i]) == "numpy_cpu" for v in g.values())
            cml = sum(max(v["coreml_cpu"][i], v["coreml_ane"][i]) < v["numpy_cpu"][i] for v in g.values())
            cpu_ane = sum(v["coreml_cpu"][i] < v["coreml_ane"][i] for v in g.values())
            print(f"{ev:7s} {'active' if active else 'full  '} {p}: exact R3 order {exact}/8, "
                  f"MLX GPU slowest {mlx_last}/8, NumPy fastest {np_first}/8, both Core ML < NumPy {cml}/8, "
                  f"Core ML CPU < CPU+NE {cpu_ane}/8")
    for e in ENG:
        es = [d for d in E["nominal"] if d["job"]["event"] == ev and d["job"]["engine"] == e]
        a = [x["age_ms"] for d in es for x in d["actions"] if x["task_active"]]
        print(f"  pooled active-only {e:11s} p50/p95/p99 " + " / ".join(f"{np.percentile(a, q):.3f}" for q in (50, 95, 99)))

print("\n## 4. Stress: age regression vs design indicators (leave-one-seed-out R^2)")
es = E["stress"]
age = np.array([np.mean([a["age_ms"] for a in d["actions"]]) for d in es])
seed = np.array([d["job"]["seed"] for d in es])
delay = np.array([d["job"]["delay_ms"] for d in es])
pol = np.array([d["job"]["policy"] for d in es])
cell = np.array([f"{a}/{b}" for a, b in zip(delay, pol)])
one = np.ones((len(es), 1))
for outcome in ("pursuit_error_px", "tracking_error_px"):
    y = np.array([d["task"][outcome] for d in es])
    m = delay == 50
    within = 1-sum(((y[cell == c]-y[cell == c].mean())**2).sum() for c in set(cell))/((y-y.mean())**2).sum()
    print(f"{outcome}: both delays age {loso_r2(np.column_stack([one, age]), y, seed):.3f}, "
          f"delay x policy indicator only {loso_r2(onehot(cell), y, seed):.3f}, indicator + age "
          f"{loso_r2(np.column_stack([onehot(cell), age]), y, seed):.3f}; between-cell variance share {within:.3f}")
    print(f"  50 ms only: age {loso_r2(np.column_stack([one[m], age[m]]), y[m], seed[m]):.3f}, policy indicator "
          f"{loso_r2(onehot(pol[m]), y[m], seed[m]):.3f}, policy + age "
          f"{loso_r2(np.column_stack([onehot(pol[m]), age[m]]), y[m], seed[m]):.3f}")
    for c in sorted(set(cell)):
        k = cell == c
        a, yy, s = age[k], y[k], seed[k]
        ac = a-np.array([a[s == v].mean() for v in s])
        yc = yy-np.array([yy[s == v].mean() for v in s])
        print(f"  cell {c:12s} n={k.sum()} age {a.min():.1f}-{a.max():.1f} ms, within-cell LOSO R2 "
              f"{loso_r2(np.column_stack([one[k], a]), yy, s):.3f}, seed-demeaned r {np.corrcoef(ac, yc)[0, 1]:.3f}")

print("\n## 5. Load bookkeeping per condition")
for run, eps in E.items():
    for ld in ("ambient", "loaded"):
        es_ = [d for d in eps if d["job"]["load"] == ld]
        la = [d["env_before"]["loadavg"][0] for d in es_]
        mm = np.array([d["background_matmuls"] for d in es_])
        wall = np.array([d["metrics"]["wall_s"] for d in es_])
        rate = (mm/wall).mean()
        print(f"{run:13s} {ld:7s} n={len(es_):3d} 1-min load before: mean {np.mean(la):.2f} "
              f"({min(la):.2f}-{max(la):.2f}); background 1024^2 matmuls/s {rate:.0f} "
              f"(~{rate*2*1024**3/1e12:.2f} TFLOP/s)")

print("\n## 6. Coverage cost of dropping stale frames (stress tracker)")
for dl in (0, 50):
    for p in ("queue", "latest"):
        es_ = [d for d in E["stress"] if d["job"]["delay_ms"] == dl and d["job"]["policy"] == p]
        c = {k: sum(d["metrics"][k] for d in es_) for k in
             ("scheduled", "applied", "dropped", "pending_at_end", "inflight_at_end")}
        print(f"{dl:2d} ms {p:6s}: applied {c['applied']}/{c['scheduled']} ({c['applied']/c['scheduled']:.2%}), "
              f"dropped {c['dropped']} ({c['dropped']/c['scheduled']:.1%}), stranded at horizon "
              f"{c['pending_at_end']}+{c['inflight_at_end']} "
              f"({(c['pending_at_end']+c['inflight_at_end'])/c['scheduled']:.1%})")
