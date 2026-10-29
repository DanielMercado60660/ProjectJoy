"""Event 07: Terra learned A* heuristic (HNet U-Net, base 32, 2.0M params), one forward pass.

One item = the learned heuristic field for one (96x96 map, goal) query. Batch 1 = one A* query, as in
eval_heuristic.py and the demo; batch 16 = the training batch, a set of goals at once.
Inputs are queries from the sibling's generated validation set (data/val.npz, fixed seed).
Task check (both batches): 32 fixed (map, start, goal) queries; run A* (alpha = 1, the project's
"Learned a=1.0") with each athlete's heuristic; compare path cost above optimal and node expansions to
MLX fp32, plus the training metric (|log-ratio error| vs exact cost-to-go labels).
"""
import mlx.core as mx
import numpy as np

import coreml_util
from models import terra as T

EVENT = dict(id="07_terra", project="07_terra", model="HNet U-Net heuristic (base 32, GroupNorm, 4 downsamplings)",
             checkpoint=T.CKPT, unit="heuristic fields (one 96x96 map + goal)", params=None,
             notes="Queries from ../07_terra/data/val.npz (generated, gitignored there; regenerable with its "
                   "make_dataset.py). Core ML packages are fp16.")
BATCHES = (1, 16)
SKIPS = {"mlx_gpu_q8": "conv net; MLX quantization covers Linear layers only (one 128x128 Linear here)",
         "mlx_gpu_q4": "conv net; MLX quantization covers Linear layers only (one 128x128 Linear here)"}
N_QUERIES = 32

_cache = {}


def _model(dtype=mx.float32):
    if ("mlx", dtype) not in _cache:
        m = T.load_mlx(dtype=dtype)
        _cache[("mlx", dtype)] = m
        if dtype == mx.float32:
            import mlx.utils
            EVENT["params"] = sum(v.size for _, v in mlx.utils.tree_flatten(m.parameters()))
    return _cache[("mlx", dtype)]


def _indices(n, seed):
    return np.random.default_rng(seed).choice(len(T.val()["goal"]), n, replace=False)


def make_inputs(batch, seed=0):
    return {"x": T.sample_inputs(_indices(batch, seed))}


def _wrap_mlx(fwd, dtype=mx.float32, stream=None):
    def fn(inp):
        x = mx.array(inp["x"]).astype(dtype)
        if stream is not None:
            with mx.stream(stream):
                r = fwd(x)
                mx.eval(r)
        else:
            r = fwd(x)
            mx.eval(r)
        return {"r": np.array(r.astype(mx.float32))}
    return fn


def coreml_packages(batch):
    x = make_inputs(batch)["x"]
    path = coreml_util.export_torch(T.torch_net(), (x,), ["x"], f"e07_terra_b{batch}_fp16")
    return {k: path for k in coreml_util.UNITS}


def athletes(batch):
    if ("ath", batch) in _cache:
        return _cache[("ath", batch)]
    m = _model()
    npn = T.NumpyHNet()
    out = {
        "numpy_cpu": lambda inp: {"r": npn(inp["x"])},
        "mlx_cpu": _wrap_mlx(m, stream=mx.cpu),
        "mlx_gpu": _wrap_mlx(m),
        "mlx_gpu_fp16": _wrap_mlx(_model(mx.float16), dtype=mx.float16),
        "mlx_gpu_compiled": _wrap_mlx(mx.compile(m)),
    }
    for name, path in coreml_packages(batch).items():
        out[name] = coreml_util.make_athlete(path, name, rename=["r"])
    _cache[("ath", batch)] = out
    return out


def agreement(o, ref):
    r, rr = np.maximum(o["r"], 0), np.maximum(ref["r"], 0)   # the planner clips r at 0
    return dict(r_max_err=float(np.abs(o["r"] - ref["r"]).max()),
                h_max_rel_err=float(np.abs(np.exp(r - rr) - 1).max()))


# ---------- task check ----------
def _queries():
    """N_QUERIES fixed (sample idx, start, goal) with octile(start, goal) >= 30, as in eval_heuristic.py."""
    if "queries" not in _cache:
        rng = np.random.default_rng(2024)
        idx = _indices(N_QUERIES, seed=99)
        qs, adjs = [], {}
        for i in idx:
            elev, kind = T.map_of(i)
            gr, gc = T.val()["goal"][i]
            o = T.octile_field(T.SIZE, T.SIZE, gr, gc)
            ok = np.flatnonzero((kind.ravel() != T.WATER) & (o.ravel() >= 30))
            s = int(rng.choice(ok))
            mi = int(T.val()["map_index"][i])
            if mi not in adjs:
                adjs[mi] = T.adjacency(elev, kind)
            g = int(gr * T.SIZE + gc)
            opt, _ = T.astar(adjs[mi], s, g, o.ravel().tolist())   # octile is admissible -> optimal cost
            qs.append(dict(i=int(i), s=s, g=g, mi=mi, opt=opt, oct=o))
        _cache["queries"], _cache["adjs"] = qs, adjs
    return _cache["queries"], _cache["adjs"]


def _run_queries(fn, batch):
    qs, adjs = _queries()
    v = T.val()
    rs = []
    for k in range(0, len(qs), batch):
        chunk = [q["i"] for q in qs[k:k + batch]]
        rs.append(fn({"x": T.sample_inputs(chunk)})["r"])
    rs = np.concatenate(rs)
    cost, exp_, mae = [], [], []
    for q, r in zip(qs, rs):
        t, m = v["target"][q["i"]].astype(np.float32), v["mask"][q["i"]]
        mae.append(float(np.abs(r - t)[m].mean()))
        h = (q["oct"] * np.exp(np.maximum(r.astype(np.float64), 0))).ravel().tolist()
        c, e = T.astar(adjs[q["mi"]], q["s"], q["g"], h)
        cost.append(c); exp_.append(e)
    opt = np.array([q["opt"] for q in qs])
    return np.array(cost), np.array(exp_), np.array(mae), opt


def task_check(name, batch):
    """A* with this athlete's heuristic on 32 fixed queries, run through the athlete at this batch size."""
    ath = athletes(batch)
    if ("ref", batch) not in _cache:
        _cache[("ref", batch)] = _run_queries(ath["mlx_gpu"], batch)
    rc, re_, rmae, opt = _cache[("ref", batch)]
    c, e, mae, _ = (rc, re_, rmae, opt) if name == "mlx_gpu" else _run_queries(ath[name], batch)
    return dict(queries=len(c),
                same_path_cost_frac=float(np.mean(np.abs(c - rc) <= 1e-9 * rc)),
                same_expansions_frac=float(np.mean(e == re_)),
                cost_above_opt_pct=float(np.mean(c / opt - 1) * 100),
                ref_cost_above_opt_pct=float(np.mean(rc / opt - 1) * 100),
                expanded_vs_ref=float(e.sum() / re_.sum()),
                val_abs_logratio_err=float(mae.mean()), ref_val_abs_logratio_err=float(rmae.mean()))


def qualifies(agreement, task):
    """Same heuristic if A* with it is as good as with MLX fp32: mean cost above optimal within 0.5 points,
    total expansions within 5%, label error within 0.005. Without a task: r max err <= 0.05."""
    if task:
        d_cost = abs(task["cost_above_opt_pct"] - task["ref_cost_above_opt_pct"])
        d_exp = abs(task["expanded_vs_ref"] - 1)
        d_mae = abs(task["val_abs_logratio_err"] - task["ref_val_abs_logratio_err"])
        ok = d_cost <= 0.5 and d_exp <= 0.05 and d_mae <= 0.005
        return ok, (f"A* cost above opt {task['cost_above_opt_pct']:.2f}% vs ref {task['ref_cost_above_opt_pct']:.2f}%, "
                    f"expansions x{task['expanded_vs_ref']:.3f}, same path cost {task['same_path_cost_frac']:.0%}, "
                    f"label err {task['val_abs_logratio_err']:.4f} vs {task['ref_val_abs_logratio_err']:.4f}")
    err = agreement["r_max_err"]
    return err <= 0.05, f"r max err {err:.2g} (tolerance 0.05)"
