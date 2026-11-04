"""Runner-run transfer test (Claude), NOT an Astra attempt: apply the two ONE-SHOT builders unchanged to the
`init` condition (θ0 = random init; every training-free merge scored ~0% there in the earlier round).

1. export checkpoints/s*/{init,init_fwd,init_bwd}.npz -> wsp/data_init/s*/{theta0,fwd,bwd}.npz (same format as wsp/data)
2. copy wsp/attempts/r6wsp_os_{a,b}/build.py -> wsp/init_transfer/r6wsp_os_{a,b}/build.py with the one-line data
   path change "data" -> "data_init" (construction logic untouched), run them
3. score on the held-out bank 31000-31019: oracle, θ0 (= random init), average, DAgger-200k from θ0
   (checkpoints/s*/distill_init_theta0.npz), fwd/bwd specialist for both labels, the two transferred builds

  ../bin/gpu_run ../.venv/bin/python wsp/init_transfer.py     -> wsp/results/init_transfer.json
"""
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys

os.environ["OMP_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
import numpy as np  # noqa: E402

HELDOUT = list(range(31000, 31020))
AK = ["a1.weight", "a1.bias", "a2.weight", "a2.bias", "a3.weight", "a3.bias"]
KEYS = AK + ["log_std"]


def export():
    for s in range(4):
        d = os.path.join(HERE, "data_init", f"s{s}")
        os.makedirs(d, exist_ok=True)
        for name, src in (("theta0", "init"), ("fwd", "init_fwd"), ("bwd", "init_bwd")):
            z = np.load(os.path.join(PROJ, "checkpoints", f"s{s}", f"{src}.npz"))
            np.savez(os.path.join(d, f"{name}.npz"), **{k: z[k].astype(np.float32) for k in KEYS},
                     norm_mean=z["norm_mean"], norm_var=z["norm_var"])


def build():
    for tag in ("r6wsp_os_a", "r6wsp_os_b"):
        d = os.path.join(HERE, "init_transfer", tag)
        os.makedirs(d, exist_ok=True)
        src = open(os.path.join(HERE, "attempts", tag, "build.py")).read()
        old = 'DATA = HERE.parent.parent / "data"'
        assert src.count(old) == 1
        open(os.path.join(d, "build.py"), "w").write(src.replace(old, 'DATA = HERE.parent.parent / "data_init"'))
        r = subprocess.run([sys.executable, os.path.join(d, "build.py")], cwd=PROJ, capture_output=True, text=True,
                           env=dict(os.environ, OMP_NUM_THREADS="4"))
        print(tag, "build exit", r.returncode, r.stdout[-300:], r.stderr[-500:], flush=True)


def nets(seed):
    from common import load_params
    import merge as M
    d = os.path.join(PROJ, "checkpoints", f"s{seed}")
    _, norm = load_params(os.path.join(d, "base.npz"))
    t0 = load_params(os.path.join(d, "init.npz"))[0]
    A, B = load_params(os.path.join(d, "init_fwd.npz"))[0], load_params(os.path.join(d, "init_bwd.npz"))[0]
    n = {"oracle": ("switch", A, B), "theta0_init": t0, "average": M.interpolate(A, B, 0.5),
         "dagger_200k": load_params(os.path.join(d, "distill_init_theta0.npz"))[0], "fwd_spec_both": A, "bwd_spec_both": B}
    for tag in ("r6wsp_os_a", "r6wsp_os_b"):
        z = np.load(os.path.join(HERE, "init_transfer", tag, f"merged_s{seed}.npz"))
        n[tag + "_transfer"] = {k: z[k].astype(np.float32) for k in AK}
    return n, norm


def job(args):
    seed, name = args
    from common import evaluate
    n, norm = nets(seed)
    p = n[name]
    return seed, name, {t: evaluate(p[1 + i] if isinstance(p, tuple) else p, norm, t, HELDOUT).tolist()
                        for i, t in enumerate(("fwd", "bwd"))}


def main():
    export()
    build()
    jobs = [(s, k) for s in range(4) for k in nets(s)[0]]
    R = {}
    with mp.get_context("spawn").Pool(4) as pool:
        for s, k, o in pool.imap_unordered(job, jobs):
            R.setdefault(str(s), {})[k] = o
    rows = {}
    for k in R["0"]:
        mr = [np.mean([np.mean(R[s][k][t]) / np.mean(R[s]["oracle"][t]) for t in ("fwd", "bwd")]) for s in R]
        rows[k] = dict(fwd=float(np.mean([np.mean(R[s][k]["fwd"]) for s in R])),
                       bwd=float(np.mean([np.mean(R[s][k]["bwd"]) for s in R])),
                       mean_ret=float(np.mean(mr)), std=float(np.std(mr)), per_seed=[float(v) for v in mr])
        print(f"{k:26s} fwd {rows[k]['fwd']:6.0f} bwd {rows[k]['bwd']:6.0f}  retained {100 * np.mean(mr):5.1f}% ± {100 * np.std(mr):4.1f}"
              f"  {' / '.join(f'{100 * v:.0f}' for v in mr)}", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(dict(bank=HELDOUT, raw=R, summary=rows), open(os.path.join(HERE, "results", "init_transfer.json"), "w"))


if __name__ == "__main__":
    main()
