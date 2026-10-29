"""Speed / memory bench, re-runnable on an idle machine:
    ../.venv/bin/python bench_rtf.py --out results/bench_rtf_idle.json   # refuses to overwrite an existing file
Per method, on all held-out clips (ground-truth mels):
  - RTF = synthesis wall time / audio duration (whole utterance, non-streaming; lower is better)
  - first-output latency = wall time to synthesize audio for the first 64 mel frames (~1.0 s of audio),
    i.e. the earliest a non-streaming vocoder could hand back a first chunk; median of 5
  - peak memory: MLX peak allocator memory (neural) / tracemalloc peak of numpy (Griffin-Lim) for the
    longest clip, plus process max RSS
The `uptime` load average is stored alongside; numbers taken while other jobs run are "under load".
"""
import argparse, json, os, resource, subprocess, sys, time, tracemalloc
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def timeit(fn, mel, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(mel); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "cache", "test.npz"))
    ap.add_argument("--methods", default="gl32,gl64,neural_gpu,neural_cpu")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "bench_rtf.json"))
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing --out file")
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.overwrite:  # review fix: the load-13.7 run was silently overwritten
        sys.exit(f"{a.out} exists; pass a new --out (e.g. results/bench_rtf_idle.json) or --overwrite")
    import mlx.core as mx
    from vocoders import gl_fn, neural_fn
    z = np.load(a.data)
    mels = [z[f"mel_{i}"] for i in range(len(z["ids"]))]
    secs = [len(z[f"wav_{i}"]) / 16000 for i in range(len(z["ids"]))]
    longest = int(np.argmax(secs))
    res = {"uptime_before": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
           "omp_threads": os.environ.get("OMP_NUM_THREADS"), "n_clips": len(mels),
           "audio_seconds": float(sum(secs)), "methods": {}}
    for name in a.methods.split(","):
        if name.startswith("gl"):
            fn = gl_fn(int(name[2:]))
        else:
            mx.set_default_device(mx.gpu if name.endswith("gpu") else mx.cpu)
            fn = neural_fn()
        fn(mels[0][:64]); fn(mels[0])  # warm-up (compiles kernels / caches)
        total = sum(timeit(fn, m, a.reps) for m in mels)
        first = timeit(fn, mels[0][:64], 5)
        if name.startswith("gl"):
            tracemalloc.start(); fn(mels[longest]); _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        else:
            mx.reset_peak_memory(); fn(mels[longest]); peak = mx.get_peak_memory()
        res["methods"][name] = {"rtf": total / sum(secs), "x_realtime": sum(secs) / total,
                                "first_output_latency_ms_64frames": first * 1000,
                                "peak_mem_mb_longest_clip": peak / 2 ** 20, "longest_clip_s": secs[longest]}
        print(name, res["methods"][name], flush=True)
    res["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20  # bytes on macOS
    res["uptime_after"] = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(res["uptime_before"], "|", res["uptime_after"])


if __name__ == "__main__":
    main()
