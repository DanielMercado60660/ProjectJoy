"""Round 6 speed: real-time factor (synthesis time / audio time, median of 3 per clip, summed over 28's 24 held-out clips)
for GL 64 (numpy CPU, 4 threads) and a generator checkpoint on MLX GPU and MLX CPU. Records uptime (load) before/after.
  OMP_NUM_THREADS=4 ../.venv/bin/python r6_bench.py --ckpt checkpoints/r6_gan.safetensors --out results_r6/bench_rtf.json
"""
import argparse, json, os, subprocess, time
import numpy as np
import mlx.core as mx
from griffin_lim import griffin_lim
from r6_vocoder import load, vocode

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(os.path.join(ROOT, "cache", "test.npz"))
    mels = [z[f"mel_{i}"] for i in range(len(z["ids"]))]
    secs = sum((len(m) - 1) * 256 for m in mels) / 16000
    up0 = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    G = load(a.ckpt)
    res = {}

    def run(name, fn):
        fn(mels[0][:64])
        tot = 0.0
        for m in mels:
            ts = []
            for _ in range(3):
                t = time.perf_counter(); fn(m); ts.append(time.perf_counter() - t)
            tot += float(np.median(ts))
        res[name] = {"rtf": tot / secs, "x_realtime": secs / tot}
        print(name, res[name], flush=True)

    run("gl64_numpy_cpu", lambda m: griffin_lim(m, n_iter=64, seed=0))
    mx.set_default_device(mx.gpu); run("generator_mlx_gpu", lambda m: vocode(G, m))
    mx.set_default_device(mx.cpu); G = load(a.ckpt); run("generator_mlx_cpu", lambda m: vocode(G, m))
    res.update({"ckpt": a.ckpt, "audio_s": secs, "uptime_before": up0,
                "uptime_after": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()})
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
