"""Speed bench: first-audio latency and real-time factor (RTF) per (acoustic, vocoder) pipeline.

UNDER LOAD: other builders train on this Mac at the same time; `uptime` is recorded before and after.
  acoustic: tts (MLX GPU) | baseline (numpy CPU);  vocoder: neural (MLX GPU) | gl (Griffin-Lim 64, numpy CPU)
  latency_full   = text -> normalized -> mel -> whole waveform
  latency_first  = text -> mel -> first 1.0 s of audio (vocoder run on the first 63 frames + 8 frames lookahead);
                   the acoustic model is non-autoregressive, so it always produces the whole mel first
  RTF            = latency_full / audio seconds (lower is faster; < 1 is faster than real time)
  ../.venv/bin/python bench.py --out results/bench.json
"""
import argparse, json, os, subprocess, time
import numpy as np
import mlx.core as mx
from synth import TTS, BaselineTTS, NeuralVocoder, gl
from text import normalize, encode

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "bench.json"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    a = ap.parse_args()
    up0 = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    z = np.load(os.path.join(a.cache, "test.npz"))
    texts = [str(t) for t in z["texts"][: a.n]]
    tts, bl, voc = TTS(), BaselineTTS(), NeuralVocoder()
    acoustic = {"tts": lambda t: tts.mel_from_tokens(encode(normalize(t)))[0],
                "baseline": lambda t: bl.mel_from_tokens(encode(normalize(t)))[0]}
    vocs = {"neural": voc, "gl": gl}
    for t in texts[:2]:  # warm-up (MLX kernels, numba)
        for f in acoustic.values():
            m = f(t)
            voc(m); voc(m[:71])
    res = {}
    for an, af in acoustic.items():
        for vn, vf in vocs.items():
            lat_full, lat_first, rtf, t_ac = [], [], [], []
            for t in texts:
                best = None
                for _ in range(a.reps if vn == "neural" else 1):
                    s0 = time.perf_counter(); m = af(t); s1 = time.perf_counter()
                    y1 = vf(m[:71]); s2 = time.perf_counter()
                    y = vf(m); s3 = time.perf_counter()
                    cur = (s1 - s0, (s1 - s0) + (s2 - s1), (s1 - s0) + (s3 - s2), len(y) / 16000)
                    best = cur if best is None or cur[2] < best[2] else best
                t_ac.append(best[0]); lat_first.append(best[1]); lat_full.append(best[2]); rtf.append(best[2] / best[3])
            res[f"{an}/{vn}"] = {"acoustic_s_median": float(np.median(t_ac)), "first_audio_s_median": float(np.median(lat_first)),
                                 "full_s_median": float(np.median(lat_full)), "rtf_median": float(np.median(rtf)),
                                 "rtf_max": float(np.max(rtf))}
            print(f"{an}/{vn}", res[f"{an}/{vn}"], flush=True)
    up1 = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    out = {"label": "UNDER LOAD (other builders training concurrently)", "uptime_before": up0, "uptime_after": up1,
           "n_sentences": a.n, "reps_neural": a.reps,
           "device": "MLX default device (GPU) for tts/neural; numpy CPU for baseline/gl", "results": res}
    json.dump(out, open(a.out, "w"), indent=1)
    print(up0, "|", up1)


if __name__ == "__main__":
    main()
