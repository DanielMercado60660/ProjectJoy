"""Round 6 diagnosis: is a generator's gap to the recording in its magnitudes or in its phase?
For each val100 clip, take the generator's per-frame output (log-magnitude, phase) on the analysis STFT grid and swap one
half with the recording's own STFT (mel.stft, the same n_fft 1024 / hop 256 / centred frames), then inverse-STFT:
  <tag>_Gmag_GTphase   generator magnitude + true phase
  <tag>_GTmag_Gphase   true magnitude + generator phase
  oracle_GTmag_GTphase true magnitude + true phase (sanity: should equal the recording)
  ../.venv/bin/python r6_oracle.py --ckpt checkpoints/followup_val3k/best.safetensors --tag old
Writes results_r6/wavs/val100/<system>/ + results_r6/metrics/val100/<system>.json (same format as r6_eval.py);
UTMOS them with r6_utmos.py.
With --gl N it instead writes <tag>_Gmag_GL<N>: the generator's magnitude with N fast Griffin-Lim iterations (momentum
0.99, as griffin_lim.py) started from the generator's own phase -- a deployable hybrid (no ground truth used).
"""
import argparse, json, os
import numpy as np
import mlx.core as mx
import soundfile as sf
from mel import stft, istft
from metrics import mrstft, mel_l1, pitch_metrics
from r6_vocoder import load

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results_r6")


def gl_refine(mag, ph0, length, n_iter, momentum=0.99):
    """Fast Griffin-Lim on a given magnitude, initialised with a given phase; returns the waveform."""
    T = len(mag)
    angles = np.exp(1j * ph0)
    prev = np.zeros_like(mag, dtype=np.complex128)
    x = istft(mag * angles, length)
    for _ in range(n_iter):
        reb = stft(x)[:T]
        angles = reb - (momentum / (1 + momentum)) * prev
        prev = reb
        angles = angles / np.maximum(np.abs(angles), 1e-16)
        x = istft(mag * angles, length)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", default="val100")
    ap.add_argument("--gl", type=int, default=0)
    a = ap.parse_args()
    z = np.load(os.path.join(ROOT, "cache_r6", f"{a.set}.npz"))
    G = load(a.ckpt)
    systems = {f"{a.tag}_Gmag_GL{a.gl}": []} if a.gl else \
        {f"{a.tag}_Gmag_GTphase": [], f"{a.tag}_GTmag_Gphase": [], "oracle_GTmag_GTphase": []}
    for s in systems:
        os.makedirs(os.path.join(R, "wavs", a.set, s), exist_ok=True)
    for i, cid in enumerate(z["ids"]):
        x, m = z[f"wav_{i}"], z[f"mel_{i}"]
        lm, ph = G.head_out(mx.array(m[None])); mx.eval(lm, ph)
        gmag = np.minimum(np.exp(np.array(lm[0])), 1e2); gph = np.array(ph[0])
        if a.gl:
            outs = {f"{a.tag}_Gmag_GL{a.gl}": gl_refine(gmag, gph, len(x), a.gl)}
        else:
            S = stft(x)[: len(m)]
            outs = {f"{a.tag}_Gmag_GTphase": gmag * np.exp(1j * np.angle(S)),
                f"{a.tag}_GTmag_Gphase": np.abs(S) * np.exp(1j * gph),
                "oracle_GTmag_GTphase": S}
        for s, spec in outs.items():
            y = spec if spec.ndim == 1 else istft(spec, len(x))
            sf.write(os.path.join(R, "wavs", a.set, s, f"{cid}.wav"), y, 16000, subtype="FLOAT")
            systems[s].append({"id": str(cid), "seconds": len(x) / 16000, "synth_s": 0.0, "mrstft": mrstft(y, x),
                               "mel_l1": mel_l1(y, x), **pitch_metrics(y, x)})
    for s, rows in systems.items():
        summ = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("id", "seconds", "synth_s")}
        os.makedirs(os.path.join(R, "metrics", a.set), exist_ok=True)
        json.dump({"set": a.set, "system": s, "ckpt": a.ckpt, "n": len(rows), "summary": summ, "per_clip": rows},
                  open(os.path.join(R, "metrics", a.set, f"{s}.json"), "w"), indent=1)
        print(s, json.dumps({k: round(v, 4) for k, v in summ.items()}), flush=True)


if __name__ == "__main__":
    main()
