"""Naturalness proxy: UTMOS22-strong MOS predictor (Saeki et al. 2022, VoiceMOS 2022 winner), run in ../.venv-sac (which
already has torch + torchaudio; nothing installed). Model code + weights come from torch.hub
"tarepan/SpeechMOS:v1.2.0" (downloaded once into ~/.cache/torch/hub, approved by the user on 2026-10-31).

  ../.venv-sac/bin/python mos.py --list results/fm/lists/test_real.json --out results/fm/mos/test_real.json
list.json: the judge format, [{"id", "wav", "ref"}, ...]. Every wav is resampled to 16 kHz (the predictor's native rate)
with scipy resample_poly and scored on its own on CPU (keeps the GPU free). Output: per-clip scores, mean and a
bootstrap 95% CI of the mean (2,000 resamples, seed 1).
"""
import argparse, json, os, time
from math import gcd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import torch
    import soundfile as sf
    from scipy.signal import resample_poly
    torch.set_num_threads(a.threads)
    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    model.eval()
    items = json.load(open(a.list))
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for it in items:
            x, sr = sf.read(it["wav"], dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                g = gcd(16000, sr)
                x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
            s = float(model(torch.from_numpy(x)[None], 16000)[0])
            rows.append({"id": it["id"], "mos": round(s, 4)})
    v = np.array([r["mos"] for r in rows])
    rng = np.random.default_rng(1)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    out = {"n": len(rows), "mos": round(float(v.mean()), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3),
           round(float(np.percentile(bs, 97.5)), 3)], "wall_s": round(time.time() - t0, 1),
           "model": "tarepan/SpeechMOS:v1.2.0 utmos22_strong", "clips": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "clips"}), flush=True)


if __name__ == "__main__":
    main()
