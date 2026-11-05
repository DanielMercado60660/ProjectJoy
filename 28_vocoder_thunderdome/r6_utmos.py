"""UTMOS22-strong naturalness proxy (Saeki et al. 2022, VoiceMOS 2022 winner), modelled on 27's mos.py but loading only
from the local torch.hub cache: the repo code from ~/.cache/torch/hub/tarepan_SpeechMOS_v1.2.0 (source="local") and the
weights from ~/.cache/torch/hub/checkpoints/utmos22_strong_step7459_v1.pt (downloaded 2026-10-31 with the owner's
approval for 27). Network access is blocked inside this process: any urllib / torch.hub download raises.
Runs in ../.venv-sac (torch lives there; MLX nets are not involved). CPU, 4 threads.

  ../.venv-sac/bin/python r6_utmos.py --wavdir results_r6/wavs/test27/gan --out results_r6/mos/test27_gan.json
  ../.venv-sac/bin/python r6_utmos.py --jobs jobs.json      # [{"wavdir": ..., "out": ...}, ...] in one model load
Each wav is scored on its own at 16 kHz (resampled with resample_poly if needed). Output: per-clip scores, mean and a
bootstrap 95% CI of the mean (2,000 resamples, seed 1), as in 27's mos.py. Paired comparisons: r6_stats.py.
"""
import argparse, glob, json, os, time
from math import gcd
import numpy as np

HUB = os.path.expanduser("~/.cache/torch/hub")
REPO = os.path.join(HUB, "tarepan_SpeechMOS_v1.2.0")
WEIGHTS = os.path.join(HUB, "checkpoints", "utmos22_strong_step7459_v1.pt")


def block_network():
    import urllib.request, socket
    def no(*a, **k):
        raise RuntimeError("r6_utmos: network access attempted; refusing (load from local cache only)")
    urllib.request.urlopen = no
    socket.create_connection = no
    import torch.hub
    torch.hub.download_url_to_file = no
    torch.hub.load_state_dict_from_url = no


def load_model(threads=4):
    import torch
    torch.set_num_threads(threads)
    assert os.path.isdir(REPO) and os.path.isfile(WEIGHTS), "UTMOS cache missing; not downloading"
    block_network()
    model = torch.hub.load(REPO, "utmos22_strong", source="local", pretrained=False, trust_repo=True)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    return model


def score_dir(model, wavdir, out, pattern="*.wav"):
    import torch
    import soundfile as sf
    from scipy.signal import resample_poly
    files = sorted(glob.glob(os.path.join(wavdir, pattern)))
    assert files, f"no wavs in {wavdir}"
    rows, t0 = [], time.time()
    with torch.no_grad():
        for f in files:
            x, sr = sf.read(f, dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            if sr != 16000:
                g = gcd(16000, sr); x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
            rows.append({"id": os.path.basename(f)[:-4], "mos": round(float(model(torch.from_numpy(x)[None], 16000)[0]), 4)})
    v = np.array([r["mos"] for r in rows])
    rng = np.random.default_rng(1)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    res = {"n": len(rows), "mos": round(float(v.mean()), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3),
           round(float(np.percentile(bs, 97.5)), 3)], "wall_s": round(time.time() - t0, 1), "wavdir": os.path.abspath(wavdir),
           "model": "tarepan/SpeechMOS:v1.2.0 utmos22_strong (local cache)", "clips": rows}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "clips"}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavdir")
    ap.add_argument("--out")
    ap.add_argument("--jobs", help="json list of {wavdir, out}")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()
    model = load_model(a.threads)
    jobs = json.load(open(a.jobs)) if a.jobs else [{"wavdir": a.wavdir, "out": a.out}]
    for j in jobs:
        if a.skip_existing and os.path.exists(j["out"]):
            continue
        score_dir(model, j["wavdir"], j["out"])


if __name__ == "__main__":
    main()
