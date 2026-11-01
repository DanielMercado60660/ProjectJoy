"""XY-Tokenizer I/O (Fudan / OpenMOSS, ACL 2026). Runs ONLY in ../.venv-xy (torch 2.15, transformers 4.57).

fnlp/XY-Tokenizer-TTSD-V0-hf (Apache-2.0): 16 kHz in, 24 kHz out, 12.5 frames/s, 8-layer RVQ x 1,024 (~1 kbps).
Loaded with trust_remote_code from the HF snapshot (modeling/feature-extraction code read before use: no network,
subprocess, eval/exec or pickle calls). CPU only, to leave the GPU to MLX training.

  ../.venv-xy/bin/python xy_io.py roundtrip --ids ids.txt --out dir --nq 4 8      # LJ ids -> dir/nq<k>/<id>.wav (24 kHz)
"""
import argparse, glob, json, os, time
import numpy as np
import soundfile as sf
import torch
torch.set_num_threads(4)  # lab rule: <= 4 CPU threads per job (uncapped torch starved MLX training)
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
LJ = os.path.join(HERE, "..", "data", "ljspeech", "LJSpeech-1.1", "wavs")
SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--fnlp--XY-Tokenizer-TTSD-V0-hf/snapshots/*"))[0]


def load():
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(SNAP, trust_remote_code=True)
    codec = AutoModel.from_pretrained(SNAP, trust_remote_code=True).eval()
    return fe, codec


def load_lj16(cid):
    x, sr = sf.read(os.path.join(LJ, cid + ".wav"), dtype="float32")
    assert sr == 22050
    return resample_poly(x, 320, 441).astype(np.float32)


@torch.no_grad()
def cmd_roundtrip(a):
    fe, codec = load()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    for nq in a.nq:
        os.makedirs(os.path.join(a.out, f"nq{nq}"), exist_ok=True)
    te = td = 0.0
    shape = None
    for cid in ids:
        x = load_lj16(cid)
        t0 = time.perf_counter()
        feats = fe(torch.from_numpy(x)[None], sampling_rate=16000, return_attention_mask=True, return_tensors="pt")
        code = codec.encode(feats)["audio_codes"]
        te += time.perf_counter() - t0
        shape = shape or [tuple(c.shape) for c in (code if isinstance(code, (list, tuple)) else [code])]
        for nq in a.nq:
            c = [cc[:nq] for cc in code] if isinstance(code, (list, tuple)) else code[:nq]
            t1 = time.perf_counter()
            out = codec.decode(c, overlap_seconds=10)
            td += time.perf_counter() - t1
            wav = out["audio_values"] if isinstance(out, dict) else out
            w = wav[0] if isinstance(wav, (list, tuple)) else wav
            sf.write(os.path.join(a.out, f"nq{nq}", cid + ".wav"), w.squeeze().cpu().numpy().astype(np.float32), 24000)
    print(json.dumps({"clips": len(ids), "codes_shape": shape, "encode_s": round(te, 2), "decode_s": round(td, 2)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("roundtrip"); r.add_argument("--ids", required=True); r.add_argument("--out", required=True)
    r.add_argument("--nq", type=int, nargs="+", default=[4, 8])
    a = ap.parse_args()
    {"roundtrip": cmd_roundtrip}[a.cmd](a)


if __name__ == "__main__":
    main()
