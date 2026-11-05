"""Check that Hugging Face's PyTorch Mimi (kyutai/mimi) reads and writes 33's 'space c' latents like moshi_mlx did.

  ../../.venv-xy/bin/python mimi_parity.py prep  --out /tmp/x      # pick LJ val clips, write their stored latents
  ../../.venv-moshi/bin/python ../mimi_io.py decode-latents --lat /tmp/x/lat.npz --out /tmp/x/mlx
  ../../.venv-xy/bin/python mimi_parity.py check --out /tmp/x      # HF decode vs MLX decode, HF encode vs stored latent
"""
import argparse, json, os, sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
D33 = os.path.join(HERE, "..")
LJ = os.path.join(D33, "..", "data", "ljspeech", "LJSpeech-1.1")


def to_c(model, wav24):
    """24 kHz mono float32 -> (frames, 512) space-c latent with transformers' MimiModel."""
    import torch
    x = torch.from_numpy(wav24)[None, None]
    with torch.no_grad():
        e = model.encoder(x)
        e = model.encoder_transformer(e.transpose(1, 2))[0].transpose(1, 2)
        z = model.downsample(e)
        q = model.quantizer
        c = torch.cat([q.semantic_residual_vector_quantizer.input_proj(z),
                       q.acoustic_residual_vector_quantizer.input_proj(z)], 1)
    return c[0].T.numpy()


def from_c(model, c):
    """(frames, 512) space-c latent -> 24 kHz wav (codebook lookup skipped)."""
    import torch
    q = model.quantizer
    x = torch.from_numpy(np.ascontiguousarray(c.T, dtype=np.float32))[None]
    with torch.no_grad():
        e = q.semantic_residual_vector_quantizer.output_proj(x[:, :256]) + \
            q.acoustic_residual_vector_quantizer.output_proj(x[:, 256:])
        e = model.upsample(e)
        e = model.decoder_transformer(e.transpose(1, 2))[0].transpose(1, 2)
        y = model.decoder(e)
    return y[0, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "check"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z = np.load(os.path.join(D33, "data", "codes32.npz"))
    idx = [i for i in range(len(z["splits"])) if z["splits"][i] == "val"][: a.n]
    names = [str(z["ids"][i]) for i in idx]
    if a.cmd == "prep":
        store = np.load(os.path.join(D33, "data", "lj_lat.npy"), mmap_mode="r")
        lats = [np.asarray(store[z["offs"][i]: z["offs"][i] + z["lens"][i]], np.float32) for i in idx]
        np.savez(os.path.join(a.out, "lat.npz"), names=np.array(names), lens=np.array([len(l) for l in lats]),
                 lat=np.concatenate(lats))
        print("wrote", names)
        return
    from transformers import MimiModel
    model = MimiModel.from_pretrained("kyutai/mimi").eval()
    lz = np.load(os.path.join(a.out, "lat.npz"))
    offs = np.concatenate([[0], np.cumsum(lz["lens"])])
    res = []
    for k, name in enumerate(names):
        stored = lz["lat"][offs[k]:offs[k + 1]]
        y_hf = from_c(model, stored)
        y_mlx, _ = sf.read(os.path.join(a.out, "mlx", name + ".wav"), dtype="float32")
        n = min(len(y_hf), len(y_mlx))
        err = y_hf[:n] - y_mlx[:n]
        snr = 10 * np.log10((y_mlx[:n] ** 2).sum() / max((err ** 2).sum(), 1e-12))
        x, sr = sf.read(os.path.join(LJ, "wavs", name + ".wav"), dtype="float32")
        c_hf = to_c(model, resample_poly(x, 160, 147).astype(np.float32))
        m = min(len(c_hf), len(stored))
        rel = float(np.linalg.norm(c_hf[:m] - stored[:m]) / np.linalg.norm(stored[:m]))
        sf.write(os.path.join(a.out, f"hf_{name}.wav"), y_hf, 24000)
        res.append({"clip": name, "len_hf": len(y_hf), "len_mlx": len(y_mlx), "decode_snr_db": round(float(snr), 1),
                    "frames_hf": len(c_hf), "frames_stored": len(stored), "encode_rel_err": round(rel, 4)})
        print(json.dumps(res[-1]), flush=True)


if __name__ == "__main__":
    sys.exit(main())
