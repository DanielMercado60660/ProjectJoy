"""Qwen3-TTS-Tokenizer-12Hz I/O through mlx-audio. Runs ONLY in ../.venv-audio (mlx-audio 0.6.3, MLX 0.32.4).

Qwen/Qwen3-TTS-Tokenizer-12Hz (Apache-2.0): 24 kHz, 12.5 frames/s, 2,048-entry codebooks, 16 usable codebooks
(1 semantic + 15 residual). Loading copies mlx-audio's own Qwen3-TTS loader (qwen3_tts.py, speech tokenizer branch).

  ../.venv-audio/bin/python qwen_io.py roundtrip --ids ids.txt --out dir --nq 4 8 16     # LJ ids -> dir/nq<k>/<id>.wav
"""
import argparse, glob, json, os, time
import numpy as np
import mlx.core as mx
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
LJ = os.path.join(HERE, "..", "data", "ljspeech", "LJSpeech-1.1", "wavs")
SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*"))[0]
SR = 24000


def load_tokenizer():
    from mlx_audio.tts.models.qwen3_tts.config import (Qwen3TTSTokenizerConfig, Qwen3TTSTokenizerDecoderConfig,
                                                       Qwen3TTSTokenizerEncoderConfig, filter_dict_for_dataclass)
    from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import Qwen3TTSSpeechTokenizer
    d = json.load(open(os.path.join(SNAP, "config.json")))
    dec = Qwen3TTSTokenizerDecoderConfig(**filter_dict_for_dataclass(Qwen3TTSTokenizerDecoderConfig, d["decoder_config"]))
    enc = Qwen3TTSTokenizerEncoderConfig(**filter_dict_for_dataclass(Qwen3TTSTokenizerEncoderConfig, d["encoder_config"]))
    cfg = Qwen3TTSTokenizerConfig(encoder_config=enc, decoder_config=dec)
    for k, v in d.items():
        if k not in ("decoder_config", "encoder_config") and hasattr(cfg, k):
            setattr(cfg, k, v)
    tok = Qwen3TTSSpeechTokenizer(cfg)
    w = {}
    for f in glob.glob(os.path.join(SNAP, "*.safetensors")):
        w.update(mx.load(f))
    tok.load_weights(list(Qwen3TTSSpeechTokenizer.sanitize(w).items()), strict=False)
    mx.eval(tok.parameters())
    tok.eval()
    q = tok.encoder_model.quantizer
    for layer in list(q.rvq_first.vq.layers) + list(q.rvq_rest.vq.layers):
        layer.codebook.update_in_place()
    return tok


def load_lj24(cid):
    x, sr = sf.read(os.path.join(LJ, cid + ".wav"), dtype="float32")
    assert sr == 22050
    return resample_poly(x, 160, 147).astype(np.float32)


def cmd_roundtrip(a):
    tok = load_tokenizer()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    for nq in a.nq:
        os.makedirs(os.path.join(a.out, f"nq{nq}"), exist_ok=True)
    te = td = 0.0
    shape = None
    for cid in ids:
        x = load_lj24(cid)
        t0 = time.perf_counter()
        codes = tok.encode(mx.array(x)[None, None])          # (1, 16, T)
        mx.eval(codes)
        te += time.perf_counter() - t0
        shape = shape or tuple(codes.shape)
        for nq in a.nq:
            c = mx.transpose(codes[:, :nq], (0, 2, 1))         # (1, T, nq)
            t1 = time.perf_counter()
            wav, _ = tok.decode(c)
            mx.eval(wav)
            td += time.perf_counter() - t1
            sf.write(os.path.join(a.out, f"nq{nq}", cid + ".wav"), np.array(wav[0], np.float32), SR)
    print(json.dumps({"clips": len(ids), "codes_shape": shape, "encode_s": round(te, 2), "decode_s": round(td, 2)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("roundtrip"); r.add_argument("--ids", required=True); r.add_argument("--out", required=True)
    r.add_argument("--nq", type=int, nargs="+", default=[4, 8, 16])
    a = ap.parse_args()
    {"roundtrip": cmd_roundtrip}[a.cmd](a)


if __name__ == "__main__":
    main()
