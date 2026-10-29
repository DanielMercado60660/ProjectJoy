"""Text -> waveform with any (acoustic, vocoder) pair; reloads the saved checkpoints.

  ../.venv/bin/python synth.py "Doctor Smith paid $15 on March 3rd, 1963." --system tts --vocoder neural --out media/x.wav
systems: tts (learned acoustic model), baseline (template concatenation); vocoders: neural (28's), gl (Griffin-Lim 64)
"""
import argparse, json, os
import numpy as np
import mlx.core as mx

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_TTS = os.path.join(ROOT, "checkpoints", "acoustic", "tts.safetensors")
CKPT_BL = os.path.join(ROOT, "checkpoints", "baseline", "baseline.npz")
CKPT_VOC = os.path.join(ROOT, "checkpoints", "vocoder28", "vocoder.safetensors")


class TTS:
    def __init__(self, path=CKPT_TTS, pace=None, rounding=None):
        from model import load_acoustic
        self.model, self.cfg = load_acoustic(path)
        self.mean = np.array(self.cfg["mel_mean"], np.float32); self.std = np.array(self.cfg["mel_std"], np.float32)
        self.pace = pace if pace is not None else self.cfg.get("pace", 1.0)
        self.rounding = rounding or self.cfg.get("rounding", "round")

    def mel_from_tokens(self, tok):
        from model import infer
        m, d = infer(self.model, mx.array(np.asarray(tok, np.int32)[None]), self.pace, self.rounding)
        return m * self.std + self.mean, d

    def __call__(self, text):
        from text import normalize, encode
        return self.mel_from_tokens(encode(normalize(text)))[0]


class BaselineTTS:
    def __init__(self, path=CKPT_BL):
        from baseline import Baseline
        self.bl = Baseline.load(path)

    def mel_from_tokens(self, tok):
        return self.bl.synth(np.asarray(tok))

    def __call__(self, text):
        from text import normalize, encode
        return self.mel_from_tokens(encode(normalize(text)))[0]


class NeuralVocoder:
    def __init__(self, path=CKPT_VOC):
        from vocoder_model import load_vocoder
        self.model, self.cfg = load_vocoder(path)

    def __call__(self, mel):
        y = self.model(mx.array(np.asarray(mel, np.float32)[None]))
        mx.eval(y)
        return np.array(y[0])


def gl(mel, n_iter=64):
    from griffin_lim import griffin_lim
    return griffin_lim(np.asarray(mel, np.float32), n_iter=n_iter, seed=0)


def write_wav(path, y, sr=16000):
    import soundfile as sf
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, np.clip(y, -1, 1), sr, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--system", default="tts", choices=["tts", "baseline"])
    ap.add_argument("--vocoder", default="neural", choices=["neural", "gl"])
    ap.add_argument("--tts_ckpt", default=CKPT_TTS)
    ap.add_argument("--out", default=os.path.join(ROOT, "media", "synth.wav"))
    a = ap.parse_args()
    from text import normalize
    print("normalized:", normalize(a.text))
    mel = (TTS(a.tts_ckpt) if a.system == "tts" else BaselineTTS())(a.text)
    y = NeuralVocoder()(mel) if a.vocoder == "neural" else gl(mel)
    write_wav(a.out, y)
    print(f"wrote {a.out} ({len(y) / 16000:.2f} s)")


if __name__ == "__main__":
    main()
