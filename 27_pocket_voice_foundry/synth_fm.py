r"""Text -> wav with the follow-up's flow-matching TTS (reloads checkpoints/fm/*; sampler = the checkpoint's val choice).

  ../.venv/bin/python synth_fm.py "Dr. Smith paid \$15 on March 3rd, 1963." --out media/fm_synth.wav
  ../.venv/bin/python synth_fm.py "..." --ckpt checkpoints/fm/tts_fm_ft --temp 0.8 --seed 3
"""
import argparse, os
from model_fm import cap_memory
from eval_fm import FM
from synth import NeuralVocoder, write_wav
from text import normalize

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    cap_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "fm", "tts_fm_lj"))
    ap.add_argument("--vocoder", default=os.path.join(ROOT, "checkpoints", "vocoder28", "vocoder.safetensors"))
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--n_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "media", "fm_synth.wav"))
    a = ap.parse_args()
    print("normalized:", normalize(a.text))
    tts = FM(a.ckpt, a.n_steps, a.temp)
    y = NeuralVocoder(a.vocoder)(tts(a.text, seed=a.seed))
    write_wav(a.out, y)
    print(f"wrote {a.out} ({len(y) / 16000:.2f} s; {tts.solver} {tts.n_steps} steps, temp {tts.temp})")


if __name__ == "__main__":
    main()
