"""Phase 0: build the independent-judge lists (val-100 and test-100) for every system we already have.

  ../.venv/bin/python phase0_prep.py          # -> runs/phase0/lists/<split>_<system>.json, runs/voc28_real/<split>/*.wav
  then for each list: ../.venv-judge/bin/python judge.py --list <list> --out results/judge/<split>_<system>.json
Systems: real recording; real mel through 28's vocoder (the mel path's ceiling); Mimi round trips at 1/2/4/8/16/32
codebooks; our r3 (Gemma, 4 cb) and scratch (random init) at their val-chosen temperatures; project 27's TTS (test only,
27 saved only test wavs). References are project 27's normalized texts.
"""
import json, os, sys
import numpy as np
import mlx.core as mx
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score import eval_sets  # noqa: E402
from vocoder_model import load_vocoder  # noqa: E402

LJ = os.path.join(HERE, "..", "data", "ljspeech", "LJSpeech-1.1", "wavs")
P27 = os.path.join(HERE, "..", "27_pocket_voice_foundry")


def main():
    sets = eval_sets()
    voc, _ = load_vocoder(os.path.join(HERE, "checkpoints", "vocoder28", "vocoder.safetensors"))
    outdir = os.path.join(HERE, "runs", "phase0", "lists")
    os.makedirs(outdir, exist_ok=True)
    for split, clips in sets.items():
        z = dict(np.load(os.path.join(P27, "cache", f"{split}.npz")))
        idx = {str(c): i for i, c in enumerate(z["ids"])}
        vdir = os.path.join(HERE, "runs", "voc28_real", split)
        os.makedirs(vdir, exist_ok=True)
        for cid, _ in clips:
            i = idx[cid]
            m = z["mels"][z["offs"][i]:z["offs"][i + 1]].astype(np.float32)
            y = voc(mx.array(m[None])); mx.eval(y)
            sf.write(os.path.join(vdir, cid + ".wav"), np.array(y[0], np.float32), 16000)
        systems = {"real": lambda c: os.path.join(LJ, c + ".wav"),
                   "voc28_real": lambda c, s=split: os.path.join(HERE, "runs", "voc28_real", s, c + ".wav")}
        for nq in (8, 16, 32):
            systems[f"mimi{nq}"] = lambda c, s=split, n=nq: os.path.join(HERE, "runs", "ceiling", s, f"nq{n}", c + ".wav")
        for nq in (1, 2, 4):
            systems[f"mimi{nq}"] = lambda c, s=split, n=nq: os.path.join(HERE, "runs", "ceiling_low", s, f"nq{n}", c + ".wav")
        tag = {"val": ("val100_t0.2", "val100_t0.4"), "test": ("test", "test")}[split]
        systems["gemma_r3"] = lambda c, t=tag[0]: os.path.join(HERE, "runs", "r3", f"eval_{t}", "wavs", c + ".wav")
        systems["scratch"] = lambda c, t=tag[1]: os.path.join(HERE, "runs", "scratch", f"eval_{t}", "wavs", c + ".wav")
        if split == "test":
            systems["p27_tts"] = lambda c: os.path.join(P27, "results", "wavs", f"{c}_tts_neural.wav")
        for name, f in systems.items():
            items = [{"id": c, "wav": f(c), "ref": ref} for c, ref in clips]
            missing = sum(not os.path.exists(it["wav"]) for it in items)
            json.dump(items, open(os.path.join(outdir, f"{split}_{name}.json"), "w"), indent=0)
            print(split, name, "missing", missing)


if __name__ == "__main__":
    main()
