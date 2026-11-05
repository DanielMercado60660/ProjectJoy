"""16 kHz LJSpeech waveform cache for the vocoder GAN fine-tune: the exact waveform mel.py analysed for cache2/lj
(28's resample_poly(320, 441), truncated to a multiple of HOP), stored as raw int16 + per-clip sample offsets, in
cache2/lj/ order. Frame t of cache2/lj mels is centered on sample t*HOP of the waveform here.

  ../.venv/bin/python prep_wav16.py      # -> cache2/lj/wav16.i16, cache2/lj/wav16_offs.npy (~2.7 GB, ~1 min)
"""
import os
from multiprocessing import Pool
import numpy as np
from mel import load_ljspeech_16k, HOP

ROOT = os.path.dirname(os.path.abspath(__file__))
LJ = os.path.join(ROOT, "..", "data", "ljspeech", "LJSpeech-1.1", "wavs")


def _work(cid):
    x = load_ljspeech_16k(os.path.join(LJ, cid + ".wav"))
    x = x[: (len(x) // HOP) * HOP]
    return np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)


def main():
    d = os.path.join(ROOT, "cache2", "lj")
    z = np.load(os.path.join(d, "index.npz"))
    offs = [0]
    with open(os.path.join(d, "wav16.i16"), "wb") as f, Pool(4) as p:
        for k, x in enumerate(p.imap(_work, [str(i) for i in z["ids"]], chunksize=16)):
            assert len(x) // HOP + 1 == z["lens"][k], (z["ids"][k], len(x), z["lens"][k])
            f.write(x.tobytes()); offs.append(offs[-1] + len(x))
    np.save(os.path.join(d, "wav16_offs.npy"), np.array(offs, np.int64))
    print("clips", len(offs) - 1, "hours", round(offs[-1] / 16000 / 3600, 2))


if __name__ == "__main__":
    main()
