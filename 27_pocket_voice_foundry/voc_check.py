"""Does 28's LJ-only vocoder render *other* speakers' mels acceptably? Real clips -> 27's mel -> vocoder28 -> wav, vs the
same real clip resampled to 16 kHz. 20 clips per source, chosen by a seeded draw: LibriTTS-R held-out val speakers (unseen
by every model here) and Hi-Fi TTS (spread over its 10 speakers), plus LJ val as the in-domain reference.

  ../bin/gpu_run ../.venv/bin/python voc_check.py && ./score_fm.sh vc_lj_real vc_lj_voc vc_libritts_real vc_libritts_voc vc_hifi_real vc_hifi_voc
"""
import json, os
import numpy as np
from model_fm import cap_memory
from synth import NeuralVocoder, write_wav


ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    cap_memory()
    voc = NeuralVocoder()
    rng = np.random.default_rng(0)
    for src, split in (("lj", "val"), ("libritts", "val"), ("hifi", "train")):
        d = os.path.join(ROOT, "cache2", src)
        z = np.load(os.path.join(d, "index.npz"))
        mels = np.memmap(os.path.join(d, "mels.f16"), np.float16, "r").reshape(-1, 80)
        cand = np.nonzero((z["splits"] == split) & (z["lens"] > 2 * 62.5))[0]
        if src == "hifi":  # 2 per speaker
            sel = np.concatenate([rng.choice(cand[z["speakers"][cand] == s], 2, replace=False) for s in sorted(set(z["speakers"][cand]))])
        else:
            sel = rng.choice(cand, 20, replace=False)
        lists = {"real": [], "voc": []}
        for i in sel:
            m = np.array(mels[z["offs"][i]:z["offs"][i + 1]], np.float32)
            uid = str(z["ids"][i])
            y = voc(m)
            lists["voc"].append({"id": uid, "wav": os.path.join(ROOT, "runs", "fm", "gen", f"vc_{src}_voc", uid + ".wav"),
                                 "ref": str(z["texts"][i]), "speaker": str(z["speakers"][i])})
            write_wav(lists["voc"][-1]["wav"], y)
            lists["real"].append({"id": uid, "wav": _real_path(src, uid), "ref": str(z["texts"][i]), "speaker": str(z["speakers"][i])})
        for k, l in lists.items():
            json.dump(l, open(os.path.join(ROOT, "results", "fm", "lists", f"vc_{src}_{k}.json"), "w"), indent=0)
        print(src, len(sel), "clips", sorted(set(x["speaker"] for x in lists["real"]))[:12], flush=True)


_HIFI = None


def _real_path(src, uid):
    """The original recording (any sample rate; judge and mos.py resample)."""
    global _HIFI
    if src == "lj":
        return os.path.join(ROOT, "..", "data", "ljspeech", "LJSpeech-1.1", "wavs", uid + ".wav")
    if src == "libritts":
        spk, book = uid.split("_")[:2]
        return os.path.join(ROOT, "..", "data", "libritts_r", "extracted", "LibriTTS_R", "train-clean-100", spk, book, uid + ".wav")
    if _HIFI is None:
        from prep_multi import rows_hifi
        _HIFI = {r["id"]: r["path"] for r in rows_hifi()}
    return _HIFI[uid]


if __name__ == "__main__":
    main()
