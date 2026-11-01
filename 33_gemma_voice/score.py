"""Intelligibility scoring with project 27's CTC recognizer proxy (copied: ctc.py, ctc_blocks.py, mel.py).

Audio path (protocol): 24 kHz wav -> resample_poly(2, 3) -> 16 kHz -> log_mel -> greedy CTC -> corpus CER
(total character edits / total reference characters, 27's definition). Real recordings go 22.05 kHz ->
resample_poly(320, 441) -> 16 kHz, exactly as 27 did.

  ../.venv/bin/python score.py ceiling      # Mimi round-trip ceiling on val-100 and test-100 (needs ../.venv-moshi)
"""
import argparse, json, os, subprocess, sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ctc import load_recognizer, transcribe, edit_distance, ref_text  # noqa: E402
from mel import log_mel, load_ljspeech_16k  # noqa: E402

LJ = os.path.join(HERE, "..", "data", "ljspeech", "LJSpeech-1.1")
REC = os.path.join(HERE, "checkpoints", "ctc27", "ctc.safetensors")
CACHE27 = os.path.join(HERE, "..", "27_pocket_voice_foundry", "cache")  # read-only: split ids + 27's normalized text
EVAL27 = os.path.join(HERE, "..", "27_pocket_voice_foundry", "results", "eval_clips.jsonl")
MOSHI_PY = os.path.join(HERE, "..", ".venv-moshi", "bin", "python")

_rec = None


def recognizer():
    global _rec
    if _rec is None:
        _rec = load_recognizer(REC)
    return _rec


def wav24_to_mel(path):
    y, sr = sf.read(path, dtype="float32")
    assert sr == 24000, (path, sr)
    return log_mel(resample_poly(y, 2, 3).astype(np.float32))


def corpus_cer(mels, texts):
    m, cfg = recognizer()
    hyps = transcribe(m, cfg, mels)
    refs = [ref_text(t) for t in texts]
    edits = [edit_distance(h, r) for h, r in zip(hyps, refs)]
    return {"cer": float(sum(edits) / sum(len(r) for r in refs)),
            "clip_cer": [e / max(1, len(r)) for e, r in zip(edits, refs)], "hyps": hyps}


def eval_sets():
    """test-100 = 27's shipped eval clips; val-100 = seed-0 sample of 27's val split. Texts = 27's normalized text."""
    test = [json.loads(l) for l in open(EVAL27)]
    v = np.load(os.path.join(CACHE27, "val.npz"))
    pick = np.sort(np.random.default_rng(0).choice(len(v["ids"]), 100, replace=False))
    val = [{"id": str(v["ids"][i]), "text": str(v["texts"][i])} for i in pick]
    return {"val": [(c["id"], c["text"]) for c in val], "test": [(c["id"], c["text"]) for c in test]}


def cmd_ceiling(a):
    sets = eval_sets()
    out = {}
    for name, clips in sets.items():
        ids_file = os.path.join(a.out, f"{name}_ids.txt")
        os.makedirs(a.out, exist_ok=True)
        open(ids_file, "w").write("\n".join(c for c, _ in clips) + "\n")
        subprocess.run([MOSHI_PY, os.path.join(HERE, "mimi_io.py"), "roundtrip", "--ids", ids_file,
                        "--out", os.path.join(a.out, name), "--nq", *map(str, a.nq)], check=True)
        texts = [t for _, t in clips]
        res = {"real": corpus_cer([log_mel(load_ljspeech_16k(os.path.join(LJ, "wavs", c + ".wav"))) for c, _ in clips], texts)}
        for nq in a.nq:
            res[f"mimi_nq{nq}"] = corpus_cer([wav24_to_mel(os.path.join(a.out, name, f"nq{nq}", c + ".wav")) for c, _ in clips], texts)
        out[name] = {k: round(100 * v["cer"], 2) for k, v in res.items()}
        print(name, out[name], flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "ceiling_nq" + "_".join(map(str, a.nq)) + ".json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("ceiling"); c.add_argument("--nq", type=int, nargs="+", default=[8, 16, 32])
    c.add_argument("--out", default=os.path.join(HERE, "runs", "ceiling"))
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    {"ceiling": cmd_ceiling}[a.cmd](a)


if __name__ == "__main__":
    main()
