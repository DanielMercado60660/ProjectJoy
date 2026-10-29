"""Choose between finalized acoustic checkpoints on the *validation* split (never test): recognizer CER on the predicted mel,
DTW log-mel L1 against the real recording, and total-length ratio, on the first 100 val clips.

  ../.venv/bin/python val_compare.py runs/run1_tts_export/tts.safetensors runs/run2_tts_export/tts.safetensors
  # -> results/val_compare.json
"""
import json, os, sys
import numpy as np
from mas import dtw_cost
from ctc import load_recognizer, transcribe, edit_distance, ref_text
from synth import TTS
from text import encode

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    paths = sys.argv[1:]
    z = np.load(os.path.join(ROOT, "cache", "val.npz"))
    n = 100
    mels = [z["mels"][z["offs"][i]:z["offs"][i + 1]].astype(np.float32) for i in range(n)]
    texts = [str(t) for t in z["texts"][:n]]
    refs = [ref_text(t) for t in texts]
    rec, rcfg = load_recognizer(os.path.join(ROOT, "checkpoints", "ctc", "ctc.safetensors"))
    out = {}
    for p in paths:
        tts = TTS(p)
        pred = [tts.mel_from_tokens(encode(t))[0] for t in texts]
        hyps = transcribe(rec, rcfg, pred)
        out[p] = {"step": tts.cfg["step"],
                  "val_cer": sum(edit_distance(h, r) for h, r in zip(hyps, refs)) / sum(len(r) for r in refs),
                  "val_dtw_logmel_l1": float(np.mean([dtw_cost(a.astype(np.float64), b.astype(np.float64))[0] for a, b in zip(pred, mels)])),
                  "val_len_ratio": float(np.exp(np.mean([np.log(len(a) / len(b)) for a, b in zip(pred, mels)]))),
                  "val_hist_last": tts.cfg.get("val_hist", [])[-1:]}
        print(p, json.dumps({k: v for k, v in out[p].items() if k != "val_hist_last"}), flush=True)
    json.dump({"n_val": n, "candidates": out}, open(os.path.join(ROOT, "results", "val_compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
