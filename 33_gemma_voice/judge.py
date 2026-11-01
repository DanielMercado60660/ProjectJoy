"""Independent intelligibility judge: Whisper large-v3-turbo (MLX). Runs ONLY in ../.venv-judge.

  ../.venv-judge/bin/python judge.py --list list.json --out results/judge_<name>.json
list.json: [{"id": ..., "wav": path, "ref": reference text}, ...]  (any sample rate; Whisper resamples via ffmpeg)

Both hypothesis and reference get the same normalization: lowercase, numbers spelled out (num2words), "&" -> "and",
punctuation dropped except in-word apostrophes, whitespace collapsed. Scores: corpus CER and WER (total edits /
total reference characters or words), plus per-clip hypotheses. Decoding: language en, temperature 0, no previous-
text conditioning, so each clip is judged on its own.
"""
import argparse, json, os, re, time
import numpy as np

MODEL = "mlx-community/whisper-large-v3-turbo"
ABBR = {"mrs": "missus", "mr": "mister", "dr": "doctor", "st": "saint", "lt": "lieutenant", "sgt": "sergeant",
        "capt": "captain", "gov": "governor", "gen": "general", "jr": "junior", "sr": "senior"}
# project 27's normalizer expands these in the references; here only when written with a period ("Mr.")


def normalize(t):
    from num2words import num2words
    t = t.lower().replace("&", " and ").replace("%", " percent")
    t = re.sub(r"(\d),(\d)", r"\1\2", t)
    t = re.sub(r"\b(1[0-9]|20)\d\d\b", lambda m: " " + num2words(int(m.group()), to="year") + " ", t)  # years like LJ
    t = re.sub(r"\d+(\.\d+)?", lambda m: " " + num2words(float(m.group()) if "." in m.group() else int(m.group())) + " ", t)
    for k, v in ABBR.items():
        t = re.sub(rf"\b{k}\.", v + " ", t)
    t = t.replace("-", " ")
    t = re.sub(r"[^a-z' ]+", " ", t)
    t = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def edits(a, b):
    """Levenshtein distance between sequences a (hyp) and b (ref)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mlx_whisper
    items = json.load(open(a.list))
    ce = cn = we = wn = 0
    rows = []
    t0 = time.time()
    for it in items:
        if not os.path.exists(it["wav"]) or os.path.getsize(it["wav"]) < 100:
            hyp = ""
        else:
            hyp = mlx_whisper.transcribe(it["wav"], path_or_hf_repo=MODEL, language="en", temperature=0.0,
                                         condition_on_previous_text=False, verbose=None)["text"]
        h, r = normalize(hyp), normalize(it["ref"])
        c, w = edits(h, r), edits(h.split(), r.split())
        ce += c; cn += len(r); we += w; wn += len(r.split())
        rows.append({"id": it["id"], "hyp_raw": hyp.strip(), "hyp": h, "ref": r,
                     "cer": c / max(1, len(r)), "wer": w / max(1, len(r.split()))})
    out = {"n": len(rows), "cer": round(100 * ce / cn, 2), "wer": round(100 * we / wn, 2),
           "wall_s": round(time.time() - t0, 1), "model": MODEL, "clips": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "clips"}), flush=True)


if __name__ == "__main__":
    main()
