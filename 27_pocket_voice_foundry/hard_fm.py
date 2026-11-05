"""Hard-sentence set scored by the Whisper judge (results/fm/judge/hard_<name>.json): a hard word fails when its best
substring match in the judge-normalized transcript has character error above 1/3 (27's rule, eval.py), with the word
put through the judge's own normalizer. Whisper writes real spellings ("colonel" for a /kernel/ sound), so unlike 27's
CTC proxy this rewards correct pronunciation over spelling pronunciation. Caveat: Whisper writes some numbers as digits
("$15.50"), which the judge expands differently from 27's normalizer, so the number sentences carry scorer noise that is
the same for every system.

  ../.venv-judge/bin/python hard_fm.py hard_p27v1 hard_fm_ft ...     (needs num2words: judge venv)
"""
import json, os, re, sys

# --- PROVENANCE: copied verbatim from ../33_gemma_voice/judge.py (ABBR + normalize) on 2026-10-31, so the hard-word check
# uses exactly the judge's normalization without importing a sibling project. Needs num2words (present in ../.venv-judge).
ABBR = {"mrs": "missus", "mr": "mister", "dr": "doctor", "st": "saint", "lt": "lieutenant", "sgt": "sergeant",
        "capt": "captain", "gov": "governor", "gen": "general", "jr": "junior", "sr": "senior"}


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
# --- end copy ---

ROOT = os.path.dirname(os.path.abspath(__file__))


def semiglobal(word, hyp):
    prev = [0] * (len(hyp) + 1)
    for i in range(1, len(word) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (word[i - 1] != hyp[j - 1]))
        prev = cur
    return min(prev)


def main():
    words = []
    for line in open(os.path.join(ROOT, "hard_sentences.txt")):
        if line.strip() and not line.startswith("#"):
            words.append([w.strip() for w in line.split("||")[1].split(",")])
    out = {}
    for name in sys.argv[1:]:
        j = json.load(open(os.path.join(ROOT, "results", "fm", "judge", name + ".json")))
        fails, failed = 0, []
        for c, ws in zip(j["clips"], words):
            for w in ws:
                wn = normalize(w)
                if semiglobal(wn, c["hyp"]) / max(1, len(wn)) > 1 / 3:
                    fails += 1; failed.append(w)
        out[name] = {"whisper_cer": j["cer"], "whisper_wer": j["wer"], "hard_word_failures": fails,
                     "hard_words": sum(map(len, words)), "failed": failed}
        print(name, json.dumps(out[name]))
    json.dump(out, open(os.path.join(ROOT, "results", "fm", "hard_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
