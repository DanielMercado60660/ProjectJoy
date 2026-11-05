"""Do systems swallow the end of the sentence? From the Whisper judge transcripts: of the clips whose reference ends with
"end quote", how many transcripts don't end with "quote"; and how many transcripts get the last word wrong.

  python3 endings_fm.py      # -> results/fm/endings.json (stdlib only)
"""
import glob, json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
out = {}
for p in sorted(glob.glob(os.path.join(ROOT, "results", "fm", "judge", "*.json"))):
    n = os.path.basename(p)[:-5]
    if not n.startswith(("val_", "test_")) or n.startswith("val_vocgan"):
        continue
    cl = json.load(open(p))["clips"]
    eq = [c for c in cl if c["ref"].endswith("end quote")]
    out[n] = {"end_quote_clips": len(eq), "end_quote_dropped": sum(1 for c in eq if not c["hyp"].endswith("quote")),
              "last_word_wrong": sum(1 for c in cl if c["hyp"].split()[-1:] != c["ref"].split()[-1:]), "clips": len(cl)}
    print(f"{n:40s} {out[n]}")
json.dump(out, open(os.path.join(ROOT, "results", "fm", "endings.json"), "w"), indent=1)
