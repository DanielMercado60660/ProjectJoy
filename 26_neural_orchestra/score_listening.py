"""Score the blind A/B listening test after you have filled in listening_sheet.csv.

  ../.venv/bin/python score_listening.py media/listening_ab/listening_sheet.csv
"""
import csv
import json
import os
import sys
from collections import defaultdict


def main(sheet):
    key = json.load(open(os.path.join(os.path.dirname(sheet), "_answer_key", "answer_key.json")))
    truth = {p["pair"]: p for p in key["pairs"]}
    wins, total = defaultdict(int), defaultdict(int)
    for row in csv.DictReader(open(sheet)):
        choice = row["preferred (A/B)"].strip().upper()
        if choice not in ("A", "B"):
            continue
        p = truth[row["pair"]]
        rival = p["B"] if p["A"] == "vae" else p["A"]
        total[rival] += 1
        wins[rival] += p[choice] == "vae"
    if not total:
        print("no answers filled in yet")
        return
    for rival in total:
        print(f"VAE preferred over {rival:10s}: {wins[rival]}/{total[rival]} ({100 * wins[rival] / total[rival]:.0f}%)")
    print(f"overall: {sum(wins.values())}/{sum(total.values())}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "media/listening_ab/listening_sheet.csv")
