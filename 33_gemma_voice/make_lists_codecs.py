"""Write judge lists for the SAC / Qwen3-16 / XY-8 round trips of one split (used by logs/phase0_run.sh)."""
import json, sys
s = sys.argv[1]
refs = {d["id"]: d["ref"] for d in json.load(open(f"runs/phase0/lists/{s}_real.json"))}
ids = [l.strip() for l in open(f"runs/ceiling/{s}_ids.txt") if l.strip()]
for name, d in [("sac", f"runs/ceiling_sac/{s}"), ("qwen16", f"runs/ceiling_qwen/{s}/nq16"), ("xy8", f"runs/ceiling_xy/{s}/nq8")]:
    json.dump([{"id": c, "wav": f"{d}/{c}.wav", "ref": refs[c]} for c in ids], open(f"runs/phase0/lists/{s}_{name}.json", "w"))
