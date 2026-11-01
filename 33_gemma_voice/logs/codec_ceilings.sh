cd /Users/danielmercado/ProjectJoy/33_gemma_voice
while pgrep -f "logs/phase0_judge.sh" >/dev/null || pgrep -f "train_mel.py --out runs/mel1" >/dev/null; do sleep 20; done
for s in val test; do
  ids=runs/ceiling/${s}_ids.txt
  ../.venv-sac/bin/python sac_io.py roundtrip --ids $ids --out runs/ceiling_sac/$s 2>/dev/null | tail -1
  ../bin/gpu_run ../.venv-audio/bin/python qwen_io.py roundtrip --ids $ids --out runs/ceiling_qwen/$s --nq 16 2>/dev/null | tail -1
  ../.venv-xy/bin/python xy_io.py roundtrip --ids $ids --out runs/ceiling_xy/$s --nq 8 2>/dev/null | tail -1
  ../.venv/bin/python - <<PY
import json
refs = {d["id"]: d["ref"] for d in json.load(open("runs/phase0/lists/${s}_real.json"))}
ids = [l.strip() for l in open("$ids") if l.strip()]
for name, d in [("sac", "runs/ceiling_sac/$s"), ("qwen16", "runs/ceiling_qwen/$s/nq16"), ("xy8", "runs/ceiling_xy/$s/nq8")]:
    json.dump([{"id": c, "wav": f"{d}/{c}.wav", "ref": refs[c]} for c in ids], open(f"runs/phase0/lists/${s}_{name}.json", "w"))
PY
  for n in sac qwen16 xy8; do echo -n "${s}_$n "; ../bin/gpu_run ../.venv-judge/bin/python judge.py --list runs/phase0/lists/${s}_$n.json --out results/judge/${s}_$n.json 2>/dev/null | tail -1; done
done
echo CEILINGS_DONE
