cd /Users/danielmercado/ProjectJoy/33_gemma_voice
for f in runs/phase0/lists/*.json; do
  n=$(basename $f .json)
  [ -f results/judge/$n.json ] && continue
  echo -n "$n "
  ../bin/gpu_run ../.venv-judge/bin/python judge.py --list $f --out results/judge/$n.json 2>/dev/null | tail -1
done
echo PHASE0_DONE
for s in val test; do
  ids=runs/ceiling/${s}_ids.txt
  ../.venv-sac/bin/python sac_io.py roundtrip --ids $ids --out runs/ceiling_sac/$s 2>/dev/null | tail -1
  ../bin/gpu_run ../.venv-audio/bin/python qwen_io.py roundtrip --ids $ids --out runs/ceiling_qwen/$s --nq 16 2>/dev/null | tail -1
  ../.venv-xy/bin/python xy_io.py roundtrip --ids $ids --out runs/ceiling_xy/$s --nq 8 2>/dev/null | tail -1
  ../.venv/bin/python make_lists_codecs.py $s
  for n in sac qwen16 xy8; do echo -n "${s}_$n "; ../bin/gpu_run ../.venv-judge/bin/python judge.py --list runs/phase0/lists/${s}_$n.json --out results/judge/${s}_$n.json 2>/dev/null | tail -1; done
done
echo CEILINGS_DONE
