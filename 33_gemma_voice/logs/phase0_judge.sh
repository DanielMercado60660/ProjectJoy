cd /Users/danielmercado/ProjectJoy/33_gemma_voice
while pgrep -f "train_mel.py --out runs/mel1" >/dev/null; do sleep 20; done
for f in runs/phase0/lists/*.json; do
  n=$(basename $f .json)
  echo -n "$n "
  ../bin/gpu_run ../.venv-judge/bin/python judge.py --list $f --out results/judge/$n.json 2>/dev/null | tail -1
done
echo PHASE0_DONE
