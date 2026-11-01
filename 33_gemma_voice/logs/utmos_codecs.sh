#!/bin/bash
# UTMOS (CPU, 4 threads) on val-100 codec round trips and r5 outputs; uses 27's mos.py in .venv-sac.
cd /Users/danielmercado/ProjectJoy/33_gemma_voice
for s in real mimi4 mimi8 mimi16 qwen16 xy8 sac voc28_real r5_bo1 r5_bo4; do
  ../.venv-sac/bin/python ../27_pocket_voice_foundry/mos.py --list runs/phase0/lists/val_$s.json --out results/utmos/val_$s.json --threads 4 > /dev/null 2>&1
  python3 -c "import json;d=json.load(open('results/utmos/val_$s.json'));print('$s', {k:v for k,v in d.items() if not isinstance(v,(list,dict))})"
done
echo UTMOS_DONE
