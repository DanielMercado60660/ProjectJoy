#!/bin/sh
# Score judge lists: Whisper judge (33's judge.py, as-is, in ../.venv-judge, through the GPU lock) + UTMOS22 (mos.py, CPU).
#   ./score_fm.sh test_fm_ft val_fm_lj_s8000 ...      (names of results/fm/lists/<name>.json)
# -> results/fm/judge/<name>.json, results/fm/mos/<name>.json. Existing outputs are kept (delete to re-score).
cd "$(dirname "$0")"
mkdir -p results/fm/judge results/fm/mos
for n in "$@"; do
  l=results/fm/lists/$n.json
  [ -f results/fm/judge/$n.json ] || ../bin/gpu_run ../.venv-judge/bin/python ../33_gemma_voice/judge.py --list $l --out results/fm/judge/$n.json 2>/dev/null | tail -1
  [ -f results/fm/mos/$n.json ] || OMP_NUM_THREADS=4 ../.venv-sac/bin/python mos.py --list $l --out results/fm/mos/$n.json 2>/dev/null | tail -1
  python3 -c "import json;j=json.load(open('results/fm/judge/$n.json'));m=json.load(open('results/fm/mos/$n.json'));print('$n', 'whisper CER', j['cer'], 'WER', j['wer'], 'UTMOS', m['mos'], m['ci95'])"
done
