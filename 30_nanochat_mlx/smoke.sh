#!/bin/bash
# End-to-end smoke check at tiny size (< 2 minutes): pretrain a d2 model for a few steps,
# save + resume, SFT on a few sandbox traces, KV-cache generation, sandbox eval, unit tests.
# Needs the tokenizer (checkpoints/tokenizer) and at least one tokenized shard (data/tokens).
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
if ! ls data/tokens/train_*.npy >/dev/null 2>&1; then
  echo "smoke.sh needs tokenized data (data/ is gitignored). Minimal setup (~200 MB download):"
  echo "  $PY data.py --download 1 && RAYON_NUM_THREADS=4 $PY data.py --tokenize"
  exit 1
fi
OUT=checkpoints/_smoke
rm -rf $OUT media/_smoke_eval.json
A=(--depth 2 --seq-len 256 --device-batch 8 --total-batch 4096 --num-iterations 20 --warmup-steps 2 --eval-every 10 --eval-batches 2)
$PY train.py "${A[@]}" --out $OUT/base --stop-at-step 10 | tail -2
$PY train.py "${A[@]}" --out $OUT/base --resume | grep -E "resuming|done"
$PY sft.py --base $OUT/base/last --out $OUT/sft --episodes 100 --epochs 0.5 --seq-len 256 --batch 8 | tail -2
$PY baseline.py --episodes 100 --epochs 2 --out $OUT/mlp | tail -1
$PY eval_sandbox.py --ckpt $OUT/sft/last --mlp $OUT/mlp --episodes 3 --policies expert,mlp,lm,lm_constrained --out media/_smoke_eval.json
$PY -m pytest -q tests
rm -rf $OUT media/_smoke_eval.json
echo "SMOKE OK"
