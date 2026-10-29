#!/usr/bin/env bash
# End-to-end smoke check at tiny size (< 2 min): unit tests -> tiny train -> eval -> GIF -> html.
# Uses the fixed val split as a stand-in training set and writes everything to a temp dir.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
TMP=$(mktemp -d)
trap 'rm -rf "$TMP" results/eval_smoke.json media/curves_smoke.png' EXIT
$PY -m pytest -q
$PY train.py --train_split val --steps 30 --batch 8 --enc_dim 64 --hid 32 --ckpt_every 30 --out "$TMP/ck" | tail -2
$PY evaluate.py --variant base --ckpt "$TMP/ck/final" --n 20 --tag smoke | tail -3
$PY demo.py --variant base --ckpt "$TMP/ck/final" --n_pool 12 --out "$TMP/media" --data_out "$TMP" | tail -2
echo "SMOKE OK"
