#!/bin/bash
# End-to-end smoke check at tiny size (< 2 min): unit tests, then train -> analyze -> demo into a temp dir.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
export OMP_NUM_THREADS=4
$PY -m pytest -q
T=$(mktemp -d)
export ALIEN_CKPT=$T/ck ALIEN_RES=$T/res ALIEN_MEDIA=$T/media
mkdir -p $ALIEN_RES $ALIEN_MEDIA
$PY train.py --stage full --method gumbel reinforce nocomm compositional holistic --P 2 --steps 300 --ckpt_dir $ALIEN_CKPT --fresh | grep done
$PY train.py --stage color --method gumbel reinforce nocomm --P 2 --steps 200 --ckpt_dir $ALIEN_CKPT --fresh | grep done
$PY analyze.py | tail -6
$PY make_demo.py
if command -v node >/dev/null; then node tests/check_demo.js $ALIEN_MEDIA/demo.html | grep match; fi
rm -rf $T
echo "SMOKE OK"
