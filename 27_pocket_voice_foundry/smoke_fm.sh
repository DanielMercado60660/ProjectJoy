#!/bin/sh
# Follow-up pipeline end to end at tiny size (~1.5 min): prep 30 clips per source -> multi-speaker pretrain 40 steps ->
# LJ fine-tune 20 steps -> resume 10 steps -> finalize (val pace) -> load the export and synthesize one sentence.
# Scoring with the Whisper judge / UTMOS is left out (separate venvs; see score_fm.sh).
set -e
cd "$(dirname "$0")"
S=runs/smoke_fm
rm -rf $S && mkdir -p $S
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=../.venv/bin/python
$PY prep_multi.py --limit 30 --out $S/cache2 | grep -v "^  " | tail -3
# tiny model; --limit keeps the val/train sets small
ARGS="--cache $S/cache2 --d 64 --enc_layers 2 --heads 2 --ff 128 --dec_d 64 --dec_layers 2 --dec_ff 128 --val_every 20 --n_val 4 --snap_every 20 --max_frames 4000"
$PY train_fm.py $ARGS --sources lj,libritts,hifi --out $S/pre --steps 40 | tail -2
$PY train_fm.py $ARGS --sources lj --init $S/pre --out $S/ft --steps 20 --warm 10 | tail -1
$PY train_fm.py $ARGS --sources lj --out $S/ft --steps 30 --resume | grep -E "resumed|done"
# finalize/eval read LJ text and mels from cache2/; point them at the smoke cache through a symlinked root
mkdir -p $S/root && for f in *.py; do ln -sf "$PWD/$f" $S/root/$f; done
ln -sfn "$PWD/$S/cache2" $S/root/cache2; ln -sfn "$PWD/cache" $S/root/cache; ln -sfn "$PWD/checkpoints" $S/root/checkpoints
mkdir -p $S/root/results $S/root/runs
LJV=$($PY -c "import numpy as np;z=np.load('$S/cache2/lj/index.npz');print(int((z['splits']=='val').sum()))")
$PY $S/root/finalize_fm.py --src $S/ft_snap30 --out $S/root/runs/export --n_val $LJV
echo "smoke: the val judge list's clips are not in the 30-clip smoke cache, so this checks that the export loads and synthesizes"
$PY -c "
import sys; sys.path.insert(0, '$S/root')
import numpy as np, mlx.core as mx
from model_fm import load_fm, infer, cap_memory
from text import encode, normalize
cap_memory()
m, cfg = load_fm('$S/root/runs/export.safetensors')
mel, d = infer(m, mx.array(np.array(encode(normalize('Hello there, this is a smoke test.')), np.int32)[None]), pace=cfg['pace'], n_steps=4)
from synth import NeuralVocoder
y = NeuralVocoder()(mel * np.array(cfg['mel_std']) + np.array(cfg['mel_mean']))
assert np.isfinite(y).all() and len(y) > 0
print('SMOKE OK: mel', mel.shape, 'wav', len(y) / 16000, 's, pace', round(cfg['pace'], 3))
"
