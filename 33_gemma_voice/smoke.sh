#!/bin/bash
# End-to-end smoke test at tiny size (< 2 min): Mimi encode 9 clips -> train 20 steps -> resume ->
# sample one utterance -> Mimi decode -> recognizer CER. Needs ../.venv-moshi and the Gemma/Mimi weights in the HF cache.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
MPY=../.venv-moshi/bin/python
rm -rf runs/smoke_sh
$MPY mimi_io.py encode --out runs/smoke_sh/codes.npz --limit 3
A="--codes runs/smoke_sh/codes.npz --steps 20 --warmup 5 --val_every 10 --save_every 10 --budget 4000 --log_every 5"
$PY train.py --out runs/smoke_sh $A --stop_at 10 | tail -1
$PY train.py --out runs/smoke_sh $A --resume | grep -E "resumed|done"
$PY - <<'EOF'
import json, subprocess, sys, numpy as np, mlx.core as mx
from evaluate import load_run
from gemma import load_tokenizer
from tts import generate
from score import corpus_cer, wav24_to_mel, MOSHI_PY
model, meta = load_run("runs/smoke_sh", "last")
ids = [2] + load_tokenizer().encode("Printing, in the only sense.", add_special_tokens=False).ids
codes, info = generate(model, ids, max_frames=40, temp=0.8, top_k=50)
np.savez("runs/smoke_sh/gen.npz", names=np.array(["gen"]), lens=np.array([codes.shape[1]]), codes=codes)
subprocess.run([MOSHI_PY, "mimi_io.py", "decode", "--codes", "runs/smoke_sh/gen.npz", "--nq", str(model.nq),
                "--out", "runs/smoke_sh/wavs"], check=True)
r = corpus_cer([wav24_to_mel("runs/smoke_sh/wavs/gen.wav")], ["printing in the only sense"])
print("SMOKE OK", json.dumps({"frames": info["frames"], "cer_untrained_expected_high": round(r["cer"], 3)}))
EOF
