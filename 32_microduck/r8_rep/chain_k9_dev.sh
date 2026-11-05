#!/bin/bash
# R8 Gate C: one r8_job slot, sequential: the 80 dev rooms at 224 px (evaluation only), then (after the previous
# recorder, pid $1, has exited) the rest of the k=9 noisy-bot set_lamp episodes on configs 0000-0269, then a
# marker that releases the LeWM lanes (data/r8_rep/train_data_ready.json: the frozen training episode count).
#   ./r8_job.sh rep-k9-dev r8_rep/chain_k9_dev.sh <pid of the running k9 recorder job>
set -u
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
$PY r8_rep/rep_record.py --plan dev --procs 4 --max-min 30 > logs/r8_rep_rec_dev.log 2>&1
if [ $# -ge 1 ]; then while kill -0 "$1" 2>/dev/null; do sleep 5; done; fi
$PY r8_rep/rep_record.py --plan train_lamp --cfg-lo 0 --cfg-hi 270 --procs 4 --max-min 40 > logs/r8_rep_rec_train3.log 2>&1
$PY - <<'EOF' > data/r8_rep/train_data_ready.json
import json, sys, time
sys.path.insert(0, "r8_rep")
import rep_data as D
tr = D.episode_list("train")
by_k = {}
for e in tr:
    by_k[str(e["k"])] = by_k.get(str(e["k"]), 0) + 1
print(json.dumps({"time": time.strftime("%F %T"), "train_episodes": len(tr), "by_k": by_k,
                  "train_ticks": sum(e["ticks"] for e in tr), "heldout_episodes": len(D.episode_list("heldout")),
                  "dev_episodes": len(D.episode_list("dev"))}))
EOF
cat data/r8_rep/train_data_ready.json
