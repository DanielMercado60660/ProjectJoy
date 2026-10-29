#!/bin/bash
# Seed-replication follow-up: after phase A (gens 1-100, started separately) finishes for
# seed $1, run phase B (gens 101-136) and phase C (gens 137-$CTARGET) with seed 0's exact
# per-phase arguments, each as its own GPU-lock job (<= 44 min each). Optional STOP=HH:MM (today)
# makes training stop by then.
#   bash seeds_chain.sh 1        (C's last gen in logs/seeds/c_target, default 191)
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
S=$1; STOP=${STOP:-}
PY=../.venv/bin/python
until grep -q '^done' logs/seeds/seed${S}_A.log; do sleep 10; done
mins_left() { [ -z "$STOP" ] && { echo 44; return; }; python3 -c "import time,datetime as d;n=d.datetime.now();h,m=map(int,'$STOP'.split(':'));print(max(0.5,min(44,(n.replace(hour=h,minute=m,second=0)-n).total_seconds()/60)))"; }
# phase A may stop on its 44-min budget just short of gen 100: finish it with phase-A settings first
if [ "$(wc -l < logs/seed$S.jsonl)" -lt 100 ]; then
  ../bin/gpu_run $PY train.py --run seed$S --seed $S --gens 100 --minutes $(mins_left) > logs/seeds/seed${S}_A2.log 2>&1
fi
../bin/gpu_run $PY train.py --run seed$S --seed $S --gens 136 --sims 200 --lr 1e-3 --minutes $(mins_left) > logs/seeds/seed${S}_B.log 2>&1
CT=$(cat logs/seeds/c_target 2>/dev/null || echo 191)
../bin/gpu_run $PY train.py --run seed$S --seed $S --gens $CT --sims 200 --lr 1e-3 --random-plies 4 --temp-moves 16 --minutes $(mins_left) > logs/seeds/seed${S}_C.log 2>&1
echo "chain done $(date +%H:%M)"
