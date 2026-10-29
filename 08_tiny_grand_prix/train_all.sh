#!/bin/bash
# Train everything one job at a time (each job takes one shared GPU/CPU slot).
# Order: ES, PPO, then grip-randomized variants of both. Logs in logs/.
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=../.venv/bin/python
RUN=../bin/gpu_run
$RUN $PY train_es.py  --gens 300 --max_minutes 30 --tag es  > logs/es.log 2>&1
$RUN $PY train_ppo.py --minutes 25 --tag ppo > logs/ppo.log 2>&1
$RUN $PY train_es.py  --gens 300 --max_minutes 20 --grip_lo 0.5 --grip_hi 1.1 --tag es_dr  > logs/es_dr.log 2>&1
$RUN $PY train_ppo.py --minutes 20 --grip_lo 0.5 --grip_hi 1.1 --tag ppo_dr > logs/ppo_dr.log 2>&1
