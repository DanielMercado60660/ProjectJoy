#!/bin/bash
# Extra seeds for the seed-variance check (same configs as the main runs, except PPO
# stops after 10 min since its best validation checkpoint comes within the first ~5 min).
# Two chains run side by side (2 slots). Logs in logs/.
# (In the recorded run only seed 2 got a slot; seed 1 was cancelled after a 75 min queue.)
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=../.venv/bin/python
RUN=../bin/gpu_run
chain() {
  s=$1
  $RUN $PY train_ppo.py --minutes 10 --seed $s --tag ppo_s$s > logs/ppo_s$s.log 2>&1
  $RUN $PY train_es.py --gens 300 --max_minutes 30 --seed $s --tag es_s$s > logs/es_s$s.log 2>&1
}
chain 1 &
chain 2 &
wait
