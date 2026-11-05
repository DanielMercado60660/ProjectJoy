#!/bin/bash
# Round 7 queue 6: after queue5 (3B documentation run): the Stage 3 arm's one look at the 80 confirmation rooms with its
# belief-conditioned control (plan: one pass per arm; the servo and the 3A controls stay off the 80; the bot's 80-room row
# is M3's first look). Two jobs. Then the method set is frozen by hand (STATUS.md) and r7_queue_fresh.sh starts.
cd "$(dirname "$0")"
until grep -q "all done (3B included)" logs/r7_queue5.log 2>/dev/null; do sleep 30; done
echo "[queue6] $(date +%H:%M:%S) queue5 done; Stage 3 arm + belief-conditioned control on the 80 rooms"
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py confirm sel_learned_s0_dagger=selector:results/r7/skillsel/learned_s0_dagger.npz > logs/r7_confirm_3b.log 2>&1 &
OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py confirm sel_freq=selector:freq:results/r7/skillsel/freq.json > logs/r7_confirm_freq.log 2>&1 &
wait
echo "[queue6] $(date +%H:%M:%S) 80-room confirmations done: $(tail -1 logs/r7_confirm_3b.log | cut -c1-120) | $(tail -1 logs/r7_confirm_freq.log | cut -c1-120)"
echo "[queue6] $(date +%H:%M:%S) all done"
