#!/bin/bash
# R8 Gate C: one r8_job slot, two things at once (4 busy processes): the set_lamp training recording continued
# (plan train_lamp: any missing k=8, then k=9 noisy bot, configs 0000-0269) with 3 workers, and (1 process) the
# probe chain: probe set -> frozen-encoder features -> probes.
#   ./r8_job.sh rep-probe-k9 r8_rep/after_k8.sh [pid-to-wait-for]
set -u
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
if [ $# -ge 1 ]; then while kill -0 "$1" 2>/dev/null; do sleep 5; done; fi   # the previous recorder must be gone
$PY r8_rep/rep_record.py --plan train_lamp --cfg-lo 0 --cfg-hi 270 --procs 3 --max-min 44 > logs/r8_rep_rec_train2.log 2>&1 &
REC=$!
$PY r8_rep/rep_probeset.py > logs/r8_rep_probeset.log 2>&1 &&
$PY r8_rep/rep_features.py dinov2_s vjepa21_l lewm_tworooms lewm_pusht lewm_cube lewm_reacher random_vit p2a r6_lewm r6_rand > logs/r8_rep_features.log 2>&1 &&
$PY r8_rep/rep_probes.py > logs/r8_rep_probes.log 2>&1
echo "probe chain rc=$? at $(date '+%T')" >> logs/r8_rep_probes.log
wait $REC
echo "recorder exit at $(date '+%T')" >> logs/r8_rep_rec_train2.log
