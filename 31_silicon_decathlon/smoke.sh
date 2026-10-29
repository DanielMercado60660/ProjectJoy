#!/bin/sh
# End-to-end smoke check, under 2 minutes: harness + report tests, e15 parity, two events at tiny settings
# (batch 1 only, no task checks, written to logs/smoke/ so results/ is untouched), then the report on those.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
export OMP_NUM_THREADS=4
$PY -m pytest -q -p no:cacheprovider tests/test_harness.py tests/test_e15_object_permanence.py
rm -rf logs/smoke
for ev in e09_alphazero e15_object_permanence; do
  $PY run_event.py $ev --no-task --batches 1 --repeats 1 --budget 0.02 --out logs/smoke 2>&1 | grep -E "^(==|  b=|wrote)"
done
$PY make_report.py --results logs/smoke --out logs/smoke
if command -v node >/dev/null; then node tests/check_demo.js logs/smoke/demo.html; else echo "node not found; demo page check skipped"; fi
echo "smoke OK"
