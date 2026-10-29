#!/bin/sh
# Smoke check (< 1 min): all tests, M0-M6 and Phase 2 (tests/test_*.py): body, room, interface, skills, bot, recorder, learner, recorder-vs-evaluator, Phase 2 learner and labels.
set -e
cd "$(dirname "$0")"
../.venv/bin/python -m pytest -q tests
