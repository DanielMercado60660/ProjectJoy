# r6wsp_b10_a

Four label-routed HalfCheetah policy merges, built by closed-form linear
algebra on synthetic inputs. Exactly 10/10 permitted evaluation calls used.
Final mean retained on training resets: 87.100%; held-out resets not measured.

Rebuild from the project directory (source weights only; no environment):

```sh
/Users/danielmercado/ProjectJoy/.venv/bin/python wsp/attempts/r6wsp_b10_a/build.py
/Users/danielmercado/ProjectJoy/.venv/bin/python wsp/attempts/r6wsp_b10_a/smoke_check.py
```

- `merged_s0.npz` through `merged_s3.npz`: final ordinary actor networks.
- `build.py`: deterministic construction; accepted choices are its defaults.
- `NOTES.md`: exact construction, synthetic distribution, all ten evaluation
  calls, acceptance decisions, measured results, and limitations.
- `smoke_check.py`, `verification.json`: synthetic routing and reproducibility
  verification; no environment calls.
- `*_diagnostics.json`: synthetic approximation errors and selected neurons.
- `baseline_*.npz`, `shared_motion_*.npz`, `derivative_motion_*.npz`: preserved
  evaluated candidates for audit; only `merged_s*.npz` are submitted.
- `ledger.jsonl`: evaluator-owned call ledger, never modified by this attempt.
- `call_03.json` through `call_10.json`: copied evaluator stdout records;
  calls 1 and 2 are recorded in the complete ledger and NOTES.md.
- `evaluate_once.py`: historical single-call/logging helper. Budget is exhausted;
  do not run it again for this attempt.
- `finalize_notes.py`: regenerates documentation from this attempt's receipts.
- `STATUS.md`: timestamped milestones.
