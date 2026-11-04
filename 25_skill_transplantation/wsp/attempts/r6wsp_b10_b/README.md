# r6wsp_b10_b

One same-size, task-routed merged actor per training seed. Construction uses deterministic closed-form linear algebra on the supplied specialist weights and invented inputs. Exactly 10 authorized evaluations were used; final training-reset mean retained is 95.8675%. Held-out performance is not measured.

From the project root:

```sh
/Users/danielmercado/ProjectJoy/.venv/bin/python wsp/attempts/r6wsp_b10_b/build.py
/Users/danielmercado/ProjectJoy/.venv/bin/python wsp/attempts/r6wsp_b10_b/verify.py
```

- `merged_s0.npz` through `merged_s3.npz`: final actors, seven float32 tensor keys each.
- `build.py`: rebuilds the final actors from `wsp/data/` alone, without environment use.
- `NOTES.md`: construction, synthetic distributions, all ten evaluation decisions, results, and limitations.
- `merged_s*.json`: selected neuron maps and synthetic error diagnostics.
- `verify.py`, `verification.json`: synthetic-only shape, routing, and evaluated-candidate identity checks.
- `ledger.jsonl`: evaluator-owned record, never manually modified.
- Other named candidate files and build logs: preserved experiment evidence; only `merged_s*.npz` are submitted actors.
