"""NumPy-only acceptance gate for the predeclared turn-2 training budget.

Run with no arguments for the paired seed-61 experiment, or --seed62 for the
optional unpaired treatment. Fail before fresh test if work/selection differ.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from io_r5 import load_checkpoint
from env_r5 import GROUPS, BUDGETS


def check_run(name, seed, entropy):
    p = Path("logs/r5")/name
    summary = json.loads((p/"summary.json").read_text())
    saved = load_checkpoint(p/"latest.npz")
    validation = [json.loads(x) for x in (p/"validation.jsonl").read_text().splitlines()]
    records = [json.loads(x) for x in (p/"train.jsonl").read_text().splitlines()]
    assert summary["target_reached"] and summary["update"] == 5600
    assert saved["update"] == 5600 and saved["env_steps"] == 22937600 and int(saved["optimizer"]["step"]) == 67200
    assert saved["config"]["seed"] == seed and saved["config"]["entropy"] == entropy
    assert [r["update"] for r in records] == list(range(1, 5601))
    assert [v["update"] for v in validation] == [1, 1000, 2000, 3000, 4000, 5000, 5600]
    for v in validation:
        assert len(v["rows"]) == 16 and all(r["n"] == 32 for r in v["rows"])
        assert {(r["group"], r["budget"]) for r in v["rows"]} == {(g, b) for g in GROUPS for b in BUDGETS}
    assert np.isfinite([[r[k] for k in ("loss", "entropy", "grad_norm", "reward_mean")] for r in records]).all()
    assert all(len(r["action_fractions"]) == 5 and np.isclose(sum(r["action_fractions"]), 1) for r in records)
    selected = load_checkpoint(summary["selected"])
    assert selected["config"] == saved["config"]
    assert selected["validation"] in validation
    assert selected["best_score"] == max(v["score"] for v in validation)
    print(name, "seconds", summary["total_seconds"], "best validation", summary["best_score"],
          "tail WAIT fraction", np.mean([r["action_fractions"][3] for r in records[-500:]]))
    return dict(saved["config"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed62", action="store_true")
    a = parser.parse_args()
    if a.seed62:
        check_run("t2_entropy62", 62, .06)
        print("UNPAIRED SEED 62 BUDGET OK")
    else:
        configs = [check_run("t2_control61", 61, .015), check_run("t2_entropy61", 61, .06)]
        assert [c.pop("entropy") for c in configs] == [.015, .06] and configs[0] == configs[1]
        print("MATCHED PAIR OK")


if __name__ == "__main__":
    main()
