"""Read-only NumPy audit of preliminary R5 receipts; writes only a new audit file."""
import argparse
from pathlib import Path
import numpy as np
from r5_core import (Budget, read_json, write_json, digest, planning_metrics, forecast_metrics,
                     uncertainty_metrics, paired_test, holm_adjust)


def close_tree(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left:
            close_tree(left[k], right[k])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            close_tree(a, b)
    elif left is None or isinstance(left, (str, bool)):
        assert left == right
    else:
        assert np.isclose(left, right, rtol=1e-12, atol=1e-12), (left, right)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=1)
    a = ap.parse_args()
    budget = Budget(a.max_minutes)
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    root = Path(a.run)
    selection, fit = read_json(root / "selection.json"), read_json(root / "fit.json")
    assert digest(fit_path := root / "fit.json") == selection["fit_sha256"]
    assert digest(Path(selection["selected"]["checkpoint"]) / "model.safetensors") == selection["weights_sha256"]
    assert fit["selected"]["parameters"] == {"restitution": 1., "separation": 1.}
    report = dict(run=str(root), selected_beta=selection["beta"], exact_law=fit["selected"], splits={})
    for split in ("headline", "old_secondary"):
        budget.check()
        directory = root / split
        summary = read_json(directory / "summary.json")
        tasks = [read_json(p) for p in sorted((directory / "tasks").glob("*.json"))]
        assert len(tasks) == summary["status"]["requested_tasks"] and summary["status"]["complete"]
        for key, stored in summary["metrics"].items():
            budget.check()
            close_tree(planning_metrics([t["rows"][key] for t in tasks]), stored)
        tests = {}
        for comparison in summary["paired_tests"]:
            left, right = comparison.split(" vs ")
            tests[comparison] = paired_test([t["rows"][left]["real"] < .06 for t in tasks],
                                            [t["rows"][right]["real"] < .06 for t in tasks])
        close_tree(holm_adjust(tests), summary["paired_tests"])
        raw, forecast = np.load(directory / "forecast_raw.npz"), read_json(directory / "forecast.json")
        for name in forecast["metrics"]:
            close_tree(forecast_metrics(raw["hybrid" if name == "hybrid_mean" else name], raw["truth"], raw["flags"]), forecast["metrics"][name])
        for h, metrics in forecast["uncertainty"].items():
            close_tree(uncertainty_metrics(raw["disagreement"][:, int(h)], raw["error"][:, int(h)]), metrics)
        assert np.array_equal(raw["fitted"], raw["truth"]), "purported exact-law reference differs from truth"
        report["splits"][split] = dict(tasks=len(tasks), planning_cells=len(summary["metrics"]),
                                      paired_tests=len(tests), forecasts=len(raw["truth"]),
                                      summary_sha256=digest(directory / "summary.json"), all_recomputed=True)
    report.update(ok=True, seconds=budget.elapsed)
    write_json(a.out, report)
    print(f"AUDIT_TURN1_OK seconds={budget.elapsed:.3f} out={a.out}")


if __name__ == "__main__":
    main()
