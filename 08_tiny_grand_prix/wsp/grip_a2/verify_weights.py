"""CPU-only audit of the four explicit edits. Does not import or run a simulator.

Run: /Users/danielmercado/ProjectJoy/.venv/bin/python verify_weights.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FILES = {
    1: "c01_speed_sqrt2.npz",
    2: "c02_speed_steer_quarter.npz",
    3: "c03_speed_wall_feedback.npz",
    4: "c04_clearance_speed135.npz",
}
SHAPES = {"W1": (13, 32), "b1": (32,), "W2": (32, 32),
          "b2": (32,), "W3": (32, 2), "b3": (2,)}


def read(name):
    with np.load(ROOT / name, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def specified_edit(start, call):
    """Reconstruct the documented edit directly from start.npz, in memory."""
    w = {k: a.copy() for k, a in start.items()}
    w["W1"][10] *= np.float32(1.35 if call == 4 else np.sqrt(2))
    if call == 2:
        w["W3"][:, 0] *= .25
        w["b3"][0] *= .25
    if call in (3, 4):
        delta = np.float32(.75) * (w["W1"][6] - w["W1"][3])
        w["W1"][9] += delta
        w["W1"][0] -= delta
    return w


def main():
    start = read("start.npz")
    assert sum(a.size for a in start.values()) == 1570
    counts = {}
    for call, name in FILES.items():
        saved = read(name)
        specified = specified_edit(start, call)
        assert set(saved) == set(SHAPES)
        for key, shape in SHAPES.items():
            assert saved[key].shape == shape
            assert saved[key].dtype == np.float32
            assert np.isfinite(saved[key]).all()
            np.testing.assert_array_equal(saved[key], specified[key])
        counts[str(call)] = sum(int(np.count_nonzero(saved[k] != start[k]))
                                for k in SHAPES)
    assert counts == {"1": 32, "2": 65, "3": 96, "4": 96}

    final = read(FILES[4])
    best = read("best.npz")
    for key in SHAPES:
        np.testing.assert_array_equal(best[key], final[key])
    unchanged_rows = [i for i in range(13) if i not in (0, 9, 10)]
    np.testing.assert_array_equal(final["W1"][unchanged_rows],
                                  start["W1"][unchanged_rows])
    for key in ("b1", "W2", "b2", "W3", "b3"):
        np.testing.assert_array_equal(final[key], start[key])

    # Hand-made probes verify the input-transformation identity, not driving skill.
    straight = [.1, .116, .174, .324, .957, .957, .324, .174, .116, .1]
    left = [.1, .12, .16, .21, .27, .4, .5, .25, .14, .11]
    near_left = [.20, .22, .27, .32, .38, .48, .62, .04, .03, .02]
    rays = [straight, left, left[::-1], near_left, near_left[::-1]]
    x = np.array([r + [v / 30, st, pd]
                  for r in rays for v in (0, 5, 10, 15, 20, 30, 40)
                  for st in (-1, 0, 1) for pd in (-1, 0, 1)], dtype=np.float64)
    virtual = x.copy()
    side_difference = x[:, 9] - x[:, 0]
    virtual[:, 3] -= .75 * side_difference
    virtual[:, 6] += .75 * side_difference
    virtual[:, 10] *= 1.35
    direct = x @ final["W1"] + final["b1"]
    equivalent = virtual @ start["W1"] + start["b1"]
    max_error = float(np.max(np.abs(direct - equivalent)))
    np.testing.assert_allclose(direct, equivalent, rtol=0, atol=3e-7)

    records = [json.loads(line) for line in (ROOT / "eval_log.jsonl").read_text().splitlines()]
    assert [r["call"] for r in records] == [1, 2, 3, 4]
    selection = json.loads((ROOT / "best.json").read_text())
    assert selection["call"] == 4
    assert records[-1]["sha"] == hashlib.sha256((ROOT / FILES[4]).read_bytes()).hexdigest()[:12]
    assert selection["sha"] == records[-1]["sha"]
    for call in FILES:
        output = json.loads((ROOT / f"call{call:02d}.json").read_text())
        assert output["call"] == call
        assert output["score_key"] == records[call - 1]["key"]

    report = {
        "passed": True, "simulator_calls_in_this_script": 0,
        "evaluation_calls_used": len(records), "evaluation_budget": 25,
        "best_call": selection["call"], "changed_parameters_by_call": counts,
        "best_changed_fraction": counts["4"] / 1570,
        "synthetic_probe_count": len(x),
        "input_transform_max_abs_error": max_error,
        "start_sha256": hashlib.sha256((ROOT / "start.npz").read_bytes()).hexdigest(),
        "best_sha256": hashlib.sha256((ROOT / "best.npz").read_bytes()).hexdigest(),
        "best_summaries": records[-1]["summaries"],
    }
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
