"""Timed, paired random search. Every candidate sees the FULL validation grid."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from baseline_r5 import AcousticPlanner, candidate
from env_r5 import make_pool, assert_disjoint, BUDGETS
from eval_r5 import suite
from io_r5 import Deadline, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=5)
    ap.add_argument("--per-group", type=int, default=32)
    ap.add_argument("--train-per-group", type=int, default=256)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--max-candidates", type=int, default=0, help="0 fills time allowance; positive only for smoke")
    ap.add_argument("--out", default="results/r5/baselines.json")
    a = ap.parse_args()
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    clock = Deadline(a.max_minutes)
    pool = make_pool("val", a.per_group)
    assert_disjoint(make_pool("train", a.train_per_group), pool)
    rng = np.random.default_rng(a.seed)
    out = dict(manifest=pool.manifest(), config=vars(a), best={}, trials=[], complete=False)
    durations = []
    index = 0
    while not a.max_candidates or index < a.max_candidates:
        # The two schedulers get the SAME sampled controller config and equal
        # numbers of fully completed candidates. Incomplete pairs never select.
        if durations and clock.remaining < 1.15*max(durations[-4:])+1:
            break
        config = candidate(rng, index)
        pair = {}
        started = time.monotonic()
        try:
            for mode in ("event", "uncertainty"):
                pair[mode] = suite(pool, lambda n, mode=mode: AcousticPlanner(n, mode, config), deadline=clock)
        except TimeoutError:
            break
        durations.append(time.monotonic()-started)
        for mode, result in pair.items():
            rec = dict(candidate=index, config=config, score=result["score"], rows=result["rows"], seconds=result["seconds"])
            out["trials"].append(dict(mode=mode, **rec))
            if mode not in out["best"] or rec["score"] > out["best"][mode]["score"]:
                out["best"][mode] = rec
        index += 1
        out.update(complete=True, candidate_pairs=index, seconds=clock.elapsed,
                   measured_pair_seconds=durations[-1], projected_total_pairs=index+int(max(0, clock.remaining)/np.mean(durations[-4:])))
        write_json(a.out, out)
        print(json.dumps({k: out[k] for k in ("candidate_pairs", "seconds", "measured_pair_seconds", "projected_total_pairs")}), flush=True)
    out.update(seconds=clock.elapsed, utilization=clock.elapsed/(a.max_minutes*60),
               stop="smoke candidate cap" if a.max_candidates and index >= a.max_candidates else "wall-clock guard")
    write_json(a.out, out)
    if not out["complete"]:
        raise SystemExit("No full paired validation completed; increase allowance")
    print(f"BASELINES COMPLETE: {index} candidates per scheduler; {clock.elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
