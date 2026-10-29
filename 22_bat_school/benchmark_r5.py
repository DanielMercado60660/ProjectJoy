"""Short NumPy-only full-validation throughput probe, not a headline result."""
import argparse
import time
from pathlib import Path
from env_r5 import make_pool, assert_disjoint
from baseline_r5 import AcousticPlanner
from eval_r5 import suite
from io_r5 import Deadline, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=.5)
    ap.add_argument("--per-group", type=int, default=32)
    ap.add_argument("--out", default="results/r5/numpy_speed.json")
    a = ap.parse_args()
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    clock = Deadline(a.max_minutes)
    train, val = make_pool("train", 256), make_pool("val", a.per_group)
    assert_disjoint(train, val)
    out = dict(setup_seconds=clock.elapsed, per_group=a.per_group, benchmark_only=True, methods={})
    for mode in ("event", "uncertainty"):
        result = suite(val, lambda n, mode=mode: AcousticPlanner(n, mode), deadline=clock)
        out["methods"][mode] = {k: result[k] for k in ("seconds", "rows", "score")}
    pair = sum(v["seconds"] for v in out["methods"].values())
    out.update(pair_seconds=pair, projected_pairs_in_five_minutes=int(max(0, 300-out["setup_seconds"])/pair),
               seconds=clock.elapsed)
    write_json(a.out, out)
    print({k: out[k] for k in ("setup_seconds", "pair_seconds", "projected_pairs_in_five_minutes", "seconds")})


if __name__ == "__main__":
    main()
