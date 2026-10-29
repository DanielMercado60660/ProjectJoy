"""Claude-only bounded follow-up: frozen heads, fresh data, validation-tuned allocation.

NumPy helpers are importable without MLX. Original round-5 receipts are never changed.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from r5_core import Search, allocation, features, mistakes, group_masks, report_moves
from r5_data import Builder, arrays, fingerprint
from r5_baselines import scores
from r5_run import METHODS, decision, write, demo

SEEDS = dict(train=50242211, val=50242223, test=50242237)
PUCTS = (.75, 1.5, 3.)
FLOORS = (8, 24)
CAPS = (64, 128)


def blocked_sources():
    """Exclude all six previous labelled splits, including cross-run overlaps."""
    keys, manifest = set(), []
    for run in ("main", "ac_large"):
        for split in ("train", "val", "test"):
            path = Path(f"results/r5/{run}/{split}.npz")
            with np.load(path) as d:
                keys.update(map(int, d["key"]))
            manifest.append(dict(path=str(path), sha256=fingerprint(path)))
    return keys, manifest


def counterfactual_moves(score, floor, cap, moves, batch=64, mean=32):
    """Replay allocation over per-position search continuations, without labels.

    A fractional final position gets its exact integer budget. The cache stores
    every budget from 8..128, so no rounding or free simulations are introduced.
    """
    answer = np.empty(len(score), int)
    for begin in range(0, len(score), batch):
        sl = slice(begin, begin+batch)
        budgets = allocation(score[sl], floor, cap, len(score[sl])*mean)
        answer[sl] = moves[np.arange(begin, min(begin+batch, len(score))), budgets]
    return answer


def error_objective(values, group, moves):
    y = mistakes(values, moves)
    masks = group_masks(values, group, y)
    names = ("all", "early", "middle", "late", "drawn")
    return float(np.mean([y[masks[k]].mean() for k in names if masks[k].any()]))


def validation_cache(data, ev, a, deadline):
    """One tree continuation supplies all exact budgets for every PUCT candidate."""
    cache = {}
    for cp in PUCTS:
        xs, mv = [], []
        for start in range(0, len(data["pos"]), 64):
            if time.monotonic() >= deadline:
                raise TimeoutError("Validation grid incomplete; no selection or test scoring")
            p, m = data["pos"][start:start+64], data["mask"][start:start+64]
            search = Search(p, m, ev, capacity=128, c_puct=cp)
            search.advance(np.full(len(p), 8))
            x, _, _ = features(p, m, search)  # all four AB depths, paid once for tuning
            track = np.zeros((len(p), 129), np.int64)
            track[:, 8] = search.moves()
            for budget in range(9, 129):
                search.advance(np.full(len(p), budget))
                track[:, budget] = search.moves()
            xs.append(x); mv.append(track)
        cache[cp] = dict(x=np.concatenate(xs), moves=np.concatenate(mv))
    return cache


def select(data, cache, learned, baselines):
    """Every arm uses the same complete-validation error objective; no test data."""
    best, trials = {}, []
    for method in METHODS:
        for cp, c in cache.items():
            # Calibration bias/scale do not change ranks. Search the 48 actual
            # rule formulas, rather than claiming 576 independent rankings.
            rules = [dict(depth=d, signal=s, weight=w, bias=-3., scale=1.)
                     for d in range(4) for s in range(3) for w in (0., .5, 1., 2.)] if method == "rule" else [None]
            for rule in rules:
                b = deepcopy(baselines)
                if rule is not None:
                    b["rule"] = rule
                risk = scores(method, c["x"], b, learned) if method != "fixed" else None
                configs = [(32, 32)] if method == "fixed" else [(f, h) for f in FLOORS for h in CAPS]
                for floor, cap in configs:
                    mv = c["moves"][:, 32] if method == "fixed" else counterfactual_moves(risk, floor, cap, c["moves"])
                    value = error_objective(data["values"], data["group"], mv)
                    row = dict(method=method, c_puct=cp, allocation_floor=floor, high=cap,
                               objective=value, mistakes=int(mistakes(data["values"], mv).sum()), rule=rule)
                    trials.append(row)
                    if method not in best or value < best[method]["objective"]:
                        best[method] = row
    groups = {k:int(v.sum()) for k,v in group_masks(data["values"], data["group"],
                                                  np.zeros(len(data["values"]), int)).items()
              if k != "probe_mistakes"}
    return dict(choices=best, trials=trials, validation_rows=len(data["values"]), validation_groups=groups,
                selection="Mean error over all + each ply group + draws; first grid entry wins ties; no timing/test selection")


def finalize(root, builder, records, config):
    # Only complete test blocks enter the headline. Retain pending raw candidates.
    n = max((max(r["indices"])+1 for r in records), default=0)
    if n == 0:
        return
    data = arrays(builder.rows["test"][:n])
    summary = {}
    for regime in ("sim", "time"):
        for method in METHODS:
            key = regime+"/"+method
            rows = [r for r in records if r["regime"] == regime and r["method"] == method]
            mv, initial = np.full(n, -1, int), np.full(n, -1, int)
            for row in rows:
                mv[row["indices"]], initial[row["indices"]] = row["moves"], row["initial"]
            assert (mv >= 0).all() and (initial >= 0).all()
            sims, elapsed = sum(r["simulations"] for r in rows), sum(r["seconds"] for r in rows)
            r = report_moves(data["values"], mv, initial, sims, elapsed)
            r["groups"] = {g: dict(n=int(t.sum()), mistakes=int(mistakes(data["values"][t], mv[t]).sum()) if t.any() else 0)
                           for g,t in group_masks(data["values"], data["group"], mistakes(data["values"], initial)).items()}
            if regime == "time":
                target = sum(x["fixed_seconds"] for x in rows)
                r.update(time_ratio=elapsed/target, matched_within_10_percent=abs(elapsed/target-1) <= .1)
            else:
                assert sims == n*32
            summary[key] = r
    np.savez_compressed(root/"test.npz", **data)
    write(root/"evaluation.json", dict(complete=True, summary=summary, records=records, config=config,
         test_sha256=fingerprint(root/"test.npz"), scored_test_rows=n,
         pending_unscored_rows=len(builder.rows["test"])-n,
         note="Fresh follow-up; selected head and logistic weights frozen. Gates have different validation-selected PUCTs, so compare absolute errors first; net rescues use each gate's own 8-sim probe."))


def main(a):
    started = time.monotonic(); deadline = started+60*a.max_minutes
    assert 0 < a.max_minutes <= 45
    run = "turn2_floor_smoke" if a.smoke else "turn2_floor"
    root = Path("results/r5")/run; root.mkdir(parents=True, exist_ok=True)
    keys, manifest = blocked_sources()
    seedset = {k:v+1000 for k,v in SEEDS.items()} if a.smoke else SEEDS
    builder = Builder(root/"data", seeds=seedset, extra_keys=keys, extra_manifest=manifest)
    source = Path("checkpoints/r5_ac_large/selected")
    head_config = json.loads((source/"head.json").read_text())["config"]
    for suffix, name in ((".safetensors", "backbone_sha256"), (".json", "backbone_config_sha256")):
        assert fingerprint(Path(head_config["backbone"]+suffix)) == head_config[name], "Frozen backbone changed"
    config = dict(source=str(source), source_sha256=fingerprint(source/"head.safetensors"),
                  baselines_sha256=fingerprint(source/"baselines.json"), seeds=seedset,
                  normalizer_sha256=fingerprint(source/"normalizer.npz"),
                  head_config_sha256=fingerprint(source/"head.json"),
                  backbone_sha256=head_config["backbone_sha256"],
                  val_rows=12 if a.smoke else 1536, batch=64, mean_simulations=32,
                  pucts=list(PUCTS), floors=list(FLOORS), caps=list(CAPS), protocol=2)
    if (root/"config.json").exists():
        assert json.loads((root/"config.json").read_text()) == config
    else:
        write(root/"config.json", config)
    from net import load_model, make_evaluator
    from r5_mlx import predictor
    net, _ = load_model(head_config["backbone"])
    ev, learned = make_evaluator(net), predictor(source)
    baselines = json.loads((source/"baselines.json").read_text())
    selection_path = root/"selection.json"
    if not selection_path.exists():
        while len(builder.rows["val"]) < config["val_rows"]:
            n = min(64, config["val_rows"]-len(builder.rows["val"]))
            progress = builder.add("val", n, deadline-10)
            builder.save()
            print(json.dumps(dict(event="fresh_validation", n=len(builder.rows["val"]), **progress)), flush=True)
            if progress["added"] == 0:
                raise TimeoutError("Full validation not obtained; resume without reducing it")
        data = arrays(builder.rows["val"])
        table = validation_cache(data, ev, a, deadline-5)
        selection = select(data, table, learned, baselines)
        write(selection_path, selection)
    selection = json.loads(selection_path.read_text())
    print(json.dumps(dict(event="validation_selected", n=selection["validation_rows"],
                     groups=selection["validation_groups"], trials=len(selection["trials"]),
                     choices=selection["choices"])), flush=True)
    blockdir = root/"blocks"; blockdir.mkdir(exist_ok=True)
    records = [row for path in sorted(blockdir.glob("block_*.json")) for row in json.loads(path.read_text())]
    done = max((max(r["indices"])+1 for r in records), default=0)
    recent = [1.]
    blocksize = 6 if a.smoke else 64
    while time.monotonic()+max(recent[-3:])*1.2+2 < deadline:
        t = time.monotonic()
        if len(builder.rows["test"]) < done+blocksize:
            builder.add("test", done+blocksize-len(builder.rows["test"]), deadline-2)
            builder.save()
        if len(builder.rows["test"]) < done+blocksize:
            break
        data = arrays(builder.rows["test"][done:done+blocksize])
        def execute(method, regime, seconds=None):
            choice = selection["choices"][method]
            cfg = SimpleNamespace(low=8, fixed=32, high=choice["high"],
                                  c_puct=choice["c_puct"], allocation_floor=choice["allocation_floor"],
                                  time_capacity=choice["high"])
            b = deepcopy(baselines)
            if choice["rule"] is not None:
                b["rule"] = choice["rule"]
            return decision(data["pos"], data["mask"], method, regime, ev, learned, b, cfg, seconds)
        # First block warms shapes and selected configurations, outside its timings.
        if not records:
            for name in METHODS:
                execute(name, "sim")
        fixed = sorted([execute("fixed", "sim") for _ in range(3)], key=lambda r:r["seconds"])[1]
        block = []
        rng = np.random.default_rng(50242249+done)
        for regime in ("sim", "time"):
            for name in rng.permutation(METHODS):
                r = fixed if name == "fixed" else execute(name, regime, fixed["seconds"])
                block.append(dict(method=str(name), regime=regime, indices=list(range(done, done+blocksize)),
                    moves=r["move"].tolist(), initial=r["initial"].tolist(),
                    per_position_sims=r["per_position_sims"].tolist(), simulations=r["simulations"],
                    seconds=r["seconds"], fixed_seconds=fixed["seconds"]))
        write(blockdir/f"block_{done:08}.json", block)
        records.extend(block); done += blocksize
        recent.append(time.monotonic()-t)
        print(json.dumps(dict(event="fresh_test_block", scored=done, seconds=recent[-1],
                         measured_positions_per_s=blocksize/recent[-1], seconds_left=deadline-time.monotonic())), flush=True)
        if a.smoke:
            break
    finalize(root, builder, records, config)
    if not (root/"evaluation.json").exists():
        raise TimeoutError("No complete test block: resume with remaining allowance")
    demo(SimpleNamespace(run=run, max_minutes=1, low=8, fixed=32, high=128, holdout=config["val_rows"], batch=64,
                         demo_out=f"media/r5_{run}.html"))
    print(json.dumps(dict(event="followup_complete", scored=done, seconds=time.monotonic()-started,
                         selection_trials=len(selection["trials"]), smoke=a.smoke)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-minutes", type=float, default=6)
    p.add_argument("--smoke", action="store_true")
    main(p.parse_args())
