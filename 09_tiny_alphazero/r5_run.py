"""Round-5 orchestration. MLX imports are lazy; --help and tests work without Metal."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import c4
from r5_core import Search, allocation, features, mistakes, calibration, group_masks, report_moves, learner_features
from r5_data import Builder, arrays, fingerprint, candidate
from r5_baselines import fit_baselines, macro_nll, scores

METHODS = ("fixed", "learned", "entropy", "uncertainty", "disagreement", "rule", "logistic")


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)); tmp.replace(path)


def log(a, event, **kw):
    row = dict(event=event, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **kw)
    print(json.dumps(row, allow_nan=False), flush=True)
    with open(Path("logs/r5") / (a.run + ".jsonl"), "a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")


def setup(a):
    assert a.max_minutes > 0 and 0 < a.low < a.fixed < a.high
    assert a.holdout >= 3 and a.batch > 0
    assert a.run.replace("_", "").replace("-", "").isalnum()
    for p in ("logs/r5", f"results/r5/{a.run}", f"checkpoints/r5_{a.run}"):
        Path(p).mkdir(parents=True, exist_ok=True)
    return Path(f"results/r5/{a.run}"), Path(f"checkpoints/r5_{a.run}")


def evaluator(a):
    # Existing evaluator is compiled over a frozen model; weights never change.
    from net import load_model, make_evaluator
    model, _ = load_model(a.backbone)
    return make_evaluator(model)


def cache_features(rows, path, ev, a, deadline):
    data = arrays(rows)
    old, xs, ms, ps = 0, [], [], []
    if path.exists():
        with np.load(path) as d:
            old = len(d["x"])
            assert old <= len(rows) and np.array_equal(d["key"], data["key"][:old])
            xs.append(d["x"]); ms.append(d["move"]); ps.append(d["policy"])
    for start in range(old, len(rows), a.batch):
        if time.monotonic() >= deadline:
            raise TimeoutError("Feature deadline: rerun same train command to resume")
        part = slice(start, start+a.batch)
        s = Search(data["pos"][part], data["mask"][part], ev, capacity=a.high)
        s.advance(np.full(s.n, a.low))
        x, move, policy = features(data["pos"][part], data["mask"][part], s)
        xs.append(x); ms.append(move); ps.append(policy)
    data.update(x=np.concatenate(xs), move=np.concatenate(ms), policy=np.concatenate(ps))
    data["y"] = mistakes(data["values"], data["move"])
    with open(path.with_suffix(".tmp"), "wb") as f:
        np.savez_compressed(f, **data)
    path.with_suffix(".tmp").replace(path)
    return data


def train(a):
    start = time.monotonic(); deadline = start + 60*a.max_minutes
    out, ckpt = setup(a)
    ev = evaluator(a)
    cfg = dict(backbone=a.backbone, backbone_sha256=fingerprint(a.backbone+".safetensors"),
               backbone_config_sha256=fingerprint(a.backbone+".json"), low=a.low,
               fixed=a.fixed, high=a.high, holdout=a.holdout, min_train=a.min_train,
               batch=a.batch, protocol=1)
    configpath = out / "config.json"
    if configpath.exists():
        assert json.loads(configpath.read_text()) == cfg, "Use a new run name for changed config"
    else:
        write(configpath, cfg)
    frozen = out / "frozen.json"
    if not frozen.exists():
        builder = Builder(out / "data")
        data_end = start + 60*(a.data_minutes if a.data_minutes is not None else a.max_minutes*.60)
        for split in ("val", "test"):
            while len(builder.rows[split]) < a.holdout:
                r = builder.add(split, min(a.batch, a.holdout-len(builder.rows[split])), data_end)
                builder.save()
                if not r["added"]:
                    raise TimeoutError("Holdout incomplete: resume data, never use a truncated validation set")
                log(a, "labels", split=split, n=len(builder.rows[split]), **r)
            cache_features(builder.rows[split], out/(split+".npz"), ev, a, deadline)
        rates = []
        while time.monotonic() < data_end or len(builder.rows["train"]) < a.min_train:
            t = time.monotonic()
            if t >= data_end:
                raise TimeoutError("Minimum train data incomplete; resume with longer data allowance")
            r = builder.add("train", a.batch, data_end)
            builder.save()
            if r["added"]:
                cache_features(builder.rows["train"], out/"train.npz", ev, a, deadline)
                rates.append(r["added"] / max(time.monotonic()-t, .001))
                projected = int(np.median(rates[-8:]) * max(data_end-time.monotonic(), 0))
                log(a, "data_rate", train=len(builder.rows["train"]), boards_per_s=rates[-1],
                    additional_boards_planned=projected, seconds_left=max(data_end-time.monotonic(), 0))
            if a.smoke and len(builder.rows["train"]) >= a.min_train:
                break
        assert len(builder.rows["train"]) >= a.min_train
        coverage = {k: {str(g): dict(accepted=sum(r["group"] == g for r in rows), unsolved=0)
                         for g in range(3)} for k, rows in builder.rows.items()}
        unsolved = out/"data"/"unsolved.jsonl"
        if unsolved.exists():
            for line in unsolved.read_text().splitlines():
                row = json.loads(line)
                coverage[row["split"]][str(row["group"])]["unsolved"] += 1
        write(frozen, dict(rows={k: len(v) for k, v in builder.rows.items()},
                          attempts=builder.attempts, coverage=coverage, exclusions=builder.manifest,
                          sha256={k: fingerprint(out/(k+".npz")) for k in ("train", "val", "test")}))
    data = {k: dict(np.load(out/(k+".npz"))) for k in ("train", "val")}
    freeze = json.loads(frozen.read_text())
    for k in data:
        assert fingerprint(out/(k+".npz")) == freeze["sha256"][k]
    tr, va = data["train"], data["val"]
    assert len(va["y"]) == a.holdout
    assert len(set(tr["key"]) & set(va["key"])) == 0
    tx, vx = learner_features(tr["x"]), learner_features(va["x"])
    mean, std = tx.mean(0), np.maximum(tx.std(0), .05)
    X, V = (tx-mean)/std, (vx-mean)/std
    from r5_mlx import Trainer, CONFIGS, predictor
    trainer = Trainer(ckpt/"resume", dict(**cfg, data_sha256=freeze["sha256"]), mean, std)
    log(a, "resume", steps=trainer.steps, data_rows=len(tr["y"]),
        validation_groups={k: int(v.sum()) for k, v in group_masks(va["values"], va["group"], va["y"]).items()})
    baselinepath = ckpt/"selected"/"baselines.json"
    if not baselinepath.exists():
        baselines = fit_baselines(tr["x"], tr["y"], va["x"], va["y"], va["values"], va["group"], deadline-2)
        write(baselinepath, baselines)
        log(a, "baseline_tuning", trials=len(baselines["trials"]),
            rule_score=baselines["rule"]["score"], logistic_score=baselines["logistic"]["score"])
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    rounds = 0
    # Measured step rate determines work; no small fixed step cap can end a run early.
    # Equal updates for six candidates, selecting on COMPLETE validation every round.
    while time.monotonic() < deadline-2 and not stop[0]:
        t = time.monotonic(); before = sum(trainer.steps)
        losses = []
        chunk = a.steps_per_round
        target = max(trainer.steps)+chunk
        for i in range(len(CONFIGS)):
            losses.append(trainer.train(i, X, tr["y"], target-trainer.steps[i],
                                        min(512, max(32, len(X))), deadline-2))
        # Only equal-update rounds may produce a selection. A partially finished
        # final round is saved for resume; the next round catches every head up.
        if all(s == target for s in trainer.steps):
            for i in range(len(CONFIGS)):
                p = trainer.predict(i, V)
                score = macro_nll(va["y"], p, va["values"], va["group"])
                trainer.select(i, score, ckpt/"selected")
        trainer.save()
        rounds += 1
        rate = (sum(trainer.steps)-before)/max(time.monotonic()-t, .001)
        log(a, "train", steps=trainer.steps, losses=losses,
            best_full_validation_macro_nll=trainer.best if np.isfinite(trainer.best) else None,
            steps_per_s=rate, planned_remaining_steps=int(rate*max(deadline-time.monotonic()-2, 0)))
        # Pollable, written only after a complete checkpoint; smoke SIGKILLs here.
        write(out/"checkpoint_ready.json", dict(steps=trainer.steps))
    trainer.save()
    if not (ckpt/"selected"/"head.json").exists():
        raise RuntimeError("No selected model: resume training before evaluation")
    learned = predictor(ckpt/"selected")
    p = learned(va["x"])
    write(out/"validation.json", {k: calibration(va["y"][m], p[m]) if m.any() else dict(n=0)
                                 for k, m in group_masks(va["values"], va["group"], va["y"]).items()})
    log(a, "train_complete", seconds=time.monotonic()-start, steps=trainer.steps, rounds_this_call=rounds)


def decision(pos, mask, name, regime, ev, learned, baselines, a, seconds=None):
    t = time.monotonic()
    capacity = a.fixed if name == "fixed" else (a.high if regime == "sim" else getattr(a, "time_capacity", max(a.high, 1024)))
    s = Search(pos, mask, ev, capacity=capacity, c_puct=getattr(a, "c_puct", 1.5))
    s.advance(np.full(s.n, a.low))
    initial = s.moves()
    p = None
    if name == "fixed":
        s.advance(np.full(s.n, a.fixed))
    else:
        depths = ((2, 4, 6, 8)[baselines["rule"]["depth"]],) if name == "rule" else ()
        x, _, _ = features(pos, mask, s, ab_depths=depths)
        p = scores(name, x, baselines, learned)
        s.sort(p)
        floor = getattr(a, "allocation_floor", a.low)
        if regime == "sim":
            targets = allocation(p, floor, a.high, s.n*a.fixed)
            s.advance(targets[s.ids])
        else:
            # Same measured end-to-end block allowance as fixed. Overhead counts.
            # Priority fraction is fixed by the sim budget, never by test labels.
            if floor != a.low:
                s.advance(np.full(s.n, floor), deadline=t+seconds)
            k = max(1, int(np.ceil(s.n*(a.fixed-floor)/(a.high-floor))))
            target = np.full(s.n, floor); target[:k] = s.capacity
            s.advance(target, deadline=t+seconds)
    move = s.moves()
    elapsed = time.monotonic()-t
    return dict(move=move, initial=initial, seconds=elapsed, simulations=int(s.count.sum()),
                per_position_sims=s.count[np.argsort(s.ids)], probability=p,
                leaf_evals=int(s.pool.n_evals),
                time_ratio=elapsed/seconds if seconds else None)


def evaluate(a):
    start = time.monotonic(); deadline = start + 60*a.max_minutes
    out, ckpt = setup(a)
    config = json.loads((out/"config.json").read_text())
    for key in ("low", "fixed", "high", "backbone", "holdout", "batch"):
        assert getattr(a, key) == config[key], f"Evaluation config mismatch: {key}"
    assert fingerprint(a.backbone+".safetensors") == config["backbone_sha256"]
    assert fingerprint(a.backbone+".json") == config["backbone_config_sha256"]
    freeze = json.loads((out/"frozen.json").read_text())
    assert fingerprint(out/"test.npz") == freeze["sha256"]["test"]
    from r5_mlx import predictor
    ev = evaluator(a); learned = predictor(ckpt/"selected")
    baselines = json.loads((ckpt/"selected"/"baselines.json").read_text())
    selected_hash = fingerprint(ckpt/"selected"/"head.safetensors")
    baseline_hash = fingerprint(ckpt/"selected"/"baselines.json")
    test = dict(np.load(out/"test.npz"))
    assert len(test["y"]) == a.holdout
    # Warm both engine and head before measuring. Test labels are not used here.
    for power in range(3, max(3, (a.batch-1).bit_length())+1):
        ev(np.zeros((2**power, 6, 7, 2), np.float32))
    for name in METHODS:
        decision(test["pos"][:a.batch], test["mask"][:a.batch], name, "sim", ev, learned, baselines, a)
    resultpath = out/"evaluation.json"
    if resultpath.exists():
        saved = json.loads(resultpath.read_text())
        assert saved["complete"] and saved["test_sha256"] == freeze["sha256"]["test"]
        assert saved["selected_sha256"] == selected_hash and saved["baselines_sha256"] == baseline_hash, "Selected artifacts changed since evaluation"
        # Resume only remaining game/demo work. Never rescore or silently replace a headline.
        finish_eval(a, ev, learned, baselines, out, start, deadline)
        return
    records, totals = [], {}
    rng = np.random.default_rng(50241103)
    order = rng.permutation(len(test["y"]))
    for begin in range(0, len(order), a.batch):
        if time.monotonic() >= deadline:
            write(out/"evaluation_partial.json", dict(records=records, complete=False))
            raise TimeoutError("Evaluation unfinished; partial output is not a headline")
        idx = order[begin:begin+a.batch]
        args = (test["pos"][idx], test["mask"][idx])
        # Repeat fixed timing on this exact block. Median is a measured comparator,
        # no interpolation from a different batch size / machine.
        fixed_runs = [decision(*args, "fixed", "sim", ev, learned, baselines, a)
                      for _ in range(1 if a.smoke else 3)]
        fixed = sorted(fixed_runs, key=lambda r: r["seconds"])[len(fixed_runs)//2]
        for regime in ("sim", "time"):
            methods = list(rng.permutation(METHODS[1:])) + ["fixed"]
            for name in methods:
                r = fixed if name == "fixed" else decision(*args, name, regime, ev, learned, baselines, a, fixed["seconds"])
                key = regime+"/"+name
                totals.setdefault(key, []).append((idx, r))
                records.append(dict(block=begin//a.batch, method=name, regime=regime,
                                    seconds=r["seconds"], fixed_seconds=fixed["seconds"],
                                    simulations=r["simulations"], leaf_evals=r["leaf_evals"],
                                    indices=idx.tolist(), moves=r["move"].tolist(),
                                    per_position_sims=r["per_position_sims"].tolist()))
        log(a, "evaluation_block", positions=min(begin+a.batch, len(order)), total=len(order))
    summary = {}
    for key, rows in totals.items():
        moves = np.zeros(len(order), int); initial = moves.copy()
        sims = 0; elapsed = 0.
        for idx, r in rows:
            moves[idx], initial[idx] = r["move"], r["initial"]
            sims += r["simulations"]; elapsed += r["seconds"]
        summary[key] = report_moves(test["values"], moves, initial, sims, elapsed)
        summary[key]["groups"] = {}
        for g, take in group_masks(test["values"], test["group"], test["y"]).items():
            summary[key]["groups"][g] = dict(n=int(take.sum()),
                mistakes=int(mistakes(test["values"][take], moves[take]).sum()) if take.any() else 0)
        if key.startswith("time/"):
            ref = sum(r["seconds"] for _, r in totals["time/fixed"])
            ratio = elapsed/ref
            summary[key].update(time_ratio=ratio, matched_within_10_percent=abs(ratio-1) <= .10)
        else:
            assert sims == len(order)*a.fixed
    probs = {name: scores(name, test["x"], baselines, learned) for name in ("learned", "rule", "logistic")}
    cal = {name: {g: calibration(test["y"][take], p[take]) if take.any() else dict(n=0)
                  for g, take in group_masks(test["values"], test["group"], test["y"]).items()}
           for name, p in probs.items()}
    write(resultpath, dict(complete=True, summary=summary, records=records, calibration=cal,
                          selected_sha256=selected_hash, baselines_sha256=baseline_hash,
                          selected_metadata=json.loads((ckpt/"selected"/"head.json").read_text()),
                          solver_coverage=freeze["coverage"],
                          policy_mistakes=int(mistakes(test["values"], test["policy"]).sum()),
                          low_probe_mistakes=int(test["y"].sum()),
                          config=config, test_sha256=freeze["sha256"]["test"],
                          units="Batched decision throughput, NOT single-game latency. All gate overhead charged.",
                          population="Fresh unique boards fully solved within 3M nodes; see unsolved.jsonl for exclusions."))
    finish_eval(a, ev, learned, baselines, out, start, deadline)


def secondary(a, ev, learned, baselines, out, deadline):
    """Full old solver split, explicitly secondary and known to overlap old replay."""
    path = out/"secondary.json"
    if path.exists() and json.loads(path.read_text()).get("complete"):
        return
    old = json.loads(Path("results/solver/positions.json").read_text())["sets"]
    solved = {}
    for line in Path("results/solver/solved.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "position" and "values" in r:
            solved[r["id"]] = r["values"]
    summary, records = {}, []
    for group, gameset in old.items():
        if a.smoke:
            gameset = gameset[:3]
        for begin in range(0, len(gameset), a.batch):
            if time.monotonic() >= deadline:
                write(path, dict(complete=False, records=records))
                return
            batch = gameset[begin:begin+a.batch]
            pm = np.asarray([c4.from_moves(x) for x in batch])
            vals = np.asarray([solved[f"{group}/{i}"] for i in range(begin, begin+len(batch))])
            assert np.isin(vals, [-999, -1, 0, 1]).all()
            for name in METHODS:
                r = decision(pm[:, 0], pm[:, 1], name, "sim", ev, learned, baselines, a)
                rec = dict(set=group, begin=begin, method=name, moves=r["move"].tolist(),
                           **report_moves(vals, r["move"], r["initial"], r["simulations"], r["seconds"]))
                records.append(rec)
                key = group+"/"+name
                summary.setdefault(key, dict(n=0, mistakes=0, simulations=0, seconds=0.))
                for field in summary[key]:
                    summary[key][field] += rec[field]
    write(path, dict(complete=True, summary=summary, records=records,
                    note="Old solver split (3 per set for smoke, otherwise all 1000); known old-replay overlap, no tuning."))


def finish_eval(a, ev, learned, baselines, out, start, deadline):
    secondary(a, ev, learned, baselines, out, deadline-5)
    # Spend remaining evaluation allowance on fresh colour-swapped game batches.
    # Both opponents are selected by validation, never by these test results.
    classical = min(("rule", "logistic"), key=lambda k: baselines[k]["score"])
    gamepath = out/"games.json"
    game_records = json.loads(gamepath.read_text())["records"] if gamepath.exists() else []
    cycle = len(game_records)
    # A separate seed for each paired batch gives both allocators identical starts,
    # and recreates the next batch exactly after resume.
    while time.monotonic() < deadline-2:
        name = ("learned", classical)[cycle % 2]
        rng = np.random.default_rng(50241201+cycle//2)
        try:
            g = games(a, ev, learned, baselines, name, rng, deadline, pairs=1 if a.smoke else 16)
        except TimeoutError:
            break
        g["paired_batch"] = cycle//2
        game_records.append(g); cycle += 1
        write(out/"games.json", dict(records=game_records, seconds=time.monotonic()-start,
              protocol="Fresh starts, both colours; equal mean simulations per decision, actual totals reported; incomplete pairs excluded"))
        log(a, "games", method=name, pairs=g["completed_pairs"], unfinished=g["unfinished_games"])
        if g["unfinished_games"] or (a.smoke and cycle >= 2):
            break
    demo(a)
    log(a, "evaluation_complete", seconds=time.monotonic()-start, game_batches=len(game_records))


def games(a, ev, learned, baselines, name, rng, deadline, pairs):
    starts = [candidate(rng, 0, deadline) for _ in range(pairs)]
    pos = np.repeat([s["pos"] for s in starts], 2).astype(np.int64)
    mask = np.repeat([s["mask"] for s in starts], 2).astype(np.int64)
    a_first = np.arange(2*pairs) % 2 == 0
    # a_first here means adaptive moves first FROM THE START POSITION.
    turn = np.ones(2*pairs, bool)
    outcome = np.full(2*pairs, -2)
    traces = [[] for _ in pos]
    totals = {name: dict(simulations=0, decisions=0, seconds=0.), "fixed": dict(simulations=0, decisions=0, seconds=0.)}
    while (outcome == -2).any() and time.monotonic() < deadline:
        alive = outcome == -2
        who = turn == a_first
        for is_a in (True, False):
            ix = np.flatnonzero(alive & (who == is_a))
            if not len(ix):
                continue
            method = name if is_a else "fixed"
            r = decision(pos[ix], mask[ix], method, "sim", ev, learned, baselines, a)
            for k in ("simulations", "seconds"):
                totals[method][k] += r[k]
            totals[method]["decisions"] += len(ix)
            for i, move in zip(ix, r["move"]):
                won = c4.winning_move(pos[i], mask[i], int(move))
                pos[i], mask[i] = c4.play(pos[i], mask[i], int(move))
                traces[i].append(int(move))
                if won:
                    outcome[i] = 1 if is_a else -1
                elif c4.is_full(mask[i]):
                    outcome[i] = 0
            turn[ix] = ~turn[ix]
    complete_pairs = (outcome.reshape(-1, 2) != -2).all(1)
    keep = np.repeat(complete_pairs, 2)
    results = outcome[keep]
    return dict(method=name, completed_pairs=int(complete_pairs.sum()), unfinished_games=int((outcome == -2).sum()),
                wins=int((results == 1).sum()), draws=int((results == 0).sum()), losses=int((results == -1).sum()),
                score=float(np.mean((results+1)/2)) if len(results) else None,
                starts=[s["moves"] for s in starts], moves=traces, results=outcome.tolist(), costs=totals)


def demo(a):
    """Offline report with real decision allocations and preserved failure boards."""
    import html
    out, _ = setup(a)
    report = json.loads((out/"evaluation.json").read_text())
    test = dict(np.load(out/"test.npz"))
    body = ["<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>When to think longer</title>",
            "<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:16px;background:#142232;color:#eee}td,th{padding:8px;text-align:right}svg{margin:8px}button{font:inherit}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}caption{text-align:left}</style></head><body>",
            "<h1>When to think longer</h1><p>Measured batched decisions. Negative avoided counts mean search introduced mistakes. Time matches outside ±10% are unqualified.</p>",
            "<table><tr><th>Method</th><th>Mistakes</th><th>Net avoided</th><th>ms</th><th>Simulations</th><th>Time match</th></tr>"]
    for key, r in report["summary"].items():
        body.append(f"<tr><td>{html.escape(key)}</td><td>{r['mistakes']}</td><td>{r['net_avoided']}</td><td>{1000*r['seconds']:.1f}</td><td>{r['simulations']}</td><td>{r.get('matched_within_10_percent', '—')}</td></tr>")
    low = report["config"].get("low", 8)
    body.append(f"</table><p>Counts and milliseconds are totals over the displayed test set. Net avoided is relative to each gate's {low}-simulation probe, not relative to fixed search. These are throughput measurements, not single-game latency.</p>")
    if report.get("note"):
        body.append("<p>"+html.escape(report["note"])+"</p>")
    if "calibration" in report:
        body.append("<h2>Mistake-predictor calibration</h2><table><tr><th>Signal</th><th>Group / N</th><th>NLL</th><th>Brier</th><th>ECE</th></tr>")
        for method, groups in report["calibration"].items():
            for group in ("all", "drawn"):
                r = groups[group]
                if r["n"]:
                    body.append(f"<tr><td>{html.escape(method)}</td><td>{group} / {r['n']}</td><td>{r['nll']:.4f}</td><td>{r['brier']:.4f}</td><td>{r['ece']:.4f}</td></tr>")
        body.append("</table>")
    if (out/"games.json").exists():
        from r5_analyze import game_summary
        gs = game_summary(json.loads((out/"games.json").read_text())["records"],
                          mean_simulations=report["config"].get("fixed", 32))
        body.append("<h2>Games against fixed search</h2><p>Equal mean simulations per decision; actual total time differs. Colours swapped at each start.</p><table><tr><th>Allocator</th><th>W / D / L</th><th>Score</th><th>Time / fixed</th></tr>")
        for method, g in gs.items():
            score = f"{g['score']:.4f}" if g['score'] is not None else "unmeasured"
            ratio = f"{g['time_ratio']:.2f}×" if g['time_ratio'] is not None else "unmeasured"
            body.append(f"<tr><td>{html.escape(method)}</td><td>{g['wins']} / {g['draws']} / {g['losses']}</td><td>{score}</td><td>{ratio}</td></tr>")
        body.append("</table>")
    body.append("<h2>Learned allocation: mistakes retained for inspection</h2><p>Yellow: player to move; red: opponent. Columns are numbered 1–7 from the left. Exact move values: +1 win, 0 draw, −1 loss, −999 full column.</p>")
    shown = 0
    for r in report["records"]:
        if r["method"] != "learned" or r["regime"] != "sim":
            continue
        for i, move, sims in zip(r["indices"], r["moves"], r["per_position_sims"]):
            if shown >= 12 or not mistakes(test["values"][[i]], np.array([move]))[0]:
                continue
            shown += 1
            p, m = int(test["pos"][i]), int(test["mask"][i])
            svg = ["<svg viewBox='0 0 280 240' width='280' aria-label='Failure board'>"]
            for c in range(7):
                for row in range(6):
                    bit = 1 << (7*c+row)
                    fill = '#ffce54' if p & bit else ('#f05d5e' if m & bit else '#304357')
                    svg.append(f"<circle cx='{20+40*c}' cy='{220-40*row}' r='16' fill='{fill}'/>")
            svg.append("</svg>")
            body.append(f"<details><summary>Position {i}: move {move+1}, {sims} sims; true move values {test['values'][i].tolist()}</summary>{''.join(svg)}</details>")
    if not shown:
        body.append("<p>No learned simulation-arm mistakes in this measured sample.</p>")
    body.append("<h2>Provenance</h2><pre>"+html.escape(json.dumps(dict(config=report["config"], test_sha256=report["test_sha256"]), indent=2))+"</pre>")
    body.append("</body></html>")
    target = Path(getattr(a, "demo_out", None) or f"media/r5_{a.run}.html")
    assert target.parent == Path("media") and target.name.startswith("r5_") and target.suffix == ".html"
    target.write_text("\n".join(body))
    print(json.dumps(dict(event="demo_written", path=str(target), bytes=target.stat().st_size,
                          positions=len(test["values"]), methods=len(report["summary"]), failure_boards=shown)), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("train", "evaluate", "demo"))
    p.add_argument("--run", default="main")
    p.add_argument("--backbone", default="checkpoints/main/gen_191")
    p.add_argument("--max-minutes", type=float, default=24)
    p.add_argument("--data-minutes", type=float)
    p.add_argument("--holdout", type=int, default=384)
    p.add_argument("--min-train", type=int, default=1024)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--low", type=int, default=8)
    p.add_argument("--fixed", type=int, default=32)
    p.add_argument("--high", type=int, default=128)
    p.add_argument("--steps-per-round", type=int, default=128)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--demo-out", help="Optional media/r5_*.html output; use a new name to preserve earlier renders")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    {"train": train, "evaluate": evaluate, "demo": demo}[args.command](args)
