"""Reconcile Claude's immutable R5 receipts; NumPy only, no new evaluation.

Outputs are derived turn-2 artifacts. Brier gains pool pair counts, while wait
fractions pool steps. Repeated budgets are repeated caves, not independent caves.
"""
import json
from pathlib import Path
import numpy as np
from env_r5 import GROUPS, BUDGETS
from eval_r5 import summarize, selection_score
from io_r5 import file_digest, load_checkpoint, write_json

RUNS = {51: ("test.json", "main"), 52: ("test_seed52.json", "seed52"),
        53: ("test_seed53.json", "seed53")}
RANGES = dict(threshold=(.16, .60), wall_threshold=(.30, .65), temperature=(.05, .24),
              decay=(.02, .30), blend=(.3, 1.), interval=(1, 6), uncertainty=(.25, .95),
              reserve=(.05, 1.), caution=(.2, .7), repeat=(0, 3), explore=(.2, 4.), moth=(.12, .5))


def aggregate(episodes):
    n = len(episodes)
    total = lambda k: sum(e[k] for e in episodes)
    result = dict(n=n, successes=total("success"), completion=total("success")/n,
                  collisions=total("collisions")/n, steps=total("steps")/n, pulses=total("pulses")/n,
                  collision_rate=total("collisions")/total("steps"),
                  wait_fraction=total("wait_steps")/total("steps"),
                  unused_budget_timeouts=sum(e["timeout"] and e["pulses"] < e["budget"] for e in episodes),
                  every_available_step_pulsed=all(e["pulses"] == min(e["steps"], e["budget"]) for e in episodes))
    for key in ("wait_steps", "wait_listens", "double_listens", "turn_listens", "translate_listens"):
        result[key] = total(key)
        result[key+"_per_episode"] = total(key)/n
        result[key+"_episodes"] = sum(e[key] > 0 for e in episodes)
    for key in ("repeat", "turn", "translate"):
        pairs = total(key+"_pairs")
        result[key+"_pairs"] = pairs
        result[key+"_brier_gain"] = total(key+"_brier_gain")/pairs if pairs else None
        result[key+"_improvement_fraction"] = total(key+"_improved")/pairs if pairs else None
    return result


def check_grid(rows, n):
    assert len(rows) == 16
    assert {(r["group"], r["budget"]) for r in rows} == {(g, b) for g in GROUPS for b in BUDGETS}
    assert all(r["n"] == n for r in rows)


def main():
    sources = {}
    def read(path, lines=False):
        sources[str(path)] = file_digest(path)
        text = Path(path).read_text()
        return [json.loads(x) for x in text.splitlines()] if lines else json.loads(text)
    baseline = read("results/r5/baselines.json")
    assert baseline["complete"]
    trials = {m: [r for r in baseline["trials"] if r["mode"] == m] for m in ("event", "uncertainty")}
    for records in trials.values():
        assert len(records) == baseline["candidate_pairs"]
        for r in records:
            check_grid(r["rows"], 32)
            assert selection_score(r["rows"]) == r["score"]
    for a, b in zip(trials["event"], trials["uncertainty"]):
        assert a["candidate"] == b["candidate"] and a["config"] == b["config"]
    audit = dict(candidate_pairs=baseline["candidate_pairs"], seconds=baseline["seconds"],
                 episode_evaluations=sum(sum(r["n"] for r in t["rows"]) for t in baseline["trials"]),
                 identical_paired_rows=sum(a["rows"] == b["rows"] for a, b in zip(*trials.values())),
                 selected=baseline["best"], numeric_search_ranges=RANGES,
                 selected_range_positions={k: (baseline["best"]["event"]["config"][k]-lo)/(hi-lo)
                                           for k, (lo, hi) in RANGES.items()})
    data, training = {}, {}
    for seed, (filename, directory) in RUNS.items():
        d = read(Path("results/r5")/filename)
        assert d["complete"] and d["manifest"]["split"] == "test"
        assert d["baseline_sha256"] == sources["results/r5/baselines.json"]
        sources[d["checkpoint"]] = file_digest(d["checkpoint"])
        assert d["checkpoint_sha256"] == sources[d["checkpoint"]]
        selected = load_checkpoint(d["checkpoint"])
        assert selected["update"] == d["selected_update"] and selected["config"]["seed"] == seed
        for result in d["methods"].values():
            assert result["complete"]
            check_grid(result["rows"], 128)
            assert len(result["episodes"]) == 2048
            assert len({(e["layout"], e["budget"]) for e in result["episodes"]}) == 2048
            assert summarize(result["episodes"]) == result["rows"]
            assert selection_score(result["rows"]) == result["score"]
        data[seed] = d
        path = Path("logs/r5")/directory
        log, val, summary = read(path/"train.jsonl", True), read(path/"validation.jsonl", True), read(path/"summary.json")
        assert [r["update"] for r in log] == list(range(1, summary["update"]+1))
        finite_keys = ("loss", "entropy", "grad_norm")
        assert np.isfinite([[r[k] for k in finite_keys] for r in log]).all()
        for r in val:
            check_grid(r["rows"], 32)
        windows = {}
        for lo, hi in ((101, 500), (501, 1000), (1001, 2000), (2001, 4000), (4001, 7000), (7001, 10000), (10001, 14000)):
            subset = [r for r in log if lo <= r["update"] <= hi]
            if subset:
                windows[f"{lo}-{min(hi, summary['update'])}"] = dict(n=len(subset), entropy_median=float(np.median([r["entropy"] for r in subset])),
                     completion_mean=float(np.mean([r.get("train_completion_last400", 0) for r in subset])))
        training[seed] = dict(summary=summary, selected_update=selected["update"],
                             selected_seconds=selected["total_seconds"], env_steps=log[-1]["env_steps"],
                             optimizer_steps=log[-1]["optimizer_step"], all_loss_entropy_grad_finite=True,
                             aggregate_updates_per_second=summary["update"]/summary["seconds"],
                             selected_val_completion=float(np.mean([r["completion"] for r in selected["validation"]["rows"]])),
                             entropy_min=min(r["entropy"] for r in log), last_record=log[-1], windows=windows,
                             validation=[dict(update=r["update"], completion=float(np.mean([x["completion"] for x in r["rows"]]))) for r in val])
    for d in data.values():
        assert d["manifest"] == data[51]["manifest"]
        assert d["methods"]["event"]["episodes"] == d["methods"]["uncertainty"]["episodes"]
        for mode in ("event", "uncertainty", "every"):
            assert d["methods"][mode]["episodes"] == data[51]["methods"][mode]["episodes"]
    names = {"event/uncertainty": data[51]["methods"]["event"], "every": data[51]["methods"]["every"]}
    names.update({f"learned_{seed}": d["methods"]["learned"] for seed, d in data.items()})
    ablations = {str(seed): aggregate(d["methods"]["learned_every"]["episodes"]) for seed, d in data.items()}
    macros = {name: aggregate(result["episodes"]) for name, result in names.items()}
    groups = {name: {g: aggregate([e for e in result["episodes"] if e["group"] == g]) for g in GROUPS}
              for name, result in names.items()}
    demo = read("media/r5_main.json")
    assert demo["checkpoint_sha256"] == data[51]["checkpoint_sha256"]
    demo_cases = [e for e in data[51]["methods"]["learned"]["episodes"] if e["group"] == "hard" and e["budget"] == 16]
    assert len(demo_cases) == 128 and sum(e["success"] for e in demo_cases) == 0
    assert demo["absent"] == ["success"] and demo["clips"][0]["episode"] == demo_cases[0]
    legacy = read("results/r5/legacy.json")
    assert legacy["checkpoint_sha256"] == data[51]["checkpoint_sha256"]
    legacy_summary = {name: aggregate(result["episodes"]) for name, result in legacy["methods"].items()}
    out = dict(sources_sha256=sources, baseline_audit=audit, training=training, macro=macros, groups=groups,
               learned_every=ablations, legacy_seed51=legacy_summary,
               demo=dict(eligible_successes=0, eligible_n=128, absent=demo["absent"], episode=demo["clips"][0]["episode"]),
               learned53_equals_forced_every=data[53]["methods"]["learned"]["episodes"] == data[53]["methods"]["learned_every"]["episodes"],
               learned_cells_below_every_classical=all(
                   r["completion"] < c["completion"] for d in data.values() for mode in ("event", "uncertainty", "every")
                   for r, c in zip(d["methods"]["learned"]["rows"], d["methods"][mode]["rows"])))
    write_json("results/r5/turn2_analysis.json", out)
    lines = ["# Round 5 turn 2: reconciled tables", "",
             "Generated by `analyze_r5.py` from `test.json` (51), `test_seed52.json`, `test_seed53.json`,",
             "`baselines.json`, and `../../logs/r5/{main,seed52,seed53}/`. Exact values and SHA-256 sources: [turn2_analysis.json](turn2_analysis.json).",
             "All test rows recomputed from episodes; 128 caves per group, each reused across four budgets. All 16 cells included.",
             "Event and uncertainty have identical episodes in all three files, so E/U denotes BOTH, counted once. Baselines repeat exactly across seeds.", "",
             "## Full sweep", "", "Each entry: **completion % / collisions per episode / pulses per episode**. 128 episodes per cell.", "",
             "| Group | Budget | E/U | Every | Learned 51 | Learned 52 | Learned 53 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    cells = {name: {(r["group"], r["budget"]): r for r in result["rows"]} for name, result in names.items()}
    for g in GROUPS:
        for b in BUDGETS:
            entries = [f"{r['completion']*100:.2f} / {r['collisions']:.2f} / {r['pulses']:.2f}"
                       for r in [v[g, b] for v in cells.values()]]
            lines.append(f"| {g} | {b} | " + " | ".join(entries) + " |")
    lines += ["", "## Pooled behavior across budgets", "", "512 episodes per group. Wait % is total WAIT steps / total steps; other counts are per episode.", "",
              "| Method | Group | Wait % | Wait+listen | Double listen | Turn+listen | Translate+listen |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name, group in groups.items():
        for g, a in group.items():
            lines.append(f"| {name} | {g} | {100*a['wait_fraction']:.2f} | {a['wait_listens_per_episode']:.3f} | {a['double_listens_per_episode']:.3f} | {a['turn_listens_per_episode']:.3f} | {a['translate_listens_per_episode']:.3f} |")
    lines += ["", "## Echo decoder Brier gains", "", "Entries: **mean gain (pairs; % improved)**, pooling gain sums / pair counts. Positive is lower error.",
              "Repeat means same position/heading within two steps (double is exactly consecutive); failed movement can qualify.",
              "Turn compares same-position, changed-heading echoes; translation compares the same edge from opposite sides.",
              "These are sensor-decoder diagnostics, not neural beliefs or causal evidence that the action helped completion.", "",
              "| Method | Group | Repeat | Turn | Translate |", "|---|---|---:|---:|---:|"]
    for name, group in groups.items():
        for g, a in group.items():
            entries = [f"{a[k+'_brier_gain']:+.5f} ({a[k+'_pairs']}; {100*a[k+'_improvement_fraction']:.1f}%)"
                       if a[k+"_pairs"] else "— (0)" for k in ("repeat", "turn", "translate")]
            lines.append(f"| {name} | {g} | " + " | ".join(entries) + " |")
    lines += ["", "## Forced-every-step sensing ablation", "", "Same learned flight weights; pulse head overridden, capped by budget. Entries: completion % / collisions / pulses.", "",
              "| Group | Budget | Seed 51 forced | Seed 52 forced | Seed 53 forced |", "|---|---:|---:|---:|---:|"]
    for g in GROUPS:
        for b in BUDGETS:
            entries = []
            for d in data.values():
                r = next(r for r in d["methods"]["learned_every"]["rows"] if r["group"] == g and r["budget"] == b)
                entries.append(f"{100*r['completion']:.2f} / {r['collisions']:.2f} / {r['pulses']:.2f}")
            lines.append(f"| {g} | {b} | " + " | ".join(entries) + " |")
    Path("results/r5/turn2_tables.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(audits="PASS", macro={k: {f: v[f] for f in ('completion', 'collisions', 'pulses')} for k, v in macros.items()},
                          training_rates={s: t["aggregate_updates_per_second"] for s, t in training.items()},
                          learned53_equals_forced_every=out["learned53_equals_forced_every"],
                          learned_cells_below_every_classical=out["learned_cells_below_every_classical"]), indent=2))


if __name__ == "__main__":
    main()
