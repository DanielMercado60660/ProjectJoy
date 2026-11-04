"""Print the markdown tables used in RESULTS_WSP.md from the saved JSON (no simulation).
    python wsp/report_wsp.py > wsp/results/tables.md
"""
import glob
import json
import os

WSP = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "round5_logs"))  # local Astra logs, not in the public repo


def usage(tag):
    p = f"{LOGS}/08_tiny_grand_prix_{tag}.jsonl"
    if not os.path.exists(p):
        return None, None
    th, u = None, None
    for l in open(p):
        try:
            e = json.loads(l)
        except Exception:
            continue
        if e.get("type") == "thread.started":
            th = e.get("thread_id")
        if e.get("type") == "turn.completed":
            u = e["usage"]
    return th, u


def attempts():
    print("### Astra turns\n")
    print("| tag | thread | calls used / budget | best call | input tok (cached) | output tok (reasoning) |")
    print("|---|---|---|---|---|---|")
    for tag in ["r6wsp_t0_smoke", "r6wsp_read_a1", "r6wsp_read_a2", "r6wsp_read_a3", "r6wsp_read_catch", "r6wsp_edit_a1", "r6wsp_edit_a2",
                "r6wsp_edit0_a1", "r6wsp_edit0_a2", "r6wsp_grip_a1", "r6wsp_grip_a2"]:
        th, u = usage(tag)
        d = os.path.join(WSP, tag.replace("r6wsp_", ""))
        calls = "-"
        best = "-"
        if os.path.exists(os.path.join(d, "budget_config.json")):
            cfg = json.load(open(os.path.join(d, "budget_config.json")))
            n = sum(1 for _ in open(os.path.join(d, "eval_log.jsonl"))) if os.path.exists(os.path.join(d, "eval_log.jsonl")) else 0
            calls = f"{n} / {cfg['budget']}"
            if os.path.exists(os.path.join(d, "best.json")):
                best = json.load(open(os.path.join(d, "best.json")))["call"]
        ut = f"{u['input_tokens']:,} ({u['cached_input_tokens']:,})" if u else "-"
        ot = f"{u['output_tokens']:,} ({u['reasoning_output_tokens']:,})" if u else "-"
        print(f"| {tag} | {th} | {calls} | {best} | {ut} | {ot} |")
    print()


def ledger(name):
    p = os.path.join(WSP, name, "eval_log.jsonl")
    print(f"#### {name}\n")
    print("| call | key | improved | strict done | strict lap | soft wall steps/lap | note |")
    print("|---|---|---|---|---|---|---|")
    for l in open(p):
        r = json.loads(l)
        s = r["summaries"]
        k = [round(x, 3) for x in r["key"]]
        if "soft_g1.0" in s:
            st, so = s["strict_g1.0"], s["soft_g1.0"]
            print(f"| {r['call']} | {k} | {r['improved']} | {st['completion']:.3f} | {st['mean_lap_time'] or float('nan'):.2f} | "
                  f"{so['wall_contact_steps_per_lap']:.2f} | {r['note'][:140].replace('|', '/')} |")
        else:
            s5, s1 = s["strict_g0.5"], s["strict_g1.0"]
            print(f"| {r['call']} | {k} | {r['improved']} | g0.5 {s5['completion']:.3f} / g1.0 {s1['completion']:.3f} | "
                  f"g0.5 {s5['mean_lap_time'] or float('nan'):.2f} / g1.0 {s1['mean_lap_time'] or float('nan'):.2f} | - | {r['note'][:140].replace('|', '/')} |")
    print()


def heldout(path, keys):
    out = json.load(open(path))
    print(f"seeds: {out['seeds']}\n")
    print("| policy | " + " | ".join(keys) + " | weights changed | rel. change | cos(change, es_best - start) |")
    print("|---|" + "---|" * (len(keys) + 3))
    for n, v in out["results"].items():
        cells = []
        for k in keys:
            s = v[k]
            lap = s["mean_lap_time"]
            cells.append(f"{100 * s['completion']:.1f}% / {lap:.2f} s" if lap == lap else f"{100 * s['completion']:.1f}% / -")
        w = v["weights"]
        ws = (f"{100 * w['frac_changed']:.1f}% | {w['rel_change']:.2f} | {w['cos_change_vs_es_best_direction']:+.2f}" if w else "- | - | -")
        print(f"| {n} | " + " | ".join(cells) + f" | {ws} |")
    print()


if __name__ == "__main__":
    attempts()
    for n in ["edit_a1", "edit_a2", "grip_a1", "grip_a2"]:
        if os.path.exists(os.path.join(WSP, n, "eval_log.jsonl")):
            ledger(n)
    for p in sorted(glob.glob(os.path.join(WSP, "results", "heldout_*.json"))):
        print(f"### {os.path.basename(p)}\n")
        task = json.load(open(p))["task"]
        keys = (["heldout_strict_g1.0", "hard_strict_g1.0", "heldout_soft_g1.0"] if task == "hack"
                else ["heldout_strict_g0.5", "heldout_strict_g1.0", "hard_strict_g0.5", "hard_strict_g1.0"])
        heldout(p, keys)
